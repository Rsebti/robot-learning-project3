#!/usr/bin/env python3
"""Live wrist-cam + joint state + MJCF FK overlay for picking a reset pose.

Connects the follower with torque OFF so you can move the arm freely by
hand, opens the gripper-cam (cv2 index 0), and overlays the current
joint angles + EE xyz (in ggand0's MJCF frame, ready to paste into
yellow_v1_*.json's ik_reset_ee_pos field).

Controls:
  s   save the current EE position. Prints the JSON line and appends
      to outputs/reset_pose_candidates.jsonl (with timestamp).
  q   quit.

Run from the ggand0 venv:
  cd ~/Desktop/eval-ggand0/hil-serl-so101 && uv run python \
    /Users/admin/Documents/ETH/M4/Robot\\ Learning\\ /Project\\ S101/robot-learning-project3/sim/hilserl/scripts/probe_reset_pose.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import numpy as np

from lerobot.robots.so101_follower.config_so101_follower import (
    SO101FollowerConfig,
)
from lerobot.robots.so101_follower.so101_follower import SO101Follower

REPO_ROOT = Path(
    "/Users/admin/Documents/ETH/M4/Robot Learning /Project S101/robot-learning-project3"
)
MJCF = str(REPO_ROOT / "sim/hilserl/assets/so101_mjcf/so101_new_calib.xml")
OUT_FILE = REPO_ROOT / "outputs/reset_pose_candidates.jsonl"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# EE workspace bounds — from our lerobot-find-joint-limits measurement.
BOUNDS_MIN = np.array([0.057, -0.244, -0.035])
BOUNDS_MAX = np.array([0.430, 0.286, 0.248])


def overlay(frame: np.ndarray, lines: list[str], color=(0, 255, 0)) -> None:
    """Draw multi-line text on the BGR frame in-place."""
    y0 = 22
    for i, line in enumerate(lines):
        y = y0 + i * 22
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )


def main() -> int:
    # MuJoCo FK setup
    m = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(m)
    ee_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    # Follower — connect, then immediately disable torque so user can move it.
    cfg = SO101FollowerConfig(
        port="/dev/tty.usbmodem5B141129871",
        id="so101_follower",
        use_degrees=True,
        calibration_dir=Path(
            "/Users/admin/.cache/huggingface/lerobot/calibration/robots/so_follower"
        ),
        cameras={},
        disable_torque_on_disconnect=True,
    )
    r = SO101Follower(cfg)
    r.connect()
    r.bus.sync_write("Torque_Enable", 0, num_retry=3)
    print("[probe] follower connected, torque OFF — free to move by hand.")

    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("[probe] ERROR: could not open wrist camera at index 0")
        r.disconnect()
        return 1
    print("[probe] camera open at index 0. Press 's' to save a pose, 'q' to quit.")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    cv2.namedWindow("reset_pose_probe", cv2.WINDOW_AUTOSIZE)

    saved_count = 0
    last_save_ts = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            # Read joints
            obs = r.get_observation()
            pos_deg = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
            qpos_deg = np.array([pos_deg[f"{j}.pos"] for j in JOINTS])

            # FK
            data.qpos[:5] = np.deg2rad(qpos_deg)
            mujoco.mj_forward(m, data)
            ee = data.site_xpos[ee_id]

            # Bounds check
            inside = bool(np.all(ee >= BOUNDS_MIN) and np.all(ee <= BOUNDS_MAX))
            color = (0, 255, 0) if inside else (0, 0, 255)

            # Overlay
            lines = [
                f"pan {qpos_deg[0]:+6.2f}  lift {qpos_deg[1]:+6.2f}  elb {qpos_deg[2]:+6.2f}",
                f"wfx {qpos_deg[3]:+6.2f}  wrl {qpos_deg[4]:+6.2f}  grp {pos_deg['gripper.pos']:+6.2f}",
                f"EE  x={ee[0]:+.3f}  y={ee[1]:+.3f}  z={ee[2]:+.3f}  m  ({'in' if inside else 'OUT OF'} bounds)",
                f"ik_reset_ee_pos: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]",
                f"[s] save   [q] quit   saved: {saved_count}",
            ]
            overlay(frame, lines, color)

            cv2.imshow("reset_pose_probe", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[probe] quit requested.")
                break
            if key == ord("s") and (time.time() - last_save_ts) > 0.5:
                rec = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "joints_deg": {j: round(float(qpos_deg[i]), 4) for i, j in enumerate(JOINTS)},
                    "gripper_deg": round(pos_deg["gripper.pos"], 4),
                    "ee_xyz_m": [round(float(ee[i]), 4) for i in range(3)],
                    "inside_bounds": inside,
                }
                with OUT_FILE.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(
                    f"[probe] saved #{saved_count + 1}: "
                    f"ik_reset_ee_pos = [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]"
                    f"  (joints: pan={qpos_deg[0]:+.1f} lift={qpos_deg[1]:+.1f} "
                    f"elb={qpos_deg[2]:+.1f} wfx={qpos_deg[3]:+.1f} wrl={qpos_deg[4]:+.1f})"
                )
                saved_count += 1
                last_save_ts = time.time()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        r.disconnect()
        print(f"[probe] disconnected. {saved_count} pose(s) saved to {OUT_FILE}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
