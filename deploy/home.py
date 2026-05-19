"""
home.py - smooth ramp to a named home pose from deploy/homes.py.

No policy, no rollout. Just connect, ramp, hold briefly, disconnect.

Usage:
    python deploy/home.py                                       # eval1_rest (universal)
    python deploy/home.py --home_pose eval1_sac_legacy          # SAC training rest
    python deploy/home.py --port COM3 --camera_index 1 --hold_s 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Windows MSMF -> DSHOW patch
_OriginalVideoCapture = cv2.VideoCapture


class _DShowVideoCapture(_OriginalVideoCapture):
    def __init__(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], int) and args[1] in (
            cv2.CAP_ANY, cv2.CAP_MSMF
        ):
            args = (args[0], cv2.CAP_DSHOW)
        elif len(args) == 1 and isinstance(args[0], int):
            args = (args[0], cv2.CAP_DSHOW)
        super().__init__(*args, **kwargs)


cv2.VideoCapture = _DShowVideoCapture

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from homes import get_home_deg, list_home_poses                        # noqa: E402
from robot_calibration import make_so101_follower_config               # noqa: E402
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.robots.utils import make_robot_from_config                # noqa: E402

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="COM3")
    p.add_argument("--camera_index", type=int, default=1)
    p.add_argument("--home_pose", default="eval1_rest",
                   choices=list_home_poses(),
                   help="Named preset in deploy/homes.py.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max_deg_per_step", type=float, default=0.7,
                   help="Max deg/step ramp speed (smaller = smoother/slower).")
    p.add_argument("--hold_s", type=float, default=1.5,
                   help="Hold time at home before disconnect.")
    p.add_argument("--max_total_s", type=float, default=25.0)
    p.add_argument("--no_camera", action="store_true",
                   help="Skip wrist-camera config (slightly faster connect).")
    args = p.parse_args()

    target = get_home_deg(args.home_pose)
    print(f"[home] target preset {args.home_pose!r}: {target.round(1).tolist()}")

    cameras = (
        {} if args.no_camera
        else {"wrist": OpenCVCameraConfig(
            index_or_path=args.camera_index, fps=30, width=640, height=480)}
    )
    cfg = make_so101_follower_config(args.port, cameras=cameras, use_degrees=True)
    robot = make_robot_from_config(cfg)
    print(f"[home] connecting to {args.port} ...")
    robot.connect()

    try:
        obs = robot.get_observation()
        q_cmd = np.array([float(obs[f"{n}.pos"]) for n in MOTOR_NAMES], dtype=float)
        print(f"[home] start: {q_cmd.round(1).tolist()}")
        period = 1.0 / args.fps
        t0_session = time.time()
        step = 0
        while True:
            err = target - q_cmd
            if np.max(np.abs(err)) < 0.5:
                break
            if time.time() - t0_session > args.max_total_s:
                print(f"[home] timeout @ {q_cmd.round(1).tolist()}")
                break
            q_cmd = q_cmd + np.clip(err, -args.max_deg_per_step, args.max_deg_per_step)
            action = {f"{n}.pos": float(q_cmd[i]) for i, n in enumerate(MOTOR_NAMES)}
            t0 = time.time()
            robot.send_action(action)
            step += 1
            if step % args.fps == 0:
                obs = robot.get_observation()
                q_live = np.array([float(obs[f"{n}.pos"]) for n in MOTOR_NAMES])
                print(f"  t={time.time() - t0_session:5.2f}s  "
                      f"cmd={q_cmd.round(1).tolist()}  live={q_live.round(1).tolist()}")
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

        # Final readout + hold
        obs = robot.get_observation()
        q_final = np.array([float(obs[f"{n}.pos"]) for n in MOTOR_NAMES])
        err = q_final - target
        print(f"\n[home] reached. final: {q_final.round(1).tolist()}  "
              f"err: {err.round(1).tolist()}  max={np.max(np.abs(err)):.1f} deg")

        if args.hold_s > 0:
            print(f"[home] holding {args.hold_s:.1f}s ...")
            t_end = time.time() + args.hold_s
            action = {f"{n}.pos": float(target[i]) for i, n in enumerate(MOTOR_NAMES)}
            while time.time() < t_end:
                robot.send_action(action)
                time.sleep(period)
    finally:
        robot.disconnect()
        print("[home] disconnected.")


if __name__ == "__main__":
    main()
