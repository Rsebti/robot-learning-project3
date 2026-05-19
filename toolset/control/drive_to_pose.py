"""
drive_to_pose.py - ramp the SO-101 follower to a target joint pose, then
optionally hold or launch a policy rollout from there.

Useful when:
  - You've just rehomed and want to verify by driving to [0,0,0,0,0,0].
  - You want a custom starting pose before infer (one not equal to the
    hardcoded REST_QPOS in the friend's scripts).

Usage examples:
    # Drive to the rehomed center (all zeros), hold 3 seconds, exit.
    python -m toolset.control.drive_to_pose --port COM3 \\
        --target_deg 0 0 0 0 0 0 --hold_s 3

    # Drive to a custom pose, then hand off to the friend's ACT/SAC infer.
    python -m toolset.control.drive_to_pose --port COM3 \\
        --target_deg 0 0 0 90 -90 60 --hold_s 1 \\
        --then "python deploy/eval1/infer_eval1.py --checkpoint deploy/eval1/eval1_ckpt.pt --no-viz --n_episodes 1"

The ramp is linear in joint-space at --max_deg_per_step deg/step, paced at
--fps Hz. If the live joint reading is more than --abort_diff_deg away from
the command at any step, the ramp aborts (safety).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import cv2
import numpy as np

# --- Windows MSMF -> DSHOW patch (same as rehome_only.py) -----------------
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
# --------------------------------------------------------------------------


JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--camera_index", type=int, default=1)
    parser.add_argument("--target_deg", type=float, nargs=6, required=True,
                        metavar=tuple(JOINT_NAMES),
                        help="Target joint positions in degrees, lerobot convention.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_deg_per_step", type=float, default=1.0,
                        help="Max degrees any joint can move per command step.")
    parser.add_argument("--max_total_s", type=float, default=20.0,
                        help="Hard cap on total drive time.")
    parser.add_argument("--hold_s", type=float, default=2.0,
                        help="Once target reached, keep sending the same command this long.")
    parser.add_argument("--abort_diff_deg", type=float, default=30.0,
                        help="Abort if any joint deviates more than this from the command.")
    parser.add_argument("--then", default=None,
                        help="Shell command to exec after the drive (e.g. an infer script).")
    args = parser.parse_args()

    target = np.array(args.target_deg, dtype=float)

    from lerobot.robots.utils import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cfg = SO101FollowerConfig(
        port=args.port, id="so101_follower",
        cameras={"wrist": OpenCVCameraConfig(
            index_or_path=args.camera_index, fps=30, width=640, height=480,
        )},
    )
    robot = make_robot_from_config(cfg)
    print(f"[drive] Connecting to {args.port} ...")
    robot.connect()

    period = 1.0 / args.fps
    max_step = float(args.max_deg_per_step)
    start_t = time.time()
    aborted = False
    try:
        obs = robot.get_observation()
        q_now = np.array([float(obs[f"{n}.pos"]) for n in JOINT_NAMES], dtype=float)
        print(f"[drive] start qpos (deg): {q_now.round(1).tolist()}")
        print(f"[drive] target   (deg): {target.round(1).tolist()}")
        print(f"[drive] max {max_step:.2f} deg/step at {args.fps} Hz "
              f"(=> {max_step * args.fps:.1f} deg/s).")

        q_cmd = q_now.copy()
        step_idx = 0
        while True:
            err = target - q_cmd
            if np.max(np.abs(err)) < 0.5:
                break
            if time.time() - start_t > args.max_total_s:
                print(f"[drive] max_total_s reached; stopping at {q_cmd.round(1).tolist()}")
                aborted = True
                break
            step = np.clip(err, -max_step, max_step)
            q_cmd = q_cmd + step
            action = {f"{n}.pos": float(q_cmd[i]) for i, n in enumerate(JOINT_NAMES)}
            t0 = time.time()
            robot.send_action(action)
            # Safety check: live reading vs command
            obs = robot.get_observation()
            q_live = np.array([float(obs[f"{n}.pos"]) for n in JOINT_NAMES], dtype=float)
            diff = np.abs(q_live - q_cmd)
            if np.max(diff) > args.abort_diff_deg:
                print(f"\n[drive] ABORT: joint {JOINT_NAMES[int(np.argmax(diff))]} "
                      f"diff {diff.max():.1f} deg > {args.abort_diff_deg:.1f}")
                aborted = True
                break
            step_idx += 1
            if step_idx % args.fps == 0:
                print(f"  t={time.time() - start_t:5.2f}s  cmd={q_cmd.round(1).tolist()}  "
                      f"live={q_live.round(1).tolist()}")
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

        if not aborted:
            obs = robot.get_observation()
            q_live = np.array([float(obs[f"{n}.pos"]) for n in JOINT_NAMES], dtype=float)
            print(f"\n[drive] reached. final live qpos (deg): {q_live.round(1).tolist()}")
            print(f"[drive] command qpos (deg):              {q_cmd.round(1).tolist()}")
            # Hold
            t_end = time.time() + args.hold_s
            action = {f"{n}.pos": float(q_cmd[i]) for i, n in enumerate(JOINT_NAMES)}
            while time.time() < t_end:
                robot.send_action(action)
                time.sleep(period)
            print(f"[drive] held for {args.hold_s}s.")
    finally:
        robot.disconnect()
        print("[drive] Disconnected.")

    if aborted:
        sys.exit(2)

    if args.then:
        print(f"\n[drive] launching follow-up: {args.then}")
        sys.exit(subprocess.call(args.then, shell=True))


if __name__ == "__main__":
    main()
