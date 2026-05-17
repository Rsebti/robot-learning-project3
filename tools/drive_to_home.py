#!/usr/bin/env python3
"""Drive the SO-101 follower to a target joint pose, interpolating smoothly
from its current pose. Used to reset the follower to a policy's training-time
home pose before launching inference, so the policy starts in-distribution.

Defaults below are the home pose of `osammotg1/projet3-eval1-bowl1-v1`
(dark-shadow training data), computed as the modal value across 39 episodes —
NOT the raw mean, because 3 reset-outlier episodes pull wrist_flex/gripper.

Override any joint from the CLI, e.g.:
  python tools/drive_to_home.py --wrist_roll 5.0 --gripper 30
"""
import argparse
import sys
import time

from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower


JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Home pose of osammotg1/projet3-eval1-bowl1-v1 (degrees).
# Source: tools/inspect_dataset_state.py over 39 episodes.
DEFAULT_TARGET = {
    "shoulder_pan":  -2.0,
    "shoulder_lift": -92.2,
    "elbow_flex":    +97.7,
    "wrist_flex":    +75.2,
    "wrist_roll":     0.0,
    "gripper":        +1.2,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/tty.usbmodem5B141129871")
    parser.add_argument("--id", default="so101_follower")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=2.5,
        help="Total time to interpolate from current pose to target.",
    )
    parser.add_argument("--hz", type=float, default=30.0, help="Waypoint send rate.")
    parser.add_argument(
        "--hold-torque",
        action="store_true",
        help="Leave torque ON after reaching the pose (default: torque OFF so you can manually verify).",
    )
    for j, v in DEFAULT_TARGET.items():
        parser.add_argument(f"--{j}", type=float, default=v)
    args = parser.parse_args()

    target = {f"{j}.pos": getattr(args, j) for j in JOINTS}

    cfg = SO101FollowerConfig(
        port=args.port,
        id=args.id,
        use_degrees=True,
        disable_torque_on_disconnect=not args.hold_torque,
    )
    robot = SOFollower(cfg)
    print(f"[drive-to-home] connecting to {args.port} ...")
    robot.connect()

    try:
        obs = robot.get_observation()
        start = {f"{j}.pos": float(obs[f"{j}.pos"]) for j in JOINTS}
        print(f"[drive-to-home] start :  " + "  ".join(f"{j}={start[f'{j}.pos']:+7.2f}" for j in JOINTS))
        print(f"[drive-to-home] target:  " + "  ".join(f"{j}={target[f'{j}.pos']:+7.2f}" for j in JOINTS))

        max_delta = max(abs(target[f"{j}.pos"] - start[f"{j}.pos"]) for j in JOINTS)
        print(f"[drive-to-home] max joint delta: {max_delta:.1f} deg over {args.duration_s}s "
              f"({max_delta / args.duration_s:.1f} deg/s)")

        n_steps = max(2, int(args.duration_s * args.hz))
        dt = 1.0 / args.hz
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            waypoint = {
                k: start[k] + alpha * (target[k] - start[k])
                for k in target
            }
            robot.send_action(waypoint)
            time.sleep(dt)

        # Let the final waypoint settle before reading the residual.
        time.sleep(0.6)
        final = robot.get_observation()
        residual = {j: final[f"{j}.pos"] - target[f"{j}.pos"] for j in JOINTS}
        print(f"[drive-to-home] final :  " + "  ".join(f"{j}={final[f'{j}.pos']:+7.2f}" for j in JOINTS))
        print(f"[drive-to-home] error :  " + "  ".join(f"{j}={residual[j]:+7.2f}" for j in JOINTS))
    finally:
        if args.hold_torque:
            print("[drive-to-home] holding torque ON (--hold-torque). Re-run without that flag to release.")
        else:
            print("[drive-to-home] disconnecting (torque OFF). Robot is now free to move by hand.")
        robot.disconnect()


if __name__ == "__main__":
    sys.exit(main() or 0)
