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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "teleop"))
from _arm_lib import JOINTS, connect_arm, drive_to  # noqa: E402

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
    parser.add_argument("--port", default=None,
                        help="Override DEFAULT_FOLLOWER_PORT from teleop/_arm_lib.py")
    parser.add_argument("--id", default="so101_follower")
    parser.add_argument(
        "--duration-s", type=float, default=2.5,
        help="Minimum interpolation duration. Actual duration may be longer "
             "if max joint delta requires it at the 25°/s velocity cap.",
    )
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument(
        "--hold-torque", action="store_true",
        help="Leave torque ON after reaching the pose. "
             "Default: torque OFF on disconnect so you can move the arm by hand.",
    )
    for j, v in DEFAULT_TARGET.items():
        parser.add_argument(f"--{j}", type=float, default=v)
    args = parser.parse_args()

    target = {j: getattr(args, j) for j in JOINTS}

    print(f"[drive-to-home] connecting to follower ...")
    arm = connect_arm("follower", port=args.port, robot_id=args.id,
                      hold_torque_on_exit=args.hold_torque)
    try:
        result = drive_to(arm, target, hz=args.hz, min_duration_s=args.duration_s)
        print(f"[drive-to-home] start :  "
              + "  ".join(f"{j}={result['start'][j]:+7.2f}" for j in JOINTS))
        print(f"[drive-to-home] target:  "
              + "  ".join(f"{j}={result['target'][j]:+7.2f}" for j in JOINTS))
        print(f"[drive-to-home] final :  "
              + "  ".join(f"{j}={result['achieved'][j]:+7.2f}" for j in JOINTS))
        print(f"[drive-to-home] error :  "
              + "  ".join(f"{j}={result['residual'][j]:+7.2f}" for j in JOINTS))
        print(f"[drive-to-home] max error: {result['max_error']:.3f}°")
    finally:
        if args.hold_torque:
            print("[drive-to-home] holding torque ON (--hold-torque). "
                  "Re-run without that flag to release.")
        else:
            print("[drive-to-home] disconnecting (torque OFF). "
                  "Robot is now free to move by hand.")
        arm.disconnect()


if __name__ == "__main__":
    sys.exit(main() or 0)
