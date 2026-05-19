"""
monitor_joints.py - live readout of all 6 SO-101 motor positions, refreshed
at ~10 Hz. Use to verify motor name <-> index mapping: gently move one
joint at a time, watch which line changes.

No camera is configured. No writes are made.

Usage:
    python -m toolset.control.monitor_joints --port COM3
    python -m toolset.control.monitor_joints --port COM3 --hz 20
    python -m toolset.control.monitor_joints --port COM3 --no_disable_torque
"""
from __future__ import annotations

import argparse
import time

import cv2

# Windows MSMF -> DSHOW patch (in case lerobot touches cv2 elsewhere)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--no_disable_torque", action="store_true",
                        help="By default torque is DISABLED so you can hand-move the "
                             "joints. Pass this flag to leave torque ON.")
    args = parser.parse_args()

    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    if str(repo / "deploy") not in sys.path:
        sys.path.insert(0, str(repo / "deploy"))
    from robot_calibration import make_so101_follower_config
    from lerobot.robots.utils import make_robot_from_config

    rconf = make_so101_follower_config(args.port, cameras={}, use_degrees=True)
    robot = make_robot_from_config(rconf)
    print(f"[monitor] Connecting to {args.port} ...")
    robot.connect()

    names_order = list(robot.bus.motors.keys())
    print(f"[monitor] motor order (matches --joint_offset_deg slot order):")
    for i, n in enumerate(names_order):
        print(f"  slot {i} -> {n}")

    try:
        if not args.no_disable_torque:
            print("\n>>> SUPPORT THE ARM PHYSICALLY before disabling torque <<<")
            input("Press Enter when ready to disable torque: ")
            robot.bus.disable_torque()
            print("[monitor] torque disabled.")

        print(f"\n[monitor] reading at {args.hz:.0f} Hz. Ctrl+C to stop.\n")
        header = "   " + "  ".join(f"{n[:10]:>10s}" for n in names_order)
        print(header)
        print("   " + "  ".join(" " * 10 for _ in names_order))

        period = 1.0 / max(0.1, args.hz)
        while True:
            t0 = time.time()
            obs = robot.get_observation()
            vals = [obs[f"{n}.pos"] for n in names_order]
            line = "   " + "  ".join(f"{v:+10.2f}" for v in vals)
            print(line, end="\r", flush=True)
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[monitor] stopped.")
    finally:
        if not args.no_disable_torque:
            try:
                robot.bus.enable_torque()
                print("[monitor] torque re-enabled.")
            except Exception as e:
                print(f"[monitor] could not re-enable torque: {e}")
        robot.disconnect()
        print("[monitor] Disconnected.")


if __name__ == "__main__":
    main()