#!/usr/bin/env python3
"""Record the SO-101 home pose.

You hold the arm physically still for 10 seconds. The script samples joint
positions at 30 Hz, computes per-joint median (the pose) and MAD (stability),
and writes teleop/anchors/home_pose.json — pinned to the SHA-256 of the
calibration file that produced it.

Usage:
  python teleop/record_home_pose.py                      # follower (default)
  python teleop/record_home_pose.py --target follower
  python teleop/record_home_pose.py --target leader      # leader only
  python teleop/record_home_pose.py --target both        # sequential

Then verify with:
  python teleop/verify_home_pose.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _arm_lib import (
    ANCHOR_PATH, JOINTS,
    connect_arm, sample_pose, save_anchor, stability_report, say_async,
)


def record_one(side: str, port: str | None, duration_s: float, hz: float) -> None:
    print(f"\n[record] ====== {side.upper()} ======")
    print(f"[record] Connecting to {side} (torque OFF — you'll hold the arm by hand)...")
    arm = connect_arm(side, port=port, hold_torque_on_exit=False)
    try:
        say_async(f"Place the {side} at the home pose. Get ready.")
        input(f"[record] Place the {side} at the desired home pose, "
              f"then press ENTER to start the {duration_s:.0f}s capture...")

        result = sample_pose(arm, duration_s=duration_s, hz=hz, say_cues=True)
        ok, warnings = stability_report(result["stats"])

        print(f"\n[record] Captured pose (median, °):")
        for j in JOINTS:
            s = result["stats"][j]
            shaky = s["mad"] > 0.5 or s["range"] > 2.0
            mark = "⚠ " if shaky else "  "
            print(f"  {mark}{j:14s} {result['joints_deg'][j]:+7.2f}    "
                  f"MAD={s['mad']:.3f}°  range={s['range']:.3f}°  n={s['n']}")

        if not ok:
            print(f"\n[record] ⚠ Stability warnings:")
            for w in warnings:
                print(w)
            print(f"[record] Saving anyway — re-run if these are too large.")
        else:
            print(f"\n[record] ✓ Stable capture.")

        save_anchor(side, result, port=port)
        print(f"[record] → {ANCHOR_PATH}")
    finally:
        arm.disconnect()
        print(f"[record] Disconnected ({side}, torque OFF).")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target", choices=["follower", "leader", "both"], default="follower")
    p.add_argument("--port-follower", default=None,
                   help="Override DEFAULT_FOLLOWER_PORT from _arm_lib.py")
    p.add_argument("--port-leader", default=None,
                   help="Override DEFAULT_LEADER_PORT from _arm_lib.py")
    p.add_argument("--duration-s", type=float, default=10.0)
    p.add_argument("--hz", type=float, default=30.0)
    args = p.parse_args()

    sides = ["follower", "leader"] if args.target == "both" else [args.target]
    for i, side in enumerate(sides):
        port = args.port_follower if side == "follower" else args.port_leader
        record_one(side, port, args.duration_s, args.hz)
        if i + 1 < len(sides):
            input(f"\n[record] Press ENTER to move on to {sides[i+1]}...")

    print("\n[record] All done.")


if __name__ == "__main__":
    sys.exit(main() or 0)
