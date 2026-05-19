#!/usr/bin/env python3
"""Verify the saved SO-101 home pose by driving the follower back to it
(or, for the leader, guiding the user to position it by hand).

Loops interactively: each ENTER re-runs the drive/guide, letting you confirm
that the arm reaches the same physical position from arbitrary start poses.
Type 'q' to quit. Every run appends a record to teleop/anchors/_drift_log.jsonl
so you can see whether residuals are growing over sessions (motor heat,
calibration drift, mechanical loosening).

Safety:
  * Refuses to drive if the saved pose's calibration_sha256 doesn't match
    the current calibration file on disk. Re-record or pass --force.
  * Refuses to drive the follower if it is more than 30° from the target
    on any joint. Move it closer by hand, or pass --force.

Usage:
  python teleop/verify_home_pose.py                       # follower
  python teleop/verify_home_pose.py --target follower
  python teleop/verify_home_pose.py --target leader       # hand-guided
  python teleop/verify_home_pose.py --target both
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _arm_lib import (
    ANCHOR_PATH, JOINTS,
    append_drift, calibration_sha256, connect_arm, drive_to, guide_to,
    load_anchor, read_pose, say_async,
)

MAX_AUTO_DELTA_DEG = 90.0


def _check_anchor(side: str) -> dict | int:
    anchor = load_anchor()
    if anchor is None:
        print(f"[verify] ERROR: {ANCHOR_PATH} not found.")
        print(f"         Run first: python teleop/record_home_pose.py --target {side}")
        return 2
    side_block = anchor.get(side)
    if side_block is None:
        print(f"[verify] ERROR: no {side!r} pose in {ANCHOR_PATH}.")
        print(f"         Run first: python teleop/record_home_pose.py --target {side}")
        return 2
    return side_block


def verify_follower(port: str | None, force: bool) -> int:
    side_block = _check_anchor("follower")
    if isinstance(side_block, int):
        return side_block
    target = side_block["joints_deg"]
    saved_sha = side_block.get("calibration_sha256")
    current_sha = calibration_sha256("follower")
    sha_match = bool(saved_sha and current_sha and saved_sha == current_sha)

    if not sha_match:
        print(f"[verify] ⚠ Calibration SHA-256 mismatch (follower):")
        print(f"           saved:   {saved_sha}")
        print(f"           current: {current_sha}")
        if not force:
            print(f"[verify] Refusing to drive. Re-record the pose, or re-run with --force.")
            return 3
        print(f"[verify] --force set: proceeding despite mismatch.")

    print(f"\n[verify] ====== FOLLOWER ======")
    print(f"[verify] Target: " + "  ".join(f"{j}={target[j]:+7.2f}" for j in JOINTS))

    arm = connect_arm("follower", port=port, hold_torque_on_exit=False)
    try:
        while True:
            current = read_pose(arm)
            max_delta = max(abs(target[j] - current[j]) for j in JOINTS)
            if max_delta > MAX_AUTO_DELTA_DEG and not force:
                print(f"[verify] ⚠ Arm is far from home "
                      f"(max delta {max_delta:.1f}° > {MAX_AUTO_DELTA_DEG:.0f}°).")
                print(f"[verify] Move it closer by hand, or pass --force.")
                return 4

            say_async("Driving follower to home.")
            result = drive_to(arm, target)
            print(f"[verify] residual: " + "  ".join(
                f"{j}={result['residual'][j]:+6.3f}" for j in JOINTS))
            tag = "OK" if result["max_error"] <= 1.0 else "WARN"
            print(f"[verify] max error: {result['max_error']:.3f}°   ({tag})")

            append_drift({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "side": "follower",
                "target": result["target"],
                "achieved": result["achieved"],
                "residual": result["residual"],
                "max_error": result["max_error"],
                "calibration_sha256_match": sha_match,
            })

            ans = input("\n[verify] ENTER to re-drive (you can perturb the arm "
                        "in between by overpowering the servos by hand), q to quit: ").strip().lower()
            if ans == "q":
                break
    finally:
        arm.disconnect()
        print(f"[verify] Disconnected (follower, torque OFF).")
    return 0


def verify_leader(port: str | None, force: bool) -> int:
    side_block = _check_anchor("leader")
    if isinstance(side_block, int):
        return side_block
    target = side_block["joints_deg"]
    saved_sha = side_block.get("calibration_sha256")
    current_sha = calibration_sha256("leader")
    sha_match = bool(saved_sha and current_sha and saved_sha == current_sha)

    if not sha_match:
        print(f"[verify] ⚠ Calibration SHA-256 mismatch (leader):")
        print(f"           saved:   {saved_sha}")
        print(f"           current: {current_sha}")
        if not force:
            print(f"[verify] Refusing to guide. Re-record the pose, or re-run with --force.")
            return 3
        print(f"[verify] --force set: proceeding despite mismatch.")

    print(f"\n[verify] ====== LEADER (hand-guided) ======")
    print(f"[verify] Target: " + "  ".join(f"{j}={target[j]:+7.2f}" for j in JOINTS))

    arm = connect_arm("leader", port=port, hold_torque_on_exit=False)
    try:
        while True:
            result = guide_to(arm, target)
            print(f"[verify] residual: " + "  ".join(
                f"{j}={result['residual'][j]:+6.3f}" for j in JOINTS))
            print(f"[verify] max error: {result['max_error']:.3f}°")

            append_drift({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "side": "leader",
                "target": result["target"],
                "achieved": result["achieved"],
                "residual": result["residual"],
                "max_error": result["max_error"],
                "calibration_sha256_match": sha_match,
            })

            ans = input("\n[verify] ENTER to re-guide (move the leader away first), "
                        "q to quit: ").strip().lower()
            if ans == "q":
                break
    finally:
        arm.disconnect()
        print(f"[verify] Disconnected (leader, torque OFF — backdrivable).")
    return 0


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target", choices=["follower", "leader", "both"], default="follower")
    p.add_argument("--port-follower", default=None)
    p.add_argument("--port-leader", default=None)
    p.add_argument("--force", action="store_true",
                   help="Skip calibration-SHA and large-delta safety checks.")
    args = p.parse_args()

    sides = ["follower", "leader"] if args.target == "both" else [args.target]
    for side in sides:
        if side == "follower":
            rc = verify_follower(args.port_follower, args.force)
        else:
            rc = verify_leader(args.port_leader, args.force)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
