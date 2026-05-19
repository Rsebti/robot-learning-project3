"""
clip_homing_offsets.py - clip every homing_offset in the lerobot
calibration JSON to within Feetech's +/-2047 limit so that
robot.connect() can proceed.

This does NOT touch the motors. It just rewrites the JSON. After running,
lerobot's calibrate() will push the clipped values to the motors during
the next connect.

Usage:
    python -m toolset.calibration_scripts.clip_homing_offsets
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_PATH = _REPO / "deploy" / "calibration" / "so101_follower.json"
LIMIT = 2047


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()

    if not args.calibration.exists():
        raise SystemExit(f"calibration not found: {args.calibration}")

    bak = args.calibration.with_suffix(".json.clip_bak")
    shutil.copy(args.calibration, bak)
    print(f"[clip] backup -> {bak}")

    with open(args.calibration) as f:
        d = json.load(f)

    changed = []
    for m, v in d.items():
        h = int(v.get("homing_offset", 0))
        if abs(h) > LIMIT:
            new = max(-LIMIT, min(LIMIT, h))
            v["homing_offset"] = int(new)
            changed.append((m, h, new))
            print(f"  {m:14s}  homing_offset {h:+5d} -> {new:+5d}  (clipped)")
        else:
            print(f"  {m:14s}  homing_offset {h:+5d}  OK")

    with open(args.calibration, "w") as f:
        json.dump(d, f, indent=4)
    print(f"\n[clip] wrote {args.calibration}")
    if changed:
        print(f"[clip] {len(changed)} motor(s) clipped. The clipped joints will be off")
        print(f"        physically by however many degrees their offset exceeded the limit.")
        print(f"        For those joints, plan to recalibrate the affected motor's mounting")
        print(f"        OR retrain with the new mounting; clipping is only a connect-unblocker.")
    else:
        print(f"[clip] nothing to clip; calibration is within limits.")


if __name__ == "__main__":
    main()