"""
manage_calibration.py — sync LeRobot follower calibration with deploy/calibration/.

Workflow after changing a motor or on a new PC:

  1. Full range calibration (interactive):
       lerobot-calibrate --robot.type=so101_follower --robot.port=COM3 --robot.id=so101_follower

  2. Save into the repo (all deploy scripts read this file):
       python -m toolset.calibration_scripts.manage_calibration pull

  3. Optional: if homing_offset exceeded +/-2047 and connect fails:
       python -m toolset.calibration_scripts.manage_calibration clip

  4. If only the zero shifted (motor remounted) but ranges are still OK:
       python -m toolset.calibration_scripts.rehome_only --port COM3 \\
           --calibration deploy/calibration/so101_follower.json

  5. Push local file to HF cache (so lerobot-record / bare CLI match deploy):
       python -m toolset.calibration_scripts.manage_calibration install

Subcommands:
    pull      HF cache  →  deploy/calibration/so101_follower.json
    install   deploy/calibration  →  HF cache
    validate  check JSON schema
    show      print homing_offset / range summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "deploy") not in sys.path:
    sys.path.insert(0, str(_REPO / "deploy"))

from robot_calibration import (  # noqa: E402
    LOCAL_CALIBRATION_FILE,
    ensure_local_calibration_dir,
    install_local_to_cache,
    pull_cache_to_local,
    resolve_hf_cache_calibration,
    validate_calibration,
)

LIMIT = 2047


def cmd_pull(args) -> None:
    pull_cache_to_local(force=args.force)


def cmd_install(args) -> None:
    install_local_to_cache(force=args.force)


def cmd_validate(args) -> None:
    path = Path(args.calibration) if args.calibration else LOCAL_CALIBRATION_FILE
    validate_calibration(path)
    print(f"[calib] OK: {path}")


def cmd_show(args) -> None:
    path = Path(args.calibration) if args.calibration else LOCAL_CALIBRATION_FILE
    with open(path) as f:
        d = json.load(f)
    print(f"[calib] {path}")
    for m, v in d.items():
        h = int(v.get("homing_offset", 0))
        flag = " CLIP!" if abs(h) > LIMIT else ""
        print(f"  {m:14s}  id={v.get('id')}  homing={h:+5d}{flag}  "
              f"range=[{v.get('range_min')}, {v.get('range_max')}]")


def cmd_clip(args) -> None:
    path = Path(args.calibration) if args.calibration else LOCAL_CALIBRATION_FILE
    ensure_local_calibration_dir()
    if not path.is_file():
        raise SystemExit(f"not found: {path}")

    bak = path.with_suffix(".json.clip_bak")
    import shutil
    shutil.copy(path, bak)
    print(f"[clip] backup -> {bak}")

    with open(path) as f:
        d = json.load(f)
    n = 0
    for m, v in d.items():
        h = int(v.get("homing_offset", 0))
        if abs(h) > LIMIT:
            new = max(-LIMIT, min(LIMIT, h))
            print(f"  {m:14s}  {h:+5d} -> {new:+5d}")
            v["homing_offset"] = int(new)
            n += 1
    with open(path, "w") as f:
        json.dump(d, f, indent=4)
    print(f"[clip] wrote {path} ({n} joint(s) clipped)")
    if n:
        print("[clip] run: python -m toolset.calibration_scripts.manage_calibration install")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("pull", help="HF cache → deploy/calibration/")
    pp.add_argument("--force", action="store_true")
    pp.set_defaults(func=cmd_pull)

    pi = sub.add_parser("install", help="deploy/calibration → HF cache")
    pi.add_argument("--force", action="store_true")
    pi.set_defaults(func=cmd_install)

    pv = sub.add_parser("validate", help="check local JSON")
    pv.add_argument("--calibration", type=Path, default=None)
    pv.set_defaults(func=cmd_validate)

    ps = sub.add_parser("show", help="print offsets and ranges")
    ps.add_argument("--calibration", type=Path, default=None)
    ps.set_defaults(func=cmd_show)

    pc = sub.add_parser("clip", help="clip homing_offset to +/-2047 (local file)")
    pc.add_argument("--calibration", type=Path, default=None)
    pc.set_defaults(func=cmd_clip)

    args = p.parse_args()
    cache = resolve_hf_cache_calibration()
    if args.cmd == "pull" and cache:
        print(f"[calib] HF cache source: {cache}")
    args.func(args)


if __name__ == "__main__":
    main()
