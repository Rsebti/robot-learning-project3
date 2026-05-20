"""
run_ckpt_short.py - thin wrapper around run_ckpt_simple.py for
C:\\Users\\hugod\\ckpt_short.pt. Homes to the SAC training rest, then
runs 450 steps at 30 Hz (~15 s). Everything else (--color, --bowl_xyz,
--wrist_roll_offset_deg, ...) is forwarded to run_ckpt_simple.

Defaults injected:
    --checkpoint              C:\\Users\\hugod\\ckpt_short.pt
    --home                    (ramp to REST_QPOS_LEGACY before rollout)
    --episode_steps           1000    (~33 s @ 30 Hz)
    --wrist_roll_offset_deg   -90     (Feetech keyway shift after motor reseat)

Examples:
    python deploy/run_ckpt_short.py
    python deploy/run_ckpt_short.py --color 3 --bowl_xyz 0.16 0.32 0.0
    python deploy/run_ckpt_short.py --wrist_roll_offset_deg -90
    python deploy/run_ckpt_short.py --no-home              # skip homing
    python deploy/run_ckpt_short.py --episode_steps 900    # ~30 s
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = Path.home() / "ckpt_short.pt"

_argv = sys.argv[1:]


def _has(*flags: str) -> bool:
    return any(flag in _argv for flag in flags)


injected: list[str] = []
if not _has("--checkpoint"):
    injected += ["--checkpoint", str(DEFAULT_CKPT)]
# Home to the SAC training rest by default (REST_QPOS_LEGACY in run_ckpt_simple).
if not _has("--home", "--no-home"):
    injected += ["--home"]
# 30 Hz default = 1000 steps = ~33 s episode.
if not _has("--episode_steps"):
    injected += ["--episode_steps", "1000"]
# Bake in the wrist keyway correction so the home pose lands at the
# training orientation. Pass --wrist_roll_offset_deg <other> to override.
if not _has("--wrist_roll_offset_deg"):
    injected += ["--wrist_roll_offset_deg", "-90"]

sys.argv = [sys.argv[0]] + injected + _argv

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from run_ckpt_simple import main  # noqa: E402

if __name__ == "__main__":
    main()
