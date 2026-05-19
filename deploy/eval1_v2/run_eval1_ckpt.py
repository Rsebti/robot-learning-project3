"""
DEPRECATED: use `python deploy/run_checkpoint.py` (same defaults).

run_eval1_ckpt.py - thin wrapper around deploy/eval1_v2/infer_sac_legacy.py
with Eval-1-friendly defaults pre-injected. Zero hardcoded fixes; every
default is overridable from the command line.

Defaults set here (vs the underlying script):
    --checkpoint  deploy/eval1_v2/ckpt.pt
    --bowl_xyz    0.16 0.32 0.00          (matches Eval-1 bowl)
    --goal_color  3                       (yellow; override per scene)
    --no-viz                              (Rerun off; pass --viz to enable)

Pass any flag to override; e.g.
    python deploy/eval1_v2/run_eval1_ckpt.py --goal_color 0 --bowl_xyz 0.10 0.30 0.0
    python deploy/eval1_v2/run_eval1_ckpt.py --no-home --action_scale 0.05
    python deploy/eval1_v2/run_eval1_ckpt.py --viz --episode_steps 300

Home preset can be selected via --home_pose (see deploy/homes.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = HERE / "ckpt.pt"

# Pre-inject defaults only if the user didn't pass them already.
_argv = sys.argv[1:]


def _has(*flags: str) -> bool:
    return any(flag in _argv for flag in flags)


injected: list[str] = []
if not _has("--checkpoint"):
    injected += ["--checkpoint", str(DEFAULT_CKPT)]
if not _has("--bowl_xyz"):
    injected += ["--bowl_xyz", "0.16", "0.32", "0.00"]
if not _has("--goal_color"):
    injected += ["--goal_color", "3"]
if not _has("--viz", "--no-viz"):
    injected += ["--no-viz"]
if not _has("--n_episodes"):
    injected += ["--n_episodes", "1"]

sys.argv = [sys.argv[0]] + injected + _argv

# Hand off to the real script. No monkey-patching, no offsets.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from infer_sac_legacy import main  # noqa: E402

if __name__ == "__main__":
    main()
