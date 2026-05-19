"""
run_best_ckpt.py — entry point for your friend's Squint SAC ``*.pt`` checkpoints.

Same as ``python deploy/run_checkpoint.py`` for local ``.pt`` files: it resolves
the path, injects sensible defaults, then runs the real-time loop. The loop
lives in ``deploy/eval1_v2/infer_sac_legacy.py`` (implementation detail — you
do not need to call that file yourself unless debugging).

Works for any filename you drop in ``~/`` or the search dirs: ``ckpt.pt``,
``ckpt_best_1.pt``, ``e1100lat.pt``, etc.

Examples:
    python deploy/run_best_ckpt.py --inspect --ckpt C:\\Users\\hugod\\e1100lat.pt
    python deploy/run_best_ckpt.py --name ckpt_best_1 --goal_color 0 --bowl_xyz 0.2 0.1 0.0
    python deploy/run_best_ckpt.py --ckpt C:\\Users\\hugod\\e1300nolat.pt --no-viz --n_episodes 1
"""
from __future__ import annotations

import sys
from pathlib import Path

_DEPLOY = Path(__file__).resolve().parent
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))

from run_checkpoint import main  # noqa: E402

if __name__ == "__main__":
    main()
