"""
DEPRECATED: use `python deploy/run_checkpoint.py --ckpt <path>`.

run_ckpt_best1.py - thin pass-through to infer_sac_legacy.main().
Nothing injected. Pass every flag yourself.

Example:
    python deploy/eval1_v2/run_ckpt_best1.py `
        --checkpoint C:/Users/hugod/ckpt_best_1.pt `
        --goal_color 3 --bowl_xyz 0.16 0.32 0.00 `
        --no-viz --n_episodes 1
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from infer_sac_legacy import main  # noqa: E402

if __name__ == "__main__":
    main()
