"""
Eval-1 real-robot deploy: 8D goal-conditioned LeRobot ACT.

Forwards to deploy/infer_eval1_act_nocube.py (the working reference script).

    python infer_eval1.py --target_color yellow --bowl_x 0.16 --bowl_y 0.32
    python infer_eval1.py ... --no-home   # skip homing

Home poses: edit deploy/homes.py

Legacy sim SAC handoff (eval1_ckpt.pt, 12-d state, delta actions):
    python infer_eval1_sac_legacy.py --checkpoint eval1_ckpt.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEPLOY = Path(__file__).resolve().parent.parent
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))

from infer_eval1_act_nocube import main  # noqa: E402

if __name__ == "__main__":
    main()
