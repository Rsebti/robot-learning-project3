"""
Eval-2/3 real-robot deploy: 8D goal-conditioned LeRobot ACT.

Forwards to deploy/infer_eval2.py (same stack as infer_eval1_act_nocube.py).

Legacy sim SAC handoff:
    python infer_sac_legacy.py --checkpoint eval2_ckpt.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEPLOY = Path(__file__).resolve().parent.parent
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))

from infer_eval2 import main  # noqa: E402

if __name__ == "__main__":
    main()
