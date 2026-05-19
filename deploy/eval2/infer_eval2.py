"""
Eval-2/3 real-robot deploy: 8D goal-conditioned LeRobot ACT.

Forwards to deploy/infer_eval2.py (same stack as infer_eval1_act_nocube.py).

    python infer_eval2.py --target_color red --bowl_x -0.155 --bowl_y 0.295

Legacy sim SAC handoff (eval2_ckpt.pt, 18-d state, delta actions):
    python infer_eval2_sac_legacy.py --checkpoint eval2_ckpt.pt --goal_color 0
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_DEPLOY = Path(__file__).resolve().parent.parent
_MODULE = "_deploy_infer_eval2_act"


def main():
    path = _DEPLOY / "infer_eval2.py"
    if _MODULE not in sys.modules:
        spec = importlib.util.spec_from_file_location(_MODULE, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE] = mod
        spec.loader.exec_module(mod)
    else:
        mod = sys.modules[_MODULE]
    mod.main()


if __name__ == "__main__":
    main()
