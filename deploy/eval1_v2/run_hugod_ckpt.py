"""
DEPRECATED: use `python deploy/run_checkpoint.py --name <stem>`.

run_hugod_ckpt.py - thin pass-through to infer_sac_legacy.main().
Only extra: --name X resolves to C:\\Users\\hugod\\X (auto-appends .pt).
Everything else is forwarded untouched.

    python deploy/eval1_v2/run_hugod_ckpt.py --name ckpt_best_1 --goal_color 3 --bowl_xyz 0.16 0.32 0.00
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--name", required=True)
own, rest = pre.parse_known_args()
raw = own.name if own.name.endswith(".pt") else own.name + ".pt"
sys.argv = [sys.argv[0], "--checkpoint", str(Path.home() / raw)] + rest

from infer_sac_legacy import main  # noqa: E402

if __name__ == "__main__":
    main()
