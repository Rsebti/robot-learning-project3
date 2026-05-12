"""Idempotent patcher for the rsl_rl LR-floor fix.

Why this exists:
    rsl_rl 3.0.1's PPO adaptive-KL controller hard-codes a learning-rate
    floor of 1e-5 in `algorithms/ppo.py:282`:

        self.learning_rate = max(1e-5, self.learning_rate / 1.5)

    The V2.18 cold run (2026-05-11_23-32-42) collapsed to that floor by
    iter 200 and stayed there for 215 iters before a single bad minibatch
    triggered VF blowup (0.009 → inf in 5 iters, run dead at iter 415).

    Raising the floor to 1e-4 prevents the geometric LR collapse and
    keeps the optimizer responsive when the policy needs to escape a
    plateau (e.g., when sparse grasp signal finally fires).

Why a script:
    The patch lives in `.venv/Lib/site-packages/rsl_rl/algorithms/ppo.py`,
    which is NOT tracked by git. Any of these wipes the patch silently:
      - rebuilding the venv
      - `pip install --upgrade rsl-rl-lib`
      - `pip install --force-reinstall isaac_so_arm101` (transitive)
      - cloning the repo onto a new machine

    This script re-applies the patch idempotently — safe to run any time.

Usage:
    cd C:/Users/user/Desktop/MA2/robot-learning-project3
    C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe \
      sim/eval2/scripts/patch_rsl_rl.py

Behaviour:
    - exits 0 if patch is already applied (no-op, idempotent)
    - exits 0 after applying patch
    - exits 1 if the target file is missing or has unexpected content
"""
from __future__ import annotations

import sys
from pathlib import Path

# The exact line we want to replace and the patched version.
ORIGINAL = "                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)"
PATCHED_LINE = "                            self.learning_rate = max(1e-4, self.learning_rate / 1.5)"

# Comment block prepended above the patched line so the change is
# self-documenting if anyone reads ppo.py directly.
PATCH_HEADER = """\
                            # PATCHED 2026-05-12 by robot-learning-project3 V2.18b
                            # (was 1e-5; raised to 1e-4 to prevent the LR-collapse
                            # cascade that ended V2.18 cold at iter 415 in VF blowup).
                            # Re-apply after any venv rebuild via:
                            #   python sim/eval2/scripts/patch_rsl_rl.py
                            # See notes/rsl_rl_patches.md for context.
"""

# Default rsl_rl install path on the fixed PC. Override via CLI arg if
# the venv lives elsewhere on a teammate machine.
DEFAULT_PPO_PATH = Path(
    r"C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101\.venv"
    r"\Lib\site-packages\rsl_rl\algorithms\ppo.py"
)


def main() -> int:
    ppo_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PPO_PATH
    )

    if not ppo_path.exists():
        print(f"[ERR] target file not found: {ppo_path}")
        print("      pass the correct path as argv[1] if your venv is elsewhere.")
        return 1

    text = ppo_path.read_text(encoding="utf-8")

    # Idempotent: bail early if already patched.
    if PATCHED_LINE in text:
        print(f"[OK] already patched: {ppo_path}")
        return 0

    if ORIGINAL not in text:
        print(f"[ERR] expected line not found in {ppo_path}")
        print(f"      looking for: {ORIGINAL.strip()}")
        print("      The rsl_rl version may have changed; manually inspect")
        print("      the adaptive-KL block (search for 'desired_kl * 2.0').")
        return 1

    new_text = text.replace(ORIGINAL, PATCH_HEADER + PATCHED_LINE)
    ppo_path.write_text(new_text, encoding="utf-8")
    print(f"[OK] patched LR floor 1e-5 → 1e-4 in {ppo_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
