"""Eval 2 v0 — training launcher.

Usage (after ``uv pip install -e <project repo>`` into the isaac_so_arm101 venv):

    uv run python -m sim.eval2.scripts.train --task Eval2-PickInBowl-v0 --headless

This thin wrapper just imports our task module (which registers the gym envs)
and then defers to the upstream rsl_rl training script.
"""

from __future__ import annotations

# Import side-effect: registers Eval2-PickInBowl-{,Play-}v0 with gymnasium.
import sim.eval2  # noqa: F401

# Reuse the upstream training entry point so we benefit from any updates there.
from isaac_so_arm101.scripts.rsl_rl.train import main


if __name__ == "__main__":
    main()
