"""Eval 2 v0 — visual playback of a trained policy.

Usage (no --headless, opens an Isaac Sim window):

    uv run python -m sim.eval2.scripts.play --task Eval2-PickInBowl-Play-v0
"""

from __future__ import annotations

import sim.eval2  # noqa: F401  (registers the gym envs)

from isaac_so_arm101.scripts.rsl_rl.play import main


if __name__ == "__main__":
    main()
