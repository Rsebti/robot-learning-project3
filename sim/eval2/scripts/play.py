"""Phase A play wrapper — same idea as `train.py` but for `play.py`.

Loads a trained checkpoint and replays it on the restricted lift task.

Example
-------
.. code-block:: powershell

    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    uv run python -m sim.eval2.scripts.play `
        --task Isaac-SO-ARM101-Lift-Cube-Restricted-Play-v0 `
        --num_envs 4 `
        --checkpoint logs\\rsl_rl\\lift\\<timestamp>\\model_999.pt
"""
import runpy

import sim.eval2  # noqa: F401  (registers our tasks)

runpy.run_module(
    "isaac_so_arm101.scripts.rsl_rl.play",
    run_name="__main__",
)
