"""Phase A scene viewer — open the env in Omniverse with zero actions.

Same wrapper trick as `train.py`/`play.py`: import sim.eval2 first so our
custom Gym tasks are registered, then delegate to isaac_so_arm101's
`zero_agent.py` script which opens an Isaac Sim window and steps the env
with zero actions (so the robot stays still and you can inspect the scene).

Example
-------
.. code-block:: powershell

    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    uv run python -m sim.eval2.scripts.view `
        --task Isaac-LeIsaac-SO101-Lift-RL-Play-v0 `
        --num_envs 4
"""
import runpy

import sim.eval2  # noqa: F401  (registers our tasks)

runpy.run_module(
    "isaac_so_arm101.scripts.zero_agent",
    run_name="__main__",
)
