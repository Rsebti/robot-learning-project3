"""Phase A smoke test launcher.

Wraps `isaac_so_arm101.scripts.rsl_rl.train` so our custom Gym tasks (e.g.
`Isaac-SO-ARM101-Lift-Cube-Restricted-v0`) are registered before the inner
script calls `gym.make`. We do not modify isaac_so_arm101 — we just import
our package first, then run isaac_so_arm101's train script with `runpy`,
forwarding all CLI arguments unchanged.

Example
-------
.. code-block:: powershell

    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    uv run python -m sim.eval2.scripts.train `
        --task Isaac-SO-ARM101-Lift-Cube-Restricted-v0 `
        --headless --num_envs 4096 --max_iterations 1500 --seed 42
"""
import runpy

# Importing `sim.eval2` triggers the gym.register calls for our tasks.
# (Also already triggered transitively by `python -m sim.eval2.scripts.train`,
# but we keep the explicit import for clarity and forward-compat.)
import sim.eval2  # noqa: F401

# Run the inner train script as if it had been invoked directly with the
# current sys.argv (Hydra + Isaac Lab AppLauncher will parse it as usual).
runpy.run_module(
    "isaac_so_arm101.scripts.rsl_rl.train",
    run_name="__main__",
)
