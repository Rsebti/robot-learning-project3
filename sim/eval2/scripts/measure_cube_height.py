"""Diagnostic: empirically measure cube z relative to base z at reset.

Used to validate the ``cube_dropped`` termination threshold. Spawns the
LeIsaac scene (state-only V27 PLAY env, 50 envs, no cameras), resets
several times, and prints the distribution of:

  - ``cube.z`` (world frame) — where the cube sits at spawn
  - ``base.z`` (world frame) — where the robot base body sits
  - ``cube.z - base.z`` (the quantity ``cube_dropped`` thresholds on)

Interpretation guide for the threshold ``drop_threshold=0.10`` (cube
dropped if ``cube.z - base.z < -0.10``):

  - if mean(``cube.z - base.z``) at spawn ≈ 0:  threshold 0.10 m is good
    (lots of margin from spawn, only catches real falls).
  - if mean ≈ +0.40 (table elevated above base):  cube needs to drop
    *below the floor* to trigger 0.10 m; threshold needs to switch to a
    world-frame z check or to a much larger relative threshold.
  - if mean ≈ -0.40 (base above table, robot mounted on a stand):
    threshold 0.10 m fires constantly, switch to world-frame.

Example
-------
.. code-block:: powershell

    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    .\\.venv\\Scripts\\Activate.ps1
    python -m sim.eval2.scripts.measure_cube_height `
        --task Isaac-LeIsaac-SO101-Lift-RL-V285-Play-v0 `
        --num_envs 50 `
        --num_resets 20 `
        --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Measure cube.z and base.z at reset for the LeIsaac Lift scene."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-LeIsaac-SO101-Lift-RL-V285-Play-v0",
    help="Task name. Default: V285 PLAY (state-only, 50 envs).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=50,
    help="Number of parallel envs (50 is plenty for stats).",
)
parser.add_argument(
    "--num_resets",
    type=int,
    default=20,
    help="How many times to reset (each reset re-randomizes spawn).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Late imports — Isaac Lab requires AppLauncher to be live.
import gymnasium as gym
import torch

import sim.eval2  # noqa: F401  (registers our tasks)
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab.assets import Articulation, RigidObject
from isaaclab_tasks.utils import parse_env_cfg


def _summarize(name: str, values: torch.Tensor) -> None:
    """Print min / max / mean / std of a 1-D tensor of measurements."""
    flat = values.flatten()
    print(
        f"  {name:24s}  "
        f"min={flat.min().item():+.4f}  "
        f"max={flat.max().item():+.4f}  "
        f"mean={flat.mean().item():+.4f}  "
        f"std={flat.std().item():.4f}"
    )


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] task            = {args_cli.task}")
    print(f"[INFO] num_envs        = {args_cli.num_envs}")
    print(f"[INFO] num_resets      = {args_cli.num_resets}")
    print(f"[INFO] obs space       = {env.observation_space}")
    print(f"[INFO] action space    = {env.action_space}")

    cube: RigidObject = env.unwrapped.scene["cube"]
    robot: Articulation = env.unwrapped.scene["robot"]
    base_idx = robot.find_bodies("base")[0][0]
    print(f"[INFO] base body index = {base_idx} (named 'base')")

    cube_z_w_all = []
    base_z_w_all = []
    diff_all = []

    for r in range(args_cli.num_resets):
        env.reset()

        # Both root_pos_w (cube) and body_pos_w (robot bodies) are populated
        # after reset. No step needed.
        cube_z_w = cube.data.root_pos_w[:, 2].detach().clone()
        base_z_w = robot.data.body_pos_w[:, base_idx, 2].detach().clone()
        diff = cube_z_w - base_z_w

        cube_z_w_all.append(cube_z_w)
        base_z_w_all.append(base_z_w)
        diff_all.append(diff)

        if r < 3 or r == args_cli.num_resets - 1:
            print(
                f"[reset {r:2d}] cube.z mean={cube_z_w.mean().item():+.4f}  "
                f"base.z mean={base_z_w.mean().item():+.4f}  "
                f"diff mean={diff.mean().item():+.4f}"
            )

    cube_z_all = torch.cat(cube_z_w_all)
    base_z_all = torch.cat(base_z_w_all)
    diff_all_t = torch.cat(diff_all)

    print()
    print(f"=== Aggregate over {args_cli.num_envs * args_cli.num_resets} samples ===")
    _summarize("cube.z (world)", cube_z_all)
    _summarize("base.z (world)", base_z_all)
    _summarize("cube.z - base.z", diff_all_t)

    print()
    print("=== Threshold sanity ===")
    drop_threshold = 0.10
    spawn_diff_mean = diff_all_t.mean().item()
    margin = spawn_diff_mean - (-drop_threshold)
    print(f"  cube_dropped triggers when (cube.z - base.z) < -{drop_threshold:.2f}")
    print(f"  spawn (cube.z - base.z) mean = {spawn_diff_mean:+.4f}")
    print(f"  margin from spawn to trigger = {margin:+.4f} m")
    if margin < 0.05:
        print("  ⚠️  Margin < 5 cm — threshold may fire on transient grasp dips")
    elif margin > 0.50:
        print("  ⚠️  Margin > 50 cm — threshold may never fire (cube can't drop that far)")
    else:
        print(f"  ✓  Margin reasonable ({margin*100:.1f} cm)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
