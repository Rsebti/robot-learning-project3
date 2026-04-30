"""Eval 2 — open the env in interactive 3D mode without training.

Sends zero actions so the robot stays put, lets you inspect the scene
(blocks, bowl, table, lighting, robot pose). Close the window or Ctrl+C
to quit.

Usage:
    uv run python -m sim.eval2.scripts.view --task Eval2-PickInClutter-v1 --num_envs 4
"""

from __future__ import annotations

import argparse

# We must launch the SimulationApp BEFORE importing isaaclab/torch.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect the Eval 2 env without training.")
parser.add_argument("--task", type=str, default="Eval2-PickInClutter-v1")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max_steps", type=int, default=10000, help="Max sim steps before auto-quit (~ 200s at 50 Hz).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force UI on so the user actually sees the window.
args.headless = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main():
    import gymnasium as gym
    import torch

    # Import side-effect: registers our gym envs.
    import sim.eval2  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()
    zero_action = torch.zeros(env.action_space.shape, device="cuda:0")

    step = 0
    while simulation_app.is_running() and step < args.max_steps:
        obs, _, _, _, _ = env.step(zero_action)
        step += 1

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
