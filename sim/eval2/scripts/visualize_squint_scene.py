"""Visualize a Squint env without training (no checkpoint needed).

Opens the Isaac Sim GUI, spawns the env at small scale, resets every
2 seconds with random actions so you can see:
- Robot in start pose
- Cube spawning in random positions (Squint spawn box -0.10/0.10)
- Bin walls (Place env only)
- Lighting variation per reset (intensity + warm/cool color)
- Wrist camera POV (visible in the side panel)

Launch
------
.. code-block:: powershell

    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
        -m sim.eval2.scripts.visualize_squint_scene `
        --task Isaac-SO101-Squint-Place-Play-v0 --num_envs 4 --enable_cameras

Add ``--task Isaac-SO101-Squint-Lift-Play-v0`` for the Lift variant.
Drop ``--enable_cameras`` if you don't need the wrist cam to be active
(saves VRAM).
"""
from __future__ import annotations

import argparse
import time

parser = argparse.ArgumentParser(description="Visualize a Squint env")
parser.add_argument("--task", type=str, default="Isaac-SO101-Squint-Place-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--reset_every", type=float, default=2.0, help="Seconds between resets")
parser.add_argument("--total_seconds", type=float, default=120.0, help="Stop after N seconds")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force GUI (drop --headless if user passed it)
args.headless = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sim.eval2  # noqa: F401,E402


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    print(f"[viz] Task={args.task}, num_envs={args.num_envs}")
    print(f"[viz] Action space: {env.unwrapped.action_space}")

    obs, _ = env.reset(seed=0)
    n_act = env.unwrapped.action_space.shape[-1]

    last_reset = time.time()
    start = time.time()
    step_count = 0

    while simulation_app.is_running():
        if time.time() - start > args.total_seconds:
            print("[viz] Total seconds elapsed, exiting.")
            break

        # Random actions in the action space range — purely for moving
        # the robot a bit so the scene isn't static.
        action = torch.zeros(args.num_envs, n_act, device=device)
        # Small random gripper movements only (avoid wild arm flailing)
        action[:, -1] = torch.empty(args.num_envs, device=device).uniform_(-1.0, 1.0)

        obs, _, _, _, _ = env.step(action)
        step_count += 1

        if time.time() - last_reset > args.reset_every:
            print(f"[viz] step={step_count} — resetting envs...")
            obs, _ = env.reset()
            last_reset = time.time()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
