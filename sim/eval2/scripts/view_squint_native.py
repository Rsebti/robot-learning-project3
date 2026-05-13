"""GUI viewer for the Squint-native env — no policy, no training.

Opens the Isaac Sim viewport, loads the env at home pose, then steps zero
actions forever. You can rotate / pan / zoom freely with the mouse to
inspect the scene. Press Ctrl+C in the terminal (or close the window) to
quit.

Launch
------
.. code-block:: powershell

    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
        -m sim.eval2.scripts.view_squint_native

Add ``--task Isaac-SquintNative-Place-v0`` to view the multi-env layout
instead of the single-env Play variant.
"""
from __future__ import annotations

import argparse

import torch  # noqa: F401  (import early to win over Isaac Kit DLLs)

parser = argparse.ArgumentParser(description="GUI viewer for Squint-native Isaac env")
parser.add_argument("--task", type=str, default="Isaac-SquintNative-Place-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--reset_every", type=int, default=200,
                    help="Reset the env every N control steps so you can see the spawn randomization.")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# Force GUI mode — overrides whatever was passed on CLI.
args.headless = False
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sim.eval2  # noqa: F401,E402  (registers the SquintNative tasks)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    # Disable the built-in HDF5 recorder so we don't fight a file lock.
    try:
        env_cfg.recorders = None
    except Exception:
        pass
    env = gym.make(args.task, cfg=env_cfg)

    n_act = env.action_space.shape[-1]
    zero = torch.zeros(args.num_envs, n_act, device=device)

    obs, _ = env.reset(seed=0)
    print(f"\n[viewer] Loaded {args.task}  (num_envs={args.num_envs}, action_dim={n_act})")
    print(f"[viewer] Stepping zero actions; reset every {args.reset_every} steps.")
    print(f"[viewer] Move/rotate the camera with right-click drag, scroll to zoom.")
    print(f"[viewer] Close the window or Ctrl+C to quit.\n")

    step = 0
    while simulation_app.is_running():
        env.step(zero)
        step += 1
        if step % args.reset_every == 0:
            env.reset()
            print(f"[viewer] step {step} — env reset (new spawn).")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
