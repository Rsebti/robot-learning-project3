"""Dump the actual USD prim hierarchy under {ENV_REGEX_NS}/Robot to see
where the wrist cam should be parented.
"""
from __future__ import annotations

import argparse

import torch  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-SquintNative-Place-Play-v0")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
args.num_envs = 1

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sim.eval2  # noqa: F401,E402


def main():
    env_cfg = parse_env_cfg(args.task, num_envs=1)
    try:
        env_cfg.recorders = None
    except Exception:
        pass
    env = gym.make(args.task, cfg=env_cfg)
    base_env = env.unwrapped

    from pxr import Usd
    stage = base_env.sim.stage
    out_path = r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/robot_prim_tree.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Prim tree under /World/envs/env_0/Robot ===\n")
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if not root.IsValid():
            f.write("ERROR: /World/envs/env_0/Robot not found\n")
        else:
            for prim in Usd.PrimRange(root):
                depth = str(prim.GetPath()).count("/") - 4
                f.write(f"{'  ' * depth}{prim.GetPath().pathString.split('/')[-1]}  [{prim.GetTypeName()}]\n")
    print(f"\n[dump] wrote {out_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
