#!/usr/bin/env python3
"""
replay_lerobot_in_isaac.py — kinematic replay of teleop demos in Isaac Lab.

Uses pre-extracted ``RLPD/data/episodes.json`` (grasp frame, cube spawn, trajectories).
No FK annotation step required on the GPU machine.

Usage (Isaac Lab env):
  cd C:\\Users\\hugod\\project3
  python RLPD/scripts/replay_lerobot_in_isaac.py --episode 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401

_RLPD = Path(__file__).resolve().parents[1]
_PROJECT = _RLPD.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

_DEFAULT_ANN = _RLPD / "data" / "episodes.json"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--annotations", type=Path, default=_DEFAULT_ANN)
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--task", type=str, default="Isaac-SquintNative-Place-Replay-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--step_dt_s", type=float, default=0.033)
parser.add_argument("--physics_substeps", type=int, default=1)
parser.add_argument("--use_action_trajectory", action="store_true")
parser.add_argument("--headless", action="store_true")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.headless:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sim.eval2  # noqa: F401,E402
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from isaac_replay_common import set_cube_pose, set_robot_qpos_deg  # noqa: E402


def main() -> None:
    ann_path = args.annotations.resolve()
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    ep_key = str(args.episode)
    if ep_key not in data["episodes"]:
        raise KeyError(f"episode {args.episode} not in {ann_path}")
    ep = data["episodes"][ep_key]

    traj_key = "trajectory_action_deg" if args.use_action_trajectory else "trajectory_observation_state_deg"
    traj = ep[traj_key]
    cube_pose = ep["cube_pose_urdf_world"]

    print(f"[replay] episode {args.episode}  frames={len(traj)}  task={args.task}")
    print(f"[replay] grasp_local={ep['grasp_frame_local']}  cube_pose_urdf_world={cube_pose}")

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    base_env = env.unwrapped

    env.reset()
    set_cube_pose(base_env, cube_pose)
    if args.physics_substeps > 0:
        for _ in range(30):
            base_env.sim.step()
            base_env.scene.update(dt=base_env.physics_dt)

    for i, qdeg in enumerate(traj):
        set_robot_qpos_deg(base_env, qdeg)
        if args.physics_substeps > 0:
            for _ in range(args.physics_substeps):
                base_env.sim.step()
                base_env.scene.update(dt=base_env.physics_dt)
        if args.step_dt_s > 0 and not args.headless:
            time.sleep(args.step_dt_s)
        if i % 30 == 0:
            print(f"  frame {i}/{len(traj)}")

    print("[replay] done — close viewer or Ctrl+C")
    try:
        while simulation_app.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
