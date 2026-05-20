#!/usr/bin/env python3
"""
replay_lerobot_in_isaac.py — joint replay of annotated LeRobot demos in Isaac Lab.

Reads ``episodes.json`` from annotate_grasp_fk.py, spawns the cube at FK-inferred
pose, and kinematically replays ``observation.state`` (motor deg → rad).

Isaac vs ManiSkill
------------------
- **ManiSkill / Squint** is the original training stack for shipped Squint checkpoints
  (external repo). This project **ports** that policy to **Isaac Lab** for Eval2/RL.
- **This script is Isaac-only** — it uses the same pattern as
  ``sim/eval2/scripts/replay_squint_trajectory.py`` (``write_joint_state_to_sim``).
- For native ManiSkill replay you would need the Squint env + USD there; we do not
  vendor ManiSkill in project3.

Requires Isaac Lab venv (not the trim conda env alone).

Usage (after annotate_grasp_fk):
  cd <isaac_so_arm101 or Isaac-enabled env>
  python <path-to-project3>/sim/demo_replay/replay_lerobot_in_isaac.py \\
      --annotations outputs/datasets/projet3_demos_v1_grasp_fk/episodes.json \\
      --episode 0 \\
      --task Isaac-SquintNative-Place-Replay-v0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--annotations", type=Path, required=True,
                    help="episodes.json from annotate_grasp_fk.py")
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--task", type=str, default="Isaac-SquintNative-Place-Replay-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--step_dt_s", type=float, default=0.033,
                    help="Wall-clock pause between frames (30 Hz ~ 0.033).")
parser.add_argument("--physics_substeps", type=int, default=1,
                    help="sim.step() calls per frame (0 = snap only).")
parser.add_argument("--use_action_trajectory", action="store_true",
                    help="Replay action[] instead of observation.state (default: observation).")
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


def _set_cube_pose(env, pose7: list[float]) -> None:
    """pose = [x, y, z, qw, qx, qy, qz] URDF world (+ env origin)."""
    cube = env.scene["cube"]
    device = env.device
    p = torch.tensor(pose7[:3], device=device, dtype=torch.float32).unsqueeze(0)
    q = torch.tensor(pose7[3:], device=device, dtype=torch.float32).unsqueeze(0)
    p = p + env.scene.env_origins
    full = torch.cat([p, q], dim=-1)
    vel = torch.zeros(1, 6, device=device)
    cube.write_root_pose_to_sim(full)
    cube.write_root_velocity_to_sim(vel)


def _set_robot_qpos_deg(env, motor_deg: list[float]) -> None:
    from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig

    robot = env.scene["robot"]
    device = env.device
    q_urdf = MotorToUrdfConfig.load().motor_to_urdf_rad(np.asarray(motor_deg, dtype=float))
    qp = torch.tensor(q_urdf, device=device, dtype=torch.float32).unsqueeze(0)
    qv = torch.zeros_like(qp)
    robot.write_joint_state_to_sim(position=qp, velocity=qv)


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
    print(f"[replay] cube_pose_urdf_world={cube_pose}")

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    base_env = env.unwrapped

    env.reset()
    _set_cube_pose(base_env, cube_pose)
    if args.physics_substeps > 0:
        for _ in range(30):
            base_env.sim.step()
            base_env.scene.update(dt=base_env.physics_dt)

    for i, qdeg in enumerate(traj):
        _set_robot_qpos_deg(base_env, qdeg)
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
