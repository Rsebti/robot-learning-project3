"""Shared helpers for Isaac demo replay / verify (reads pre-built episodes.json)."""
from __future__ import annotations

import numpy as np
import torch


def set_cube_pose(env, pose7: list[float]) -> None:
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


def set_robot_qpos_deg(env, motor_deg: list[float]) -> None:
    from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig

    robot = env.scene["robot"]
    device = env.device
    q_urdf = MotorToUrdfConfig.load().motor_to_urdf_rad(np.asarray(motor_deg, dtype=float))
    qp = torch.tensor(q_urdf, device=device, dtype=torch.float32).unsqueeze(0)
    qv = torch.zeros_like(qp)
    robot.write_joint_state_to_sim(position=qp, velocity=qv)


def tip_user_from_motor_deg(motor_deg: list[float]) -> np.ndarray:
    from toolset.kinematics.config import KinematicsConfig
    from toolset.kinematics.ik_relative import urdf_xyz_to_user
    from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig
    from toolset.kinematics.urdf_fk import SO101FK

    kcfg = KinematicsConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)
    mcfg = MotorToUrdfConfig.load()
    q = mcfg.motor_to_urdf_rad(np.asarray(motor_deg, dtype=float).reshape(-1)[:6])
    return urdf_xyz_to_user(fk.fk(q, target="gripper_tip")["position"])
