"""Termination terms — replicate Squint's success criterion exactly.

Squint's success (``envs/place.py:465``):

    success = is_item_above_bin
              & (~robot_touching_item)
              & is_robot_static
              & (~robot_touching_bin)

is_item_above_bin is XY-ONLY (no z check) — place.py:449-451.

We use the same contact/proximity gates as ``squint_rewards.py`` so the
reward-side success and the termination-side success are bit-identical.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Same constants as squint_rewards.py — keep in sync.
_BOWL_HALF_X = 0.0745
_BOWL_HALF_Y = 0.0745
_BOWL_HALF_Z = 0.0265
_BOWL_RADIUS = (_BOWL_HALF_X ** 2 + _BOWL_HALF_Y ** 2) ** 0.5
_CONTACT_FORCE_MIN = 0.5
_TOUCHING_FORCE_LIGHT = 0.01
_CUBE_NEAR_GRIPPER = 0.03


def _contact_force_mag(sensor: ContactSensor) -> torch.Tensor:
    forces = sensor.data.net_forces_w
    if forces is None:
        return torch.zeros(sensor.num_instances, device=sensor.device)
    return torch.linalg.norm(forces, dim=-1).sum(dim=-1)


def success(
    env: "ManagerBasedRLEnv",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    robot_static_qvel_thresh: float = 0.2,
) -> torch.Tensor:
    """Boolean (N,) — Squint canonical place.py:465 success predicate."""
    cube: RigidObject = env.scene[cube_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    cube_pos = cube.data.root_pos_w
    bowl_pos = bowl.data.root_pos_w
    gripper_idx = robot.body_names.index("gripper")
    tcp_pos = robot.data.body_pos_w[:, gripper_idx]
    env_origins = env.scene.env_origins

    # is_item_above_bin (XY only — matches place.py:449-451).
    dx = cube_pos[:, 0] - bowl_pos[:, 0]
    dy = cube_pos[:, 1] - bowl_pos[:, 1]
    is_item_above_bowl = (torch.abs(dx) < _BOWL_HALF_X) & (torch.abs(dy) < _BOWL_HALF_Y)

    # is_robot_static — qvel-arm norm threshold.
    qvel_arm = robot.data.joint_vel[:, :-1]
    is_robot_static = torch.linalg.norm(qvel_arm, dim=-1) <= robot_static_qvel_thresh

    # Contact + proximity gates (same as squint_rewards.py).
    gripper_sensor: ContactSensor = env.scene.sensors["gripper_contact"]
    jaw_sensor: ContactSensor = env.scene.sensors["jaw_contact"]
    F_g = _contact_force_mag(gripper_sensor)
    F_j = _contact_force_mag(jaw_sensor)
    any_contact = (F_g >= _TOUCHING_FORCE_LIGHT) | (F_j >= _TOUCHING_FORCE_LIGHT)

    cube_to_gripper = torch.linalg.norm(tcp_pos - cube_pos, dim=-1)
    robot_touching_item = any_contact & (cube_to_gripper < _CUBE_NEAR_GRIPPER)

    gripper_above_table = tcp_pos[:, 2] - env_origins[:, 2]
    robot_touching_bowl = any_contact & (
        (torch.linalg.norm(tcp_pos[:, :2] - bowl_pos[:, :2], dim=-1) < _BOWL_RADIUS + 0.03)
        & (gripper_above_table < 2.0 * _BOWL_HALF_Z)
    )

    return is_item_above_bowl & (~robot_touching_item) & is_robot_static & (~robot_touching_bowl)
