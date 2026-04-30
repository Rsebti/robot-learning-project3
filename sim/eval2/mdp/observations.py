"""Observation functions for Eval 2.

Each function returns a ``(num_envs, D)`` tensor that becomes one term of the
policy's observation vector. Isaac Lab concatenates them at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# v0 helpers (single block)
# ---------------------------------------------------------------------------
def block_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    """Position of the (single) block in the robot's base frame (3D)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    block: RigidObject = env.scene[block_cfg.name]
    block_pos_w = block.data.root_pos_w[:, :3]
    block_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], block_pos_w
    )
    return block_pos_b


def bowl_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    """Position of the bowl center in the robot's base frame (3D)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]
    bowl_pos_w = bowl.data.root_pos_w[:, :3]
    bowl_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], bowl_pos_w
    )
    return bowl_pos_b


# ---------------------------------------------------------------------------
# v1 helpers (two colored blocks + target color)
# ---------------------------------------------------------------------------
def block_red_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    block_cfg: SceneEntityCfg = SceneEntityCfg("block_red"),
) -> torch.Tensor:
    return _block_position_helper(env, robot_cfg, block_cfg)


def block_blue_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    block_cfg: SceneEntityCfg = SceneEntityCfg("block_blue"),
) -> torch.Tensor:
    return _block_position_helper(env, robot_cfg, block_cfg)


def _block_position_helper(env, robot_cfg, block_cfg):
    robot: RigidObject = env.scene[robot_cfg.name]
    block: RigidObject = env.scene[block_cfg.name]
    block_pos_w = block.data.root_pos_w[:, :3]
    block_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], block_pos_w
    )
    return block_pos_b


def target_color_one_hot(
    env: ManagerBasedRLEnv,
    num_classes: int = 2,
) -> torch.Tensor:
    """One-hot encoding of the per-env target color index (shape (num_envs, num_classes)).

    Reads ``env.target_color`` (set by ``reset_target_color`` event term). If it
    hasn't been set yet (very first call before any reset), defaults to all zeros
    on class 0.
    """
    if not hasattr(env, "target_color"):
        env.target_color = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    one_hot = torch.zeros(env.num_envs, num_classes, device=env.device)
    one_hot.scatter_(1, env.target_color.unsqueeze(-1), 1.0)
    return one_hot
