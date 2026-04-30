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


def block_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    """Position of the block in the robot's base frame (3D)."""
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
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
) -> torch.Tensor:
    """Position of the bowl in the robot's base frame (3D).

    For Eval 2 the bowl xyz is the runtime-configurable target (TA spec). In
    v0 the bowl is fixed; in v1+ it will be randomized per episode and this
    observation lets the policy see where to drop the block.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]
    bowl_pos_w = bowl.data.root_pos_w[:, :3]
    bowl_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], bowl_pos_w
    )
    return bowl_pos_b
