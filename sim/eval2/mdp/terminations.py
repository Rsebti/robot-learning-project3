"""Termination functions for Eval 2."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def success_block_in_bowl(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.04,
    z_max_above_bowl: float = 0.05,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
) -> torch.Tensor:
    """Episode ends with success when the block is inside the bowl."""
    block: RigidObject = env.scene[block_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]

    block_pos = block.data.root_pos_w
    bowl_pos = bowl.data.root_pos_w

    xy_distance = torch.norm(block_pos[:, :2] - bowl_pos[:, :2], dim=1)
    inside_xy = xy_distance < xy_threshold

    dz = block_pos[:, 2] - bowl_pos[:, 2]
    inside_z = (dz > -0.01) & (dz < z_max_above_bowl)

    return inside_xy & inside_z
