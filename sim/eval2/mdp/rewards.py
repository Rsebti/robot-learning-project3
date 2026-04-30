"""Reward functions for Eval 2 (pick-and-place in a bowl).

The functions follow Isaac Lab's conventions: each takes the env as first
argument plus task-specific parameters via ``RewTerm.params``, and returns a
``(num_envs,)`` tensor of per-env reward values for that term. The total
reward is the weighted sum across active terms (weights set in ``RewardsCfg``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Reaching: gripper close to the (target) block.
# ---------------------------------------------------------------------------
def block_ee_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Tanh-shaped reward for the gripper getting close to the block.

    Returns ~1.0 when the gripper is on the block, decays to 0 with distance.
    """
    block: RigidObject = env.scene[block_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    block_pos_w = block.data.root_pos_w  # (num_envs, 3)
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)
    distance = torch.norm(block_pos_w - ee_pos_w, dim=1)  # (num_envs,)

    return 1.0 - torch.tanh(distance / std)


# ---------------------------------------------------------------------------
# Lifting: block above a minimal height.
# ---------------------------------------------------------------------------
def block_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    """Binary reward (1.0/0.0) — block has been lifted above ``minimal_height``."""
    block: RigidObject = env.scene[block_cfg.name]
    return torch.where(block.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


# ---------------------------------------------------------------------------
# Placement: block close to the bowl, conditioned on having lifted it first.
# ---------------------------------------------------------------------------
def block_to_bowl_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
) -> torch.Tensor:
    """Tanh-shaped reward for the block being close to the bowl.

    Multiplied by the lifted-condition so this only rewards the agent once it
    has actually picked the block up — prevents the policy from getting reward
    by sliding the block on the table.
    """
    block: RigidObject = env.scene[block_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]

    block_pos_w = block.data.root_pos_w  # (num_envs, 3)
    bowl_pos_w = bowl.data.root_pos_w  # (num_envs, 3)
    distance = torch.norm(block_pos_w - bowl_pos_w, dim=1)  # (num_envs,)

    lifted = (block_pos_w[:, 2] > minimal_height).float()
    return lifted * (1.0 - torch.tanh(distance / std))


# ---------------------------------------------------------------------------
# Success bonus: block is inside the bowl.
# ---------------------------------------------------------------------------
def block_in_bowl(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.04,
    z_max_above_bowl: float = 0.05,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
) -> torch.Tensor:
    """Sparse 0/1 reward — block is positioned inside the bowl.

    "Inside" is approximated as: xy distance to bowl center < ``xy_threshold``
    AND block z is between bowl z and bowl z + ``z_max_above_bowl``.
    """
    block: RigidObject = env.scene[block_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]

    block_pos = block.data.root_pos_w  # (num_envs, 3)
    bowl_pos = bowl.data.root_pos_w  # (num_envs, 3)

    xy_distance = torch.norm(block_pos[:, :2] - bowl_pos[:, :2], dim=1)
    inside_xy = xy_distance < xy_threshold

    dz = block_pos[:, 2] - bowl_pos[:, 2]
    inside_z = (dz > -0.01) & (dz < z_max_above_bowl)

    return (inside_xy & inside_z).float()
