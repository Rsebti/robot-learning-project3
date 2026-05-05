"""Reward functions for Eval 2 (pick-and-place in a bowl).

The functions follow Isaac Lab's conventions: each takes the env as first
argument plus task-specific parameters via ``RewTerm.params``, and returns a
``(num_envs,)`` tensor of per-env reward values for that term. The total
reward is the weighted sum across active terms (weights set in ``RewardsCfg``).

v0 functions operate on a single block ``"block"``.
v1 functions operate on two blocks ``"block_red"`` and ``"block_blue"`` and
dispatch to the correct one according to ``env.target_color`` (a long tensor
of shape (num_envs,) with values 0 = red, 1 = blue).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ===========================================================================
# Helpers
# ===========================================================================
def _target_block_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the world position of the per-env target block (num_envs, 3)."""
    red: RigidObject = env.scene["block_red"]
    blue: RigidObject = env.scene["block_blue"]
    target_idx = env.target_color  # (num_envs,) long
    # torch.where on per-env basis to pick red or blue
    is_red = (target_idx == 0).unsqueeze(-1)  # (num_envs, 1)
    return torch.where(is_red, red.data.root_pos_w, blue.data.root_pos_w)


def _distractor_block_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the world position of the *non*-target block."""
    red: RigidObject = env.scene["block_red"]
    blue: RigidObject = env.scene["block_blue"]
    target_idx = env.target_color
    is_red = (target_idx == 0).unsqueeze(-1)
    # If target is red, distractor is blue, and vice versa.
    return torch.where(is_red, blue.data.root_pos_w, red.data.root_pos_w)


# ===========================================================================
# v0 single-block rewards (kept for backwards-compat with Eval2-PickInBowl-v0)
# ===========================================================================
def block_ee_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    block: RigidObject = env.scene[block_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    block_pos_w = block.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(block_pos_w - ee_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def block_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    block: RigidObject = env.scene[block_cfg.name]
    return torch.where(block.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def block_to_bowl_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    block: RigidObject = env.scene[block_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]
    block_pos_w = block.data.root_pos_w
    bowl_pos_w = bowl.data.root_pos_w
    distance = torch.norm(block_pos_w - bowl_pos_w, dim=1)
    lifted = (block_pos_w[:, 2] > minimal_height).float()
    return lifted * (1.0 - torch.tanh(distance / std))


def block_in_bowl(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.04,
    z_max_above_bowl: float = 0.05,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    block: RigidObject = env.scene[block_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]
    block_pos = block.data.root_pos_w
    bowl_pos = bowl.data.root_pos_w
    xy_distance = torch.norm(block_pos[:, :2] - bowl_pos[:, :2], dim=1)
    inside_xy = xy_distance < xy_threshold
    dz = block_pos[:, 2] - bowl_pos[:, 2]
    inside_z = (dz > -0.01) & (dz < z_max_above_bowl)
    return (inside_xy & inside_z).float()


# ===========================================================================
# v1 two-block rewards (target-aware)
# ===========================================================================
def target_block_ee_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Tanh reward for the gripper getting close to the *target* block."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    target_pos_w = _target_block_pos(env)
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(target_pos_w - ee_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def target_block_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
) -> torch.Tensor:
    """Binary reward — the *target* block is lifted above ``minimal_height``."""
    target_pos_w = _target_block_pos(env)
    return torch.where(target_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def target_block_to_bowl_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    """Tanh reward for the *target* block being close to the bowl, gated on lift."""
    bowl: RigidObject = env.scene[bowl_cfg.name]
    target_pos_w = _target_block_pos(env)
    bowl_pos_w = bowl.data.root_pos_w
    distance = torch.norm(target_pos_w - bowl_pos_w, dim=1)
    lifted = (target_pos_w[:, 2] > minimal_height).float()
    return lifted * (1.0 - torch.tanh(distance / std))


def target_block_in_bowl(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.04,
    z_max_above_bowl: float = 0.05,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    """Sparse 0/1 reward — *target* block is positioned inside the bowl."""
    bowl: RigidObject = env.scene[bowl_cfg.name]
    target_pos = _target_block_pos(env)
    bowl_pos = bowl.data.root_pos_w
    xy_distance = torch.norm(target_pos[:, :2] - bowl_pos[:, :2], dim=1)
    inside_xy = xy_distance < xy_threshold
    dz = target_pos[:, 2] - bowl_pos[:, 2]
    inside_z = (dz > -0.01) & (dz < z_max_above_bowl)
    return (inside_xy & inside_z).float()


def distractor_block_disturbed(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.025,
) -> torch.Tensor:
    """Penalty signal — the *distractor* (wrong-color) block has been lifted.

    Returns 1.0 when the distractor is moving up (wrong action), 0.0 otherwise.
    Apply with a *negative* weight in ``RewardsCfg`` to penalize.
    """
    distractor_pos_w = _distractor_block_pos(env)
    return torch.where(distractor_pos_w[:, 2] > height_threshold, 1.0, 0.0)


def action_l2_norm(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 norm of the last action, per env (shape: (num_envs,)).

    Pairs with a *negative* weight in ``RewardsCfg`` to discourage the actor
    from learning very large action magnitudes. Without this, the previous
    1k-iter run drove the actor mean to ~+/-15, which after the 0.5 action
    scale saturates the joint targets at ~+/-7.5 rad — well past joint
    limits — and prevents the policy from refining a real solution.
    """
    actions = env.action_manager.action  # (num_envs, action_dim)
    return torch.linalg.norm(actions, dim=-1)


# ---------------------------------------------------------------------------
# v1.4 milestone rewards: explicit sparse signals at the key task waypoints
# (grasp succeeded, block lifted above bowl, block dropped in bowl).
# Without these, PPO only sees a smooth dense reward landscape and can't
# tell that the "grasp -> lift -> place" chain is the actual goal — it just
# tries to maximize the sum of dense terms, which favors hovering near the
# bowl forever.
# ---------------------------------------------------------------------------
def target_block_grasped(
    env: ManagerBasedRLEnv,
    gripper_closed_threshold: float = 0.15,
    ee_to_block_threshold: float = 0.04,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Binary signal — gripper is closed AND the target block is within
    ``ee_to_block_threshold`` of the end-effector frame.

    A reasonable proxy for "the target block is currently being held":
    gripper joint position below the closed threshold means the jaws are
    pressing on something, and a small EE-to-block distance means that
    something is the block.

    Pairs with a *positive* weight (e.g. 50) in ``RewardsCfg`` to give a
    clear sparse signal when the policy first achieves a successful grasp,
    even before any lifting.
    """
    robot = env.scene["robot"]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Gripper joint position (binary action open=0.5, close=0.0 in our cfg).
    gripper_idx = robot.data.joint_names.index("gripper")
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    is_closed = gripper_pos < gripper_closed_threshold

    # Distance from end-effector to target block (in world frame).
    target_pos_w = _target_block_pos(env)
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(ee_pos_w - target_pos_w, dim=1)
    is_close = dist < ee_to_block_threshold

    return (is_closed & is_close).float()


def target_block_above_bowl(
    env: ManagerBasedRLEnv,
    height_above: float = 0.05,
    xy_threshold: float = 0.10,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    """Binary signal — target block is at least ``height_above`` meters above
    the bowl's z, AND its xy is within ``xy_threshold`` of the bowl center.

    Triggers right before the placement: the policy is hovering the block
    over the bowl, ready to drop. Pairs with a positive weight (e.g. 100) to
    encourage the policy to bring the block to that staging position before
    the final release.
    """
    bowl: RigidObject = env.scene[bowl_cfg.name]
    target_pos = _target_block_pos(env)
    bowl_pos_w = bowl.data.root_pos_w

    is_above = target_pos[:, 2] > bowl_pos_w[:, 2] + height_above
    xy_dist = torch.norm(target_pos[:, :2] - bowl_pos_w[:, :2], dim=1)
    is_above_bowl = xy_dist < xy_threshold

    return (is_above & is_above_bowl).float()
