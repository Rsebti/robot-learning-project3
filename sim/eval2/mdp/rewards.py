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


# ===========================================================================
# v1.7 — Stage transition tracking (Action #1 from convergence_methods.md)
#
# Replaces the v1.4/v1.5 "independent milestone bonuses" with a single
# stage-progression reward. Each env tracks the HIGHEST stage reached so far
# in the episode (env.episode_max_stage). The reward at every step is the
# value of that max stage. Once you reach a stage, you can't lose its reward
# during the same episode (unless you regress, which is penalized).
#
# Why this beats independent bonuses (v1.4/v1.5):
#   - PPO can't game by hovering an empty gripper above the bowl: it never
#     reaches stage 4 without first reaching stage 2 (grasped) and 3 (lifted).
#   - The reward landscape is monotonic in task progress: every step in the
#     wrong direction loses its potential. PPO has no consolation plateau to
#     stall on.
#
# Stages:
#   0 default              nothing achieved
#   1 approach             gripper close to target block
#   2 grasped              gripper closed and at the block
#   3 grasped + lifted     block held above the table
#   4 grasped + above bowl block held + xy near bowl center
#   5 success              block in bowl AND released
# ===========================================================================
def _ensure_stage_buffer(env: "ManagerBasedRLEnv") -> None:
    """Lazily create env.episode_max_stage (per-env long tensor)."""
    if not hasattr(env, "episode_max_stage"):
        env.episode_max_stage = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )


def _compute_current_stage(
    env: "ManagerBasedRLEnv",
    gripper_closed_threshold: float = 0.15,
    ee_to_block_threshold: float = 0.04,
    approach_threshold: float = 0.05,
    lift_threshold: float = 0.05,
    above_bowl_xy_threshold: float = 0.10,
    in_bowl_xy_threshold: float = 0.06,
    in_bowl_z_max: float = 0.10,
) -> torch.Tensor:
    """Compute the per-env current task stage (long tensor (N,) in [0..5])."""
    robot = env.scene["robot"]
    bowl: RigidObject = env.scene["bowl_floor"]
    ee_frame: FrameTransformer = env.scene["ee_frame"]

    target_pos_w = _target_block_pos(env)                              # (N, 3)
    bowl_pos_w = bowl.data.root_pos_w                                  # (N, 3)
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]                   # (N, 3)

    # Grasp: gripper closed AND EE near target block
    gripper_idx = robot.data.joint_names.index("gripper")
    is_closed = robot.data.joint_pos[:, gripper_idx] < gripper_closed_threshold
    ee_dist = torch.norm(ee_pos_w - target_pos_w, dim=1)
    grasped = is_closed & (ee_dist < ee_to_block_threshold)            # (N,) bool

    # Bowl placement geometry
    xy_dist_bowl = torch.norm(target_pos_w[:, :2] - bowl_pos_w[:, :2], dim=1)
    dz = target_pos_w[:, 2] - bowl_pos_w[:, 2]
    in_bowl = (xy_dist_bowl < in_bowl_xy_threshold) & (dz > -0.01) & (dz < in_bowl_z_max)

    # Stage assignments — each is 0 (not reached) or its index (reached).
    s1 = (ee_dist < approach_threshold).long() * 1
    s2 = grasped.long() * 2
    lifted = (target_pos_w[:, 2] > lift_threshold).long()
    s3 = grasped.long() * lifted * 3
    s4 = grasped.long() * lifted * (xy_dist_bowl < above_bowl_xy_threshold).long() * 4
    s5 = (in_bowl & ~grasped).long() * 5

    return torch.stack([s1, s2, s3, s4, s5], dim=-1).max(dim=-1).values


def stage_progress_reward(
    env: "ManagerBasedRLEnv",
    transition_bonuses: tuple[float, ...] = (0.0, 5.0, 20.0, 50.0, 100.0, 500.0),
) -> torch.Tensor:
    """One-shot bonus when the policy transitions to a NEW (higher) stage.

    Returns ``transition_bonuses[new_stage]`` only on the step where
    ``env.episode_max_stage`` strictly increases, otherwise 0. Updates the
    buffer in place.

    Why one-shot (delta) instead of constant per-step:
        Per-step rewards on the max stage let the policy farm a low stage
        forever. e.g. with weights (1, 5, 20, 100, 1000) per step and a
        300-step horizon, "stage 4 max forever" yields 100*300 = 30 000
        cumulative reward, while "reach stage 5 then terminate" yields
        only 1 000 (since the success termination ends the episode at
        step S < 300). The policy learns to never commit to the final
        release. With one-shot transition bonuses, there is NO cumulative
        advantage in dwelling on a stage — every reward only fires once
        per episode, and the only way to maximize total reward is to
        traverse all 5 transitions.

    Pair with weight=1.0 in RewardsCfg. Bonuses heavily favour the final
    transition (500 vs 100 for stage 4) so that committing to the success
    sequence dominates any partial progress.
    """
    _ensure_stage_buffer(env)
    current = _compute_current_stage(env)
    transitioned = current > env.episode_max_stage
    env.episode_max_stage = torch.maximum(env.episode_max_stage, current)
    bonuses = torch.tensor(transition_bonuses, device=env.device, dtype=torch.float32)
    return bonuses[current] * transitioned.float()


def stage_regression_penalty(
    env: "ManagerBasedRLEnv",
) -> torch.Tensor:
    """Penalty signal (1.0 per step) when the current stage drops below the
    episode's high-water mark (e.g. block dropped after grasp).

    Pair with a NEGATIVE weight (e.g. -10) in RewardsCfg. This creates the
    recovery learning signal — without it, a drop just pauses the
    stage_progress reward but doesn't actively punish.
    """
    _ensure_stage_buffer(env)
    current = _compute_current_stage(env)
    return (current < env.episode_max_stage).float()


def target_block_above_bowl_conditional(
    env: ManagerBasedRLEnv,
    height_above: float = 0.05,
    xy_threshold: float = 0.10,
    gripper_closed_threshold: float = 0.15,
    ee_to_block_threshold: float = 0.04,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Above-bowl reward CONDITIONAL on a successful grasp at the same step.

    Returns 1 only when BOTH:
      - the target block is hovering above the bowl (same condition as
        ``target_block_above_bowl``), AND
      - the gripper is currently grasping the target block (same condition
        as ``target_block_grasped``).

    This blocks the v1.4 gaming loop where the policy picked up the
    "above_bowl" milestone reward by hovering an EMPTY gripper above the
    bowl, never grasping anything. With the AND-gating, hovering without a
    grasp scores 0, so PPO must grasp before it can collect the bonus.

    See ``target_block_above_bowl`` and ``target_block_grasped`` for the
    individual conditions. Pair with the same weight as the original
    above_bowl term (e.g. 100).
    """
    # Above-bowl geometry condition
    bowl: RigidObject = env.scene[bowl_cfg.name]
    target_pos = _target_block_pos(env)
    bowl_pos_w = bowl.data.root_pos_w
    is_above = target_pos[:, 2] > bowl_pos_w[:, 2] + height_above
    xy_dist = torch.norm(target_pos[:, :2] - bowl_pos_w[:, :2], dim=1)
    is_above_bowl_geom = xy_dist < xy_threshold

    # Grasp condition (gripper closed AND EE close to target block)
    robot = env.scene["robot"]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    gripper_idx = robot.data.joint_names.index("gripper")
    is_closed = robot.data.joint_pos[:, gripper_idx] < gripper_closed_threshold
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    is_close = torch.norm(ee_pos_w - target_pos, dim=1) < ee_to_block_threshold
    is_grasped = is_closed & is_close

    return (is_above & is_above_bowl_geom & is_grasped).float()
