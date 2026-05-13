"""Squint-port reward functions for SO-101 Lift and Place tasks.

Translates the dense reward shaping from
``envs/lift.py::Lift.compute_dense_reward`` and
``envs/place.py::Place.compute_dense_reward`` (from
https://github.com/aalmuzairee/squint) into Isaac Lab reward terms.

ManiSkill returns a single big tensor from ``compute_dense_reward`` after
internally combining multi-stage logic (``if grasped: ...``, ``if above
bin: ...``). Isaac Lab's manager-based pattern composes the same reward
from independent RewardTermCfg entries with weights. We split each Squint
stage into a separate function:

Squint Lift = sum of
  - 1 - tanh(5 d_ee_cube)        (reach, weight 1.0)
  - is_grasped                   (grasp bonus, weight 1.0)
  - exp(-2 * d_qpos_rest) * is_grasped  (place-back bonus, weight 1.0)
  - -3 * robot_touching_table    (penalty)
  - -(~is_lifted)                (encourage fast lifting)

Squint Place = sum of
  - 2 * (1 - tanh(5 d_ee_cube))                          (reach, weight 1.0)
  - 3 * is_grasped                                       (grasp bonus)
  - place_final = 1 - tanh(5 d_cube_goal)                (always)
  - place_z_far = 1 - tanh(10 d_z_above)                 (when xy NOT close)
  - place_z_close = 1 - tanh(10 d_z_to_goal)             (when xy IS close)
  - 4 * is_above_bin                                     (stage bonus)
  - gripper_openness * is_above_bin                      (drop incentive)
  - 9 * success                                          (terminal bonus)
  - -6 * robot_touching_table
  - -(~is_lifted)
"""
from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _goal_pos_world(env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return commanded goal position in WORLD frame (B, 3)."""
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_b = command[:, :3]
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_b
    )
    return goal_pos_w


def _ee_pos_world(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg, target_idx: int = 1) -> torch.Tensor:
    """Return the jaw fingertip (target index 1 by LeIsaac convention) world pos (B, 3)."""
    ee_frame = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, target_idx, :]


# ---------------------------------------------------------------------------
# Squint Lift reward terms
# ---------------------------------------------------------------------------


def squint_reach_dense(
    env: ManagerBasedRLEnv,
    sharpness: float = 5.0,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Squint reaching reward: ``1 - tanh(sharpness * d_ee_cube)``.

    Replaces Isaac Lab's ``object_ee_distance(std=...)`` which uses
    ``1 - tanh(d / std)``. Same shape, just different parametrization.
    Squint defaults to sharpness=5.0 (equivalent to std=0.2).
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_pos_w = _ee_pos_world(env, ee_frame_cfg)
    distance = torch.norm(cube.data.root_pos_w - ee_pos_w, dim=-1)
    return 1.0 - torch.tanh(sharpness * distance)


def squint_not_lifted_penalty(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.05,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    robot_base_name: str = "base",
) -> torch.Tensor:
    """Returns 1.0 when cube is NOT lifted, 0.0 when lifted.

    Use with a negative weight (e.g. -1.0) — Squint's "encourage picking
    item fast" term. Pays a per-step cost as long as the cube hasn't been
    raised, which accelerates the policy past the reach plateau.
    """
    from .rewards import cube_lifted_above_base

    is_lifted = cube_lifted_above_base(
        env,
        height_threshold=height_threshold,
        cube_cfg=cube_cfg,
        robot_cfg=robot_cfg,
        robot_base_name=robot_base_name,
    )
    return 1.0 - is_lifted


def squint_place_back_bonus(
    env: ManagerBasedRLEnv,
    rest_qpos: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    decay: float = 2.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    diff_threshold: float = 0.02,
    grasp_threshold: float = 0.26,
) -> torch.Tensor:
    """``exp(-decay * |qpos - rest_qpos|) * is_grasped``.

    Squint Lift's "lift back to rest pose while holding cube" reward.
    Without this, the policy can grasp+hover but doesn't learn the lift-up
    motion. With it, picking up and returning to a known pose (typically
    arm-up) is the dominant strategy.
    """
    from .rewards import cube_grasped

    robot: Articulation = env.scene[robot_cfg.name]
    qpos = robot.data.joint_pos[:, :-1]  # drop gripper joint (last)
    rest = torch.tensor(rest_qpos[:qpos.shape[-1]], device=env.device, dtype=qpos.dtype)
    distance = torch.norm(qpos - rest.unsqueeze(0), dim=-1)
    is_grasped = cube_grasped(
        env,
        diff_threshold=diff_threshold,
        grasp_threshold=grasp_threshold,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        cube_cfg=cube_cfg,
    )
    return torch.exp(-decay * distance) * is_grasped


# ---------------------------------------------------------------------------
# Squint Place reward terms (in addition to Lift terms above)
# ---------------------------------------------------------------------------


def squint_place_final_dense(
    env: ManagerBasedRLEnv,
    sharpness: float = 5.0,
    command_name: str = "object_pose",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``1 - tanh(sharpness * d_cube_to_goal)`` — always active.

    Squint Place's ``place_reward_final``. Drives the cube toward the
    goal position (which we use as a virtual bin marker).
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    goal_pos_w = _goal_pos_world(env, command_name, robot_cfg)
    distance = torch.norm(cube.data.root_pos_w - goal_pos_w, dim=-1)
    return 1.0 - torch.tanh(sharpness * distance)


def squint_place_z_staged(
    env: ManagerBasedRLEnv,
    bin_xy_radius: float = 0.05,
    hover_offset: float = 0.06,
    sharpness: float = 10.0,
    command_name: str = "object_pose",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squint Place's two-stage Z reward.

    - When the cube xy is FAR from goal xy (``> bin_xy_radius``): reward
      goes up as cube approaches the *hover* point (goal_z + hover_offset).
      This shapes the trajectory through "lift above the bin".
    - When cube xy is CLOSE to goal xy (``<= bin_xy_radius``): reward goes
      up as cube approaches the *exact* goal z (descent into bin).

    Result is ``1 - tanh(sharpness * d_z_staged)`` where ``d_z_staged``
    is the appropriate z target depending on xy distance.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    goal_pos_w = _goal_pos_world(env, command_name, robot_cfg)

    diff = cube.data.root_pos_w - goal_pos_w
    xy_dist = torch.norm(diff[:, :2], dim=-1)
    z_dist_close = torch.abs(diff[:, 2])               # exact goal z
    z_dist_far = torch.abs(diff[:, 2] - hover_offset)  # hover above goal z

    is_close_xy = xy_dist <= bin_xy_radius
    z_target = torch.where(is_close_xy, z_dist_close, z_dist_far)
    return 1.0 - torch.tanh(sharpness * z_target)


def squint_is_above_bin(
    env: ManagerBasedRLEnv,
    bin_half_x: float = 0.05,
    bin_half_y: float = 0.05,
    command_name: str = "object_pose",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Returns 1.0 if cube xy is inside the bin footprint, else 0.0.

    Used as the gate for the Squint Place ``above_bin`` stage bonus.
    The bin half-extents default to 5×5 cm (small bowl).
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    goal_pos_w = _goal_pos_world(env, command_name, robot_cfg)
    diff = cube.data.root_pos_w - goal_pos_w
    inside_x = torch.abs(diff[:, 0]) < bin_half_x
    inside_y = torch.abs(diff[:, 1]) < bin_half_y
    return (inside_x & inside_y).float()


def squint_place_success(
    env: ManagerBasedRLEnv,
    bin_half_x: float = 0.05,
    bin_half_y: float = 0.05,
    z_tolerance: float = 0.02,
    static_qvel_threshold: float = 0.1,
    command_name: str = "object_pose",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Place success bonus (Squint's terminal reward = 9).

    Requires: cube inside bin xy AND cube z near goal z (placed, not
    hovering) AND robot static (released the cube).
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    goal_pos_w = _goal_pos_world(env, command_name, robot_cfg)

    diff = cube.data.root_pos_w - goal_pos_w
    inside_x = torch.abs(diff[:, 0]) < bin_half_x
    inside_y = torch.abs(diff[:, 1]) < bin_half_y
    z_close = torch.abs(diff[:, 2]) < z_tolerance
    qvel_norm = torch.norm(robot.data.joint_vel, dim=-1)
    robot_static = qvel_norm < static_qvel_threshold
    return (inside_x & inside_y & z_close & robot_static).float()
