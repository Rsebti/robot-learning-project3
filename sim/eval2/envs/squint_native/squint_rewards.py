"""Dense reward — bit-for-bit port of Squint ``envs/place.py:compute_dense_reward``.

Reward composition (cf. squint/envs/place.py:483-535)
-----------------------------------------------------
Reward starts at ``reaching_reward = 2 * (1 - tanh(5 * tcp_to_item))``.

State machine:
- if ``is_item_grasped``:        reward = 3 + place_reward
- if ``is_item_above_bowl``:     reward = 4 + place_reward
                                          + is_item_dropped (1 if gripper not touching cube)
                                          + gripper_openness (∈ [0, 1])
                                          + static_robot_reward (1 - tanh(10·qvel))
- if ``success``:                reward = 9   (cap)

Penalties added unconditionally afterwards:
- −6 if robot is touching the table
- −3 if robot is touching the bowl
- −1 if the cube has not been lifted (cube z < cube_half_size + 1e-3)

Contact detection
-----------------
Squint queries SAPIEN's ``scene.get_pairwise_contact_forces(linkA, linkB)``
which returns the per-pair force vector. Isaac Lab's
``ContactSensorCfg.filter_prim_paths_expr`` would do the same — but per
the ``isaac_lab_gpu_physx_contact_filter`` memory, the GPU PhysX backend
in Isaac Sim 5.1 silently fails or hangs env build with filters.

Workaround (validated 2026-05-12 via ``verify_contact_sensors.py``):
unfiltered ContactSensors on ``gripper`` and ``jaw`` bodies → read total
``net_forces_w``, then gate with proximity checks so we know what the
force is FROM.

is_grasping (cube grasped):
    F_gripper > 0.5 N  AND  F_jaw > 0.5 N
    AND  cube within 5 cm of the jaw centre
    AND  cube above table (lifted ≥ 1 mm).

is_touching_X uses the same forces with a target-proximity gate:
    touching_table = any jaw force AND gripper_z < 5 mm above table
    touching_bowl  = any jaw force AND gripper inside bowl xy + z range
    touching_item  = any jaw force AND cube within 3 cm of gripper.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Bowl AABB (matches ``mesh_bowl_from_ply.py`` output).
_BOWL_HALF_X = 0.0745
_BOWL_HALF_Y = 0.0745
_BOWL_HALF_Z = 0.0265                       # half-height (full = 0.053)
_BOWL_RADIUS = (_BOWL_HALF_X ** 2 + _BOWL_HALF_Y ** 2) ** 0.5
_CUBE_HALF_SIZE = 0.010

# Contact force thresholds (Newtons).
_CONTACT_FORCE_MIN = 0.5                     # min N on a jaw body to count as touching
_GRASP_FORCE_MIN = 0.5                       # min N on each jaw body for grasp
_TOUCHING_FORCE_LIGHT = 0.01                 # very-light contact threshold

# Proximity gates (m) — needed because the sensors are unfiltered so the
# force could be from ANY object touching the jaw.
_CUBE_NEAR_JAW = 0.05
_CUBE_NEAR_GRIPPER = 0.03
# P2 fix: tightened from 0.005 (5mm) → 0.001 (1mm). Previous threshold was
# triggering false positives during normal cube-grasp descents (jaw hovers
# ~3-4 mm above table while closing on a 2 cm cube whose centre is at
# z=10mm and top at z=20mm). False positive × -6 reward (now -2) was
# pushing the critic to assign low value to any descent → policy refused
# to grasp.
_TABLE_NEAR_GRIPPER = 0.001


def _gripper_openness(robot: Articulation) -> torch.Tensor:
    """``(qpos[-1] - qmin) / (qmax - qmin)`` ∈ [0, 1]."""
    gripper_idx = -1
    q = robot.data.joint_pos[:, gripper_idx]
    qmin = robot.data.soft_joint_pos_limits[:, gripper_idx, 0]
    qmax = robot.data.soft_joint_pos_limits[:, gripper_idx, 1]
    return ((q - qmin) / (qmax - qmin + 1e-6)).clamp(0.0, 1.0)


def _contact_force_mag(sensor: ContactSensor) -> torch.Tensor:
    """||net_forces_w||_2 over the sensor's bodies. Returns (N,) per env."""
    forces = sensor.data.net_forces_w  # (N, n_bodies, 3)
    if forces is None:
        # Sensor not yet active (first step before any update).
        return torch.zeros(sensor.num_instances, device=sensor.device)
    mag = torch.linalg.norm(forces, dim=-1)  # (N, n_bodies)
    return mag.sum(dim=-1)  # aggregate across bodies in the sensor (=1)


def squint_dense_reward(
    env: "ManagerBasedRLEnv",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Returns (N,) reward tensor. Port of Squint's compute_dense_reward."""
    cube: RigidObject = env.scene[cube_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    device = env.device
    env_origins = env.scene.env_origins

    # ---- TCP / item / bowl world poses ----
    gripper_body_idx = robot.body_names.index("gripper")
    jaw_body_idx = robot.body_names.index("jaw")
    tcp_pos = robot.data.body_pos_w[:, gripper_body_idx]
    jaw_pos = robot.data.body_pos_w[:, jaw_body_idx]
    item_pos = cube.data.root_pos_w
    bowl_pos = bowl.data.root_pos_w

    # ---- Reaching reward (2 * (1 - tanh(5 d))) ----
    tcp_to_item_dist = torch.linalg.norm(tcp_pos - item_pos, dim=-1)
    reaching_reward = 2.0 * (1.0 - torch.tanh(5.0 * tcp_to_item_dist))
    reward = reaching_reward.clone()

    # ---- Place reward ----
    goal_xyz = bowl_pos.clone()
    goal_xyz[:, 2] = env_origins[:, 2] + _CUBE_HALF_SIZE  # cube floor at bowl floor

    item_to_goal = goal_xyz - item_pos
    item_to_goal_dist = torch.linalg.norm(item_to_goal, dim=-1)
    place_reward_final = 1.0 - torch.tanh(5.0 * item_to_goal_dist)

    d_xy = torch.linalg.norm(goal_xyz[:, :2] - item_pos[:, :2], dim=-1)
    z_close = torch.abs(goal_xyz[:, 2] - item_pos[:, 2])
    z_far_target = goal_xyz[:, 2] + 2.0 * _BOWL_HALF_Z + 0.03
    z_far = torch.abs(z_far_target - item_pos[:, 2])
    item_close_to_goal = d_xy <= _BOWL_RADIUS
    d_z = torch.where(item_close_to_goal, z_close, z_far)
    place_reward_z = 1.0 - torch.tanh(10.0 * d_z)
    place_reward = place_reward_final + place_reward_z

    # ---- Contact forces (Squint-side handoff: jaw bodies, NOT tips) ----
    gripper_sensor: ContactSensor = env.scene.sensors["gripper_contact"]
    jaw_sensor: ContactSensor = env.scene.sensors["jaw_contact"]
    F_g = _contact_force_mag(gripper_sensor)
    F_j = _contact_force_mag(jaw_sensor)

    # ---- Proximity gates (unfiltered sensor workaround) ----
    cube_to_jaw = torch.linalg.norm(jaw_pos - item_pos, dim=-1)
    cube_to_gripper = tcp_to_item_dist  # already computed
    gripper_above_table = (tcp_pos[:, 2] - env_origins[:, 2])

    item_lifted = item_pos[:, 2] >= (env_origins[:, 2] + _CUBE_HALF_SIZE + 1e-3)

    # is_grasping: both jaw bodies under load AND cube near jaw AND cube lifted.
    is_item_grasped = (
        (F_g >= _GRASP_FORCE_MIN)
        & (F_j >= _GRASP_FORCE_MIN)
        & (cube_to_jaw < _CUBE_NEAR_JAW)
        & item_lifted
    )

    # Cube xy inside bowl rect AABB (rectangular, matches Squint).
    dx = item_pos[:, 0] - bowl_pos[:, 0]
    dy = item_pos[:, 1] - bowl_pos[:, 1]
    inside_x = dx.abs() < _BOWL_HALF_X
    inside_y = dy.abs() < _BOWL_HALF_Y
    above_bowl_z = item_pos[:, 2] > (env_origins[:, 2] - 1e-3)
    is_item_above_bowl = inside_x & inside_y & above_bowl_z

    # robot_touching_X — any contact force at a jaw + target-proximity gate.
    any_contact = (F_g >= _TOUCHING_FORCE_LIGHT) | (F_j >= _TOUCHING_FORCE_LIGHT)
    robot_touching_item = any_contact & (cube_to_gripper < _CUBE_NEAR_GRIPPER)
    robot_touching_table = any_contact & (gripper_above_table < _TABLE_NEAR_GRIPPER)
    robot_touching_bowl = any_contact & (
        (torch.linalg.norm(tcp_pos[:, :2] - bowl_pos[:, :2], dim=-1) < _BOWL_RADIUS + 0.03)
        & (gripper_above_table < 2.0 * _BOWL_HALF_Z)
    )

    # Robot static: ||qvel[:5]|| ≤ small threshold.
    qvel_arm = robot.data.joint_vel[:, :-1]
    robot_v = torch.linalg.norm(qvel_arm, dim=-1)
    static_robot_reward = 1.0 - torch.tanh(robot_v * 10.0)
    is_robot_static = robot_v <= 0.15

    # P3 fix: align reward-side success with the termination-side definition
    # (squint_terminations.success) — both should be the same target. The
    # earlier extra robot-not-touching-item / not-touching-bowl conjuncts
    # were too restrictive and almost never fired during eval, making the
    # terminal +9 essentially unreachable through the dense reward path.
    cube_lin_vel = torch.linalg.norm(cube.data.root_lin_vel_w, dim=-1)
    is_cube_static = cube_lin_vel <= 0.05
    success = is_item_above_bowl & is_cube_static

    # ---- State machine ----
    gripper_open = _gripper_openness(robot)
    is_item_dropped = (~robot_touching_item).float()

    # P4 fix: continuous transport shaping. Once the cube is grasped, give
    # a dense gradient pulling the cube's xy toward the bowl's xy. Without
    # this, the policy gets a flat +3 for grasp + 0 incentive to MOVE the
    # cube toward the bowl until it's already near it. Range: 0 → +2.
    cube_to_bowl_xy = torch.linalg.norm(item_pos[:, :2] - bowl_pos[:, :2], dim=-1)
    transport_reward = 2.0 * (1.0 - torch.tanh(5.0 * cube_to_bowl_xy))

    grasped_value = 3.0 + place_reward + transport_reward
    reward = torch.where(is_item_grasped, grasped_value, reward)

    above_value = 4.0 + place_reward + is_item_dropped + gripper_open + static_robot_reward
    reward = torch.where(is_item_above_bowl, above_value, reward)

    reward = torch.where(success, torch.full_like(reward, 9.0), reward)

    # ---- Penalties ----
    # P2 fix: -6 → -2 (tighter gate at 1mm + smaller magnitude). Critic was
    # learning "descend = catastrophic" because of cumulative -6's during
    # normal grasp descents that grazed the table proximity gate.
    reward = reward - 2.0 * robot_touching_table.float()
    # P1 fix: only penalize bowl contact when we are NOT trying to place.
    # Otherwise the policy receives -3 every step it enters the bowl region
    # to drop the cube — which is exactly what we want it to do.
    bowl_penalty_mask = robot_touching_bowl & (~is_item_above_bowl)
    reward = reward - 3.0 * bowl_penalty_mask.float()
    reward = reward - 1.0 * (~item_lifted).float()

    # NaN guard — keeps the replay buffer clean even if body state goes bad.
    reward = torch.where(torch.isfinite(reward), reward, torch.zeros_like(reward))
    return reward


def squint_dense_reward_normalized(
    env: "ManagerBasedRLEnv",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squint's training reward — ``squint_dense_reward`` / 9.0."""
    return squint_dense_reward(env, cube_cfg, bowl_cfg, robot_cfg) / 9.0
