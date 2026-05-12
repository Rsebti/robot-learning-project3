"""Custom observation terms for our LeIsaac-based RL tasks.

We expose `wrist_image_features`: a 512-D feature vector produced by the
frozen ResNet-18 encoder applied to the wrist camera RGB output. This is
the key piece of Phase B: the policy gets visual context without paying
the cost of carrying a 224×224×3 image through the rollout buffer.

Why pre-encode in the observation pipeline rather than inside the policy:
the rsl_rl rollout buffer holds num_envs × num_steps × obs_dim floats. A
raw image obs (4096 × 50 × 150528 × 4 bytes ≈ 123 GB) does not fit; a
512-D encoded feature (4096 × 50 × 512 × 4 bytes ≈ 0.4 GB) does. We pay
one ResNet forward per env per step (~4 ms on RTX 5070 for 4096 images),
which is acceptable for training and keeps the policy MLP small.
"""
from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ..policy.visual_encoder import encode_image


def wrist_image_features(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist"),
) -> torch.Tensor:
    """Read RGB from the named camera, encode with frozen ResNet-18.

    Args:
        env: The Isaac Lab env.
        sensor_cfg: SceneEntityCfg pointing at the wrist `TiledCamera`.

    Returns:
        ``(num_envs, 512)`` tensor of visual features. Returns zeros at
        the very first step if the camera buffer hasn't populated yet
        (Isaac Lab fills it on the first sim render after env construction).
    """
    sensor = env.scene[sensor_cfg.name]
    image = sensor.data.output["rgb"]  # (num_envs, H, W, 3) uint8 typically

    # Defensive: if the camera buffer is unexpectedly empty (very first
    # step, before the first render), return zeros instead of crashing.
    if image is None or image.numel() == 0:
        return torch.zeros(env.num_envs, 512, device=env.device, dtype=torch.float32)

    return encode_image(image)


# ---------------------------------------------------------------------------
# Phase C compatibility placeholders — used in V2.10 to pre-allocate obs
# dimensions for `target_color_one_hot` and `bowl_xyz` so a V2.10 checkpoint
# can be warm-started in Phase C without obs-dim mismatch.
# ---------------------------------------------------------------------------


def target_color_zero(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Placeholder for the Phase C target color one-hot (6 colors).

    Returns ``(num_envs, 6)`` zeros. In Phase C this term will be replaced
    with the actual one-hot encoding of the requested color.
    """
    return torch.zeros(env.num_envs, 6, device=env.device, dtype=torch.float32)


def bowl_xyz_zero(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Placeholder for the Phase C bowl xyz position (in robot root frame).

    Returns ``(num_envs, 3)`` zeros. In Phase C this term will be replaced
    with the bowl's actual xyz coordinates from the bowl asset.
    """
    return torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# V2.12 — explicit relative vectors (EE→cube, cube→goal).
#
# Why expose these to the policy: the canonical Isaac Lab Lift obs gives
# `object_position` (cube in robot root) + `target_object_position`
# (goal in robot root) separately. The MLP must subtract them mentally
# (cube - ee, goal - cube) to derive the directions to move. Small MLPs
# (256-128-128) take many iters just to learn this trivial subtraction —
# wasted training capacity.
#
# Pre-computing the difference vectors and feeding them as obs gives the
# policy direct access to the geometric relationships, accelerating
# convergence. This is what the HW4 ETH Zürich SO-100 reference includes
# in its observation, and why ManiSkill / robosuite manipulation tasks
# expose `tcp_to_obj` and `obj_to_goal` directly.
# ---------------------------------------------------------------------------


def ee_to_cube_vector(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Vector from the EE target frame to the cube, in world frame (3D).

    Same EE target index (0) as ``lift_mdp.object_ee_distance`` so both
    terms agree on which point on the gripper is "the EE". Robot is
    static in our scene (not navigating), so the world-frame difference
    is equivalent to a robot-root-frame difference modulo a fixed
    rotation that the MLP first layer can absorb trivially.

    Returns:
        ``(num_envs, 3)`` tensor: ``cube_pos_w - ee_pos_w``.
    """
    cube = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    cube_pos_w = cube.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    return cube_pos_w - ee_pos_w


def cube_to_goal_vector(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "object_pose",
) -> torch.Tensor:
    """Vector from the cube to the commanded goal pose, in world frame (3D).

    The goal pose is published by ``UniformPoseCommandCfg`` in the robot
    root frame; we project it into world frame (matching what the
    `cube_to_goal_distance_above_base` reward does) and then take the
    difference with the cube's world position.

    Returns:
        ``(num_envs, 3)`` tensor: ``goal_pos_w - cube_pos_w``.
    """
    from isaaclab.utils.math import combine_frame_transforms

    cube = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_b = command[:, :3]
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_b
    )
    return goal_pos_w - cube.data.root_pos_w
