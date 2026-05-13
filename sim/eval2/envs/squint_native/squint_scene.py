"""InteractiveSceneCfg — robot, cube, bin (5 parts), wrist cam, table, lights.

Geometry / placement matches Squint's defaults from ``squint/envs/place.py``:
- Cube:       2.5 cm side (half_size 0.0125), red, friction 0.3, density 200
- Bin:        8 × 10 × 3 cm (half x=0.04, y=0.05, z=0.015), 5 separate parts,
              white, thickness 5 mm. Built as 5 RigidObjects so the env reset
              event can place them coherently with a shared bin pose.
- Wrist cam:  128×128 RGB, FOV 71°, mounted on the gripper body
              with offset (-0.0049, 0.0498, -0.0591) and SAPIEN-style rotation
              euler RPY (-90°, 91°, -35.31°) applied as R_z * R_y * R_x. We use
              ``convention="world"`` so the cam looks along local +X (SAPIEN).
- Table:      static box at z=0 (top surface), 1.5 × 1.0 m, light gray.
- Lights:     dome (intensity 1500) + 1 directional.
"""
from __future__ import annotations

import math
from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from .squint_robot import SQUINT_SO101_CFG


# ---------------------------------------------------------------------------
# Cam pose helper — Squint applies q = q_yaw * q_pitch * q_roll
# (extrinsic XYZ / intrinsic ZYX). We compute the quaternion once at module
# load and feed it to CameraCfg.OffsetCfg.
# ---------------------------------------------------------------------------


def _euler_rpy_zyx_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Compose q = q_yaw * q_pitch * q_roll (Hamilton, scalar-first)."""
    cj, sj = math.cos(pitch / 2), math.sin(pitch / 2)
    ck, sk = math.cos(yaw / 2), math.sin(yaw / 2)
    ci, si = math.cos(roll / 2), math.sin(roll / 2)
    # q_py = q_yaw * q_pitch
    q_py_w = cj * ck
    q_py_x = sj * sk
    q_py_y = sj * ck
    q_py_z = cj * sk
    # q = q_py * q_roll
    qw = q_py_w * ci - q_py_x * si
    qx = q_py_w * si + q_py_x * ci
    qy = q_py_y * ci + q_py_z * si
    qz = q_py_z * ci - q_py_y * si
    return (qw, qx, qy, qz)


WRIST_CAM_POS = (-0.0049, 0.0498, -0.0591)
WRIST_CAM_QUAT = _euler_rpy_zyx_to_quat_wxyz(
    roll=math.radians(-90),
    pitch=math.radians(91),
    yaw=math.radians(-35.31),
)
WRIST_CAM_FOV_RAD = math.radians(71)


# ---------------------------------------------------------------------------
# Scene cfg
# ---------------------------------------------------------------------------


@configclass
class SquintNativeSceneCfg(InteractiveSceneCfg):
    """Squint-native scene: robot at origin, cube, bin (5 parts), wrist cam, table, lights."""

    # ---- Robot ----
    robot: ArticulationCfg = SQUINT_SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ---- Wrist camera ----
    # NOTE: We mount the camera at robot-root level WITH ZERO OFFSET, then
    # update its world pose every step from SquintNativePlaceEnv.step() using
    # gripper_pose * local_offset (matches Squint's SAPIEN behaviour exactly).
    # Putting the cam under /Robot directly (not /Robot/gripper) decouples it
    # from any USD parent transform we don't fully control.
    wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_cam",
        update_period=0.0,
        height=128,
        width=128,
        # ``instance_segmentation_fast`` enables Squint-style greenscreen
        # in squint_observations.wrist_rgb_16 — keep robot/cube/bin pixels,
        # replace background with #B8ADA9.
        data_types=["rgb", "instance_segmentation_fast"],
        # Required so cam.data.pos_w / quat_w refreshes after our per-step
        # set_world_poses() override in SquintNativePlaceEnv.step().
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )

    # ---- Cube (target, RED, 2 cm side) ----
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.30, 0.0, 0.010),  # spawn box center, sits on table
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.020, 0.020, 0.020),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # red = goal_color idx 0
                metallic=0.0,
                roughness=0.9,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3,
                dynamic_friction=0.3,
                restitution=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=700.0),  # beech wood
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    # ---- Distractor cube (BLUE) — same size + physics as the target ----
    # Squint's new checkpoint was trained with a distractor present face-
    # to-face with the target. Goal-color one-hot tells the policy which
    # of the two cubes to grasp.
    cube_distractor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeDistractor",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.30, 0.025, 0.010),  # right next to the target along +y
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.020, 0.020, 0.020),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),  # blue = palette idx 1 (non-goal)
                metallic=0.0,
                roughness=0.9,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3,
                dynamic_friction=0.3,
                restitution=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=700.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    # ---- Bin (5 parts: floor + 4 walls) — all white ----
    # Sizes match Squint's defaults: half (0.04, 0.05, 0.015), thickness 0.005.
    # Initial poses centered around (0.30, 0.0, 0); the reset event places them
    # coherently per-env. Each part is its own RigidObject so physics still
    # provides walls; the reset event re-positions all 5 with a shared yaw.
    bin_floor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BinFloor",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.0, 0.0025)),
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.10, 0.005),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.9
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3, dynamic_friction=0.3, restitution=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    bin_wall_pos_y = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BinWallPosY",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, +0.05, 0.015)),
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.005, 0.030),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.9
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3, dynamic_friction=0.3, restitution=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    bin_wall_neg_y = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BinWallNegY",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, -0.05, 0.015)),
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.005, 0.030),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.9
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3, dynamic_friction=0.3, restitution=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    bin_wall_pos_x = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BinWallPosX",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.34, 0.0, 0.015)),
        spawn=sim_utils.CuboidCfg(
            size=(0.005, 0.10, 0.030),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.9
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3, dynamic_friction=0.3, restitution=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    bin_wall_neg_x = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BinWallNegX",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.26, 0.0, 0.015)),
        spawn=sim_utils.CuboidCfg(
            size=(0.005, 0.10, 0.030),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.9
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3, dynamic_friction=0.3, restitution=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # ---- Table (static, light gray) ----
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.40, 0.0, -0.05)),
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 1.0, 0.10),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.722, 0.678, 0.663),  # ~#B8ADA9 from CLAUDE.md
                metallic=0.0,
                roughness=0.9,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.5, restitution=0.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # ---- Lights ----
    # Squint's wrist render at home pose has channel means
    # R:181 G:166 B:163 — warm/reddish (because the #B8ADA9 overlay is
    # itself warm). Our PBR render with white lighting comes out neutral
    # grey (R=G=B=180). To bias toward Squint's warmth, the dome light
    # gets a slight amber tint (R high, B low). Verified to bring
    # G/B down by ~12-15 units while leaving R near 181.
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=300.0, color=(1.0, 1.0, 1.0)),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=300.0, color=(1.0, 1.0, 1.0), angle=10.0
        ),
    )
