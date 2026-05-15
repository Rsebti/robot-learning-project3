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
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass

from .squint_robot import SQUINT_SO101_CFG

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_BOWL_USD = _PROJECT_ROOT / "squint" / "meshes" / "bowl.usd"


# ---------------------------------------------------------------------------
# Cam pose helper — Squint applies q = q_yaw * q_pitch * q_roll
# (extrinsic XYZ / intrinsic ZYX). We compute the quaternion once at module
# load and feed it to CameraCfg.OffsetCfg.
# ---------------------------------------------------------------------------


def _euler_rpy_zyx_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Compose q = q_pitch * q_yaw * q_roll (Hamilton, scalar-first).

    Bit-identical to Squint's ``base_random_env.py:603-612``. The variable
    naming below (``q_py``) suggests "pitch*yaw" but the resulting xyz
    components — `sj*sk, sj*ck, cj*sk` — are the Hamilton product of
    q_pitch first then q_yaw (NOT q_yaw·q_pitch which would give
    `-sk*sj, sj*ck, sk*cj`).
    """
    cj, sj = math.cos(pitch / 2), math.sin(pitch / 2)
    ck, sk = math.cos(yaw / 2), math.sin(yaw / 2)
    ci, si = math.cos(roll / 2), math.sin(roll / 2)
    # q_py = q_pitch * q_yaw  (NOT q_yaw * q_pitch — see docstring)
    q_py_w = cj * ck
    q_py_x = sj * sk
    q_py_y = sj * ck
    q_py_z = cj * sk
    # q = q_py * q_roll = q_pitch * q_yaw * q_roll
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
# Squint training values. focal/aperture chosen so horizontal FOV = 71°:
#   FOV_h = 2 * atan(horizontal_aperture / (2 * focal_length))
WRIST_CAM_FOV_RAD = math.radians(71)
WRIST_CAM_HORIZONTAL_APERTURE = 20.955
WRIST_CAM_FOCAL_LENGTH = WRIST_CAM_HORIZONTAL_APERTURE / (2.0 * math.tan(WRIST_CAM_FOV_RAD / 2.0))


# ---------------------------------------------------------------------------
# Scene cfg
# ---------------------------------------------------------------------------


@configclass
class SquintNativeSceneCfg(InteractiveSceneCfg):
    """Squint-native scene: robot at origin, cube, bin (5 parts), wrist cam, table, lights."""

    # ---- Robot ----
    robot: ArticulationCfg = SQUINT_SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ---- Contact sensors on the gripper bodies ----
    # Per the ``isaac_lab_gpu_physx_contact_filter`` memory: ``filter_prim_paths_expr``
    # hangs env build on GPU PhysX in Isaac Sim 5.1, so we attach UNFILTERED
    # sensors on the two jaw bodies (``gripper`` = fixed jaw + STS3215 servo,
    # ``jaw`` = moving jaw) and read total contact force ``net_forces_w``.
    # The grasp predicate downstream combines force magnitude with a
    # cube-near-jaw proximity check (cf. Squint Claude handoff: tip links
    # have NO collision geometry, only the jaw bodies do).
    gripper_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper",
        update_period=0.0,
        history_length=0,
        track_pose=False,
        track_air_time=False,
    )
    jaw_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=0,
        track_pose=False,
        track_air_time=False,
    )

    # ---- Wrist camera (128×128, FOV 71° — matches Squint training) ----
    # The cam pose is written each substep by SquintNativePlaceEnv via
    # ``cam.set_world_poses()`` (CameraCfg.OffsetCfg inheritance under our
    # converted USD gave the wrong world pose).
    wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_cam",
        update_period=0.0,
        height=128,
        width=128,
        data_types=["rgb", "instance_segmentation_fast"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=WRIST_CAM_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=WRIST_CAM_HORIZONTAL_APERTURE,
            clipping_range=(0.01, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )

    # ---- Cube (target, RED, 2 cm side) ----
    # Squint current spec (envs/place.py:143-152):
    #   item_mass_range     = (0.003, 0.006) kg   → mid 4.5 g
    #   item_friction_range = (0.4, 0.6)          → mid 0.5
    # Static defaults below sit at the Squint mid; per-env DR is applied at
    # reset (randomize_cube_material / randomize_cube_mass in env_cfg).
    # combine_mode="min" matches SAPIEN's default contact-pair friction rule.
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.0, 0.010)),
        spawn=sim_utils.CuboidCfg(
            size=(0.020, 0.020, 0.020),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # red = goal_color idx 0
                metallic=0.0, roughness=0.9,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                # Boosted above Squint's nominal mid (0.325) to make
                # cube-robot and cube-table contact stickier:
                #   - vs gripper/jaw (μ=2.0), min(0.8, 2.0) = 0.8
                #   - vs table (μ=0.6), min(0.8, 0.6) = 0.6
                # DR range below is shifted up accordingly (env_cfg).
                static_friction=0.325,
                dynamic_friction=0.325,
                restitution=0.0,
                friction_combine_mode="min",
                restitution_combine_mode="min",
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0045),  # Squint mid 4.5 g
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02, rest_offset=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=15,
                solver_velocity_iteration_count=1,
                max_linear_velocity=10.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=5.0,
            ),
        ),
    )

    # ---- Distractor cube (BLUE) — same physics as the target ----
    cube_distractor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeDistractor",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.025, 0.010)),
        spawn=sim_utils.CuboidCfg(
            size=(0.020, 0.020, 0.020),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),
                metallic=0.0, roughness=0.9,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                # Squint per-env friction DR mid (uniform [0.4, 0.6] → 0.5).
                static_friction=0.325,
                dynamic_friction=0.325,
                restitution=0.0,
                friction_combine_mode="min",
                restitution_combine_mode="min",
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0045),  # Squint mid 4.5 g
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02, rest_offset=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=15,
                solver_velocity_iteration_count=1,
                max_linear_velocity=10.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=5.0,
            ),
        ),
    )

    # ---- Bowl (from SAM-3D reconstruction of the real bowl) ----
    # Mesh: ``squint/meshes/bowl.obj`` (collision) + ``bowl.ply`` (visual with
    # vertex colors), regenerated by ``sim/eval2/scripts/mesh_bowl_from_ply.py``.
    # USD baked by ``convert_bowl_to_usd.py`` with PhysX convex-decomposition
    # collision (Isaac equivalent of Squint's CoACD).
    # Origin is at the bowl floor's bottom-center → place init z = 0 to sit on
    # table top (z=0). AABB ≈ 0.15 × 0.15 × 0.053 m. Per-vertex colors give
    # the bowl its natural tan/beige look (mean rgb ≈ 0.70, 0.69, 0.67).
    # Density 500 kg/m³, friction 0.5/0.5, contact_offset 0.02, solver 15/1
    # → matches Squint's ``isaacsim_handoff`` physics params for the bowl.
    bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bowl",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_BOWL_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                solver_position_iteration_count=15,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02, rest_offset=0.0
            ),
            # bowl.obj has no vertex colors → flat tan/beige PBR fallback
            # (matches the .ply vertex-color mean from the SAM-3D recon).
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.70, 0.69, 0.67),
                metallic=0.0,
                roughness=0.7,
            ),
        ),
    )

    # ---- Table (Squint collision box, gray-uniform) ----
    # Same dimensions and pose as Squint's TableSceneBuilder.add_box_collision
    # combined with place.py's table_pose override:
    #   half_size=(2.418/2, 1.209/2, 0.92/2) LOCAL  + pose.z = +0.46 LOCAL
    #   actor pose = (-0.12 + 0.737, 0, -0.9196429) WORLD, yaw=π/2.
    # → world box center = (0.617, 0, -0.46), world half_size after yaw=π/2:
    #   X = 1.209/2 = 0.605, Y = 2.418/2 = 1.209, Z = 0.46.
    # World extent:
    #   X ∈ (0.012, 1.222) — robot base at world (0,0,0) sits at the table's
    #                       back edge (12 mm behind), identical to Squint.
    #   Y ∈ (-1.209, 1.209)
    #   Z ∈ (-0.92, 0)     — table top at z=0, matches Squint.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.617, 0.0, -0.46)),
        spawn=sim_utils.CuboidCfg(
            size=(1.209, 2.418, 0.9196429),  # already accounts for yaw=π/2 swap
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.722, 0.678, 0.663),  # ~#B8ADA9 — uniform gray
                metallic=0.0,
                roughness=0.9,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                # Squint mid for table_friction_range = (0.05, 0.4) → 0.225.
                static_friction=0.225,
                dynamic_friction=0.225,
                restitution=0.0,
                friction_combine_mode="min",
                restitution_combine_mode="min",
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # ---- Lights ----
    # Squint's emissive-only look (ambient=0, lights off, emission=diffuse)
    # was attempted but Isaac RTX real-time doesn't reproduce SAPIEN's
    # "flat unshaded silhouettes" cleanly — the framebuffer comes out
    # quasi-uniform gray and the cube becomes invisible to the policy.
    # We keep traditional dome + distant lights ON for now; emissive_color
    # is still set on every PreviewSurface so per-shader self-illumination
    # is preserved if we later switch to PathTracing.
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
