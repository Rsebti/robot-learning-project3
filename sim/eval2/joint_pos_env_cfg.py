"""Eval 2 — SO-101 specializations.

Provides the SO-101 wiring for both the v0 (single block) and v1 (two colored
blocks + target color) generic configs.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaac_so_arm101.robots import SO_ARM101_CFG  # noqa: F401

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from . import mdp
from .pick_env_cfg import PickInBowlEnvCfg
from .pick_in_clutter_env_cfg import (
    PickInClutterEnvCfg,
    PickInClutterSceneCfgWithCam,
)


# Common SO-101 wiring shared between v0 and v1.
def _wire_so101_actions_and_ee_frame(self):
    self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    self.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
        scale=0.5,
        use_default_offset=True,
    )
    self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        open_command_expr={"gripper": 0.5},
        close_command_expr={"gripper": 0.0},
    )

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    self.scene.ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                name="end_effector",
                offset=OffsetCfg(pos=[0.01, 0.0, -0.09]),
            ),
        ],
    )


# ===========================================================================
# v0 — single-block, single-bowl (smoke test target).
# ===========================================================================
@configclass
class Eval2PickInBowlEnvCfg_v0(PickInBowlEnvCfg):
    """Training config: SO-101 + standard cube + flat bowl, 4096 parallel envs."""

    def __post_init__(self):
        super().__post_init__()
        _wire_so101_actions_and_ee_frame(self)

        self.scene.block = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Block",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.20, 0.0, 0.015], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.5, 0.5, 0.5),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )


@configclass
class Eval2PickInBowlEnvCfg_v0_PLAY(Eval2PickInBowlEnvCfg_v0):
    """Play config: same task with fewer envs and no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ===========================================================================
# v1 — two adjacent colored blocks + open-top bowl, target color goal.
# ===========================================================================
def _make_colored_block_cfg(prim_name: str, init_pos: list[float], rgb: tuple[float, float, float]) -> RigidObjectCfg:
    """Build a small colored cuboid block with rigid body physics.

    Matches the team's real wooden cubes: 2x2x2 cm.
    """
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.020, 0.020, 0.020),  # 2 cm cube — matches the real wooden blocks
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, metallic=0.0),
        ),
    )


@configclass
class Eval2PickInClutterEnvCfg_v1(PickInClutterEnvCfg):
    """Training config: SO-101 + 2 colored blocks (red, blue) + open-top bowl."""

    def __post_init__(self):
        super().__post_init__()
        _wire_so101_actions_and_ee_frame(self)

        # Two blocks placed adjacent on the table at y = +/- 0.02 (4 cm apart).
        # The cluster center is at (0.20, 0.0); reset randomization in EventCfg
        # adds +/- 5 cm noise per axis.
        # The two blocks form a "flat cluster" (TA spec for Eval 2:
        #   "Two blocks of different colors are placed adjacent to each other
        #    (flat cluster)").
        # Strict "touching" interpretation: blocks share a face at y=0, centers
        # at y=+/- block_half (= 0.010 m). The cluster is aligned along y, so
        # the gripper can still grasp either block by approaching perpendicular
        # to the cluster line (along x): one finger in front of the target,
        # one behind, without contacting the adjacent block.
        self.scene.block_red = _make_colored_block_cfg(
            "BlockRed", init_pos=[0.20, 0.010, 0.010], rgb=(0.85, 0.10, 0.10)
        )
        self.scene.block_blue = _make_colored_block_cfg(
            "BlockBlue", init_pos=[0.20, -0.010, 0.010], rgb=(0.10, 0.20, 0.85)
        )


@configclass
class Eval2PickInClutterEnvCfg_v1_PLAY(Eval2PickInClutterEnvCfg_v1):
    """Play config: same task with fewer envs and no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ===========================================================================
# v2 — same as v1 with the SO-101 wrist camera mounted on wrist_roll_link.
#
# Camera placement is the placeholder from TheRobotStudio's official
# Wrist_Cam_Mount_32x32_UVC_Module spec (4 cm forward, 3 cm up, 90 deg yaw
# to point optical axis along link's x-forward, 12 deg downward pitch).
# Quaternion (w, x, y, z) computed from those Euler angles.
# Sensor specs match the LeRobot real-robot config (UVC 32x32 module at
# 640x480 @ 30 fps), but we render at lower resolution by default to keep
# PhysX + render throughput tractable on the Blackwell-bug-affected RTX
# 5070 (CameraCfg uses the standard Camera, not TiledCamera, since the
# tiled variant hangs on sm_120).
# ===========================================================================


@configclass
class Eval2PickInClutterEnvCfg_v2(Eval2PickInClutterEnvCfg_v1):
    """v1 task + wrist camera. We override ``scene`` to use the camera-aware
    scene cfg (PickInClutterSceneCfgWithCam) and fill the camera in __post_init__.
    Single-line inheritance from v1 (no diamond) to keep configclass MRO simple.
    """

    scene: PickInClutterSceneCfgWithCam = PickInClutterSceneCfgWithCam(
        num_envs=4096, env_spacing=2.5
    )

    def __post_init__(self):
        # Run v1's __post_init__ to fill robot, ee_frame, blocks, gripper action.
        super().__post_init__()

        # Hide the ee_frame XYZ gizmo in v2 — it blocks the wrist camera's view
        # because the camera is mounted right where the gizmo is anchored.
        # The gizmo stays useful in v1 (third-person inspection) so we only
        # disable it on v2.
        self.scene.ee_frame.debug_vis = False

        # Wrist camera mounted on gripper_link (= the link downstream of the
        # wrist_roll joint in this URDF; the mount STL the doc references is
        # part of this very link).
        #
        # Offset was tuned interactively in the Isaac Sim Property panel
        # while the robot was held in a "scan" pose (gripper hovering above
        # the workspace pointing down). Final values that frame both blocks
        # and the bowl correctly:
        #   Translate  (-0.02208,  0.05825,  0.03013)  meters
        #   Orient XYZ (-15.544,  -9.931,   -90.069)   degrees
        #
        # The Euler angles converted to a (w, x, y, z) quaternion (intrinsic
        # XYZ convention) are below. Calibrate against the real wrist mount
        # before any serious sim-to-real attempt.
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_cam",
            update_period=0.1,           # 10 Hz — easier on rendering than 30 Hz
            height=240,
            width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(-0.02208, 0.05825, 0.03013),
                rot=(0.70586, -0.03453, -0.15594, -0.69009),  # Euler XYZ (-15.544, -9.931, -90.069) deg
                # "opengl" = no convention transform applied, our quaternion
                # is written directly to the prim's xformOp:orient. This way
                # the values we hardcode match 1:1 what the user tuned in the
                # Isaac Sim Property panel.
                convention="opengl",
            ),
        )


@configclass
class Eval2PickInClutterEnvCfg_v2_PLAY(Eval2PickInClutterEnvCfg_v2):
    """Play config for v2: fewer envs, no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
