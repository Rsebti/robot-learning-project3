"""Eval 2 v0 — SO-101 specialization of ``PickInBowlEnvCfg``.

Plugs in:
- the SO-ARM101 articulation config from ``isaac_so_arm101.robots``,
- the joint-position arm action + binary gripper action,
- the cube object as the block,
- the end-effector frame transformer for distance computations.
"""

from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
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


@configclass
class Eval2PickInBowlEnvCfg_v0(PickInBowlEnvCfg):
    """Training config: SO-101 + standard cube + bowl, 4096 parallel envs."""

    def __post_init__(self):
        super().__post_init__()

        # Robot.
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Actions: 5D arm + 1D gripper (binary).
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

        # Block (the same dex_cube used by the upstream Lift task).
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

        # End-effector frame tracker (matches the upstream Lift v101).
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


@configclass
class Eval2PickInBowlEnvCfg_v0_PLAY(Eval2PickInBowlEnvCfg_v0):
    """Play config: same task with fewer envs and no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
