"""Eval 2 v0 — single block, single bowl. Generic env config.

Mirror of ``isaac_so_arm101.tasks.lift.lift_env_cfg`` but with a real bowl as
the placement target instead of a virtual command pose. Robot-specific bits
(SO-101 articulation, gripper action, block USD) are filled in by the
``joint_pos_env_cfg`` subclass.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class PickInBowlSceneCfg(InteractiveSceneCfg):
    """Scene with robot, ee_frame tracker, one block, one bowl, table, ground, light."""

    # Filled by the robot-specific subclass.
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    block: RigidObjectCfg = MISSING

    # Bowl approximated as a flat kinematic cylinder. We use a primitive
    # (vs. an external USD) so the prim has built-in RigidBodyAPI — Isaac Lab
    # requires it for any RigidObjectCfg, and YCB meshes don't ship with it.
    # Asset name is ``bowl_floor`` for symmetry with v1 (which uses 5 prims).
    bowl_floor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BowlFloor",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.30, -0.20, 0.02], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CylinderCfg(
            radius=0.05,
            height=0.04,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # bowl stays put even if hit
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.5, 0.3), metallic=0.0
            ),
        ),
    )

    # Standard Isaac Lab manipulation table.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# ---------------------------------------------------------------------------
# Managers
# ---------------------------------------------------------------------------
@configclass
class ActionsCfg:
    """Action specs — concrete actions are filled by the robot-specific subclass."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group (concatenated into a flat vector)."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        block_position = ObsTerm(func=mdp.block_position_in_robot_frame)
        bowl_position = ObsTerm(func=mdp.bowl_position_in_robot_frame)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset behavior: scene to default + randomize block position on the table."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_block_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("block"),
        },
    )


@configclass
class RewardsCfg:
    """Dense reward shaping. Weights tuned to match upstream Lift task."""

    # Approach the block (~1 when on it).
    reaching_block = RewTerm(
        func=mdp.block_ee_distance_tanh,
        params={"std": 0.05},
        weight=1.0,
    )

    # Big bonus for getting the block off the table.
    lifting_block = RewTerm(
        func=mdp.block_is_lifted,
        params={"minimal_height": 0.025},
        weight=15.0,
    )

    # Bring the lifted block close to the bowl (multiplied by lifted-condition).
    block_to_bowl_coarse = RewTerm(
        func=mdp.block_to_bowl_distance_tanh,
        params={"std": 0.30, "minimal_height": 0.025},
        weight=16.0,
    )

    # Same but with a much tighter kernel — bonus only when very close.
    block_to_bowl_fine = RewTerm(
        func=mdp.block_to_bowl_distance_tanh,
        params={"std": 0.05, "minimal_height": 0.025},
        weight=5.0,
    )

    # Sparse success bonus: block actually inside the bowl.
    success_bonus = RewTerm(
        func=mdp.block_in_bowl,
        params={"xy_threshold": 0.04, "z_max_above_bowl": 0.05},
        weight=50.0,
    )

    # Smooth-action penalties.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Episode endings: timeout, block falling off the world, success."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    block_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("block")},
    )

    success = DoneTerm(
        func=mdp.success_block_in_bowl,
        params={"xy_threshold": 0.04, "z_max_above_bowl": 0.05},
    )


# ---------------------------------------------------------------------------
# Top-level env config
# ---------------------------------------------------------------------------
@configclass
class PickInBowlEnvCfg(ManagerBasedRLEnvCfg):
    """Generic pick-in-bowl env. Robot-specific bits filled by subclass."""

    scene: PickInBowlSceneCfg = PickInBowlSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # Step decimation: 1 policy step = N physics steps.
        self.decimation = 2
        # Episode length in seconds.
        self.episode_length_s = 5.0
        # Default viewer position (when not headless).
        self.viewer.eye = (2.5, 2.5, 1.5)
        # Sim settings.
        self.sim.dt = 0.01  # 100 Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
