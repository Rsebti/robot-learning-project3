"""Eval 2 v1 — two adjacent colored blocks + concave bowl, target color goal.

Differences vs v0:
- The bowl is built from 5 kinematic primitives (1 floor + 4 walls) so blocks
  actually get caught when released above it (v0's flat disc let blocks roll
  away).
- Two blocks side by side, ``block_red`` and ``block_blue``, both spawned
  via ``CuboidCfg`` with a colored ``PreviewSurfaceCfg``.
- An extra observation term ``target_color_one_hot`` lets the policy know
  which block to pick. The target color is sampled per-episode in an event
  term and stored on ``env.target_color``.
- Rewards dispatch to the target block (reach/lift/place) and add a small
  penalty for disturbing the distractor block.
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
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from . import mdp


# Bowl geometry (all in meters). The bowl center is at (BOWL_X, BOWL_Y, ~floor).
# The real bowl on the team's setup is ~10-12 cm in inner diameter and holds
# ~12 wooden cubes of 2x2x2 cm — we match that for sim-to-real alignment.
# Position chosen to keep the bowl well within the SO-101's ~30 cm reach
# from the robot base at the world origin: distance = sqrt(0.20^2 + 0.15^2)
# = ~25 cm, with the cluster at (0.20, 0) sitting just to the +y side.
BOWL_X = 0.20
BOWL_Y = -0.15
BOWL_INNER_HALF = 0.06  # 12x12 cm internal floor (>= 10 cm spec, + margin)
BOWL_FLOOR_THICKNESS = 0.005
BOWL_WALL_THICKNESS = 0.008
BOWL_WALL_HEIGHT = 0.025  # ~ block height so blocks clear the rim with a small lift


# ---------------------------------------------------------------------------
# Bowl helper: build a "kinematic open-top box" out of 5 cuboids.
# ---------------------------------------------------------------------------
def _bowl_floor_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BowlFloor",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[BOWL_X, BOWL_Y, BOWL_FLOOR_THICKNESS / 2],
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(2 * BOWL_INNER_HALF, 2 * BOWL_INNER_HALF, BOWL_FLOOR_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.40, 0.25), metallic=0.0
            ),
        ),
    )


def _bowl_wall_cfg(name: str, axis: str, sign: int) -> RigidObjectCfg:
    """Build one of the 4 walls. ``axis`` is 'x' or 'y'; ``sign`` is +1 or -1."""
    half_extent_along_wall = BOWL_INNER_HALF + BOWL_WALL_THICKNESS  # so corners overlap
    wall_z = BOWL_FLOOR_THICKNESS + BOWL_WALL_HEIGHT / 2
    if axis == "x":
        size = (BOWL_WALL_THICKNESS, 2 * half_extent_along_wall, BOWL_WALL_HEIGHT)
        pos = [BOWL_X + sign * (BOWL_INNER_HALF + BOWL_WALL_THICKNESS / 2), BOWL_Y, wall_z]
    else:  # 'y'
        size = (2 * half_extent_along_wall, BOWL_WALL_THICKNESS, BOWL_WALL_HEIGHT)
        pos = [BOWL_X, BOWL_Y + sign * (BOWL_INNER_HALF + BOWL_WALL_THICKNESS / 2), wall_z]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.40, 0.25), metallic=0.0
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class PickInClutterSceneCfg(InteractiveSceneCfg):
    """Scene with robot, ee_frame tracker, two colored blocks, an open-top bowl,
    table, ground, and light."""

    # Filled by the robot-specific subclass.
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    block_red: RigidObjectCfg = MISSING
    block_blue: RigidObjectCfg = MISSING

    # Bowl primitives. The "bowl" the rewards reference is bowl_floor (its position
    # marks the center of the bowl). The walls just provide collision so the
    # blocks get physically caught when released above.
    bowl_floor: RigidObjectCfg = _bowl_floor_cfg()
    bowl_wall_xp: RigidObjectCfg = _bowl_wall_cfg("BowlWallXP", axis="x", sign=+1)
    bowl_wall_xn: RigidObjectCfg = _bowl_wall_cfg("BowlWallXN", axis="x", sign=-1)
    bowl_wall_yp: RigidObjectCfg = _bowl_wall_cfg("BowlWallYP", axis="y", sign=+1)
    bowl_wall_yn: RigidObjectCfg = _bowl_wall_cfg("BowlWallYN", axis="y", sign=-1)

    # Table — primitive cuboid with the exact color specified by the TAs:
    # "A light gray table (approximately #B8ADA9)" → RGB(184, 173, 169) = (0.722, 0.678, 0.663).
    # Replaces the upstream USD lab table so we have full control over the
    # color (UsdFileCfg.visual_material doesn't reliably override sub-prim
    # materials baked into the SeattleLabTable USD).
    #
    # Geometry: 80 cm (along x, robot reach axis) x 1 m (along y) x 4 cm thick.
    # Centered at (0.40, 0) so the effective x extent is (0.0, 0.8). This puts
    # the robot base at the near edge of the table, gives 20 cm clearance for
    # the cluster at x=0.20 + the +/-5 cm cluster randomization, and keeps the
    # bowl at (0.30, -0.20) well inside the table. Without this the upstream
    # ~60 cm-wide footprint had the cluster sitting on the table edge and any
    # negative xy noise would spawn the blocks off the table.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.40, 0, -0.02]),
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 1.00, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.722, 0.678, 0.663),  # #B8ADA9
                metallic=0.0,
            ),
        ),
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
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group, concatenated into a flat 30-D vector."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)                          # 6
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)                          # 6
        block_red_position = ObsTerm(func=mdp.block_red_position_in_robot_frame)   # 3
        block_blue_position = ObsTerm(func=mdp.block_blue_position_in_robot_frame) # 3
        bowl_position = ObsTerm(func=mdp.bowl_position_in_robot_frame)       # 3
        target_color = ObsTerm(func=mdp.target_color_one_hot)                # 2
        actions = ObsTerm(func=mdp.last_action)                              # 6

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset behavior: scene to default + randomize blocks + sample target color."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Sample a fresh target color (0=red, 1=blue) uniformly at every reset.
    randomize_target_color = EventTerm(
        func=mdp.reset_target_color,
        mode="reset",
        params={"num_classes": 2},
    )

    # Randomize the *cluster* of blocks together (same xy shift applied to
    # both). Keeps them adjacent — required by the TA spec.
    randomize_block_cluster = EventTerm(
        func=mdp.reset_cluster_uniform,
        mode="reset",
        params={
            "position_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            "asset_names": ("block_red", "block_blue"),
        },
    )

    # Randomize the bowl position too — TA spec: "Bowls placed at randomized
    # positions in the robot base frame". The same xy shift is applied to all
    # 5 bowl primitives so the bowl shape (floor + 4 walls) stays intact.
    # Range kept modest in y to avoid overlap with the cluster.
    randomize_bowl_position = EventTerm(
        func=mdp.reset_cluster_uniform,
        mode="reset",
        params={
            "position_range": {"x": (-0.04, 0.04), "y": (-0.02, 0.02)},
            "asset_names": (
                "bowl_floor",
                "bowl_wall_xp",
                "bowl_wall_xn",
                "bowl_wall_yp",
                "bowl_wall_yn",
            ),
        },
    )


@configclass
class RewardsCfg:
    """Dense reward shaping, target-aware."""

    # Approach the target block (block size is 2.5 cm so std=0.05 is ~2 block widths).
    reaching_target = RewTerm(
        func=mdp.target_block_ee_distance_tanh,
        params={"std": 0.05},
        weight=1.0,
    )

    # Two-stage lifting: a low threshold to easily reward "block off the table",
    # plus a higher one to reward "block clears the bowl rim". The low threshold
    # is critical for early training — without it lifting_target plateaus at ~0
    # because the policy never randomly clears 5 cm.
    lifting_target_low = RewTerm(
        func=mdp.target_block_is_lifted,
        params={"minimal_height": 0.025},  # ~ block height — easy to trigger
        weight=10.0,
    )

    lifting_target_high = RewTerm(
        func=mdp.target_block_is_lifted,
        params={"minimal_height": 0.05},  # block clears the bowl walls
        weight=10.0,
    )

    # Bring the block toward the bowl. Gated on the LOW lift threshold so the
    # policy gets reward as soon as it lifts and moves laterally.
    target_to_bowl_coarse = RewTerm(
        func=mdp.target_block_to_bowl_distance_tanh,
        params={"std": 0.30, "minimal_height": 0.025},
        weight=16.0,
    )

    # BOOSTED for the long-convergence run: was 5.0. The fine-grained drop is
    # the actual bottleneck (lifting + transport already work at iter ~500),
    # so this term needs a louder voice to teach the policy to commit to a
    # precise descent over the bowl.
    target_to_bowl_fine = RewTerm(
        func=mdp.target_block_to_bowl_distance_tanh,
        params={"std": 0.05, "minimal_height": 0.025},
        weight=25.0,
    )

    # ----- v1.4 milestone (sparse) rewards ----------------------------------
    # The previous runs only had dense + final-success signals: PPO could
    # plateau by maximizing the dense terms ("hover the block near the bowl
    # forever") without ever committing to the actual sparse goal. These
    # three milestones break the chain into intermediate snap-points so each
    # phase has a clear, sparse reward of its own.

    # +50 the first time the gripper actually grabs the target block.
    grasp_success = RewTerm(
        func=mdp.target_block_grasped,
        params={"gripper_closed_threshold": 0.15, "ee_to_block_threshold": 0.04},
        weight=50.0,
    )

    # +100 the first time the target block is hovering above the bowl
    # (within 10 cm xy of bowl center, at least 5 cm above bowl top).
    above_bowl = RewTerm(
        func=mdp.target_block_above_bowl,
        params={"height_above": 0.05, "xy_threshold": 0.10},
        weight=100.0,
    )

    # +200 when the target block is fully placed inside the bowl (the actual
    # task success criterion). Largest single-term reward so PPO learns to
    # value it above all the dense intermediate rewards combined.
    success_bonus = RewTerm(
        func=mdp.target_block_in_bowl,
        params={"xy_threshold": BOWL_INNER_HALF, "z_max_above_bowl": 0.10},
        weight=200.0,
    )

    # Discourage moving the wrong block.
    distractor_disturbed = RewTerm(
        func=mdp.distractor_block_disturbed,
        params={"height_threshold": 0.025},
        weight=-5.0,
    )

    # Smoothness penalties + action magnitude regularization.
    # action_rate (penalize abrupt changes between consecutive actions) was
    # already there. action_l2 is NEW: it penalizes the magnitude of each
    # action directly, so the actor mean is pulled toward small values
    # instead of saturating the joint targets at +/-7.5 rad after the 0.5
    # action scale.
    #
    # action_l2 weight bumped from -1e-2 to -1e-1 (v1.4): the previous run
    # showed std blowing up from 0.5 -> 4.65 by iter 3030, which means the
    # entropy bonus + reward landscape ambiguity was overpowering the -1e-2
    # action penalty. 10x stronger penalty now creates a much clearer
    # gradient toward "small actions" so PPO can commit to a tight policy.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    action_l2 = RewTerm(func=mdp.action_l2_norm, weight=-1e-1)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    block_red_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("block_red")},
    )
    block_blue_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("block_blue")},
    )

    success = DoneTerm(
        func=mdp.success_target_block_in_bowl,
        params={"xy_threshold": BOWL_INNER_HALF, "z_max_above_bowl": 0.10},
    )


# ---------------------------------------------------------------------------
# Top-level env config
# ---------------------------------------------------------------------------
@configclass
class PickInClutterEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickInClutterSceneCfg = PickInClutterSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 6.0  # slightly longer — placement is harder than v0
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.sim.dt = 0.01  # 100 Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # Bumped from 16K -> 64K because v1 has more prims per env (2 blocks +
        # 5 bowl primitives + table + robot) and at 4096 envs we were seeing
        # PhysX 'missing interactions' errors that stalled lifting in training.
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


# ===========================================================================
# v2 — same as v1, plus the SO-101 wrist camera (RGB).
#
# We keep v1 (state-based, fast) as the production training task and add v2
# specifically for: (a) visual development / debug, (b) training the perception
# module against ground-truth, (c) the real-robot deploy script (which mirrors
# the wrist cam observation 1:1 with the live SO-101 camera).
#
# Camera mounting follows TheRobotStudio's official Wrist_Cam_Mount_32x32_UVC
# placeholder values: ~4 cm forward, ~3 cm up of the wrist_roll_link, looking
# forward with a 12 deg downward pitch. To be remeasured / calibrated before
# any serious sim-to-real attempt (see the team doc).
# ===========================================================================


@configclass
class PickInClutterSceneCfgWithCam(PickInClutterSceneCfg):
    """v1 scene + a wrist-mounted camera. Camera prim is filled by the
    robot-specific subclass so that the placeholder for ``robot.prim_path``
    can be substituted into the camera's prim_path.
    """

    wrist_cam: CameraCfg = MISSING


@configclass
class PickInClutterEnvCfgV2(PickInClutterEnvCfg):
    """Same managers/rewards/events as v1 (training stays state-based and
    fast). The only structural change is the scene gains a wrist camera."""

    scene: PickInClutterSceneCfgWithCam = PickInClutterSceneCfgWithCam(
        num_envs=4096, env_spacing=2.5
    )
