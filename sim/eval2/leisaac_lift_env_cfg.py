"""LeIsaac LiftCube + RL augmentation — Phase A smoke test (state-only).

LeIsaac's `LiftCubeEnvCfg` ships with the scene, the SO-101 robot, both
cameras (wrist + front), state observations, the FrameTransformer, the
success termination (cube > 0.20 m above robot base), and domain
randomization on object/camera poses. But it has:

- No rewards (`SingleArmRewardsCfg` is empty)
- No commands (no goal pose)
- `actions.arm_action` / `actions.gripper_action` are `MISSING`
- 25 s episode length (tuned for teleoperation)
- `decimation = 1` (very fast control, sensible for IL)
- `dynamic_reset_gripper_effort_limit = True` (teleop-only feature)

This module adds the missing RL pieces:

1. `ActionsCfg`: JointPositionAction (5 arm joints, scale 0.5) + Binary gripper
2. `CommandsCfg`: UniformPoseCommandCfg for the lift goal pose, restricted to
   the SO-101 reachable workspace (~22 cm)
3. `RewardsCfg`: canonical Isaac Lab Lift shaping reused verbatim:
   reaching_object, lifting_object, object_goal_tracking (gated on lifted),
   object_goal_tracking_fine_grained, action_rate, joint_vel
4. `EventsCfg`: extends parent's `reset_all` with a uniform cube spawn
   randomization on reset
5. State-only observations override: drop the `wrist` and `front` image
   terms (Phase B will add them back through a frozen ResNet encoder),
   concatenate the rest into a single vector
6. Timing: `episode_length_s = 5.0`, `decimation = 2`, no curriculum

Frame / entity naming differs from `isaac_so_arm101` because LeIsaac's USD
uses bare `gripper`, `jaw`, `base` (no `_link` suffix) and names the cube
asset `cube` (not `object`). Reward terms therefore pass
`SceneEntityCfg("cube")` and the goal command targets `body_name="gripper"`.
"""
import math

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs.mdp import UniformPoseCommandCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from leisaac.tasks.lift_cube.lift_cube_env_cfg import LiftCubeEnvCfg
from leisaac.tasks.template.single_arm_env_cfg import SingleArmEventCfg

from . import mdp as eval2_mdp


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@configclass
class CommandsCfg:
    """Goal pose for the lift task — same shape as Isaac Lab Lift, but
    range tightened to the SO-101 reachable workspace (max ~22 cm)."""

    object_pose = UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper",  # LeIsaac uses "gripper", not "gripper_link"
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.05),
            pos_y=(-0.20, -0.10),
            pos_z=(0.10, 0.20),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# ---------------------------------------------------------------------------
# Rewards — canonical Isaac Lab Lift shaping
# ---------------------------------------------------------------------------


@configclass
class RewardsCfg:
    """Canonical pick-and-place reward shaping from Isaac Lab Lift-Cube.

    The fine-grained tracking term reuses the same `object_goal_distance`
    function as the coarse one but with a much tighter `std`, so the policy
    only collects the high-resolution reward right next to the goal.

    All weights are positive (rewards) except `action_rate` and `joint_vel`
    which are small regularizers. No curriculum — keeping the regularizer
    weights constant avoids the local-optimum collapse we observed when
    isaac_so_arm101 ramps them to -1e-1 mid-training.
    """

    # 1. DENSE reaching reward (Isaac Lab canonical, std 5 cm). Tells the
    # policy "the closer the gripper is to the cube, the better".
    reaching_object = RewTerm(
        func=lift_mdp.object_ee_distance,
        params={
            "std": 0.05,
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=1.0,
    )

    # 2. BINARY grasp reward (LeIsaac's `object_grasped`). Bridges the gap
    # between "near cube" (reaching reward saturating) and "lifted" (next
    # term). Without this, the policy has to make a discrete leap from
    # reaching saturation to the lift threshold without intermediate
    # signal — empirically this stalls Phase B around lifting=0.6-0.8.
    # Weight 5.0 keeps it strictly below `lifting_object` so the policy
    # is incentivized to actually pick up after grasping.
    grasping_cube = RewTerm(
        func=eval2_mdp.cube_grasped,
        params={
            "diff_threshold": 0.02,
            "grasp_threshold": 0.26,
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cube_cfg": SceneEntityCfg("cube"),
        },
        weight=5.0,
    )

    # 3. BINARY lift reward — relative to robot base (LeIsaac semantics),
    # not absolute world z. Avoids the "fire from frame 0 because the
    # table is elevated" pitfall that bit our first run.
    lifting_object = RewTerm(
        func=eval2_mdp.cube_lifted_above_base,
        params={
            "height_threshold": 0.05,  # cube must be 5cm above robot base
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": "base",
        },
        weight=15.0,
    )

    object_goal_tracking = RewTerm(
        func=eval2_mdp.cube_to_goal_distance_above_base,
        params={
            "std": 0.3,
            "height_threshold": 0.05,
            "command_name": "object_pose",
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": "base",
        },
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=eval2_mdp.cube_to_goal_distance_above_base,
        params={
            "std": 0.05,
            "height_threshold": 0.05,
            "command_name": "object_pose",
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": "base",
        },
        weight=5.0,
    )

    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-1e-4,
    )


# ---------------------------------------------------------------------------
# Events — extend LeIsaac's reset_all with cube position randomization
# ---------------------------------------------------------------------------


@configclass
class EventsCfg(SingleArmEventCfg):
    """Inherits `reset_all` (resets robot + scene to defaults) and adds a
    uniform cube spawn randomization. Order matters: `reset_all` runs first
    (puts everything in default pose), then `reset_cube_position` perturbs
    the cube xy."""

    reset_cube_position = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.10, 0.10),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


# ---------------------------------------------------------------------------
# State-only observations for Phase A
# ---------------------------------------------------------------------------


@configclass
class StatePolicyCfg(ObsGroup):
    """Compact state vector — joints + cube pose + goal pose + last action.

    Phase A: no images. Phase B will add wrist cam encoded by a frozen
    ResNet, concatenated alongside this state vector.
    """

    joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
    object_position = ObsTerm(
        func=lift_mdp.object_position_in_robot_root_frame,
        params={"object_cfg": SceneEntityCfg("cube")},
    )
    target_object_position = ObsTerm(
        func=base_mdp.generated_commands,
        params={"command_name": "object_pose"},
    )
    actions = ObsTerm(func=base_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


# ---------------------------------------------------------------------------
# Top-level env config
# ---------------------------------------------------------------------------


@configclass
class LeIsaacLiftCubeRLEnvCfg(LiftCubeEnvCfg):
    """LeIsaac LiftCube + everything PPO needs (state-only, Phase A)."""

    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Set the actions that the parent left as MISSING.
        self.actions.arm_action = base_mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = base_mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 0.5},
            close_command_expr={"gripper": 0.0},
        )

        # Replace the parent's image-heavy PolicyCfg with our state-only one.
        self.observations.policy = StatePolicyCfg()

        # Strip the cameras from the scene for Phase A (state-only smoke test).
        # LeIsaac always declares wrist + front TiledCameraCfg in
        # `SingleArmTaskSceneCfg` — they trigger an error without
        # `--enable_cameras` and slow training ~5-10x even when unused. Phase B
        # will re-add the wrist cam and route it through a frozen ResNet
        # encoder.
        self.scene.wrist = None
        self.scene.front = None
        # LiftCubeEnvCfg.__post_init__ adds two domain-randomization events
        # via `domain_randomization()`: index 0 randomizes the cube pose,
        # index 1 randomizes the front camera. Disable the camera one since
        # we removed the camera; keep the cube one (it's a superset of our
        # own reset_cube_position event, no harm).
        self.events.domain_randomize_1 = None

        # PPO timing — short episode, lower control rate, more envs.
        self.episode_length_s = 5.0
        self.decimation = 2
        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5

        # Disable teleop-specific runtime tweak.
        self.dynamic_reset_gripper_effort_limit = False

        # Replace LeIsaac's default success termination
        # (`cube_height_above_base`, threshold 0.20 m) with one that checks
        # whether the cube has reached the commanded goal pose. The default
        # is misaligned with our goal range pos_z=(0.10, 0.20): even
        # perfectly tracked goals at z<0.20 m never count as "success",
        # making the metric blind to actual policy quality (Phase A run
        # showed `success=0.01%` with `lifting_object=8.4/15`).
        self.terminations.success = DoneTerm(
            func=eval2_mdp.cube_reached_goal,
            params={
                "distance_threshold": 0.05,
                "command_name": "object_pose",
                "cube_cfg": SceneEntityCfg("cube"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )


@configclass
class LeIsaacLiftCubeRLEnvCfg_PLAY(LeIsaacLiftCubeRLEnvCfg):
    """Smaller scene + no obs corruption for replaying a checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Phase B — visual variant: wrist cam + frozen ResNet-18 features
# ---------------------------------------------------------------------------


@configclass
class VisualPolicyCfg(ObsGroup):
    """State + ResNet-18 wrist features. 28 + 512 = 540-D obs."""

    joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
    object_position = ObsTerm(
        func=lift_mdp.object_position_in_robot_root_frame,
        params={"object_cfg": SceneEntityCfg("cube")},
    )
    target_object_position = ObsTerm(
        func=base_mdp.generated_commands,
        params={"command_name": "object_pose"},
    )
    actions = ObsTerm(func=base_mdp.last_action)
    wrist_features = ObsTerm(
        func=eval2_mdp.wrist_image_features,
        params={"sensor_cfg": SceneEntityCfg("wrist")},
    )

    def __post_init__(self):
        # Disable corruption: the ResNet features carry their own noise
        # robustness, and adding extra Gaussian noise on top of the 512-D
        # features distorts the manifold the encoder produces.
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class LeIsaacLiftCubeRLVisualEnvCfg(LeIsaacLiftCubeRLEnvCfg):
    """Phase B — restore the wrist camera and feed its features into the
    policy through a frozen ResNet-18.

    Phase A's parent (`LeIsaacLiftCubeRLEnvCfg`) sets
    ``self.scene.wrist = None`` to skip the cost of camera rendering. We
    override that here by re-creating the canonical LeIsaac wrist
    `TiledCameraCfg` and observe its RGB output via
    ``eval2_mdp.wrist_image_features``, which runs each batch through the
    pretrained ResNet-18 in `eval()` mode (frozen weights).

    Render resolution is set to 224x224 to match ResNet's native input
    size; this avoids a CPU-side resize on every step (PyTorch
    interpolate would still kick in if rendered larger).

    ⚠️ Launch with ``--enable_cameras`` — Isaac Lab refuses to run
    `TiledCamera` sensors otherwise.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # Re-add the wrist camera. Same canonical pose as LeIsaac's
        # SingleArmTaskSceneCfg, but rendered at 224x224 directly.
        # Use lazy local imports so unrelated places (Phase A) don't pay
        # the import cost or risk Isaac Sim being already running.
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import TiledCameraCfg

        self.scene.wrist = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(-0.001, 0.1, -0.04),
                rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
                convention="ros",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=36.5,
                focus_distance=400.0,
                horizontal_aperture=36.83,
                clipping_range=(0.01, 50.0),
                lock_camera=True,
            ),
            width=224,
            height=224,
            update_period=1 / 30.0,
        )

        # Replace the state-only PolicyCfg with the visual one.
        self.observations.policy = VisualPolicyCfg()

        # Camera rendering blows up VRAM far faster than just compute time.
        # Each TiledCamera comes with its own DLSS / G-buffer / ray-tracing
        # pipeline. On RTX 5070 (12 GB VRAM, 16 GB BAR1), 1024 envs at
        # 224x224 saturates BAR1 and crashes at scene init (we hit
        # "Out of GPU memory allocating DLSS Output" + a CUDA illegal
        # memory access on first step). Empirically 256 envs is the
        # comfortable ceiling on this hardware.
        # Override at the CLI with --num_envs N for other GPUs.
        self.scene.num_envs = 256


@configclass
class LeIsaacLiftCubeRLVisualEnvCfg_PLAY(LeIsaacLiftCubeRLVisualEnvCfg):
    """Smaller scene + no obs corruption for replaying a Phase B checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# V2.7 — flick-exploit fixes: lift gated on grasp, tracking gated on grasp,
# grasp thresholds relaxed, lifting weight reduced.
# ---------------------------------------------------------------------------


@configclass
class RewardsCfgV27:
    """V2.7 RewardsCfg — closes the flick exploit observed in V2/V2.5/V2.6.

    Diagnostic from the V2.6 run at iter 107:
      - lifting_object frequency ~7%, but grasping_cube frequency ~0.02%
      - reaching_object peaks at iter 30 then DECLINES
      - position_error stuck at ~0.18 m, success_rate = 0

    Interpretation: the policy learned to scoop the cube above the base
    threshold (5 cm) without grasping it, harvesting +15/step from
    `lifting_object` plus partial credit from the coarse goal_tracking
    (gated only on lifted, not grasped). Reach + grasp paid 1+5=6 max
    while flick paid 15+12=27, so the policy abandoned reach.

    Three structural fixes:

    1. ``lifting_object`` now uses ``cube_lifted_and_grasped`` (AND of
       LeIsaac's lift check and grasp check). The flick stops paying.
       Weight reduced 15 → 10 so it doesn't dominate the goal-tracking
       signal once it actually fires.

    2. Both ``object_goal_tracking`` terms use
       ``cube_to_goal_distance_grasped_and_lifted`` — same gate. A cube
       in ballistic trajectory toward the goal no longer leaks reward.

    3. ``grasping_cube`` thresholds relaxed:
         - ``diff_threshold``  : 0.02 → 0.04 m (gripper-to-cube margin
           during the binary close transient)
         - ``grasp_threshold`` : 0.26 → 0.35 rad (capture closing, not
           just fully-closed)
       LeIsaac's strict defaults rarely co-fire on the binary gripper
       at decimation=2 — V2.6 shows grasp reward at 0.0014, indicating
       the grasp condition is essentially never satisfied.

    4. NEW ``success_bonus`` (weight 200): sparse one-shot bonus when
       ``cube_reached_goal`` fires. Compensates for the discounted future
       dense reward (~1900 pts at γ=0.99) that an early-terminating
       episode loses, breaking the "hover at edge of success sphere"
       degenerate optimum that would emerge once the flick is closed.

    Reaching, action_rate, joint_vel: unchanged from RewardsCfg.
    """

    reaching_object = RewTerm(
        func=lift_mdp.object_ee_distance,
        params={
            "std": 0.05,
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=1.0,
    )

    grasping_cube = RewTerm(
        func=eval2_mdp.cube_grasped,
        params={
            "diff_threshold": 0.04,         # V2.7: was 0.02 (relaxed)
            "grasp_threshold": 0.35,        # V2.7: was 0.26 (relaxed)
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cube_cfg": SceneEntityCfg("cube"),
        },
        weight=5.0,
    )

    # V2.7: was eval2_mdp.cube_lifted_above_base, weight 15.
    # Now requires grasp AND lift (both conditions). Weight reduced so
    # it doesn't dwarf the goal-tracking gradient once it actually fires.
    lifting_object = RewTerm(
        func=eval2_mdp.cube_lifted_and_grasped,
        params={
            "height_threshold": 0.05,
            "diff_threshold": 0.04,         # match grasping_cube
            "grasp_threshold": 0.35,        # match grasping_cube
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_base_name": "base",
        },
        weight=10.0,
    )

    # V2.7: was eval2_mdp.cube_to_goal_distance_above_base.
    # Same coarse std=0.3 but now gated on grasp+lift, not just lift.
    object_goal_tracking = RewTerm(
        func=eval2_mdp.cube_to_goal_distance_grasped_and_lifted,
        params={
            "std": 0.3,
            "height_threshold": 0.05,
            "diff_threshold": 0.04,
            "grasp_threshold": 0.35,
            "command_name": "object_pose",
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_base_name": "base",
        },
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=eval2_mdp.cube_to_goal_distance_grasped_and_lifted,
        params={
            "std": 0.05,
            "height_threshold": 0.05,
            "diff_threshold": 0.04,
            "grasp_threshold": 0.35,
            "command_name": "object_pose",
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_base_name": "base",
        },
        weight=5.0,
    )

    # V2.7 NEW: sparse bonus on success. Compensates for the discounted
    # future reward the policy loses by terminating early via
    # `cube_reached_goal`. Weight = 200 ≈ 7 steps of the dense reward
    # rate at the goal (~30/step), enough to make "succeed and end" net
    # preferable to "hover at edge of success sphere", but small enough
    # that PPO doesn't see a discontinuity at success.
    # MUST keep distance_threshold in sync with TerminationsCfg.success.
    success_bonus = RewTerm(
        func=eval2_mdp.cube_at_goal,
        params={
            "distance_threshold": 0.05,
            "command_name": "object_pose",
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
        weight=200.0,
    )

    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-1e-4,
    )


@configclass
class LeIsaacLiftCubeRLEnvCfgV27(LeIsaacLiftCubeRLEnvCfg):
    """V2.7 state-only env: V2.6 env + grasp-gated rewards."""

    rewards: RewardsCfgV27 = RewardsCfgV27()


@configclass
class LeIsaacLiftCubeRLEnvCfgV27_PLAY(LeIsaacLiftCubeRLEnvCfgV27):
    """Smaller scene + no obs corruption for replaying a V2.7 checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV27(LeIsaacLiftCubeRLVisualEnvCfg):
    """V2.7 visual env: V2.6 visual env + grasp-gated rewards."""

    rewards: RewardsCfgV27 = RewardsCfgV27()


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV27_PLAY(LeIsaacLiftCubeRLVisualEnvCfgV27):
    """Smaller scene + no obs corruption for replaying a V2.7 visual ckpt."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# V2.8.5 (env-side) — empirically-grounded fixes to V27 reward shaping +
# cube_dropped failure termination.
#
# Empirical measurements (`audit_scene.py` on the V285 PLAY scene):
#
#   cube.z (world) at spawn  = +0.0615 m  (cube on table, no z noise)
#   base.z (world)           = +0.0100 m  (robot base ~at floor level)
#   cube.z - base.z at spawn = +0.0515 m → settles to +0.0460 after 1 step
#   |jaw target → cube| at reset (default joint pose) = +0.24 m
#   per-step dense reward at goal (all gates fire) ≈ +37
#
# Four changes vs RewardsCfgV27:
#
# (1) Lift/tracking height_threshold 0.05 → 0.08
#     The previous 0.05 sat right at the spawn diff, so the lift gate was
#     essentially open at frame 0 — semantic bug that made the lift signal
#     noisy and the goal-tracking gate leak reward. New threshold sits
#     +3 cm above spawn (clean off-the-table semantic) and 2 cm below
#     goal-z-min so the lift signal still fires before reaching the goal.
#
# (2) reaching_object std 0.05 → 0.15
#     Audit shows the EE is 24 cm from the cube at reset (default joint
#     pose). With std=0.05, reward = 1 - tanh(0.24/0.05) = 0.0001 per step
#     — essentially zero gradient until the EE is within ~10 cm. This
#     produced the "policy wandering" behavior in V2.6/V2.7/V2.8: no
#     directional signal until the policy got lucky. With std=0.15, same
#     distance gives reward = 1 - tanh(1.6) = 0.08, ×730 more signal.
#
# (3) success_bonus weight 200 → 1500
#     Recalculation: hovering at d = 5.1 cm (just outside the success
#     sphere) yields ~30/step dense reward. Over 100 remaining steps with
#     gamma=0.99, that's Σ_{k=0..99} 0.99^k × 30 ≈ +1900 of foregone
#     discounted reward by terminating early. The previous weight=200
#     was 10× too small, so PPO would still prefer hovering. Weight=1500
#     covers ~80% of the foregone reward — enough to make
#     "succeed and end" net preferable while not creating a TD-target
#     discontinuity that would destabilize the value head.
#
# (4) cube_dropped failure termination (world-z < 0.04)
#     Defined in `mdp/terminations.py` and wired in __post_init__ below.
#     Cuts off zero-reward tails that pollute the advantage estimate.
#
# PPO config is V2.8 unchanged (same hyperparams, only experiment_name
# differs to keep logs separated under `lift_v2_8_5/`).
# ---------------------------------------------------------------------------


# Lifting/tracking height threshold for V2.8.5. Needs to be above the
# empirical spawn diff (+0.0515) by enough margin to ignore wobble
# (target ≥ +0.025 of margin) and below the goal z-min (+0.10) so the
# gate signals "in-progress lift", not "at goal".
_V285_LIFT_HEIGHT_THRESHOLD = 0.08


@configclass
class RewardsCfgV285:
    """V2.8.5 RewardsCfg — empirically-grounded fixes to RewardsCfgV27.

    Three changes vs RewardsCfgV27 (all motivated by `audit_scene.py`
    measurements — see the section comment above for full reasoning):

      1. ``height_threshold`` (lift gate): 0.05 → 0.08 in the 3 lift/
         tracking terms. Prevents the gate from firing at spawn.
      2. ``reaching_object.std``: 0.05 → 0.15. Provides a smooth gradient
         across the entire 24 cm reset distance, instead of a "blind"
         zone outside ~10 cm.
      3. ``success_bonus.weight``: 200 → 1500. Compensates the ~1900 of
         discounted future dense reward lost by terminating, so PPO
         prefers finishing over hovering at the edge of the success
         sphere.

    All other params, weights, gate predicates and function references
    are unchanged.
    """

    reaching_object = RewTerm(
        func=lift_mdp.object_ee_distance,
        params={
            "std": 0.15,                                        # V2.8.5: was 0.05
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=1.0,
    )

    grasping_cube = RewTerm(
        func=eval2_mdp.cube_grasped,
        params={
            "diff_threshold": 0.04,
            "grasp_threshold": 0.35,
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cube_cfg": SceneEntityCfg("cube"),
        },
        weight=5.0,
    )

    lifting_object = RewTerm(
        func=eval2_mdp.cube_lifted_and_grasped,
        params={
            "height_threshold": _V285_LIFT_HEIGHT_THRESHOLD,  # V2.8.5: was 0.05
            "diff_threshold": 0.04,
            "grasp_threshold": 0.35,
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_base_name": "base",
        },
        weight=10.0,
    )

    object_goal_tracking = RewTerm(
        func=eval2_mdp.cube_to_goal_distance_grasped_and_lifted,
        params={
            "std": 0.3,
            "height_threshold": _V285_LIFT_HEIGHT_THRESHOLD,  # V2.8.5: was 0.05
            "diff_threshold": 0.04,
            "grasp_threshold": 0.35,
            "command_name": "object_pose",
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_base_name": "base",
        },
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=eval2_mdp.cube_to_goal_distance_grasped_and_lifted,
        params={
            "std": 0.05,
            "height_threshold": _V285_LIFT_HEIGHT_THRESHOLD,  # V2.8.5: was 0.05
            "diff_threshold": 0.04,
            "grasp_threshold": 0.35,
            "command_name": "object_pose",
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_base_name": "base",
        },
        weight=5.0,
    )

    success_bonus = RewTerm(
        func=eval2_mdp.cube_at_goal,
        params={
            "distance_threshold": 0.05,
            "command_name": "object_pose",
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
        weight=1500.0,                                          # V2.8.5: was 200
    )

    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-1e-4,
    )


@configclass
class LeIsaacLiftCubeRLEnvCfgV285(LeIsaacLiftCubeRLEnvCfgV27):
    """V2.8.5 state-only env: corrected rewards + cube_dropped failure cutoff."""

    rewards: RewardsCfgV285 = RewardsCfgV285()

    def __post_init__(self) -> None:
        super().__post_init__()

        # New failure-side DoneTerm. Episode ends as soon as the cube's
        # world z drops below 0.04 m. The LeIsaac scene puts the robot
        # base at z≈0.01 (essentially at the floor) and the cube on the
        # table at z≈0.0615, so a relative-to-base threshold can't
        # distinguish "cube on table" from "cube on floor". Threshold in
        # world frame is unambiguous: 0.04 sits between the spawn z
        # (0.0615) and what the cube would reach on the floor (~0.02).
        # Watch `Episode_Termination/cube_dropped` in TensorBoard — if
        # it sustains > 20 % during training, the policy has learned to
        # drop on purpose to escape regularizers (give-up exploit) and
        # we add a -1.0 RewTerm tied to the same predicate.
        self.terminations.cube_dropped = DoneTerm(
            func=eval2_mdp.cube_dropped,
            params={
                "world_z_threshold": 0.04,
                "cube_cfg": SceneEntityCfg("cube"),
            },
        )


@configclass
class LeIsaacLiftCubeRLEnvCfgV285_PLAY(LeIsaacLiftCubeRLEnvCfgV285):
    """Smaller scene + no obs corruption for replaying a V2.8.5 checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV285(LeIsaacLiftCubeRLVisualEnvCfgV27):
    """V2.8.5 visual env: corrected rewards + cube_dropped failure cutoff."""

    rewards: RewardsCfgV285 = RewardsCfgV285()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.terminations.cube_dropped = DoneTerm(
            func=eval2_mdp.cube_dropped,
            params={
                "world_z_threshold": 0.04,
                "cube_cfg": SceneEntityCfg("cube"),
            },
        )


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV285_PLAY(LeIsaacLiftCubeRLVisualEnvCfgV285):
    """Smaller scene + no obs corruption for replaying a V2.8.5 visual ckpt."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# V2.9 (env-side) — V2.8.5 + cube_dropped penalty (-5.0).
#
# V2.8.5 added the `cube_dropped` DoneTerm but with no explicit reward
# cost. Result observed at iter 80: ~41 % of episodes terminate via
# cube_dropped because bumping the cube off the table during grasp
# attempts is net-positive EV (+4.8 reach reward over 30 steps before
# the drop, vs +0 for an episode that grasps wrong and times out at
# step 150 with no further dense reward beyond reach).
#
# V2.9 pairs the DoneTerm with a `cube_dropped_penalty` RewTerm
# (weight=-5.0, fires once at the drop step). With this, dropping the
# cube becomes net negative (+4.8 -5.0 = -0.2), so the policy is
# incentivized to grasp gently rather than bump.
#
# All other V2.8.5 fixes preserved unchanged (RewardsCfgV285, lift
# threshold 0.08, reaching std 0.15, success_bonus 1500). PPO config is
# V2.8 unchanged (only experiment_name differs to keep logs separated).
# ---------------------------------------------------------------------------


_V29_CUBE_DROPPED_WEIGHT = -5.0


@configclass
class LeIsaacLiftCubeRLEnvCfgV29(LeIsaacLiftCubeRLEnvCfgV285):
    """V2.9 state-only env: V2.8.5 + cube_dropped penalty + reach fix.

    Inherits ``RewardsCfgV285`` (lift threshold 0.08, reaching std 0.15,
    success_bonus 1500) and the ``cube_dropped`` DoneTerm. Adds:

      1. ``cube_dropped_penalty`` RewTerm (weight=-5.0) — pairs with the
         existing DoneTerm to make drops net negative EV.
      2. **Cube spawn reach fix** — disables LeIsaac's `domain_randomize_0`
         (which silently overrode our event with std≈0.043 centered on
         the default LeIsaac cube position 28 cm from the robot, max
         distance 36 cm = at SO-101's reach limit) and shifts our
         ``reset_cube_position`` event's pose_range so the cube spawns
         12–19 cm from the robot — comfortable reach with dexterity
         margin for fine grasp control.

    Audit-derived numbers:
      - LeIsaac default cube y_env-local = -0.361
      - Robot y_env-local ≈ -0.639 (back of the table)
      - Pre-fix |cube - robot| mean=0.286 m, max=0.363 m (edge of reach)
      - Post-fix shift cube by ~13 cm in -y → mean ≈ 0.15 m, max ≈ 0.20 m

    Watch ``Episode_Termination/cube_dropped`` after launch:
      - if rate > 25 % at iter 80 → bump cube_dropped_penalty to -10
      - if rate < 5 % but grasping still flat → relax to -2 (too timid)
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # === Reach fix (env-side) ===
        # Disable LeIsaac's domain_randomize_0 — it was overriding our
        # explicit reset_cube_position event with its own std≈0.043
        # randomization centered on the LeIsaac default cube position,
        # putting the cube at the edge of the SO-101 reach envelope.
        # The previous comment "it's a superset of our event, no harm"
        # was wrong: LeIsaac uses ±0.075 in both axes vs our ±0.05/±0.10
        # — different distribution, AND it overrides ours rather than
        # composing.
        self.events.domain_randomize_0 = None

        # Shift cube spawn ~13 cm in -y direction (toward the robot).
        # Range tightened from (-0.05,+0.05)x(-0.10,+0.10) to symmetric
        # ±3 cm to keep the cube reliably within reach across all spawns.
        self.events.reset_cube_position.params["pose_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.16, -0.10),
            "z": (0.0, 0.0),
        }

        # === cube_dropped penalty (V2.9) ===
        # MUST keep world_z_threshold in sync with TerminationsCfg.cube_dropped
        # (set on V2.8.5 to 0.04 — empirically validated).
        self.rewards.cube_dropped_penalty = RewTerm(
            func=eval2_mdp.cube_dropped_float,
            params={
                "world_z_threshold": 0.04,
                "cube_cfg": SceneEntityCfg("cube"),
            },
            weight=_V29_CUBE_DROPPED_WEIGHT,
        )


@configclass
class LeIsaacLiftCubeRLEnvCfgV29_PLAY(LeIsaacLiftCubeRLEnvCfgV29):
    """Smaller scene + no obs corruption for replaying a V2.9 checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV29(LeIsaacLiftCubeRLVisualEnvCfgV285):
    """V2.9 visual env: V2.8.5 visual + cube_dropped penalty + reach fix."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # === Reach fix (same as state-only V29) ===
        self.events.domain_randomize_0 = None
        self.events.reset_cube_position.params["pose_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.16, -0.10),
            "z": (0.0, 0.0),
        }

        # === cube_dropped penalty ===
        self.rewards.cube_dropped_penalty = RewTerm(
            func=eval2_mdp.cube_dropped_float,
            params={
                "world_z_threshold": 0.04,
                "cube_cfg": SceneEntityCfg("cube"),
            },
            weight=_V29_CUBE_DROPPED_WEIGHT,
        )


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV29_PLAY(LeIsaacLiftCubeRLVisualEnvCfgV29):
    """Smaller scene + no obs corruption for replaying a V2.9 visual ckpt."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# V2.10 (env-side) — smoothness-focused fixes after V2.9 diagnostic.
#
# V2.9 deterministic policy saturated joint velocities at 10 rad/s on every
# step (verified via play_diagnose.py). The training found a chaotic motion
# strategy that succeeded ~32 % stochastically but is unusable for deploy.
#
# V2.10 forces smooth motion with 4 env-side fixes:
#   1. action_rate weight: -1e-4 → -5e-2  (×500 stronger smoothing penalty)
#   2. joint_vel weight:   -1e-4 → -1e-2  (×100 — penalize absolute speed)
#   3. action scale on arm joints: 0.5 → 0.25 (each step's target moves 50%
#      less in joint space, mechanically restraining policy)
#   4. joint_acc_l2 reward (NEW): penalize joint acceleration directly
#
# Table color override (#B8ADA9 per spec) is **deferred to Phase D** — the
# LeIsaac scene loads the table as a static mesh inside a global Scene USD,
# not as a configclass attribute, so it can't be overridden via a simple
# `self.scene.table.spawn.visual_material = ...`. Cosmetic for Phase B.
#
# PPO config V2.10 also reduces init_noise_std (1.0 → 0.4) and
# entropy_coef (0.005 → 0.002) to discourage exploration through chaos.
# ---------------------------------------------------------------------------


# RGB normalized from #B8ADA9 (CLAUDE.md project spec). Kept here for
# Phase D reference; not currently applied — see TODO in __post_init__.
_TABLE_COLOR_RGB = (184 / 255.0, 173 / 255.0, 169 / 255.0)


@configclass
class RewardsCfgV210(RewardsCfgV285):
    """V2.10 RewardsCfg — V285 with stronger smoothness penalties.

    Inherits everything from RewardsCfgV285 (same gating, same weights for
    grasp/lift/track/success). Overrides only the regularizer weights and
    adds a `joint_acc_l2` term.
    """

    # action_rate: ×500 stronger to enforce step-to-step smoothness.
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-5e-2)

    # joint_vel_l2: ×100 stronger to penalize absolute joint speeds.
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-1e-2,
    )

    # NEW joint_acc_l2: penalize joint acceleration (= step-to-step velocity
    # change). Complementary to action_rate which is action-space; this is
    # joint-space and catches the 78 rad/s impulsive spikes observed in V2.9.
    joint_acc = RewTerm(
        func=base_mdp.joint_acc_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-1e-3,
    )


@configclass
class LeIsaacLiftCubeRLEnvCfgV210(LeIsaacLiftCubeRLEnvCfgV29):
    """V2.10 state-only env: V2.9 + smoothness fixes + wider cube randomization."""

    rewards: RewardsCfgV210 = RewardsCfgV210()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Smoothness fix #3: halve the action scale so each step's joint
        # target delta is ±0.25 rad max instead of ±0.5 rad. Mechanically
        # caps the per-step motion and forces the policy to plan multi-step
        # trajectories instead of single-step lunges.
        self.actions.arm_action.scale = 0.25

        # V2.10 — adopt LeIsaac's stock pose_range for the cube reset event,
        # plus yaw randomization (±30°). LeIsaac's `domain_randomize_0` (which
        # we keep disabled) used these exact params; we trust their workspace
        # calibration. Range is 15 cm × 15 cm per axis — wider than V2.9's
        # 5 cm — and adds yaw rotation variability.
        self.events.reset_cube_position.params["pose_range"] = {
            "x": (-0.075, 0.075),
            "y": (-0.075, 0.075),
            "z": (0.0, 0.0),
            "yaw": (-30 * math.pi / 180, 30 * math.pi / 180),
        }

        # V2.10 — Phase C compatibility placeholders. Adds 9D of zero-valued
        # obs (target_color_one_hot 6D + bowl_xyz 3D) so the policy network
        # has the right input shape for warm-starting Phase C training.
        # The MLP weights for these dims will be near-zero after V2.10
        # (no gradient signal), so when Phase C provides real values, the
        # network treats them as small inputs initially and learns from there.
        self.observations.policy.target_color_placeholder = ObsTerm(
            func=eval2_mdp.target_color_zero
        )
        self.observations.policy.bowl_xyz_placeholder = ObsTerm(
            func=eval2_mdp.bowl_xyz_zero
        )

        # V2.10 — compensate cube init z for the new 2cm size.
        # The cube was 3cm (half-extent 0.015) and spawned with center at
        # z=0.0615 → bottom rested on the table at z=0.0465. After the
        # USD-level scale to 2cm (half-extent 0.010), keeping the same
        # center z would float the cube 5 mm above the table. We lower
        # the init_state.pos.z by 5 mm so the cube bottom stays on the
        # table top.
        try:
            old_pos = self.scene.cube.init_state.pos
            self.scene.cube.init_state.pos = (old_pos[0], old_pos[1], old_pos[2] - 0.005)
        except (AttributeError, KeyError, TypeError):
            # If init_state structure differs across LeIsaac versions, leave
            # as-is (cube will fall 5 mm at sim start, harmless).
            pass

        # TODO Phase D — table color override (#B8ADA9 per project spec).
        # LeIsaac loads the table as a static mesh inside `self.scene.scene`
        # USD (not as a configclass-exposed sub-attribute), so a simple
        # `self.scene.table.spawn.visual_material = ...` does not target it.
        # Deferred to Phase D — cosmetic for Phase B training.


@configclass
class LeIsaacLiftCubeRLEnvCfgV210_PLAY(LeIsaacLiftCubeRLEnvCfgV210):
    """Smaller scene + no obs corruption for replaying a V2.10 checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class LeIsaacLiftCubeRLEnvCfgV210_VIEW(LeIsaacLiftCubeRLEnvCfgV210):
    """Single-env visual inspection cfg: 2s episodes for fast cube respawn.

    Used with `view.py` (zero-action agent) to visually verify cube spawn
    positions, randomization range, and reachability without any policy.
    Episode auto-resets every 2 s instead of 5 s, so a new random cube
    pose appears every 2 s.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 2.0  # fast respawn for visual inspection


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV210(LeIsaacLiftCubeRLVisualEnvCfgV29):
    """V2.10 visual env: V2.9 visual + smoothness + wider cube randomization."""

    rewards: RewardsCfgV210 = RewardsCfgV210()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Smoothness fix #3: action scale halved.
        self.actions.arm_action.scale = 0.25

        # LeIsaac stock pose_range + yaw (same as V210 RL env).
        self.events.reset_cube_position.params["pose_range"] = {
            "x": (-0.075, 0.075),
            "y": (-0.075, 0.075),
            "z": (0.0, 0.0),
            "yaw": (-30 * math.pi / 180, 30 * math.pi / 180),
        }

        # Compensate init_state.pos.z by -5 mm for the 2cm cube (same fix
        # as V210 RL env — cube center lowered to keep bottom on table top).
        try:
            old_pos = self.scene.cube.init_state.pos
            self.scene.cube.init_state.pos = (old_pos[0], old_pos[1], old_pos[2] - 0.005)
        except (AttributeError, KeyError, TypeError):
            pass

        # Phase C compatibility placeholders (target_color_one_hot 6D +
        # bowl_xyz 3D) — same as V210 RL env. Allows warm-starting Phase C
        # from a V2.10 visual checkpoint without obs-dim mismatch.
        self.observations.policy.target_color_placeholder = ObsTerm(
            func=eval2_mdp.target_color_zero
        )
        self.observations.policy.bowl_xyz_placeholder = ObsTerm(
            func=eval2_mdp.bowl_xyz_zero
        )

        # TODO Phase D — table color (#B8ADA9). Deferred (same architectural
        # blocker as for the cube before USD-level resize).


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV210_PLAY(LeIsaacLiftCubeRLVisualEnvCfgV210):
    """Smaller scene + no obs corruption for replaying a V2.10 visual ckpt."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# V2.10b — V2.10 with joint_acc_l2 weight rebalanced (-1e-3 → -3e-5, ÷33).
#
# Diagnosis from V2.10 cold-start (failed at iter ~50, paralysis at 1%
# reaching) and warm-start from V2.9 (failed at iter ~95, reaching 1.5% →
# 0.42% — policy actively *un-learns* V2.9 reach behaviour):
#
# Per-episode penalty contributions at iter 95 of the warm-start run:
#   - action_rate_l2 (-5e-2):  -0.71/episode
#   - joint_vel_l2  (-1e-2):  -0.71/episode
#   - joint_acc_l2  (-1e-3): -50.27/episode  ←  70× the others
#
# joint_acc dominates. Its gradient ("don't accelerate") drowns the
# `reaching_object` gradient ("approach the cube"). Since reaching from a
# resting pose requires *accelerating*, the policy converges toward "don't
# move" — a local optimum where every smoothness term contributes ~0 but
# task reward is also ~0.
#
# V2.10b rebalances joint_acc_l2 from -1e-3 to -3e-5 so its per-episode
# contribution lands ~ -1.5, in parity with the other two smoothness
# terms. Total smoothness budget drops from ~-52/ep to ~-3/ep, leaving
# headroom for the task gradient to lead the policy.
#
# All other V2.10 changes (USD cube 2cm, table #B8ADA9, LeIsaac stock
# randomization, action.scale=0.25, init_noise=0.4, entropy=0.002, Phase C
# placeholders) are inherited unchanged. Only one number changes.
# ---------------------------------------------------------------------------


@configclass
class RewardsCfgV210b(RewardsCfgV210):
    """V2.10b — V2.10 rewards with joint_acc rebalanced (÷33).

    Inherits action_rate (-5e-2) and joint_vel (-1e-2) from V2.10. Overrides
    only joint_acc weight to bring its contribution into parity with the
    other two smoothness terms.
    """

    joint_acc = RewTerm(
        func=base_mdp.joint_acc_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-3e-5,
    )


@configclass
class LeIsaacLiftCubeRLEnvCfgV210b(LeIsaacLiftCubeRLEnvCfgV210):
    """V2.10b state-only env: V2.10 with joint_acc rebalanced."""

    rewards: RewardsCfgV210b = RewardsCfgV210b()


@configclass
class LeIsaacLiftCubeRLEnvCfgV210b_PLAY(LeIsaacLiftCubeRLEnvCfgV210b):
    """Smaller scene + no obs corruption for replaying a V2.10b checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV210b(LeIsaacLiftCubeRLVisualEnvCfgV210):
    """V2.10b visual env: V2.10 visual with joint_acc rebalanced."""

    rewards: RewardsCfgV210b = RewardsCfgV210b()


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV210b_PLAY(LeIsaacLiftCubeRLVisualEnvCfgV210b):
    """Smaller scene + no obs corruption for replaying a V2.10b visual ckpt."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# V2.10c — V2.10b + dense linear distance reward EE↔cube.
#
# Diagnosis from V2.10b (cold-start, 22 iters):
#   - Smoothness penalties OK (joint_acc ÷33 worked: ~-0.23/ep, no paralysis).
#   - All penalties decreasing monotonically (policy IS smoothing).
#   - BUT `reaching_object` flat at 0.017/ep and slightly *decreasing* —
#     the policy drifts AWAY from the cube (EE settles at d ≈ 50 cm) and
#     finds a "stand still in non-cube position" optimum.
#
# Root cause: `reaching_object = 1 - tanh(d/0.15)` gives ~0 gradient past
# d ≈ 60 cm. Once the policy drifts to that range, no reward signal pulls
# it back. The smoothness gradient ("don't move") wins by default.
#
# Fix: add a linear distance term `weight × (-d)` that provides a
# CONSTANT gradient across the workspace. With `weight = -1.0`:
#   - At d = 0.45 m (where V2.10b stuck): -0.45/step = -67/ep (always-on)
#   - At d = 0.05 m (near cube): -0.05/step = -7/ep (small)
#   - At d = 0 (touching): 0
#
# γ=0.99 GAE bootstraps a global "value gradient toward cube" in V(s):
#   V(d=0.44) - V(d=0.45) ≈ +1 (vs r_t cost ~ -0.06) → advantage +1 ✓
#
# The term self-extinguishes during grasp/lift/transport (||EE - cube|| → 0
# when held), so it does not interfere with downstream rewards.
# ---------------------------------------------------------------------------


@configclass
class RewardsCfgV210c(RewardsCfgV210b):
    """V2.10c — V2.10b rewards + dense linear distance EE↔cube.

    Inherits all of V2.10b (smoothness rebalanced, V2.8.5 reward shaping).
    Adds one new term: `ee_to_cube_distance` which penalizes the raw L2
    distance between the EE and the cube. With weight -1.0, the term
    provides a constant ~-1/m gradient across the workspace, fixing the
    V2.10b "policy drifts away from cube" failure mode.
    """

    ee_to_cube_distance = RewTerm(
        func=eval2_mdp.object_ee_distance_l2,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=-1.0,
    )


@configclass
class LeIsaacLiftCubeRLEnvCfgV210c(LeIsaacLiftCubeRLEnvCfgV210b):
    """V2.10c state-only env: V2.10b + linear distance reward."""

    rewards: RewardsCfgV210c = RewardsCfgV210c()


@configclass
class LeIsaacLiftCubeRLEnvCfgV210c_PLAY(LeIsaacLiftCubeRLEnvCfgV210c):
    """Smaller scene + no obs corruption for replaying a V2.10c checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV210c(LeIsaacLiftCubeRLVisualEnvCfgV210b):
    """V2.10c visual env: V2.10b visual + linear distance reward."""

    rewards: RewardsCfgV210c = RewardsCfgV210c()


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV210c_PLAY(LeIsaacLiftCubeRLVisualEnvCfgV210c):
    """Smaller scene + no obs corruption for replaying a V2.10c visual ckpt."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# V2.11 — aligned with Isaac Lab Lift defaults + targeted SO-101 fixes.
#
# Motivation: 2 weeks of iterating reward shaping (V2.5 → V2.10c) drifted
# us far from published practice. Research synthesis (ManiSkill3, Isaac Lab,
# robosuite, DextrAH-RGB, IndustReal) flagged 4 specific divergences:
#
# 1. Smoothness via REWARD is the wrong tool. Published practice puts
#    smoothness in the ACTION SPACE (action.scale, velocity clipping,
#    geometric fabrics). Our V2.10 weights (action_rate=-5e-2, joint_vel
#    =-1e-2, joint_acc=-1e-3) were 100-500× the Isaac Lab Lift defaults
#    (-1e-4 each, no joint_acc) — paralyzed exploration.
#
# 2. Gating tracking by `is_grasped` is brittle on robots without clean
#    contact detection (LeIsaac SO-101 jaws). Isaac Lab Lift gates by
#    `lifted` only; this is robot-agnostic and recommended for SO-101.
#
# 3. Stage envelope ordering matters: max(reach) ≤ min(lift) ≤ min(track).
#    With our weights (1, 5, 10, 16) this is fine. Just verify lift-only
#    gating doesn't reintroduce the V2.6 flick exploit.
#
# 4. The V2.7 flick fix (`cube_lifted_and_grasped`) was needed because
#    V2.6 had no smoothness AND scale=0.5 (chaotic action space). With
#    V2.11's mild action.scale=0.25 + linear distance penalty, the
#    chaotic motor program that produced flicks should be naturally
#    suppressed — making lift-only gating safe again.
#
# Single env-side innovation we keep: `ee_to_cube_distance` linear penalty
# (V2.10c). It worked in V2.10c (reaching ×2.4 vs V2.10b) and provides
# the always-on global gradient that pure tanh lacks at large distances.
#
# Sources:
#   - https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/.../lift_env_cfg.py
#   - https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/pick_cube.py
#   - https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/pick_place.py
#   - DextrAH-RGB (arXiv:2412.01791)
# ---------------------------------------------------------------------------


@configclass
class RewardsCfgV211(RewardsCfgV210c):
    """V2.11 — Isaac Lab Lift defaults + V2.8.5 fixes + V2.10c linear distance.

    Inherits the V2.10c structure (which has placeholders, USD compat, and
    the linear distance term). Overrides the smoothness weights back to
    Isaac Lab Lift defaults, drops `joint_acc_l2`, and reverts the gating
    of `lifting_object` and tracking terms from `grasp ∧ lift` to
    `lift only`.

    Why we revert the V2.7 flick fix: that fix was needed in a regime with
    chaotic actions (no smoothness, action.scale=0.5). V2.11 retains the
    V2.10 mechanical action-space caps (action.scale=0.25), the V2.10c
    linear distance penalty (which discourages flick because it moves EE
    away from cube), and adds mild Isaac-Lab-default smoothness
    (-1e-4 weights). Together these prevent the flick motor program from
    forming, so lift-only gating is again safe.

    `lifting_object` weight: 15.0 (Isaac Lab default), back from V2.7's
    reduced 10.0. The V2.7 reduction was conservative under contact-based
    gating; lift-only gating with smoothness suppression doesn't need it.
    """

    # V2.11 v3: medium smoothness, calibrated to kill V2.9-style yeet
    # without paralyzing the policy.
    #
    # At yeet dynamics (V2.9 reference: |q_dot|max ~8.66 rad/s, |q_dot|²
    # total per step ~450), penalty per step:
    #   joint_vel × |q_dot|² = -5e-3 × 450 = -2.25/step → -340/ep
    # Vs reaching+lift+grasp briefly during a successful yeet ~ +100/ep:
    # yeet net = -240/ep → unprofitable, suppressed.
    #
    # At smooth motion (|q_dot|² total per step ~1):
    #   joint_vel × 1 = -5e-3 → -0.75/ep, negligible.
    #
    # Compared to V2.10's -1e-2 (which paralyzed via joint_acc dominance,
    # not via these terms specifically): ÷2 reduction. Compared to Isaac
    # Lab Lift default -1e-4: ×50. The default works for Franka because
    # Franka's action space (impedance/IK) doesn't allow chaotic yeet —
    # JointPositionAction on SO-101 with scale=0.5 does.
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-5e-3)

    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-5e-3,
    )

    # Effectively disable joint_acc by setting weight=0. Isaac Lab Lift
    # does not include this term; locomotion templates use -1e-7. Our
    # V2.10/V2.10b/V2.10c attempts to use it as a smoothness lever
    # backfired (overweight by 70× even at -3e-5).
    joint_acc = RewTerm(
        func=base_mdp.joint_acc_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=0.0,
    )

    # Revert lift-and-grasp gating to lift-only. Same V2.8.5
    # height_threshold (0.08 m above robot base) since that's the spawn-
    # margin fix from audit_scene.py — orthogonal to the gating choice.
    lifting_object = RewTerm(
        func=eval2_mdp.cube_lifted_above_base,
        params={
            "height_threshold": _V285_LIFT_HEIGHT_THRESHOLD,
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": "base",
        },
        weight=15.0,                                           # Isaac Lab default
    )

    # Tracking — gated by lifted only (not grasp+lift). Same std and
    # height_threshold as V2.8.5; only the gating function changes.
    object_goal_tracking = RewTerm(
        func=eval2_mdp.cube_to_goal_distance_above_base,
        params={
            "std": 0.3,
            "height_threshold": _V285_LIFT_HEIGHT_THRESHOLD,
            "command_name": "object_pose",
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": "base",
        },
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=eval2_mdp.cube_to_goal_distance_above_base,
        params={
            "std": 0.05,
            "height_threshold": _V285_LIFT_HEIGHT_THRESHOLD,
            "command_name": "object_pose",
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": "base",
        },
        weight=5.0,
    )

    # Lighter drop penalty (research recommends ≈ -1.0 instead of -5.0).
    # The DoneTerm itself already terminates the episode; the RewTerm just
    # needs to make drop net-negative vs continuing. Calibration:
    #   reaching at d=0.24 m gives ~0.16/step × 30 steps before drop = +4.8
    #   so penalty -1.0 makes drop barely net negative (-1 < +4.8) — wait,
    # this would make drop net POSITIVE again. Recalibrating:
    # reaching reward 0.16/step × 30 steps = 4.8. To make drop net-negative
    # we need penalty > 4.8. Keep -5.0 from V2.9.
    cube_dropped_penalty = RewTerm(
        func=eval2_mdp.cube_dropped_float,
        params={
            "world_z_threshold": 0.04,
            "cube_cfg": SceneEntityCfg("cube"),
        },
        weight=-5.0,
    )


@configclass
class LeIsaacLiftCubeRLEnvCfgV211(LeIsaacLiftCubeRLEnvCfgV210c):
    """V2.11 state-only env — Isaac Lab aligned + ee_to_cube linear.

    Sets `action.scale = 0.5` (V2.9 default, ±28.6° per-joint articular
    range from default). The earlier V2.11 attempt at scale=0.20 was
    based on a misunderstanding: I thought scale capped the **per-step
    velocity**, but `JointPositionActionCfg` with `use_default_offset=True`
    is **absolute** control — `target = scale × action + default_pos`.
    So scale=0.20 caps the joint *range* to ±11.5° from home, not the
    velocity per step. Diagnostic on model_200 confirmed the bug: the
    arm couldn't extend laterally to reach the cube (action_sat_count =
    150/150 every episode, gripper stuck 22 cm from cube).

    With scale=0.5 the joint range is ±28.6° — V2.9 used this and could
    reach successfully. Yeet protection now comes from the linear
    `ee_to_cube_distance` reward (penalizes EE far from cube), not from
    a mechanical cap that was never a cap.
    """

    rewards: RewardsCfgV211 = RewardsCfgV211()

    def __post_init__(self) -> None:
        super().__post_init__()
        # V2.11 v2 fix: revert to V2.9's scale=0.5 for full articular range.
        # ABSOLUTE control mode (use_default_offset=True) means
        # joint_target = scale × action + default_pos. With scale=0.20 from
        # the earlier V2.11 attempt, joint range was capped to ±0.20 rad
        # (±11.5°) — too restrictive, the SO-101 couldn't extend laterally
        # to reach a cube at 24cm distance. Diagnostic on model_200
        # showed action_sat_count=150/150 (always saturated), min_jaw_z
        # reached cube level vertically but min_grip_z stayed 12cm above.
        # Reverting to scale=0.5 gives ±28.6° per joint range — V2.9 used
        # this and could reach. Yeet protection comes from the linear
        # `ee_to_cube_distance` reward (penalizes EE far from cube).
        self.actions.arm_action.scale = 0.5


@configclass
class LeIsaacLiftCubeRLEnvCfgV211_PLAY(LeIsaacLiftCubeRLEnvCfgV211):
    """Smaller scene + no obs corruption for replaying a V2.11 checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV211(LeIsaacLiftCubeRLVisualEnvCfgV210c):
    """V2.11 visual env — Isaac Lab aligned + ee_to_cube linear.

    Same `action.scale = 0.5` (V2.9 default, ±28.6° per joint articular
    range) as the state-only V2.11 env.
    """

    rewards: RewardsCfgV211 = RewardsCfgV211()

    def __post_init__(self) -> None:
        super().__post_init__()
        # V2.11 v2 fix: scale 0.20 was too restrictive for ABSOLUTE control
        # (target = scale*action + default → ±0.20 rad joint range only).
        # See LeIsaacLiftCubeRLEnvCfgV211.__post_init__ for full rationale.
        self.actions.arm_action.scale = 0.5


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV211_PLAY(LeIsaacLiftCubeRLVisualEnvCfgV211):
    """Smaller scene + no obs corruption for replaying a V2.11 visual ckpt."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
