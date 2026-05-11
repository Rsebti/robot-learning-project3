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
        debug_vis=False,  # set to True to draw the goal pose RGB axis arrows
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


# ---------------------------------------------------------------------------
# V2.12 — V2.9 reward/PPO + DELTA action control (vmax mechanical cap).
#
# Strategy: V2.9 converged on the task (60% success deterministic on 3cm
# cube, 30% on 2cm). The only real failure was action quality (joints
# saturated at 10 rad/s, yeet/scoop motor program). V2.10 → V2.11 spent
# 2 weeks trying to fix the velocity issue via reward shaping, all
# failed (paralysis, plateau, or yeet returned).
#
# Root cause finally identified: ``JointPositionActionCfg`` with
# ``use_default_offset=True`` is **absolute** control (target = scale *
# action + default_pos), so ``scale`` bounds the *joint range*, not the
# velocity. There's no structural velocity cap.
#
# V2.12 fix: switch to ``RelativeJointPositionActionCfg`` (delta control,
# target = current_pos + scale * action). Now ``scale`` is a true
# per-step delta cap. With scale=0.20 and dt_ctrl=1/30s, vmax = 6 rad/s
# = Feetech STS3215 limit. The action space *cannot* command faster
# than the real servo can move — sim-to-real aligned by construction.
#
# With the velocity issue resolved structurally, we revert all the
# reward/PPO patches that were masking the problem:
#   - Smoothness rewards back to Isaac Lab default (-1e-4 each)
#   - Drop joint_acc_l2 entirely (Isaac Lab Lift doesn't use it)
#   - Restore V2.9 PPO config (init_noise=1.0, entropy=0.005)
#   - Keep V2.9 reward shaping (grasp+lift gating, lifting=10, etc.)
#
# Single env-side innovation we keep from V2.10c:
# `ee_to_cube_distance` (linear -d, weight=-1.0) — provides global value
# function gradient toward cube, accelerates exploration. V2.9 didn't
# need it because it had `init_noise=1.0` for broad exploration; we keep
# it as a bootstrap aid.
#
# References:
#   - https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html
#     (RelativeJointPositionAction docs)
#   - DextrAH-RGB (arXiv:2412.01791) — delta control with jerk limits
#   - ManiSkill3 PickCube — uses delta control
# ---------------------------------------------------------------------------


@configclass
class RewardsCfgV212(RewardsCfgV285):
    """V2.12 reward — V2.8.5 base + ee_to_cube_distance + 2 exploit fixes.

    Inherits V2.8.5 directly (V2.9 doesn't define a separate
    `RewardsCfgV29` — V2.9 added `cube_dropped_penalty` via the env's
    `__post_init__`). We declare `cube_dropped_penalty` here at class
    level with weight -50 (V2.12 fix), but the V2.9 env's
    `__post_init__` would otherwise overwrite it back to -5. To prevent
    that, `LeIsaacLiftCubeRLEnvCfgV212.__post_init__` re-asserts the
    -50 weight after `super().__post_init__()` runs.

    V2.8.5 brings: V2.7 grasp+lift gating (cube_lifted_and_grasped),
    audit fixes (lift_height_threshold=0.08, reaching std=0.15,
    success_bonus=1500), and noise-floor smoothness (-1e-4 each).

    Three modifications vs V2.9:
      1. NEW `ee_to_cube_distance` linear term (V2.10c innovation kept).
         Constant -1/m gradient via value function for early exploration.
      2. `success_bonus` weight 1500 → 2500 (math fix). At hover-near-
         goal (d=5.1cm), discounted future dense reward ≈ 1900. With
         success_bonus=1500, finishing nets 1500 < hover-100-steps net
         1900 → PPO prefers hover. Bumping to 2500 ensures
         success ≥ hover + safety margin.
      3. `cube_dropped_penalty` weight -5 → -15 (Touch-and-Yeet fix).
         A brief grasp+lift+drop trajectory under V2.9 weights gave net
         +30 (reach 5 + grasp 10 + lift 20 - drop 5). Bumping drop
         penalty to -15 makes such yeet trajectories net-negative.
    """

    ee_to_cube_distance = RewTerm(
        func=eval2_mdp.object_ee_distance_l2,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=-1.0,
    )

    # V2.12 fix #1 — math-driven: hover at d=5.1cm gives discounted
    # ~+1900 over remaining 100 steps (γ=0.99). Bump bonus above 1900.
    success_bonus = RewTerm(
        func=eval2_mdp.cube_at_goal,
        params={
            "distance_threshold": 0.05,
            "command_name": "object_pose",
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
        weight=2500.0,
    )

    # V2.12 fix #2 — Touch-and-Yeet exploit, REVISED to -50.
    #
    # Correct math: a typical yeet trajectory accumulates
    #   reach   30 steps × 0.16 = +5
    #   grasp    3 steps × 5    = +15  (briefly fires during contact)
    #   lift     2 steps × 10   = +20  (gated by grasp ∧ lift)
    #   total positive          = +40 in just a few steps
    #
    # At drop_penalty = -15, net yeet = +25 (still attractive — exploitable).
    # At -30, net = +10 (still slightly positive).
    # At -50, net = -10 (clearly net-negative — yeet suppressed).
    #
    # Risk of "fear of grasping" (policy avoids grasp attempts because of
    # accidental drops): with DELTA action control limiting impact
    # speeds, accidental drops should be rare (~2-5 %), so expected
    # penalty per episode is ~-2.5 vs +500-1500 expected from a sustained
    # grasp — not crippling. The -50 weight makes drop strictly
    # net-negative across all realistic yeet trajectories.
    #
    # Watch `Episode_Termination/cube_dropped` rate during V2.12 training:
    # if it stabilizes >30% sustained, the policy is dropping too often
    # and -50 may be excessive — would dial back to -30 in V2.13.
    cube_dropped_penalty = RewTerm(
        func=eval2_mdp.cube_dropped_float,
        params={
            "world_z_threshold": 0.04,
            "cube_cfg": SceneEntityCfg("cube"),
        },
        weight=-50.0,
    )


@configclass
class LeIsaacLiftCubeRLEnvCfgV212(LeIsaacLiftCubeRLEnvCfgV210):
    """V2.12 state-only env — V2.9 reward + DELTA action control.

    Inherits V2.10 for the env infrastructure (USD edits cube 2cm + table
    #B8ADA9, LeIsaac stock pose_range ±7.5cm + yaw ±30°, Phase C
    placeholders, cube init z compensation). Overrides:

      1. `rewards`: V2.12 (V2.9 + ee_to_cube_distance) instead of V2.10's
         heavy smoothness + V210 placeholder structure.
      2. `actions.arm_action`: replace V2.10's ``JointPositionActionCfg
         (scale=0.25, use_default_offset=True)`` with
         ``RelativeJointPositionActionCfg(scale=0.20, use_zero_offset=True)``.
         vmax_mechanical = 0.20 / (1/30) = 6.0 rad/s = Feetech limit.

    Why scale=0.20 here works (vs failed in V2.11 v1):
      - V2.11 v1 used scale=0.20 in ABSOLUTE mode → joint range capped to
        ±0.20 rad (±11.5°), too restrictive, robot couldn't reach.
      - V2.12 uses scale=0.20 in DELTA mode → per-step delta capped to
        0.20 rad/step, but no range restriction (joint can accumulate
        deltas to reach any position over multiple steps). vmax bounded.
    """

    rewards: RewardsCfgV212 = RewardsCfgV212()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Replace the inherited JointPositionActionCfg (absolute) with the
        # RelativeJointPositionActionCfg (delta). Same joint set, scale
        # tuned for vmax = 6 rad/s mechanical at 30 Hz control.
        self.actions.arm_action = base_mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex",
                         "wrist_flex", "wrist_roll"],
            scale=0.20,
            use_zero_offset=True,
        )

        # V2.12 — explicit relative vectors. The MLP would otherwise have
        # to learn (cube_pos - ee_pos) and (goal_pos - cube_pos) implicitly
        # from the absolute positions. Pre-computing these as obs terms
        # accelerates convergence on small networks (ETH HW4 SO-100 ref,
        # ManiSkill `tcp_to_obj` and `obj_to_goal`).
        self.observations.policy.ee_to_cube_vec = ObsTerm(
            func=eval2_mdp.ee_to_cube_vector,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        self.observations.policy.cube_to_goal_vec = ObsTerm(
            func=eval2_mdp.cube_to_goal_vector,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "robot_cfg": SceneEntityCfg("robot"),
                "command_name": "object_pose",
            },
        )

        # V2.12 fix #3 — Wandering Cutoff (fail-fast termination).
        # Episode terminates if EE drifts to >50 cm from cube. Saves
        # the ~100 wasted sim steps on lost episodes during early iters
        # where the bras explores randomly and ends up far from the cube.
        self.terminations.ee_far_from_cube = DoneTerm(
            func=eval2_mdp.ee_far_from_cube,
            params={
                "distance_threshold": 0.5,
                "cube_cfg": SceneEntityCfg("cube"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )

        # V2.12 fix #2 (re-asserted) — V2.9 env's __post_init__ chain
        # assigns `self.rewards.cube_dropped_penalty = RewTerm(weight=-5)`
        # which OVERWRITES our class-level RewardsCfgV212.cube_dropped_penalty
        # = -50 declaration. We re-create the term here after super() to
        # restore the -50 weight. Same with success_bonus (V285 declares
        # 1500 at class level; our V212 class-level override should win
        # via @configclass MRO, but we re-assert defensively).
        self.rewards.cube_dropped_penalty = RewTerm(
            func=eval2_mdp.cube_dropped_float,
            params={
                "world_z_threshold": 0.04,
                "cube_cfg": SceneEntityCfg("cube"),
            },
            weight=-50.0,
        )
        self.rewards.success_bonus = RewTerm(
            func=eval2_mdp.cube_at_goal,
            params={
                "distance_threshold": 0.05,
                "command_name": "object_pose",
                "cube_cfg": SceneEntityCfg("cube"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
            weight=2500.0,
        )


@configclass
class LeIsaacLiftCubeRLEnvCfgV212_PLAY(LeIsaacLiftCubeRLEnvCfgV212):
    """Smaller scene + no obs corruption for replaying a V2.12 checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV212(LeIsaacLiftCubeRLVisualEnvCfgV210):
    """V2.12 visual env — V2.9 reward + DELTA action control.

    Same structure as `LeIsaacLiftCubeRLEnvCfgV212` (state-only) but
    with the wrist camera + ResNet-18 features re-enabled (inherits
    V210 visual). Action class swapped to delta mode the same way.
    """

    rewards: RewardsCfgV212 = RewardsCfgV212()

    def __post_init__(self) -> None:
        super().__post_init__()
        # See LeIsaacLiftCubeRLEnvCfgV212.__post_init__ for full rationale.
        self.actions.arm_action = base_mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex",
                         "wrist_flex", "wrist_roll"],
            scale=0.20,
            use_zero_offset=True,
        )

        # V2.12 explicit relative vectors (see state-only env for rationale).
        self.observations.policy.ee_to_cube_vec = ObsTerm(
            func=eval2_mdp.ee_to_cube_vector,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        self.observations.policy.cube_to_goal_vec = ObsTerm(
            func=eval2_mdp.cube_to_goal_vector,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "robot_cfg": SceneEntityCfg("robot"),
                "command_name": "object_pose",
            },
        )

        # V2.12 fix #3 — Wandering Cutoff (see state-only env for rationale).
        self.terminations.ee_far_from_cube = DoneTerm(
            func=eval2_mdp.ee_far_from_cube,
            params={
                "distance_threshold": 0.5,
                "cube_cfg": SceneEntityCfg("cube"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )

        # V2.12 fix #2 / fix #1 re-asserted (defeats V29 __post_init__ override).
        # See LeIsaacLiftCubeRLEnvCfgV212.__post_init__ for rationale.
        self.rewards.cube_dropped_penalty = RewTerm(
            func=eval2_mdp.cube_dropped_float,
            params={
                "world_z_threshold": 0.04,
                "cube_cfg": SceneEntityCfg("cube"),
            },
            weight=-50.0,
        )
        self.rewards.success_bonus = RewTerm(
            func=eval2_mdp.cube_at_goal,
            params={
                "distance_threshold": 0.05,
                "command_name": "object_pose",
                "cube_cfg": SceneEntityCfg("cube"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
            weight=2500.0,
        )


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV212_PLAY(LeIsaacLiftCubeRLVisualEnvCfgV212):
    """Smaller scene + no obs corruption for replaying a V2.12 visual ckpt."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# V2.13 — V2.12 + 3 fixes for snake/scoop pathology and action saturation.
#
# Diagnostic V2.12 model_900 (play_diagnose_v2 + visual play):
#   - Action histogram: shoulder_lift p95=+2.66, wrist_flex p25=-2.52,
#     gripper p05=-5.94 — actions saturate FAR beyond [-1, 1] expected
#     range. The DELTA action class has no clip → vmax breach (peak
#     16-21 rad/s observed, vs Feetech limit 6 rad/s).
#   - Posture: gripper-down score = +0.17 (avg) / -0.27 at GRASP.
#     Steps with gripper down (>0.5): 0/150. Min jaw_z BEFORE first
#     grasp = 5.5cm (table top is 4.15cm → jaw scrapes table). The
#     policy approaches HORIZONTALLY with the jaw at table level and
#     the gripper actually pointing UPWARD at grasp moment. Sim-to-
#     real impossible (table friction model differs).
#
# V2.13 = V2.12 + 3 fixes:
#   1. Action `clip={".*": (-1.0, 1.0)}` — finally enforces the
#      mechanical vmax = 6 rad/s that V2.12 promised but didn't deliver.
#   2. NEW `gripper_orientation_penalty` (weight -1.0) — penalty for
#      gripper z-axis not pointing down. Implemented as PENALTY (not
#      bonus) to avoid the "free reward for standing still in good
#      posture" exploit. Neutral state = pointing down, deviation costs.
#   3. NEW `scoop_grasp_penalty` (weight -10.0) — penalty when
#      EE-palm height > wrist height (geometric inversion of top-down
#      grasp). Threshold-free, robust across robot configs.
#
# Together (2)+(3) make the top-down posture the only stable optimum:
# gripper must point down AND wrist must stay above the gripper.
# All other V2.12 reward shaping preserved (grasp+lift gating, weights,
# success_bonus=2500, cube_dropped_penalty=-50, ee_to_cube_distance,
# relative obs vectors, ee_far_from_cube DoneTerm).
#
# PPO config unchanged from V2.12 (V2.9 baseline that proved convergent).
# Cold-start only — V2.12 is committed to scoop, can't be unlearned.
# ---------------------------------------------------------------------------


@configclass
class RewardsCfgV213(RewardsCfgV212):
    """V2.13 v2 — V2.12 + posture/scoop penalties (calibrated softer).

    History: V2.13 v1 (initial) had gripper_orientation_penalty=-1.0,
    scoop_grasp_penalty=-10.0, cube_dropped=-150, ee_far_from_cube
    DoneTerm active. At iter 0-50 the policy DIVERGED via give-up
    exploit: episodes terminated at step 18-30 by triggering
    ee_far_from_cube (no penalty paired) to escape the heavy
    per-step penalty regime. noise_std went UP (1.0 → 1.13) and
    reaching went DOWN (10% → 0.5%).

    V2.13 v2 fixes (this class):
      - scoop_grasp_penalty: -10 → -5 (less brutal but still bites scoop)
      - reaching_object: 1.0 → 1.5 (boost positive signal so policy
        has clear net-positive baseline near cube)
      - cube_dropped_penalty: -150 → -30 (in env __post_init__)
      - REMOVE ee_far_from_cube DoneTerm (in env __post_init__) —
        eliminates the give-up exploit channel
      - gripper_orientation_penalty: keep -1.0 (gentle guide)

    With reaching boost +1.5 and softer penalties, baseline at random
    init is ~0/ep (reach +75 vs orientation -75 + scoop -25 = balanced).
    Removes give-up incentive. PPO must explore toward grasp/lift to
    find positive territory.
    """

    # V2.13 v2 — boost reaching to make baseline non-negative at random init.
    # Inherited from V2.8.5: weight=1.0. Override to +1.5.
    reaching_object = RewTerm(
        func=lift_mdp.object_ee_distance,
        params={
            "std": 0.15,
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=1.5,
    )

    gripper_orientation_penalty = RewTerm(
        func=eval2_mdp.gripper_orientation_penalty,
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=-1.0,
    )

    scoop_grasp_penalty = RewTerm(
        func=eval2_mdp.scoop_grasp_penalty,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names="wrist"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=-5.0,
    )


@configclass
class LeIsaacLiftCubeRLEnvCfgV213(LeIsaacLiftCubeRLEnvCfgV212):
    """V2.13 state-only env: V2.12 + clip on action + posture penalties."""

    rewards: RewardsCfgV213 = RewardsCfgV213()

    def __post_init__(self) -> None:
        super().__post_init__()
        # V2.13 fix #1: add clip to action term.
        # V2.12 used RelativeJointPositionActionCfg(scale=0.20) without
        # clip → policy outputs raw actions up to ±6 → joint delta up to
        # 1.2 rad/step → vmax peak 36 rad/s (way above Feetech 6 rad/s).
        # With clip={".*": (-1.0, 1.0)}, raw action is bounded to ±1
        # before scale → joint delta capped at 0.20 rad/step → vmax = 6
        # rad/s mechanical (Feetech-aligned, sim-to-real safe).
        self.actions.arm_action = base_mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex",
                         "wrist_flex", "wrist_roll"],
            scale=0.20,
            use_zero_offset=True,
            clip={".*": (-1.0, 1.0)},
        )

        # V2.13 v2 reward overrides:
        # cube_dropped_penalty: -150 (v1) → -30 (v2). Still strong enough
        # to make Touch-and-Yeet net-negative (yeet ~+25 reward, drop -30
        # → net -5). Less crippling for accidental drops during learning.
        self.rewards.cube_dropped_penalty = RewTerm(
            func=eval2_mdp.cube_dropped_float,
            params={
                "world_z_threshold": 0.04,
                "cube_cfg": SceneEntityCfg("cube"),
            },
            weight=-30.0,
        )
        self.rewards.success_bonus = RewTerm(
            func=eval2_mdp.cube_at_goal,
            params={
                "distance_threshold": 0.05,
                "command_name": "object_pose",
                "cube_cfg": SceneEntityCfg("cube"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
            weight=2500.0,
        )

        # V2.13 v2 critical fix — REMOVE ee_far_from_cube DoneTerm.
        # V2.13 v1 had this fail-fast active (inherited from V2.12). With
        # V2.13's per-step penalties active, the policy DIVERGED by
        # learning to wander out (>0.5m from cube) to trigger this
        # DoneTerm and escape the penalty regime — episodes ended at
        # step 18 vs max 150. This DoneTerm has no associated reward
        # penalty, so triggering it is "free escape" → unstoppable
        # give-up exploit. Removing the DoneTerm forces episodes to
        # run full 150 steps; policy must find positive reward (grasp,
        # lift, track) to maximize, can't escape via wandering.
        # V2.12 showed this DoneTerm rarely fires naturally (0/16 ep
        # at model_900) so removing it has no downside in normal play.
        self.terminations.ee_far_from_cube = None


@configclass
class LeIsaacLiftCubeRLEnvCfgV213_PLAY(LeIsaacLiftCubeRLEnvCfgV213):
    """Smaller scene + no obs corruption for replaying a V2.13 checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV213(LeIsaacLiftCubeRLVisualEnvCfgV212):
    """V2.13 visual env: V2.12 visual + clip on action + posture penalties."""

    rewards: RewardsCfgV213 = RewardsCfgV213()

    def __post_init__(self) -> None:
        super().__post_init__()
        # V2.13 fix #1: clip on action (see state-only env for rationale).
        self.actions.arm_action = base_mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex",
                         "wrist_flex", "wrist_roll"],
            scale=0.20,
            use_zero_offset=True,
            clip={".*": (-1.0, 1.0)},
        )

        # V2.13 v2 (see state-only env for rationale):
        # cube_dropped_penalty -150 → -30, remove ee_far_from_cube DoneTerm.
        self.rewards.cube_dropped_penalty = RewTerm(
            func=eval2_mdp.cube_dropped_float,
            params={
                "world_z_threshold": 0.04,
                "cube_cfg": SceneEntityCfg("cube"),
            },
            weight=-30.0,
        )
        self.rewards.success_bonus = RewTerm(
            func=eval2_mdp.cube_at_goal,
            params={
                "distance_threshold": 0.05,
                "command_name": "object_pose",
                "cube_cfg": SceneEntityCfg("cube"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
            weight=2500.0,
        )

        # V2.13 v2 critical fix — REMOVE ee_far_from_cube DoneTerm
        # (see state-only env for full rationale).
        self.terminations.ee_far_from_cube = None


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV213_PLAY(LeIsaacLiftCubeRLVisualEnvCfgV213):
    """Smaller scene + no obs corruption for replaying a V2.13 visual ckpt."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# V2.13 v3 — sign-bug fix on gripper_orientation_penalty + reward landscape
# overhaul to escape the "hover above cube" local optimum that V2.13 v2
# converged on.
#
# Diagnostic post-mortem (V2.13 v2 model_100, iter 128, crashed VF blowup):
#   - 16/16 episodes TIMEOUT, 0% success, 0% grasp, 0% drop
#   - min jaw.z = +0.173m sustained (EE hovers 12cm ABOVE the 5cm cube top)
#   - gripper_orientation_penalty stuck at -0.10/ep — looked "good" but
#     was actually a LIE: the old quat-based formula returned 0 (no penalty)
#     when gripper pointed UP and 2 when DOWN, because the gripper frame's
#     local +z axis points BACKWARD (toward base), not toward the jaw.
#     Confirmed by dump_scene_frames: gripper frame at home has local +z →
#     -world_y (back into robot).
#   - Visual replay (2026-05-11): policy converged on "shoulder_lift at
#     +1.745 max, wrist_flex at -1.658 min, gripper pointing UP, hovering"
#   - reaching_object tanh saturated at 0.72 from iter 60+, no descent
#     gradient (tanh asymptote)
#   - scoop_grasp_penalty at -0.27/ep but RANDOM rollout shows 0% fire
#     on gripper-down samples → redundant with the new orientation penalty
#     when the sign is fixed
#
# V2.13 v3 = 4 fixes (env-side only, PPO unchanged):
#   1. gripper_orientation_penalty REWRITTEN (in mdp/rewards.py) to use
#      jaw_z vs palm_z directly. Unambiguous, position-based, no quat math.
#      Returns 0 when jaw ≤ palm (top-down OK), ~1 when jaw is 5cm above
#      palm (fully up). Weight = -5.0 (strong enough to overcome any
#      scoop bias).
#   2. scoop_grasp_penalty DROPPED (weight 0). Random rollout confirmed
#      it fires 0% on gripper-down samples — redundant.
#   3. reaching_object tanh DROPPED (weight 0). Saturated at 0.72 from
#      iter 60+, no descent gradient remaining.
#   4. ee_to_cube_distance BOOSTED (weight -1.0 → -3.0). Becomes the
#      only non-saturating driver toward the cube. Linear in distance,
#      drives vertical descent reliably.
#
# Cold-start required: model_100 has internalized the gripper-UP pose,
# irrecoverable. PPO config unchanged (still uses lift_v2_13 experiment
# name — V213v3 logs will go into the same logs/rsl_rl/lift_v2_13/ dir,
# distinguished by timestamp + env.yaml weights).
# ---------------------------------------------------------------------------


@configclass
class RewardsCfgV213v3(RewardsCfgV213):
    """V2.13 v3 — sign-fixed gripper_orientation_penalty + reward overhaul.

    Inherits RewardsCfgV213 (V2.13 v2 stack) and overrides 4 weights:
      - reaching_object              : 1.5 → 3.0   (boosted for suicide insurance)
      - ee_to_cube_distance          : -1.0 → -3.0 (linear driver, boosted)
      - gripper_orientation_penalty  : -1.0 → -5.0 (sign-fixed in rewards.py)
      - scoop_grasp_penalty          : -5.0 → 0.0  (redundant, dropped)

    Other terms (grasping_cube, lifting_object, object_goal_tracking,
    success_bonus, cube_dropped_penalty, action_rate, joint_vel) inherit
    unchanged from RewardsCfgV213.

    Why reaching_object = +3.0 instead of 0 (anti-suicide insurance):
        With reach=0 + linear=-3 + orient=-5, the per-step cost at cold-start
        is ≈ -3.22 (d≈0.24m, orient random ≈0.5). Total timeout cost = -483/ep.
        Suicide-by-drop at step 20 costs only -94 (-30 drop + 64 partial cost).
        PPO advantage would prefer suicide ×5 over slow learning.
        Reaching=+3 reactivates a positive bonus near the cube: at d=0.05m
        with tanh(d/0.15) → reach=+2.04/step. Combined with orient=0 (gripper
        down by V213v3 penalty -5) and linear=-0.15/step → net +1.89/step.
        Stay-near saturates at +283/ep, which dominates suicide at -94/ep.
        The orient=-5 penalty prevents the hover-at-elevation V2.13 v2 mode
        because gripper-up costs heavily — policy must descend toward cube
        to stay near in down posture.
    """

    reaching_object = RewTerm(
        func=lift_mdp.object_ee_distance,
        params={
            "std": 0.15,
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=3.0,   # V213v3: 1.5 → 3.0 (anti-suicide insurance, see class docstring)
    )

    gripper_orientation_penalty = RewTerm(
        func=eval2_mdp.gripper_orientation_penalty,
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=-5.0,  # V213v3: -1.0 → -5.0 (sign-fixed formula, strong push to down)
    )

    scoop_grasp_penalty = RewTerm(
        func=eval2_mdp.scoop_grasp_penalty,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names="wrist"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=0.0,   # V213v3: -5.0 → 0.0 (redundant with sign-fixed orientation)
    )


@configclass
class LeIsaacLiftCubeRLEnvCfgV213v3(LeIsaacLiftCubeRLEnvCfgV213):
    """V2.13 v3 state-only env — sign-fixed orientation + reward overhaul."""

    rewards: RewardsCfgV213v3 = RewardsCfgV213v3()

    def __post_init__(self) -> None:
        super().__post_init__()
        # V213v3: boost ee_to_cube_distance linear driver to -3.0.
        # Inherited from V285 chain → was -1.0. We override it here after
        # super() because the class-level field on RewardsCfgV213v3 above
        # would be enough, but @configclass MRO can be subtle so we
        # re-assert defensively.
        self.rewards.ee_to_cube_distance = RewTerm(
            func=eval2_mdp.object_ee_distance_l2,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
            weight=-3.0,
        )


@configclass
class LeIsaacLiftCubeRLEnvCfgV213v3_PLAY(LeIsaacLiftCubeRLEnvCfgV213v3):
    """Smaller scene + no obs corruption for replaying a V2.13 v3 checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV213v3(LeIsaacLiftCubeRLVisualEnvCfgV213):
    """V2.13 v3 visual env — sign-fixed orientation + reward overhaul."""

    rewards: RewardsCfgV213v3 = RewardsCfgV213v3()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.ee_to_cube_distance = RewTerm(
            func=eval2_mdp.object_ee_distance_l2,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
            weight=-3.0,
        )


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV213v3_PLAY(LeIsaacLiftCubeRLVisualEnvCfgV213v3):
    """Smaller scene + no obs corruption for replaying a V2.13 v3 visual ckpt."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# V2.14 — "Slow & Precise" fix for V2.13 v3's Bang-Bang smash exploit.
#
# Diagnostic V2.13 v3 model_100 (play_diagnose 2026-05-11):
#   - GOOD: 50% phase=GRASPED (up from 0% in V213v2). Sign-fixed orientation
#     penalty works — gripper is geometrically top-down (jaw below palm).
#     No suicide-by-drop (drop rate 0%). Reach signal active.
#   - BAD: "Bang-Bang vertical smash" emerged. Policy commits shoulder_lift
#     to clip+1 (max DOWN) every step → joints saturate at +1.745 rad limit
#     in ~5-7 steps → jaws crash table at 75 rad/s (12× the Feetech 6 rad/s
#     limit; the clip on the RAW action only limits commanded velocity, not
#     the qdot spike induced by contact constraints).
#   - Symptoms:
#     - time-to-first-grasp = 9.9 steps = 0.33s (way too fast)
#     - table sliding 147/150 steps (98%, jaw_z < 0.06m)
#     - tip below table 5/16 ep (31%, physics breach)
#     - max |qdot| up to 75 rad/s (no chance for Feetech servo to track)
#     - grasps brief & unstable: 9 grasp events, all 9 ended in grasp_LOST
#       (slip=4, ejection=5), 0 lifts despite 8/16 ep reaching GRASPED phase
#
# Root cause (PPO bang-bang control):
#   ee_to_cube_distance at -3.0 per step creates a strong time-pressure: every
#   step spent far from cube costs reward. PPO's optimal is to traverse the
#   distance AS FAST AS POSSIBLE — even if that means smashing into the table.
#   The reward landscape says "rush down" because nothing penalizes the rush
#   strongly enough to overcome the per-step distance cost.
#
# V2.14 = V2.13 v3 + 4 changes (env-side only, PPO unchanged):
#   1. episode_length_s 5.0 → 10.0s (300 steps): aligns with teleop pacing
#      (10-15s episodes), gives the policy time to be slow without missing
#      grasp opportunities.
#   2. arm_action.scale 0.20 → 0.10: HARD speed cap at vmax = 0.10 / (1/30s)
#      = 3 rad/s (= 50% of Feetech 6 rad/s limit). Even when the policy
#      commands +1 raw, the actuator delta is now ±0.10 rad/step → physical
#      maximum velocity falls to 3 rad/s. Sim-to-real even safer.
#   3. joint_vel_l2 weight -1e-4 → -1e-3 (×10): soft smoothness penalty as
#      backup to the hard cap. Calibrated 10× below the V2.10 paralysis
#      threshold (-1e-2 caused full paralysis cold-start).
#   4. action_rate_l2 weight -1e-4 → -1e-3 (×10): discourages saccades and
#      "anticipate-then-slam" Δaction patterns. Encourages anticipating
#      table contact and decelerating.
#
# Cold-start required: V213v3's rush strategy is internalized.
# ETA unchanged (~14h on RTX 5070): num_steps_per_env=50 stays the same,
# only fewer episodes complete per iter but same training data quantity.
# ---------------------------------------------------------------------------


@configclass
class RewardsCfgV214(RewardsCfgV213v3):
    """V2.14 — V2.13 v3 + boosted smoothness penalties (joint_vel & action_rate ×10).

    Inherits RewardsCfgV213v3 (sign-fixed orient + reach+3 + ee_to_cube-3 +
    scoop dropped) and overrides 2 weights:
      - joint_vel_l2   : -1e-4 → -1e-3 (×10 soft smoothness penalty)
      - action_rate_l2 : -1e-4 → -1e-3 (×10 discourage saccades)

    All other terms unchanged from V213v3.

    Why ×10 and not ×50 (other LLM's "Aérofreins" proposal):
        V2.10 used joint_vel_l2 = -1e-2 (×100 from default) → full paralysis
        at cold-start (reaching dropped from 1.5% to 0.4% in 95 iters).
        V2.10b used -3e-5 (÷33) → smoothness OK but reaching plateau at 1.7%.
        V2.11 v3 attempted -5e-3 (medium) but was annulled before testing.
        ×10 = -1e-3 sits 5× below the proven paralysis threshold, giving
        margin for the harder constraint we have on top: action.scale=0.10
        (hard cap vmax 3 rad/s). The hard cap does most of the work; this
        soft penalty just discourages residual fast spikes.
    """

    action_rate = RewTerm(
        func=base_mdp.action_rate_l2,
        weight=-1e-3,   # V214: -1e-4 → -1e-3 (×10)
    )

    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-1e-3,   # V214: -1e-4 → -1e-3 (×10)
    )


@configclass
class LeIsaacLiftCubeRLEnvCfgV214(LeIsaacLiftCubeRLEnvCfgV213v3):
    """V2.14 state-only env — V2.13 v3 + 10s episodes + scale 0.10 + smoothness ×10.

    Designed to kill the Bang-Bang smash exploit by:
      (a) extending episode length to 10s (matching teleop pacing),
      (b) hard-capping commanded vmax at 3 rad/s via action scale halving,
      (c) softly penalizing residual high velocity / saccades via reward ×10.
    """

    rewards: RewardsCfgV214 = RewardsCfgV214()

    def __post_init__(self) -> None:
        super().__post_init__()
        # V214 fix #1: longer episodes (5s → 10s = 300 steps at 30 Hz).
        # Aligns with teleop pacing (10-15s for pick-and-place).
        # NB: this also changes the rsl_rl normalization of Episode_Reward/X
        # (= per-ep cumsum / max_episode_length_s) so reported values will be
        # smaller for the same per-step rate.
        self.episode_length_s = 10.0

        # V214 fix #2: hard speed cap via action scale halving.
        # Original V213 scale=0.20 → vmax = 0.20 / (1/30s) = 6 rad/s = Feetech limit.
        # V214 scale=0.10  → vmax = 0.10 / (1/30s) = 3 rad/s = 50% of Feetech.
        # Even with raw action saturated at clip ±1, per-step joint delta is
        # capped at ±0.10 rad → physical maximum velocity = 3 rad/s.
        self.actions.arm_action = base_mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex",
                         "wrist_flex", "wrist_roll"],
            scale=0.10,
            use_zero_offset=True,
            clip={".*": (-1.0, 1.0)},
        )


@configclass
class LeIsaacLiftCubeRLEnvCfgV214_PLAY(LeIsaacLiftCubeRLEnvCfgV214):
    """Smaller scene + no obs corruption for replaying a V2.14 checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV214(LeIsaacLiftCubeRLVisualEnvCfgV213v3):
    """V2.14 visual env — V2.13 v3 visual + 10s episodes + scale 0.10 + smoothness ×10."""

    rewards: RewardsCfgV214 = RewardsCfgV214()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 10.0
        self.actions.arm_action = base_mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex",
                         "wrist_flex", "wrist_roll"],
            scale=0.10,
            use_zero_offset=True,
            clip={".*": (-1.0, 1.0)},
        )


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV214_PLAY(LeIsaacLiftCubeRLVisualEnvCfgV214):
    """Smaller scene + no obs corruption for replaying a V2.14 visual ckpt."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# V2.15 — "Strict Top-Down" — fixes V2.14's snake-mode local optimum.
#
# Visual replay diagnostic V2.14 model_100 (2026-05-11) :
#   - Bang-Bang killed (qdot 27 → 5 mean, 0% physics breach). ✅
#   - But policy converged on SNAKE/HORIZONTAL approach: shoulder lift +0.97,
#     wrist_flex +0.87, elbow +0.82 → arm extends low and forward, gripper
#     skims table at jaw_z ~0.05m, approaches cube laterally instead of from
#     above.
#   - 98% of steps with jaw_z < 0.06m (table-sliding).
#   - 0% LIFTING phase reached.
#   - The V213v3 `gripper_orientation_penalty` (clamp((jaw_z - palm_z)/0.05))
#     returns 0 in BOTH top-down (jaw below palm) AND horizontal (jaw beside
#     palm at same z). Two zero-penalty local optima exist; PPO chose
#     horizontal by exploration luck.
#
# V2.15 = V2.14 + 2 fixes (env-side only, PPO unchanged):
#   1. REPLACE gripper_orientation_penalty function with the direction-
#      based version `gripper_pointing_direction_penalty`. Uses the
#      world-frame palm→jaw direction (normalized). Returns 0 (top-down),
#      1 (horizontal), 2 (gripper-up). Same weight -5.0.
#      This catches both gripper-up AND horizontal/snake.
#   2. ADD `jaw_below_cube_penalty` reward term. Penalizes jaw_z going
#      below the live cube bottom (= cube_z - 0.010m). Weight -50.
#      Doesn't fire on V214's current behavior (min jaw_z=0.042 vs
#      cube_bottom=0.031, margin 1.1cm). Acts as safety net against
#      future regression toward physics breaches.
#
# Cold-start required : V2.14's snake strategy is internalized.
# ETA ~14-19h. Same PPO config (LiftCubePPORunnerCfgV213).
#
# Empirical dimension verifications (from dump_scene_frames):
#   - cube.root_pos_w.z = 0.041 (constant at reset, USD-defined)
#   - cube_half_size = 0.010 (USD bbox)
#   - cube_bottom_z = 0.031 (= table top, matches LeIsaac scene)
#   - cube_top_z = 0.051
#   - V214 min jaw_z (model_100) = 0.042 → 1.1cm above cube_bottom → no
#     jaw_below_cube penalty fires on current behavior.
#   - V213v3 min jaw_z (model_100) = 0.010 → 2.1cm BELOW cube_bottom →
#     penalty would have cost ~30/ep, deterring physics breach.
# ---------------------------------------------------------------------------


@configclass
class RewardsCfgV215(RewardsCfgV214):
    """V2.15 — V2.14 + strict-top-down (palm→jaw direction) + jaw_below_cube.

    Inherits RewardsCfgV214 (V213v3 weights + smoothness ×10) and:
      - REPLACES `gripper_orientation_penalty` term with the direction-
        based function (same name in MDP namespace; weight unchanged at
        -5.0).
      - ADDS new `jaw_below_cube_penalty` term at weight -50.

    All other terms inherited from V214 unchanged.
    """

    gripper_orientation_penalty = RewTerm(
        func=eval2_mdp.gripper_pointing_direction_penalty,   # V215 new fn
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=-5.0,
    )

    jaw_below_cube_penalty = RewTerm(
        func=eval2_mdp.jaw_below_cube_penalty,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cube_half_size": 0.010,
        },
        weight=-50.0,
    )


@configclass
class LeIsaacLiftCubeRLEnvCfgV215(LeIsaacLiftCubeRLEnvCfgV214):
    """V2.15 state-only env — V2.14 + strict-top-down + jaw_below_cube."""

    rewards: RewardsCfgV215 = RewardsCfgV215()


@configclass
class LeIsaacLiftCubeRLEnvCfgV215_PLAY(LeIsaacLiftCubeRLEnvCfgV215):
    """Smaller scene + no obs corruption for replaying a V2.15 checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV215(LeIsaacLiftCubeRLVisualEnvCfgV214):
    """V2.15 visual env — V2.14 visual + strict-top-down + jaw_below_cube."""

    rewards: RewardsCfgV215 = RewardsCfgV215()


@configclass
class LeIsaacLiftCubeRLVisualEnvCfgV215_PLAY(LeIsaacLiftCubeRLVisualEnvCfgV215):
    """Smaller scene + no obs corruption for replaying a V2.15 visual ckpt."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
