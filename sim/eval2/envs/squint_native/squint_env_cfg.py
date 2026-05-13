"""ManagerBasedRLEnvCfg for the Squint-native Place task.

Wires together:
- Scene  (squint_scene.SquintNativeSceneCfg)
- Action (delta-target joint position from squint_actions)
- Observations (state vector matching Squint, from squint_observations)
- Events (reset robot to home + spawn cube/bin from squint_events)
- No reward / no termination (deploy mode — we only want to evaluate the
  ManiSkill checkpoint, not retrain). A timeout terminates each episode.

Sim timing matches Squint exactly:
- sim_freq    = 100  Hz (sim.dt = 0.01)
- control_freq = 10  Hz (decimation = 10)
- max_episode_steps = 50 (= 5 s)
"""
from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import squint_events, squint_observations
from .squint_actions import DeltaTargetJointPositionActionCfg
from .squint_scene import SquintNativeSceneCfg


# ---------------------------------------------------------------------------
# MDP groups
# ---------------------------------------------------------------------------


@configclass
class SquintActionsCfg:
    """Single delta-target joint position action covering all 6 joints.

    Bounds match Squint's ``pd_joint_target_delta_pos`` config:
        arm joints:    ±0.1 rad
        gripper joint: ±0.2 rad
    """

    arm_and_gripper = DeltaTargetJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        bounds=[0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
    )


@configclass
class SquintObservationsCfg:
    """Mirrors EXACTLY what Squint's policy reads after FlattenRGBDObservationWrapper:

    - obs['state'] = [qpos(6), target_qpos(6)]   (12 floats)
    - obs['rgb']   = wrist_rgb downsampled to 16x16 (uint8)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """18-d state vector for the new flattable_woodcube checkpoint:
        [qpos(6), target_qpos(6), goal_color_one_hot(6)].

        Older 12-d checkpoints can be deployed by slicing the state to
        ``[:12]`` in the deploy script.
        """

        qpos = ObsTerm(func=squint_observations.joint_pos_with_noise)
        target_qpos = ObsTerm(func=squint_observations.controller_target_qpos)
        goal_color_one_hot = ObsTerm(
            func=squint_observations.goal_color_one_hot,
            params={"goal_color_idx": 0},  # 0=red (matches our red cube)
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class RgbCfg(ObsGroup):
        """16x16 wrist RGB — RAW (no greenscreen).

        The current checkpoint was trained on Squint with apply_overlay=False
        and a table/background colored to match Isaac's natural render, so
        the policy never sees a greenscreened image. Compositing one here
        would push the obs out of the training distribution.
        """

        rgb = ObsTerm(
            func=squint_observations.wrist_rgb_16,
            params={"greenscreen_bg_rgb": None},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    rgb: RgbCfg = RgbCfg()


@configclass
class SquintEventsCfg:
    """Reset events. Order matters: reset robot first, then place items.

    The ``make_robot_matte_black`` startup event repaints every Shader prim
    under the robot articulation to a flat near-black material (Squint's
    training scene uses a black robot).
    """

    matte_robot = EventTerm(
        func=squint_events.make_robot_matte_black,
        mode="startup",
        params={"rgb": (0.02, 0.02, 0.02)},
    )

    reset_robot = EventTerm(
        func=squint_events.reset_robot_to_home,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "qpos_noise_std": 0.02,
        },
    )
    reset_cube_and_bin = EventTerm(
        func=squint_events.reset_scene_squint,
        mode="reset",
        params={"cube_cfg": SceneEntityCfg("cube")},
    )


@configclass
class SquintTerminationsCfg:
    """Episode timeout only — deploy mode, no task termination."""

    time_out = DoneTerm(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True)


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------


@configclass
class SquintNativePlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Squint-native Place env (training shape: many parallel envs)."""

    scene: SquintNativeSceneCfg = SquintNativeSceneCfg(num_envs=256, env_spacing=2.0)
    actions: SquintActionsCfg = SquintActionsCfg()
    observations: SquintObservationsCfg = SquintObservationsCfg()
    events: SquintEventsCfg = SquintEventsCfg()
    terminations: SquintTerminationsCfg = SquintTerminationsCfg()
    # No commands, no rewards, no curriculum (deploy mode).
    commands: object = None
    rewards: object = None
    curriculum: object = None

    def __post_init__(self):
        # ============================================================
        # 30 Hz inference (was 10 Hz to match Squint training).
        # Physics dt 1/300 (300 Hz), decimation 10 -> control rate 30 Hz.
        # Same number of physics substeps per control step (10) preserves
        # solver quality; wall-clock cost ~3x.
        #
        # Action delta bounds KEPT at the trained values [0.1, ..., 0.2]
        # — the per-step delta is unchanged, so the integrated target_qpos
        # advances 3x faster across one second of sim time. The robot will
        # physically move ~3x faster between training-time and 30 Hz.
        # ============================================================
        self.decimation = 10
        self.sim.dt = 1.0 / 300.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 5.0  # 150 control steps at 30 Hz

        # PhysX params — aligned with Squint/SAPIEN runtime so deploy
        # matches training-time physics (see notes/squint_friend_dumps/
        # physics_params.txt).
        # Squint:                  Ours (now):
        #   bounce_threshold = 2.0   2.0
        #   solver_pos_iter  = 15    15 (per-articulation, set below)
        #   solver_vel_iter  = 1     1 (per-articulation, set below)
        #   contact_offset   = 0.02  0.02
        #   rest_offset      = 0.0   0.0
        #   enable_pcm       = True  True
        self.sim.physx.bounce_threshold_velocity = 2.0
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        # SAPIEN-style contact offsets
        try:
            self.sim.physx.gpu_max_rigid_contact_count = 524288
            self.sim.physx.gpu_max_rigid_patch_count = 81920
        except Exception:
            pass


@configclass
class SquintNativePlaceEnvCfg_PLAY(SquintNativePlaceEnvCfg):
    """Single-env eval shape — for debug_full_audit and deploy_isaac."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5


@configclass
class SquintNativePlaceEnvCfg_REPLAY(SquintNativePlaceEnvCfg_PLAY):
    """Identical to PLAY but cube + bin spawn at the EXACT positions
    Squint observes at seed=0 (per the canonical audit). Used to test
    deploy on a known-good Squint trajectory baseline.

    Cube xy = (+0.205, -0.066), bin xy = (+0.340, +0.060). Yaws set to 0
    (Squint's actual yaws are random, but for one-shot canonical replay
    we want a deterministic baseline).
    """

    def __post_init__(self):
        super().__post_init__()
        # Override the reset_cube_and_bin event with fixed positions.
        from . import squint_events
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg
        self.events.reset_cube_and_bin = EventTerm(
            func=squint_events.reset_scene_squint,
            mode="reset",
            params={
                "cube_cfg": SceneEntityCfg("cube"),
                "fixed_cube_xy": (0.205, -0.066),
                "fixed_bin_xy": (0.340, +0.060),
                "fixed_cube_yaw": 0.0,
                "fixed_bin_yaw": 0.0,
            },
        )
