"""ManagerBasedRLEnvCfg for the Squint-native Place task.

Wires together:
- Scene  (squint_scene.SquintNativeSceneCfg)
- Action (delta-target joint position from squint_actions)
- Observations (state vector matching Squint, from squint_observations)
- Events (reset robot to home + spawn cube/bowl from squint_events)
- No reward / no termination (deploy mode — we only want to evaluate the
  ManiSkill checkpoint, not retrain). A timeout terminates each episode.

Sim timing matches Squint exactly:
- sim_freq    = 100  Hz (sim.dt = 0.01)
- control_freq = 10  Hz (decimation = 10)
- max_episode_steps = 50 (= 5 s)
"""
from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import events as mdp_events
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import squint_events, squint_observations, squint_rewards, squint_terminations
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
        # Match Squint envs/robot/so101.py:99-100 — arm caps were ±0.1 in an
        # older revision but the trained ckpt was built with ±0.05.
        bounds=[0.05, 0.05, 0.05, 0.05, 0.05, 0.2],
    )


@configclass
class SquintObservationsCfg:
    """Mirrors EXACTLY what Squint's policy reads after FlattenRGBDObservationWrapper:

    - obs['state'] = [qpos(6), target_qpos(6)]   (12 floats)
    - obs['rgb']   = wrist_rgb downsampled to 16x16 (uint8)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """21-d state vector for the Eval-2 training port (goal-conditioned
        + bowl position privileged input):
            [qpos(6), target_qpos(6), goal_color_one_hot(6), bowl_xyz(3)].

        Bowl xyz is appended because the wrist cam FOV at home pose covers
        the cube spawn zone but not necessarily the bowl spawn zone — we
        give the policy the bowl pose directly so it can plan the place
        phase without first "looking around" to find the bowl.

        Older Squint ckpts (18-d state) warmstart via zero-pad on the
        state_proj input columns for indices 18..20 (see train script).
        """

        # Squint PlaceRandomizationConfig.robot_qpos_noise_std = deg2rad(5)
        # ≈ 0.0873 rad. Set to 0 for deterministic deploy/eval.
        qpos = ObsTerm(
            func=squint_observations.joint_pos_with_noise,
            params={"noise_std": 0.0873},
        )
        target_qpos = ObsTerm(func=squint_observations.controller_target_qpos)
        goal_color_one_hot = ObsTerm(func=squint_observations.goal_color_one_hot)
        bowl_xyz = ObsTerm(func=squint_observations.bowl_xyz_world)

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
            params={"greenscreen_bg_rgb": None, "apply_jitter": True},
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

    align_squint_materials = EventTerm(
        func=squint_events.align_squint_materials,
        mode="startup",
    )

    zero_finger_tips = EventTerm(
        func=squint_events.zero_finger_tip_masses,
        mode="startup",
    )

    fix_link_coms = EventTerm(
        func=squint_events.fix_link_coms,
        mode="startup",
    )

    # NOTE: ``make_scene_emissive`` removed — Squint training did NOT use
    # an emissive-only scene. It used ambient ∈ [0.2, 0.5] + 2 directional
    # lights producing fully shaded views. Stacking emission on top of our
    # dome+distant lighting double-illuminated objects and pushed the
    # wrist cam obs out of training distribution.

    randomize_colors = EventTerm(
        func=squint_events.reset_goal_and_distractor_colors,
        mode="reset",
    )

    reset_robot = EventTerm(
        func=squint_events.reset_robot_to_home,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "qpos_noise_std": 0.02,
        },
    )
    reset_cube_and_bowl = EventTerm(
        func=squint_events.reset_scene_squint,
        mode="reset",
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "bowl_cfg": SceneEntityCfg("bowl"),
        },
    )

    # ---- Domain randomization (Squint PlaceRandomizationConfig) ----
    # Cube friction: Squint envs/place.py:143-152 → uniform [0.4, 0.6].
    randomize_cube_material = EventTerm(
        func=mdp_events.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,  # dyn ≤ stat
        },
    )
    randomize_distractor_material = EventTerm(
        func=mdp_events.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_distractor"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    # Table friction DR skipped: Squint pins the table friction at 0.5 (fixed
    # range (0.5, 0.5), no DR — envs/place.py). Our scene already sets the
    # table to 0.5 statically; no event needed.
    # Cube mass per env: Squint item_mass_range = (0.003, 0.006) kg sampled
    # directly (mid 4.5 g). NOT a density DR — values are absolute kg.
    randomize_cube_mass = EventTerm(
        func=mdp_events.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "mass_distribution_params": (0.003, 0.006),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    randomize_distractor_mass = EventTerm(
        func=mdp_events.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_distractor"),
            "mass_distribution_params": (0.003, 0.006),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )


@configclass
class SquintTerminationsCfg:
    """Episode ends ONLY on time-out — matches Squint ``partial_reset=False``
    (``ManiSkillVectorEnv(ignore_terminations=True)``).

    Squint's ``success`` flag is still computed in ``squint_terminations.success``
    and read directly by the training loop / reward function — but it does NOT
    end the episode. The policy must keep being immobile post-success to
    accumulate ``static_robot_reward = 1 - tanh(10 · qvel)`` until the timeout.
    """

    time_out = DoneTerm(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True)


@configclass
class SquintRewardsCfg:
    """Squint's dense reward (normalized to roughly [-1, +1] for C51 stability).

    Bit-for-bit port of ``envs/place.py:compute_dense_reward`` / 9. Returned
    as a single Term with weight=1 so the state-machine logic stays inside
    the function — splitting into weighted terms wouldn't preserve the
    if-grasped / if-above-bowl conditional structure.
    """

    dense = RewTerm(
        func=squint_rewards.squint_dense_reward_normalized,
        weight=1.0,
    )


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
    rewards: SquintRewardsCfg = SquintRewardsCfg()
    # No commands, no curriculum.
    commands: object = None
    curriculum: object = None

    def __post_init__(self):
        # ============================================================
        # 10 Hz control — matches Squint training-time SimConfig.
        #   sim.dt       = 1/100   (sim_freq = 100 Hz)
        #   decimation   = 10      (control_freq = 10 Hz)
        #   episode      = 50 control steps = 5.0 s
        # ============================================================
        # 10 Hz control — matches Squint training (`pd_joint_target_delta_pos`
        # at control_freq=10 Hz, sim_freq=100 Hz). Required for warmstart from
        # ckpt 8 which was trained at this rate.
        # ⚠️ For deploy / GUI viewing, override render_interval = 1 in the
        # launching script if you want smooth visual playback (200 fps vs 10 fps
        # viewport). For training, keep render_interval = decimation = 10 so
        # multi-env throughput isn't crippled by per-substep rendering.
        self.sim.dt = 1.0 / 100.0
        self.decimation = 10
        self.sim.render_interval = self.decimation
        # P5: bumped from 7.5s to 10s. Squint trained on 75 control steps
        # in SAPIEN, but their policy never had to handle our slightly
        # different placement dynamics. 100 control steps gives the policy
        # extra slack for: approach → grasp → lift → transport → drop →
        # retreat → settle (qvel→0). With 7.5s the static-settle phase
        # often ran out of time before the cube came to rest.
        self.episode_length_s = 10.0  # 100 control steps at 10 Hz

        # Disable DLSS — at 128x128 the RTX renderer falls back to a 74x74
        # internal render then DLSS-upscales (see the
        # "Render resolution of (74, 74) is below minimal input resolution
        # of 300" warning in our audit logs). Force DLSS off + native
        # resolution rendering so the policy obs matches what Squint
        # produces in SAPIEN raw.
        try:
            import carb
            settings = carb.settings.get_settings()
            settings.set("/rtx/post/dlss/execMode", 0)            # 0 = OFF
            settings.set("/rtx/post/aa/op", 0)                    # disable AA mode 0
            settings.set("/rtx/sceneDb/ambientLightIntensity", 1.0)
        except Exception:
            pass

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
        # Match Squint's per-shape contact_offset of 0.02 m, also expose
        # PCM/TGS/friction-every-iter flags the Squint Claude flagged.
        self.sim.physx.bounce_threshold_velocity = 2.0
        try:
            self.sim.physx.sleep_threshold = 0.005  # SAPIEN default
        except Exception:
            pass
        # gpu_total_aggregate_pairs_capacity scales with num_envs: at 2048
        # parallel envs PhysX needs ~22528 pairs. 64K gives headroom up to
        # ~6000 envs without hitting the broad-phase ceiling.
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        # Match Squint / ManiSkill SceneConfig default — Isaac default is
        # 0.00625 (4× smaller), which makes grasps less stable.
        self.sim.physx.friction_correlation_distance = 0.025
        # Bumped to match the Squint-side fix for the contact buffer overflow
        # observed at 2048 parallel envs with the original 60-hull bowl CoACD.
        # Squint's SimConfig now sets:
        #   max_rigid_contact_count   = 2**20 (default 2**19)
        #   max_rigid_patch_count     = 2**19 (default 2**18)
        #   found_lost_pairs_capacity = 2**26 (default 2**25)
        try:
            self.sim.physx.gpu_max_rigid_contact_count = 2**20      # 1,048,576
            self.sim.physx.gpu_max_rigid_patch_count = 2**19         # 524,288
            self.sim.physx.gpu_found_lost_pairs_capacity = 2**26     # 67,108,864
            self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
        except Exception:
            pass
        # Per Squint runtime: PCM ON, TGS ON, friction-every-iter ON, CCD OFF.
        try:
            self.sim.physx.enable_stabilization = True
        except Exception:
            pass
        try:
            self.sim.physx.enable_pcm = True
        except Exception:
            pass
        try:
            self.sim.physx.enable_tgs = True
        except Exception:
            pass
        try:
            self.sim.physx.enable_friction_every_iteration = True
        except Exception:
            pass
        try:
            self.sim.physx.enable_ccd = False
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
    """Identical to PLAY but cube + bowl spawn at the EXACT positions
    Squint observes at seed=0 (per the canonical audit). Used to test
    deploy on a known-good Squint trajectory baseline.

    Cube xy = (+0.205, -0.066), bowl xy = (+0.340, +0.060). Yaws set to 0
    (Squint's actual yaws are random, but for one-shot canonical replay
    we want a deterministic baseline).
    """

    def __post_init__(self):
        super().__post_init__()
        # Override the reset_cube_and_bowl event with fixed positions.
        from . import squint_events
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg
        self.events.reset_cube_and_bowl = EventTerm(
            func=squint_events.reset_scene_squint,
            mode="reset",
            params={
                "cube_cfg": SceneEntityCfg("cube"),
                "bowl_cfg": SceneEntityCfg("bowl"),
                "fixed_cube_xy": (0.205, -0.066),
                "fixed_bowl_xy": (0.340, +0.060),
                "fixed_cube_yaw": 0.0,
                "fixed_bowl_yaw": 0.0,
            },
        )
