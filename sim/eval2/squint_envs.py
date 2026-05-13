"""Squint-port env configs for SO-101 Lift and Place on Isaac Lab.

Implements **Option B** from the planning discussion:
- LeIsaac scaffold (robot, table, cube asset) — kept as-is
- Squint reward shape (in ``mdp/squint_rewards.py``)
- Wrist camera (16x16 RGB after trainer-side downsample from 64x64 raw)
- Cube domain randomization (friction + mass)
- Spawn box geometry matching Squint
- Gripper stiffness DR
- Visual bin marker for Place (cuboid, no physical walls — see notes)

DELIBERATE SIMPLIFICATIONS vs Squint original env:
- **No physical bin walls in Place**. Squint builds 5 boxes per env at scene
  creation (1 floor + 4 walls). Isaac Lab makes per-env unique assets
  awkward. We render a single visual cuboid marker at the goal pose;
  the "is in bin" check is geometric (xy inside 5x5 cm footprint),
  which works for RL convergence (walls only matter for sim2real
  cube-bounce-off behavior). To add walls: spawn 4 additional
  RigidObjectCfg with ``kinematic_enabled=True``.
- **No per-env cube size variation**. Squint randomizes cube_half_size in
  [0.011, 0.014] m via per-env builders. Isaac Lab clones one asset
  across envs, so we keep cube size fixed at LeIsaac default
  (~3-4 cm) and randomize ONLY mass and friction. The policy still
  learns the contact dynamics through mass/friction noise.
- **No robot color DR / lighting DR**. Squint randomizes per-episode.
  Isaac Lab needs custom shader manipulation. Skipped for v1.
"""
import math

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs.mdp import UniformPoseCommandCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp

from . import mdp as eval2_mdp
from .leisaac_lift_env_cfg import LeIsaacLiftCubeRLEnvCfg


# ---------------------------------------------------------------------------
# Wrist RGB observation group — raw 64x64 RGB, downsampled by trainer to 16x16
# ---------------------------------------------------------------------------


def _wrist_rgb_raw(env, sensor_cfg: SceneEntityCfg) -> "torch.Tensor":
    """Read raw RGB tensor from a TiledCamera sensor — module-level
    function (Isaac Lab's ObsTerm doesn't accept lambdas)."""
    return env.scene[sensor_cfg.name].data.output["rgb"]


def randomize_dome_light(
    env,
    env_ids,
    light_prim_path: str = "/World/light",
    intensity_range: tuple = (1500.0, 3000.0),
    color_warm_cool_range: tuple = (-0.15, 0.15),
):
    """Randomize the global dome light intensity + warm/cool color tint.

    Squint's lighting DR randomizes the *ambient* lighting per episode.
    Isaac Lab doesn't ship a builtin event for lights, so we write our
    own using ``omni.usd`` to mutate the light prim attributes.

    NOTE: this is a GLOBAL randomization (one light shared across all
    parallel envs), not per-env. Per-env lighting would require
    ``num_envs`` light prims and a different scene structure. For RL
    convergence the global variation is enough — the policy still has
    to be robust to varying light conditions across episodes.

    Args:
        env: ManagerBasedRLEnv (passed by EventManager).
        env_ids: Tensor of env indices being reset (we only sample once
            per call regardless — global light).
        light_prim_path: USD prim path to the dome light.
        intensity_range: (low, high) for uniform intensity sample.
        color_warm_cool_range: (low, high) for a single warm↔cool tint.
            -0.15 = cool (more blue), 0.15 = warm (more orange).
    """
    import omni.usd
    from pxr import UsdLux

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    light_prim = stage.GetPrimAtPath(light_prim_path)
    if not light_prim.IsValid():
        return

    light_api = UsdLux.LightAPI(light_prim)
    intensity = float(
        intensity_range[0]
        + (intensity_range[1] - intensity_range[0]) * float(__import__("random").random())
    )
    tint = float(
        color_warm_cool_range[0]
        + (color_warm_cool_range[1] - color_warm_cool_range[0]) * float(__import__("random").random())
    )
    # Color: shift R/B by ±tint relative to neutral white (1,1,1).
    color = (1.0 + tint, 1.0, 1.0 - tint)

    light_api.GetIntensityAttr().Set(intensity)
    light_api.GetColorAttr().Set(color)


def make_robot_matte_black(env, env_ids,
                           robot_prim_path: str = "/World/envs/env_.*/Robot",
                           rgb: tuple = (0.02, 0.02, 0.02)) -> None:
    """Override every visible material under the robot prim to matte black.

    Matte = roughness=1.0, metallic=0.0, no specular reflection. With the
    default LeIsaac SO-101 USD materials, the robot looks shiny grey under
    Isaac RTX lighting. Squint training expects the robot pixel value to
    be uniformly dark (no highlights), which matters when the policy is
    image-conditioned and the wrist cam captures a lot of robot meshes.

    Runs ONCE at startup; modifies the USD material attributes via
    omni.usd. Walks all Shader prims under the robot prim path.
    """
    import omni.usd
    from pxr import UsdShade, Sdf

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    # Scene has many env_* clones; iterate all of them.
    import re
    pattern = re.compile(robot_prim_path.replace(".*", r"[\w_]+"))
    n_overridden = 0
    for prim in stage.Traverse():
        ppath = str(prim.GetPath())
        if not pattern.match(ppath):
            continue
        # Look for shaders under this robot prim.
        for sub in prim.GetAllChildren():
            for descendant in [sub] + [d for d in stage.Traverse() if str(d.GetPath()).startswith(str(sub.GetPath()))]:
                if descendant.GetTypeName() == "Shader":
                    shader = UsdShade.Shader(descendant)
                    # OmniPBR/MDL shader inputs.
                    for attr_name, value, value_type in [
                        ("inputs:diffuse_color_constant", rgb, Sdf.ValueTypeNames.Color3f),
                        ("inputs:metallic_constant", 0.0, Sdf.ValueTypeNames.Float),
                        ("inputs:reflection_roughness_constant", 1.0, Sdf.ValueTypeNames.Float),
                        ("inputs:specular_level", 0.0, Sdf.ValueTypeNames.Float),
                    ]:
                        try:
                            inp = shader.GetInput(attr_name.split(":")[-1])
                            if inp:
                                inp.Set(value)
                            else:
                                shader.CreateInput(attr_name.split(":")[-1], value_type).Set(value)
                            n_overridden += 1
                        except Exception:
                            pass
    if n_overridden > 0:
        print(f"[squint_envs] make_robot_matte_black: overrode {n_overridden} material attributes")


def reset_bin_coherent(env, env_ids, pose_range_xy: dict, asset_names: tuple) -> None:
    """Sample ONE (dx, dy) per env and apply it to every bin part together.

    Squint's bin is a single articulated actor; in our port the bin is 5
    independent rigid bodies (floor + 4 walls). To keep the bin rigid as
    a whole during randomization we sample a SHARED xy offset and write
    it to every part's root state.

    pose_range_xy keys: "x" and "y", each a 2-tuple (lo, hi).
    """
    if len(env_ids) == 0:
        return
    import torch  # local import keeps event-imports lazy
    n = len(env_ids)
    device = env.device
    lo_x, hi_x = pose_range_xy["x"]
    lo_y, hi_y = pose_range_xy["y"]
    dx = torch.empty(n, device=device).uniform_(lo_x, hi_x)
    dy = torch.empty(n, device=device).uniform_(lo_y, hi_y)

    for asset_name in asset_names:
        obj = env.scene[asset_name]
        default = obj.data.default_root_state[env_ids].clone()  # (n, 13)
        # Add the SHARED offset to x and y. z and quat unchanged.
        default[:, 0] += dx
        default[:, 1] += dy
        # Translate to world frame (default state is already in world but
        # uses env_origins offset internally; write_root_pose_to_sim
        # expects world-frame pos so we add env_origins).
        default[:, 0:3] += env.scene.env_origins[env_ids]
        obj.write_root_pose_to_sim(default[:, 0:7], env_ids)
        # Zero velocities just in case.
        zero_vel = torch.zeros((n, 6), device=device)
        obj.write_root_velocity_to_sim(zero_vel, env_ids)


@configclass
class WristRgbObsCfg(ObsGroup):
    """Single-term group exposing the wrist camera RGB output as ``obs["wrist"]``."""

    rgb = ObsTerm(
        func=_wrist_rgb_raw,
        params={"sensor_cfg": SceneEntityCfg("wrist")},
    )

    def __post_init__(self):
        # Camera RGB is uint8 (B, H, W, 3) — don't normalize, don't concat
        # (CNN encoder expects raw uint8).
        self.enable_corruption = False
        self.concatenate_terms = False


# ---------------------------------------------------------------------------
# Squint Lift — five-term reward, lift to a held height
# ---------------------------------------------------------------------------


@configclass
class SquintLiftRewardsCfg:
    """Five-term reward matching ``squint/envs/lift.py``:

        r = (1 - tanh(5 d_ee_cube))                  # reach
          + is_grasped                               # grasp bonus
          + exp(-2 d_qpos_rest) * is_grasped         # return-to-rest
          - 1 * (1 - is_lifted)                      # fast-lift incentive
          + small action regularizers
    """

    reach = RewTerm(
        func=eval2_mdp.squint_reach_dense,
        params={
            "sharpness": 5.0,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cube_cfg": SceneEntityCfg("cube"),
        },
        weight=1.0,
    )

    grasping = RewTerm(
        func=eval2_mdp.cube_grasped,
        params={
            "diff_threshold": 0.02,
            "grasp_threshold": 0.26,
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cube_cfg": SceneEntityCfg("cube"),
        },
        weight=1.0,
    )

    place_back = RewTerm(
        func=eval2_mdp.squint_place_back_bonus,
        params={
            "rest_qpos": (0.0, 0.0, 0.0, 0.0, 0.0),
            "decay": 2.0,
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cube_cfg": SceneEntityCfg("cube"),
        },
        weight=1.0,
    )

    not_lifted_pen = RewTerm(
        func=eval2_mdp.squint_not_lifted_penalty,
        params={
            "height_threshold": 0.05,
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": "base",
        },
        weight=-1.0,
    )

    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-1e-4,
    )


@configclass
class SquintLiftEventsCfg:
    """Squint-style domain randomization on top of the LeIsaac base events.

    The LeIsaac parent provides `reset_all` and `reset_cube_position`.
    We extend with:
    - cube friction + restitution randomization at scene creation (startup)
    - cube mass perturbation per reset
    - gripper actuator gain randomization per reset
    """

    # Cube spawn: ±10 cm around the cube's default position (set in
    # SquintLiftEnvCfg.__post_init__ to the Squint spawn-box center).
    # This matches Squint's spawn_box_half_size=0.1.
    reset_cube_position = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.10, 0.10),
                "y": (-0.10, 0.10),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )

    # Robot initial joint pos noise — Squint's `initial_qpos_noise_scale=0.02`.
    randomize_initial_qpos = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.02, 0.02),   # rad
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_cube_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "static_friction_range": (0.1, 0.5),
            "dynamic_friction_range": (0.1, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Robot to matte black at startup — kills the shiny grey look that
    # the LeIsaac default OmniPBR materials produce under Isaac RTX.
    robot_matte_black = EventTerm(
        func=make_robot_matte_black,
        mode="startup",
        params={
            "robot_prim_path": "/World/envs/env_.*/Robot",
            "rgb": (0.02, 0.02, 0.02),
        },
    )

    randomize_cube_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "mass_distribution_params": (0.8, 1.2),  # multiplicative
            "operation": "scale",
        },
    )

    randomize_gripper_stiffness = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
        },
    )

    randomize_lighting = EventTerm(
        func=randomize_dome_light,
        mode="reset",
        params={
            "light_prim_path": "/World/light",
            # Fixed intensity 1500 — DR disabled. The randomize_dome_light
            # uses Python's global RNG which is non-deterministic and
            # under some sample sequences produced all-black frames. Once
            # the RNG is migrated to a properly seeded source, re-enable
            # DR with range like (1500, 2500).
            "intensity_range": (1500.0, 1500.0),
            "color_warm_cool_range": (0.0, 0.0),
        },
    )


@configclass
class SquintLiftEnvCfg(LeIsaacLiftCubeRLEnvCfg):
    """SO-101 Lift task — Squint-style reward + DR + wrist camera.

    Inherits LeIsaac scaffold (scene, actions, commands, state obs)
    and adds:
    - Wrist TiledCamera at 64x64
    - WristRgbObsCfg observation group exposing ``obs["wrist"]``
    - Squint multi-term dense reward
    - Cube friction/mass DR + gripper stiffness DR
    - Spawn box matching Squint (-0.10/0.10 × -0.10/0.10)
    """

    rewards: SquintLiftRewardsCfg = SquintLiftRewardsCfg()
    events: SquintLiftEventsCfg = SquintLiftEventsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # ----------------------------------------------------------------
        # 0) ROBOT USD OVERRIDE — replace LeIsaac SO-101 USD with the USD
        #    converted from Squint's URDF (renamed so link names match
        #    LeIsaac's bare-name convention). This eliminates the link-
        #    transform / joint-axis differences that caused the visible
        #    π/2 gripper rotation.
        #
        #    Pipeline (run once after URDF changes):
        #      python -m sim.eval2.scripts.rename_squint_urdf
        #      python -m sim.eval2.scripts.convert_squint_urdf \
        #          --urdf <renamed.urdf> --out_name so101_squint_renamed.usd
        # ----------------------------------------------------------------
        from pathlib import Path
        SQUINT_USD = Path(
            r"C:/Users/user/Desktop/MA2/robot-learning-project3/squint/converted_usd/so101_squint_renamed.usd"
        )
        if SQUINT_USD.exists():
            self.scene.robot.spawn.usd_path = str(SQUINT_USD)
            print(f"[SquintLiftEnvCfg] robot USD overridden -> {SQUINT_USD.name}")
        else:
            print(f"[SquintLiftEnvCfg] WARNING: Squint USD not found at {SQUINT_USD}; "
                  f"falling back to LeIsaac default. Run rename_squint_urdf + convert_squint_urdf.")

        # ----------------------------------------------------------------
        # Align scene to Squint training conventions
        # (https://github.com/aalmuzairee/squint, envs/robot/so101.py +
        # envs/base_random_env.py). This makes Isaac scenes visually and
        # geometrically close to ManiSkill's so Squint-style policies are
        # transferable. Three alignments below: home pose, wrist cam offset,
        # cube spawn-in-view.
        # ----------------------------------------------------------------

        # 1) HOME POSE — Squint's "start" keyframe (cam up, gripper open).
        #    See squint/envs/robot/so101.py:55-60:
        #      qpos = [0, 0, 0, np.pi/2, -np.pi/2, 60°]
        #    Joint order: shoulder_pan, shoulder_lift, elbow_flex,
        #                 wrist_flex, wrist_roll, gripper.
        SQUINT_START_QPOS = {
            "shoulder_pan":  0.0,
            "shoulder_lift": 0.0,
            "elbow_flex":    0.0,
            "wrist_flex":    math.pi / 2.0,
            "wrist_roll":   -math.pi / 2.0,
            "gripper":       60.0 * math.pi / 180.0,
        }
        # Override the SO-101 articulation's default init_state.joint_pos.
        # We REPLACE the dict wholesale because LeIsaac's parent may set
        # a regex entry like `{".*": 0.0}` which would otherwise compete
        # with per-joint entries depending on Isaac Lab's match order.
        self.scene.robot.init_state.joint_pos = dict(SQUINT_START_QPOS)

        # 2) WRIST CAMERA OFFSET — Squint's exact mount pose + FOV.
        #    See squint/envs/base_random_env.py:497-499:
        #      WRIST_CAMERA_BASE_POS = (-0.0049, 0.0498, -0.0591)
        #      WRIST_CAMERA_BASE_ROT_RAD = (-90°, 91°, -35.31°)  (roll, pitch, yaw)
        #      WRIST_CAMERA_FOV = 71° (horizontal)
        #
        #    Squint's euler→quat formula (base_random_env.py:562-571) is
        #    q = q_pitch * q_yaw * q_roll  (hamilton, scalar-first). We
        #    bake the resulting (w, x, y, z) inline.
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import TiledCameraCfg

        squint_wrist_pos = (-0.0049, 0.0498, -0.0591)

        roll, pitch, yaw = math.radians(-90.0), math.radians(91.0), math.radians(-35.31)
        ci, si = math.cos(roll / 2), math.sin(roll / 2)
        cj, sj = math.cos(pitch / 2), math.sin(pitch / 2)
        ck, sk = math.cos(yaw / 2), math.sin(yaw / 2)
        # Squint's exact quat (SAPIEN convention).
        q_py_w, q_py_x, q_py_y, q_py_z = cj * ck, sj * sk, sj * ck, cj * sk
        sw = q_py_w * ci - q_py_x * si
        sx = q_py_w * si + q_py_x * ci
        sy = q_py_y * ci + q_py_z * si
        sz = q_py_z * ci - q_py_y * si

        # Audit observation: with `convention="world"` (forward = local
        # -Z), Squint's quat puts local +Y straight down in world. To
        # reuse this DOWN direction as the cam's forward, compose with a
        # +90° rotation around local X — this maps the new local -Z onto
        # the OLD local +Y direction.
        # q_R = q(+90° around X) = (cos(45°), sin(45°), 0, 0)
        rw, rx, ry, rz = math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0
        qw = sw * rw - sx * rx - sy * ry - sz * rz
        qx = sw * rx + sx * rw + sy * rz - sz * ry
        qy = sw * ry - sx * rz + sy * rw + sz * rx
        qz = sw * rz + sx * ry - sy * rx + sz * rw
        squint_wrist_quat = (qw, qx, qy, qz)

        # Horizontal FOV 71° → focal_length for the Isaac PinholeCameraCfg.
        # FOV = 2 * atan(H_aperture / (2 * f))  →  f = H_aperture / (2*tan(FOV/2))
        h_aperture = 24.0  # mm, Isaac/USD default
        focal_length = h_aperture / (2.0 * math.tan(math.radians(71.0) / 2.0))

        # NOTE on convention: Squint/SAPIEN uses OpenGL convention for
        # cameras (optical axis = local -Z). Use Isaac Lab `convention=
        # "opengl"` to apply the Squint quat with the SAME semantics —
        # the optical axis world direction will be quat * (0,0,-1)
        # exactly as in Squint training.
        self.scene.wrist = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=squint_wrist_pos,
                rot=squint_wrist_quat,
                convention="world",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=400.0,
                horizontal_aperture=h_aperture,
                clipping_range=(0.01, 50.0),
                lock_camera=True,
            ),
            width=64,
            height=64,
            update_period=1 / 30.0,
        )

        # Add the rgb obs group — accessible from the trainer via
        # ``obs["wrist"]``. The state-based ``policy`` group (set by the
        # parent) stays unchanged.
        self.observations.wrist = WristRgbObsCfg()

        # 3) SPAWN BOX — match Squint exactly.
        #
        #    Squint's `spawn_box_pos=(0.3, 0)`, `spawn_box_half_size=0.1`:
        #    a 20×20 cm box centered 30 cm in front of the robot, where
        #    both cube AND bin are uniformly sampled per episode (with
        #    a non-overlap constraint that we approximate here by placing
        #    them in disjoint sub-ranges).
        #
        #    Isaac robot at world (0.35, -0.64), facing +Y world (audit:
        #    arm extends in +y from base). So Squint robot-frame (+0.3,
        #    0) maps to world (0.35, -0.34).
        # User constraint: the cube MUST be in the wrist cam view at the
        # very first frame (home pose). Audit shows the cam optical axis
        # at home pose hits the table at world (0.28, -0.30). Anchor the
        # cube there with tight ±2 cm spawn so it always lands inside the
        # camera frustum.
        spawn_center_world = (0.28, -0.30)
        try:
            self.scene.cube.init_state.pos = (spawn_center_world[0],
                                              spawn_center_world[1],
                                              0.051)
        except (AttributeError, KeyError, TypeError):
            pass
        self.events.reset_cube_position.params["pose_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (0.0, 0.0),
        }

        # ----------------------------------------------------------------
        # 4) ACTUATOR GAINS — match ManiSkill SO-101 PD controller.
        # LeIsaac default actuator gains (stiffness=17.8, damping=0.60)
        # are MASSIVELY weaker than ManiSkill's (stiffness=1000, damping=
        # 100). Without this override, the joints lag the policy's
        # commanded targets by huge amounts → the policy issues correct
        # actions but joints never reach the requested positions.
        try:
            for actuator_name, actuator in self.scene.robot.actuators.items():
                actuator.stiffness = 1000.0
                actuator.damping = 100.0
                actuator.effort_limit_sim = 100.0
                actuator.velocity_limit_sim = 100.0
        except (AttributeError, KeyError, TypeError):
            pass
        # IMPORTANT: LeIsaac's parent adds `domain_randomize_0` which
        # re-randomizes the cube around the *LeIsaac* default pose,
        # silently overriding our pose_range above. Disable it so our
        # `reset_cube_position` is the only cube spawn event.
        if hasattr(self.events, "domain_randomize_0"):
            self.events.domain_randomize_0 = None

        # Squint Lift: goal is to lift cube to a held height.
        self.commands.object_pose.ranges = UniformPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.05),
            pos_y=(-0.20, -0.10),
            pos_z=(0.15, 0.20),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        )

        # Camera + DR pushes VRAM. Cap at 256 envs on RTX 5070 (12 GB).
        self.scene.num_envs = 256


@configclass
class SquintLiftEnvCfg_PLAY(SquintLiftEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Squint Place — visual bin marker, place ON the table at the goal location
# ---------------------------------------------------------------------------


@configclass
class SquintPlaceRewardsCfg:
    """Eight-term reward matching ``squint/envs/place.py``:

        r = 2 * (1 - tanh(5 d_ee_cube))                # weight 2.0 reach
          + 3 * is_grasped                             # weight 3.0 grasp
          + (1 - tanh(5 d_cube_goal))                  # weight 1.0 final dense
          + place_z_staged                             # weight 1.0 (hover/descent)
          + 4 * is_above_bin                           # weight 4.0 stage
          + 9 * success                                # weight 9.0 terminal
          - 1 * (1 - is_lifted)                        # weight -1.0
          + small regularizers
    """

    reach = RewTerm(
        func=eval2_mdp.squint_reach_dense,
        params={
            "sharpness": 5.0,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cube_cfg": SceneEntityCfg("cube"),
        },
        weight=2.0,
    )

    grasping = RewTerm(
        func=eval2_mdp.cube_grasped,
        params={
            "diff_threshold": 0.02,
            "grasp_threshold": 0.26,
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cube_cfg": SceneEntityCfg("cube"),
        },
        weight=3.0,
    )

    place_final = RewTerm(
        func=eval2_mdp.squint_place_final_dense,
        params={
            "sharpness": 5.0,
            "command_name": "object_pose",
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
        weight=1.0,
    )

    place_z = RewTerm(
        func=eval2_mdp.squint_place_z_staged,
        params={
            # bin_xy_radius = max(half_x, half_y) of physical bin (5 cm).
            "bin_xy_radius": 0.05,
            "hover_offset": 0.06,
            "sharpness": 10.0,
            "command_name": "object_pose",
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
        weight=1.0,
    )

    above_bin = RewTerm(
        func=eval2_mdp.squint_is_above_bin,
        params={
            # Match physical bin footprint (8 × 10 cm) — Squint mid-range.
            "bin_half_x": 0.04,
            "bin_half_y": 0.05,
            "command_name": "object_pose",
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
        weight=4.0,
    )

    success_bonus = RewTerm(
        func=eval2_mdp.squint_place_success,
        params={
            "bin_half_x": 0.04,
            "bin_half_y": 0.05,
            "z_tolerance": 0.02,
            "static_qvel_threshold": 0.1,
            "command_name": "object_pose",
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
        weight=9.0,
    )

    not_lifted_pen = RewTerm(
        func=eval2_mdp.squint_not_lifted_penalty,
        params={
            "height_threshold": 0.05,
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": "base",
        },
        weight=-1.0,
    )

    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-1e-4,
    )


@configclass
class SquintPlaceEnvCfg(SquintLiftEnvCfg):
    """SO-101 Place task — Squint-style place reward + visual bin marker.

    Same scene/cam/DR as SquintLiftEnvCfg, plus:
    - Goal range adjusted: place ON the table (not above)
    - Visual cuboid marker at goal pose (white-ish, kinematic, no
      collision) so the policy + viewer see the target location
    - Place-specific reward composition (8 terms)
    - Episode 7 s (vs 5 s for Lift) — Place needs more time
    """

    rewards: SquintPlaceRewardsCfg = SquintPlaceRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Goal pose tracks the FIXED bin position at world (0.34, -0.30).
        # With robot at world (0.35, -0.64), 180° z-rot:
        #   x_robot = -(0.34 - 0.35) = +0.01
        #   y_robot = -(-0.30 - (-0.64)) = -0.34
        #   z_robot ≈ 0.044 (cube settled on bin floor)
        self.commands.object_pose.ranges = UniformPoseCommandCfg.Ranges(
            pos_x=(0.01, 0.01),
            pos_y=(-0.34, -0.34),
            pos_z=(0.04, 0.05),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        )

        # Real bin = floor + 4 walls, all kinematic. Geometry matches
        # Squint's mid-range (envs/place.py:33-35) — 8×10×3 cm with 5 mm
        # walls. Pure white visual material matches Squint `bin_color =
        # [1,1,1,1]`.
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg

        # Bin position — placed close to the cube spawn area but not
        # overlapping. With cube at world (0.28, -0.30) ± 2 cm and bin
        # 8 × 10 cm at world (0.34, -0.30), the cube's max x = 0.30 +
        # 0.015 = 0.315 vs bin's min x = 0.30 → 1.5 cm gap, no overlap.
        # Both fit in the cam frustum at home pose (~32 cm visible at
        # 22 cm height, FOV 71°).
        bin_x, bin_y = 0.34, -0.30
        table_top_z = 0.036
        bin_half_x = 0.04        # 8 cm wide  (Squint mid-range)
        bin_half_y = 0.05        # 10 cm long (Squint mid-range)
        wall_thick = 0.005       # Squint thickness ✓
        wall_height = 0.030      # 3 cm tall walls (Squint mid-range)
        floor_center_z = table_top_z + 0.0025
        wall_center_z = table_top_z + 0.005 + wall_height / 2.0

        # Pure white visual material (matches Squint bin_color [1,1,1,1]).
        bin_white = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))

        # Bin floor.
        self.scene.bin_floor = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/bin_floor",
            spawn=sim_utils.CuboidCfg(
                size=(bin_half_x * 2, bin_half_y * 2, 0.005),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=bin_white,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(bin_x, bin_y, floor_center_z)),
        )

        # 4 walls — left/right (along Y), front/back (along X). All white.
        wall_color = bin_white
        wall_rigid = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        wall_collision = sim_utils.CollisionPropertiesCfg()

        self.scene.bin_wall_left = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/bin_wall_left",
            spawn=sim_utils.CuboidCfg(
                size=(bin_half_x * 2 + wall_thick * 2, wall_thick, wall_height),
                rigid_props=wall_rigid,
                collision_props=wall_collision,
                visual_material=wall_color,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(bin_x, bin_y - bin_half_y - wall_thick / 2, wall_center_z),
            ),
        )
        self.scene.bin_wall_right = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/bin_wall_right",
            spawn=sim_utils.CuboidCfg(
                size=(bin_half_x * 2 + wall_thick * 2, wall_thick, wall_height),
                rigid_props=wall_rigid,
                collision_props=wall_collision,
                visual_material=wall_color,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(bin_x, bin_y + bin_half_y + wall_thick / 2, wall_center_z),
            ),
        )
        self.scene.bin_wall_back = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/bin_wall_back",
            spawn=sim_utils.CuboidCfg(
                size=(wall_thick, bin_half_y * 2, wall_height),
                rigid_props=wall_rigid,
                collision_props=wall_collision,
                visual_material=wall_color,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(bin_x - bin_half_x - wall_thick / 2, bin_y, wall_center_z),
            ),
        )
        self.scene.bin_wall_front = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/bin_wall_front",
            spawn=sim_utils.CuboidCfg(
                size=(wall_thick, bin_half_y * 2, wall_height),
                rigid_props=wall_rigid,
                collision_props=wall_collision,
                visual_material=wall_color,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(bin_x + bin_half_x + wall_thick / 2, bin_y, wall_center_z),
            ),
        )

        # Bin position FIXED at world (bin_x, bin_y). Randomization is
        # disabled here because the user wants cube + bin both visible
        # from frame 0; with random bin spawn the bin can drift out of
        # the cam frustum.
        # (To re-enable: uncomment below and set asset_names + pose_range.)

        # Place needs more time than Lift.
        self.episode_length_s = 7.0


@configclass
class SquintPlaceEnvCfg_PLAY(SquintPlaceEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
