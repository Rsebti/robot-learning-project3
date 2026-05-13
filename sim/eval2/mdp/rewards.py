"""Reward terms for our LeIsaac-based RL tasks.

This module **composes existing implementations** rather than rewriting
them. The lineage:

1. **Isaac Lab Lift canonical** (``isaaclab_tasks...manipulation.lift.mdp``)
   provides the dense reaching reward (`object_ee_distance`,
   tanh-shaped). Reused as-is in ``RewardsCfg.reaching_object`` —
   imported there directly, no wrapper needed here.

2. **LeIsaac** (``leisaac.tasks.lift_cube.mdp``) provides:
   - ``terminations.cube_height_above_base`` : ``(cube.z_w - robot_base.z_w) > threshold``,
     measured relative to the robot base instead of absolute world z.
     This is the right semantic on the LeIsaac scene where the table is
     elevated several cm above world z = 0 — the canonical Isaac Lab
     ``object_is_lifted`` (which compares to absolute z) fires
     trivially on cube spawn.
   - ``observations.object_grasped`` : ``bool`` indicating that the
     cube is within 2 cm of the EE frame's "jaw" target AND the gripper
     joint is closed enough (`< 0.26 rad`).

   We expose both as float-valued reward terms via thin wrappers below.

3. **Our combined gated tracking reward** is the only piece we have to
   write from scratch: the canonical
   ``object_goal_distance(env, std, minimal_height, ...)`` does
   ``(z_w > minimal_height) * (1 - tanh(distance / std))``, but with
   absolute z. We rewrite it using LeIsaac's relative-to-base check.

Resulting reward functions exposed here:

- ``cube_lifted_above_base``       — wraps ``cube_height_above_base``
- ``cube_grasped``                 — wraps ``object_grasped``
- ``cube_to_goal_distance_above_base`` — gated dense tracking
"""
from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

# Reused from LeIsaac. Both functions return bool tensors of shape
# (num_envs,); we cast to float for use as PPO reward terms.
from leisaac.tasks.lift_cube.mdp.observations import object_grasped
from leisaac.tasks.lift_cube.mdp.terminations import cube_height_above_base

# Reused from our own terminations module — paired with the matching
# DoneTerms (cube_at_goal pairs with cube_reached_goal for the success
# bonus; cube_dropped_float pairs with cube_dropped for the V2.9 penalty).
from .terminations import cube_dropped, cube_reached_goal


# ---------------------------------------------------------------------------
# Lifted check — wraps LeIsaac's existing termination function as a reward.
# ---------------------------------------------------------------------------


def cube_lifted_above_base(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.05,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    robot_base_name: str = "base",
) -> torch.Tensor:
    """Binary lift reward (1.0 if the cube is lifted, 0.0 otherwise).

    Delegates the geometry check to LeIsaac's ``cube_height_above_base``,
    so we always agree with the success termination on what "lifted"
    means.

    Args:
        height_threshold: meters above the robot base to count as lifted.
            5 cm by default — generous enough that the policy gets the
            reward as soon as it has picked the cube off the table, even
            without lifting it all the way to the goal pose.
    """
    return cube_height_above_base(
        env,
        cube_cfg=cube_cfg,
        robot_cfg=robot_cfg,
        robot_base_name=robot_base_name,
        height_threshold=height_threshold,
    ).float()


# ---------------------------------------------------------------------------
# Grasp check — wraps LeIsaac's existing observation function as a reward.
# ---------------------------------------------------------------------------


def cube_grasped(
    env: ManagerBasedRLEnv,
    diff_threshold: float = 0.02,
    grasp_threshold: float = 0.26,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Binary grasp reward (1.0 if the gripper is closed on the cube).

    Wraps LeIsaac's ``object_grasped``: requires both that the cube is
    within ``diff_threshold`` meters of the EE frame's "jaw" target
    (target index 1 in the FrameTransformer) AND that the gripper joint
    angle is below ``grasp_threshold`` radians (i.e. fingers closed).

    Useful as an **intermediate reward signal** between the dense
    reaching reward and the lifting reward — without it, the policy has
    to make a discrete leap from "near cube, fingers open" to "above
    base 5 cm" with nothing in between.

    Default thresholds match LeIsaac's own defaults; tune
    ``diff_threshold`` larger if your gripper has a wider effective
    grasping zone.
    """
    return object_grasped(
        env,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=cube_cfg,
        diff_threshold=diff_threshold,
        grasp_threshold=grasp_threshold,
    ).float()


# ---------------------------------------------------------------------------
# Goal tracking — gated on lifted, tanh-shaped distance to commanded pose.
# ---------------------------------------------------------------------------


def cube_to_goal_distance_above_base(
    env: ManagerBasedRLEnv,
    std: float,
    height_threshold: float = 0.05,
    command_name: str = "object_pose",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    robot_base_name: str = "base",
) -> torch.Tensor:
    """Gated dense reward that tracks the cube toward the commanded goal.

    Functionally equivalent to Isaac Lab's
    ``object_goal_distance(env, std, minimal_height, command_name, ...)``
    but with the gating check delegated to LeIsaac's
    ``cube_height_above_base`` (relative to the robot base, not absolute
    world z). Without this fix the gate fires from frame zero on the
    LeIsaac elevated table and the dense tracking reward leaks into
    every random rollout.

    The standard pattern in pick-and-place reward shaping is to have
    TWO copies of this term with different ``std`` values:

    - ``std=0.3`` (coarse): reward starts climbing far away, guides the
      policy toward the goal hemisphere.
    - ``std=0.05`` (fine):  reward only when very close to the goal,
      pushes the policy to be precise about the final placement.

    Returns:
        Tensor of shape ``(num_envs,)``. Zero on environments where the
        cube is not yet lifted; otherwise ``1 - tanh(d / std)`` where
        ``d`` is the world-frame distance between cube and goal pose.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Goal pose is published in the robot root frame: shape (B, 7) — xyz + quat.
    command = env.command_manager.get_command(command_name)
    goal_pos_b = command[:, :3]

    # Project the goal into world frame so we can diff against cube.root_pos_w.
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_b
    )
    distance = torch.norm(cube.data.root_pos_w - goal_pos_w, dim=-1)

    # Reuse LeIsaac's relative-to-base height check.
    is_lifted = cube_height_above_base(
        env,
        cube_cfg=cube_cfg,
        robot_cfg=robot_cfg,
        robot_base_name=robot_base_name,
        height_threshold=height_threshold,
    ).float()
    return is_lifted * (1.0 - torch.tanh(distance / std))


# ---------------------------------------------------------------------------
# V2.7 — flick-exploit fixes: gate lift and tracking on grasp + lift.
# ---------------------------------------------------------------------------


def cube_lifted_and_grasped(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.05,
    diff_threshold: float = 0.04,
    grasp_threshold: float = 0.35,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_base_name: str = "base",
) -> torch.Tensor:
    """Binary reward: 1.0 only if the cube is BOTH grasped AND lifted.

    Closes the "flick exploit" observed in V2/V2.5/V2.6 runs: the policy
    learned to scoop the cube above the base threshold without actually
    grasping it (e.g. a fast sweep that briefly tosses the cube into the
    air). The original ``cube_lifted_above_base`` term fires on those
    ballistic micro-trajectories because it only checks the cube's height
    relative to the robot base.

    By AND-ing the lift check with ``object_grasped``, the policy can
    only collect the lift reward by actually closing the gripper around
    the cube. Default thresholds match the V2.7 RewardsCfg:

    - ``diff_threshold=0.04`` (4 cm, vs LeIsaac default 2 cm) — gives the
      gripper margin to close around a slightly off-axis cube without
      losing the grasp during the binary close transition.
    - ``grasp_threshold=0.35`` rad (vs LeIsaac default 0.26) — captures
      the closing transient too, not just the fully-closed final state.
    """
    is_lifted = cube_height_above_base(
        env,
        cube_cfg=cube_cfg,
        robot_cfg=robot_cfg,
        robot_base_name=robot_base_name,
        height_threshold=height_threshold,
    )
    is_grasped = object_grasped(
        env,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=cube_cfg,
        diff_threshold=diff_threshold,
        grasp_threshold=grasp_threshold,
    )
    return (is_lifted & is_grasped).float()


def cube_to_goal_distance_grasped_and_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    height_threshold: float = 0.05,
    diff_threshold: float = 0.04,
    grasp_threshold: float = 0.35,
    command_name: str = "object_pose",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_base_name: str = "base",
) -> torch.Tensor:
    """Like ``cube_to_goal_distance_above_base`` but also gated on grasp.

    Without the grasp gate, a cube in ballistic trajectory toward the
    goal (e.g. from the same flick exploit) keeps paying the dense
    tracking reward as it flies past. With the grasp+lift gate, the
    only path to this reward is to physically carry the cube — which is
    the actual task.

    Same coarse/fine pattern as the original (call once with
    ``std=0.30`` and once with ``std=0.05`` in RewardsCfgV27).
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    command = env.command_manager.get_command(command_name)
    goal_pos_b = command[:, :3]
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_b
    )
    distance = torch.norm(cube.data.root_pos_w - goal_pos_w, dim=-1)

    is_lifted = cube_height_above_base(
        env,
        cube_cfg=cube_cfg,
        robot_cfg=robot_cfg,
        robot_base_name=robot_base_name,
        height_threshold=height_threshold,
    )
    is_grasped = object_grasped(
        env,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=cube_cfg,
        diff_threshold=diff_threshold,
        grasp_threshold=grasp_threshold,
    )
    gate = (is_lifted & is_grasped).float()
    return gate * (1.0 - torch.tanh(distance / std))


# ---------------------------------------------------------------------------
# V2.7 — sparse success bonus to compensate for early-termination penalty.
# ---------------------------------------------------------------------------


def cube_at_goal(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.05,
    command_name: str = "object_pose",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Float-cast version of ``cube_reached_goal`` for use as a RewTerm.

    Same predicate as the success termination — paired with it, this fires
    once on the step where the episode terminates with success.

    Why this term is needed: ``cube_reached_goal`` is wired as a DoneTerm,
    so a successful episode ends immediately with no terminal reward. The
    policy then loses the discounted future dense reward it would otherwise
    collect by *hovering* near the goal — with γ=0.99, ~30 pt/step dense
    reward, and ~100 remaining steps, that's ~1900 pts of foregone reward
    per success. Without a compensating bonus, PPO is mathematically
    incentivized to hover at the edge of the success sphere rather than
    reach it.

    Pair this with the same ``distance_threshold`` as the matching DoneTerm
    so reward and termination agree on what "at goal" means.
    """
    return cube_reached_goal(
        env,
        distance_threshold=distance_threshold,
        command_name=command_name,
        cube_cfg=cube_cfg,
        robot_cfg=robot_cfg,
    ).float()


# ---------------------------------------------------------------------------
# V2.9 — cube_dropped penalty as RewTerm (paired with the DoneTerm).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# V2.10c — raw L2 distance EE↔cube as a reward term.
# ---------------------------------------------------------------------------


def object_ee_distance_l2(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Raw L2 distance from the EE target to the cube, in meters.

    Returns the **positive distance** ``||EE - cube||`` per env. Used as a
    reward term with **negative weight** (e.g. -1.0) to create a constant
    "always-on" gradient toward the cube — independent of how far the EE
    currently is, unlike the saturating ``1 - tanh(d/std)`` reward.

    Why this term exists (V2.10c rationale): with V2.10b we observed that
    the policy stagnates at ``reaching_object ≈ 0.017`` (per-step ~ 0.0001,
    consistent with EE drifting to ~50 cm from cube). The tanh-based
    ``reaching_object`` (``std=0.15``) gives essentially zero gradient
    past 60 cm — so the policy has no reward signal pulling it back
    toward the cube once it has drifted away.

    A linear distance reward ``-d`` gives a constant gradient ``-1/m``
    everywhere in workspace. Combined with γ=0.99 GAE bootstrapping,
    this creates a global drive in the value function: V(s_close) is
    consistently greater than V(s_far), independent of the saturation
    region of the tanh.

    The term **self-extinguishes** during grasp / lift / transport: once
    the EE is on the cube (``d ≈ 0.02 m``), this term contributes only
    ``-3/ep`` while ``grasping_cube + lifting + tracking`` together
    contribute ``+4500+/ep`` — three orders of magnitude smaller. So it
    drives approach without interfering with downstream phases.

    We use the same ``ee_frame.target_pos_w[..., 0, :]`` index as the
    canonical ``lift_mdp.object_ee_distance`` so the two reward terms
    measure consistent geometry.

    Args:
        object_cfg: SceneEntityCfg for the cube (``"cube"``).
        ee_frame_cfg: SceneEntityCfg for the EE FrameTransformer
            (``"ee_frame"``).

    Returns:
        Tensor ``(num_envs,)`` of raw distances (always ≥ 0).
    """
    cube: RigidObject = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    cube_pos_w = cube.data.root_pos_w
    return torch.norm(cube_pos_w - ee_pos_w, dim=-1)


# ---------------------------------------------------------------------------
# V2.13 — posture-shaping rewards to force top-down grasp.
#
# Diagnostic V2.12 (model_900) showed the policy converged on a "scoop"
# motor program: jaw at z ≈ 5cm (table level), gripper z-axis pointing
# upward (-0.27 score), wrist tucked under the gripper, fingers reaching
# horizontally / from below. This is unsuitable for sim-to-real transfer
# (table friction will not match in real world) and looks unnatural.
#
# V2.13 adds two soft constraints:
#
# 1. `gripper_orientation_penalty` — penalty when the gripper z-axis
#    deviates from the world -z direction (pointing down). Implemented
#    as a PENALTY (not a bonus) so the policy can't farm a free "+1/step
#    for staying still while pointing down". The neutral state is
#    "pointing perfectly down"; any deviation costs reward.
#
# 2. `scoop_grasp_penalty` — penalty when the EE (gripper palm) is at a
#    higher world z than the wrist body. In a top-down grasp the wrist
#    is ALWAYS above the gripper (kinematic chain: ... → wrist → gripper
#    → jaw). In a scoop grasp, the wrist twists under so its z falls
#    below the gripper's z. This penalty is purely geometric — no
#    threshold tuning needed.
#
# Together they force the only stable optimum to be a top-down grasp.
# ---------------------------------------------------------------------------


def gripper_orientation_penalty(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty when gripper points UP (jaw above palm in world frame).

    **V2.13 v3 rewrite** — the previous formulation used quaternion math
    on the "gripper" body's local +z axis, assuming +z = the gripping
    direction. ``dump_scene_frames`` revealed that on SO-101 / LeIsaac
    the gripper frame's local +z actually points BACKWARD (toward the
    robot base), not toward the jaws. So ``1 + z_world[..., 2]`` returned
    0 when the gripper was pointing UP and 2 when pointing DOWN — the
    sign was INVERTED. With weight=-1.0 the policy was rewarded for
    pointing the gripper UP, and that's exactly the pose V2.13 v2
    converged on at iter 60+ (visually confirmed: gripper jaws facing
    the ceiling, EE hovering 17cm above the cube).

    V2.13 v3 uses a position-based formulation that is **unambiguous**
    regardless of URDF axis conventions: compare the world-frame Z of
    the palm (ee_frame target[0]) and the jaw (target[1]). In a clean
    top-down grasp the jaw is BELOW the palm; in a gripper-up pose the
    jaw is ABOVE.

    Returns ``clamp((jaw_z - palm_z) / 0.05, min=0)``:
      - 0.0   when jaw ≤ palm (top-down or horizontal — no penalty)
      - ~1.0  when jaw is ~5 cm above palm (fully gripper-up)
      - linearly interpolates in between

    Use with **NEGATIVE weight** (e.g. -5.0 in V2.13 v3). The neutral
    pose (top-down) costs nothing; any tilt toward "fingers reaching up"
    actively costs reward.

    NB: pairs with the old ``scoop_grasp_penalty`` only conceptually now
    — in V2.13 v3 the scoop term is dropped (weight=0) because random
    rollouts showed it fires only 0% of the time on gripper-down samples
    (i.e., it's redundant with this new orientation penalty once the
    sign is fixed).
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    palm_z = ee_frame.data.target_pos_w[..., 0, 2]   # target[0] = "gripper"
    jaw_z = ee_frame.data.target_pos_w[..., 1, 2]    # target[1] = "jaw"
    return torch.clamp((jaw_z - palm_z) / 0.05, min=0.0)


def scoop_grasp_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="wrist"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty when the EE (gripper palm) is higher than the wrist body.

    In a clean top-down grasp the kinematic chain has wrist above
    gripper-palm above jaw → wrist_z >= palm_z >= jaw_z. In a scoop
    grasp the wrist twists underneath the gripper (palm), so the
    geometric inversion ee_z > wrist_z is the diagnostic of "fingers
    reaching up from below" instead of "fingers descending from above".

    Returns 0.0 when wrist_z >= ee_z (top-down geometry — no penalty).
    Returns the positive overshoot (= ee_z - wrist_z) when ee is above
    wrist (scoop geometry).

    Use with **NEGATIVE weight** (e.g. -10.0). Threshold-free — works
    for any table height, any goal pose, any robot config. The
    constraint is purely geometric.

    Pairs with `gripper_orientation_penalty`. Together they force the
    top-down grasp posture as the only stable optimum.

    Body name `wrist`: confirmed via `dump_body_names.py` on the SO-101
    LeIsaac articulation (body chain: base → shoulder → upper_arm →
    lower_arm → wrist → gripper → jaw).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    wrist_z = robot.data.body_pos_w[:, robot_cfg.body_ids[0], 2]
    ee_z = ee_frame.data.target_pos_w[..., 0, 2]
    return torch.clamp(ee_z - wrist_z, min=0.0)


def gripper_pointing_direction_penalty(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """V2.15 — Penalty when the palm→jaw vector doesn't point DOWN in world.

    Computes the normalized vector from palm (ee_frame target[0]) to jaw
    (ee_frame target[1]) in world frame, and uses its z-component as the
    "down score". This is unambiguous regardless of URDF axis conventions
    because it uses real WORLD POSITIONS — no quaternion math, no local
    axis interpretation.

    Returns:
        0.0   when palm→jaw points straight DOWN (top-down grasp posture).
        1.0   when palm→jaw is horizontal (snake/sideways approach).
        2.0   when palm→jaw points UP (gripper-up posture).

    Use with **NEGATIVE weight** (e.g. -5.0 in V2.15). This catches BOTH
    gripper-up (V2.13 v2 failure) AND horizontal/snake (V2.14 failure),
    forcing strict top-down posture as the only zero-penalty configuration.

    Why this supersedes the V2.13 v3 `gripper_orientation_penalty` (jaw_z
    vs palm_z only):
        The Z-only formula returns 0 in BOTH top-down (jaw below palm in
        z) AND horizontal (jaw beside palm at same z). PPO can converge
        to either with equal preference. V2.14 visual replay confirmed
        the policy chose snake/horizontal — gripper extended forward
        skimming the table, jaws beside the cube laterally.
        The direction formula here distinguishes them : top-down → 0,
        horizontal → 1, up → 2. Only top-down avoids the penalty.

    Verified at home pose via dump_scene_frames (2026-05-11) :
        palm = (0.3294, -0.3625, 0.2769), jaw = (0.3306, -0.2691, 0.2761).
        delta = (0.0012, +0.0934, -0.0008), delta_norm.z ≈ -0.009.
        Return ≈ 1.0 (horizontal as expected at home — gripper extends
        forward in +y direction, almost parallel to table).
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    palm = ee_frame.data.target_pos_w[..., 0, :]
    jaw = ee_frame.data.target_pos_w[..., 1, :]
    delta = jaw - palm
    # Normalize. Add small epsilon to avoid div-by-zero in the degenerate
    # case where palm and jaw coincide exactly (physically impossible).
    delta_norm = delta / (torch.norm(delta, dim=-1, keepdim=True) + 1e-6)
    # delta_norm.z ∈ [-1, +1]. Top-down: jaw is below palm → delta.z < 0 →
    # delta_norm.z = -1 → return 0. Horizontal: delta.z = 0 → return 1.
    # Gripper-up: delta.z > 0 → delta_norm.z = +1 → return 2.
    return 1.0 + delta_norm[..., 2]


def jaw_below_cube_penalty(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_half_size: float = 0.010,
) -> torch.Tensor:
    """V2.15 — Penalty when jaw goes BELOW the cube's bottom (= into table).

    Live cube position is used so the threshold adapts as the cube is
    lifted: when the cube is on the table (z=0.041), the threshold is
    0.031 (table top). When the cube is lifted to z=0.20, the threshold
    becomes 0.19 (no false positive during transport).

    Returns:
        0.0 when jaw_z ≥ cube_bottom (= cube_z - cube_half_size).
        positive (in meters) when jaw is below cube_bottom — equal to
        how far below the cube's bottom face the jaw has gone.

    Typical values with weight=-50 (V2.15):
        jaw at cube top (0.051m) :        0   → 0 penalty (grasp height OK)
        jaw at cube center (0.041m):      0   → 0 (touching cube OK)
        jaw at cube bottom (0.031m):      0   → 0 (table level, tolerated)
        jaw 5 mm below bottom (0.026m):   0.005 → -0.25/step (warning zone)
        jaw 1 cm below bottom (0.021m):   0.010 → -0.50/step (clear breach)
        jaw 2 cm below bottom (0.011m):   0.020 → -1.00/step (deep in table)

    Calibration of ``weight=-50`` (V2.15):
        Empirical V2.14 model_100: min jaw_z = +0.042m (1.1cm above
        cube_bottom 0.031m). No penalty fires on current behavior.
        Empirical V2.13 v3 model_100: min jaw_z = +0.010m (2.1cm BELOW
        cube_bottom). With weight -50, this would have cost ~30/ep,
        deterring the physics-breach behavior.

    Cube half size 0.010 = 1 cm (cube is 2 cm side, root_pos_w is center).
    No threshold tuning needed across scene rescales — formula reads the
    live cube state.
    """
    cube = env.scene[cube_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    cube_z = cube.data.root_pos_w[..., 2]
    jaw_z = ee_frame.data.target_pos_w[..., 1, 2]
    cube_bottom = cube_z - cube_half_size
    return torch.clamp(cube_bottom - jaw_z, min=0.0)


def cube_height_above_spawn(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    spawn_z: float = 0.041,
    max_height: float = 0.20,
) -> torch.Tensor:
    """V2.16 — Dense linear reward proportional to cube z above its spawn.

    Designed to break the V2.15 "grasp-only local optimum": PPO gets a
    smooth gradient for partial lift starting from the very first mm
    above the table, instead of the binary `lifting_object` (z_rel >
    0.08m) which only fires for full 5+ cm lifts.

    Returns:
        clamp(cube.z - spawn_z, min=0, max=0.20), i.e., in [0, 0.20] m.
            0.000 — cube on table (no lift)
            0.005 — cube 5 mm above (first signal!)
            0.050 — cube 5 cm above table
            0.200 — cube 20 cm above (capped at goal height)

    Typical values with weight=+30 (V2.16):
        Cube 1 cm above spawn:  0.010 → +0.30/step
        Cube 5 cm above:        0.050 → +1.50/step
        Cube 10 cm above:       0.100 → +3.00/step
        Cube 20 cm above:       0.200 → +6.00/step (cap, ≈ goal_z range)

    The cap at 0.20 m prevents the policy from gaming "lift cube to
    unreasonable height" — once at goal range, no extra incentive.

    ``spawn_z = 0.041`` is the constant cube spawn z empirically verified
    via dump_scene_frames on 2026-05-11 (LeIsaac scene has cube center
    1 cm above the table top at z=0.031, i.e., cube root_pos_w.z = 0.041).
    Cube spawn z does NOT randomize in V2.15 (constant across resets),
    so the offset is stable.
    """
    cube = env.scene[cube_cfg.name]
    height_above = cube.data.root_pos_w[..., 2] - spawn_z
    return torch.clamp(height_above, min=0.0, max=max_height)


def cube_dropped_float(
    env: ManagerBasedRLEnv,
    world_z_threshold: float = 0.04,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Float-cast version of ``cube_dropped`` for use as a RewTerm.

    Pair with the matching DoneTerm — ``world_z_threshold`` MUST stay in
    sync between this RewTerm and ``TerminationsCfg.cube_dropped``.

    Why this penalty is needed (V2.8.5 → V2.9 motivation): on V2.8.5 with
    ``cube_dropped`` as a free DoneTerm (no cost), we observed ~41% of
    episodes terminating via cube_dropped because bumping the cube off
    the table during grasp attempts was net-positive EV: the policy
    collected ~5 reward from `reaching_object` over ~30 steps before the
    drop, then the episode ended with V=0 — comparable to a 150-step
    time-out. The policy never paid for the drop, so it kept
    accidentally bumping the cube without learning to grasp gently.

    Calibration of ``weight=-5.0``: with reaching_object std=0.15 giving
    ~0.16 reward/step at typical reset distance, 30 steps of reach pay
    +4.8. Penalty -5.0 makes the drop net negative (+4.8 -5.0 = -0.2),
    barely below zero — enough to discourage casual drops without making
    the policy too timid to explore grasp attempts. Watch
    ``Episode_Termination/cube_dropped`` after V2.9 launch:
      - if rate > 25 % at iter 80 → bump weight to -10
      - if rate < 5 % but grasping still flat → relax to -2 (policy too timid)

    Returns ``(num_envs,)`` float tensor of {0.0, 1.0}.
    """
    return cube_dropped(
        env,
        world_z_threshold=world_z_threshold,
        cube_cfg=cube_cfg,
    ).float()


# ===========================================================================
# V2.18 — "Precision Landing" reward stack (Claude search design)
#
# Bounded-magnitude, multiplicatively-gated rewards designed to:
#   1. Force precision pre-grasp positioning (palm above cube + aligned).
#   2. Use a strict 6-condition grasp predicate (cube_grasped_strict)
#      that requires geometric containment between the jaws.
#   3. Keep lift/goal/fine rewards ZERO unless the strict grasp predicate
#      fires (multiplicative gating).
#   4. Penalize jaw approaching the table top WITHOUT a grasp (anti-smash).
#   5. Keep all dense weights ≤ 2.0 (per-step magnitude budget |r| ≤ 5
#      for PPO stability with γ=0.99 × 300-step episodes).
#
# Design rationale, calibration math, citations, risk runbook: see
# `notes/eval2_pipeline.md` § "V2.18 design" (or the original Claude
# search artifact archived in `notes/v218_design_claude_search.md`).
# ===========================================================================


def cube_grasped_strict(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_joint_name: str = "gripper",
) -> torch.Tensor:
    """Strict 6-condition geometric containment predicate (V2.18).

    Replaces the loose ``cube_grasped`` predicate (jaw<4cm + gripper
    closed) which was a false-positive when the cube was BESIDE the
    jaws (visual confirmed on V2.15 model_300, 16/16 episodes had
    grasp_LOST/ejection on lift attempt).

    All six conditions must be simultaneously true :
      (a) palm strictly above cube top by ≥ 1 cm safety margin
      (b) jaw at or below cube top (jaws have descended past the top)
      (c) jaw at or above cube bottom (catches scoop-from-below)
      (d) cube xy within 1.5 cm of the palm-jaw midpoint (lateral
          containment between the fingers)
      (e) gripper joint closed past 70% of its closing travel
      (f) cube lateral velocity < 0.50 m/s (cube not ejecting)

    Returns BoolTensor (num_envs,) — used as a multiplicative gate
    in lift_height_gated, goal_tracking_gated_*, and as the binary
    grasp reward source.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    cube_pos_w = cube.data.root_pos_w                  # (N, 3)
    cube_vel_w = cube.data.root_lin_vel_w              # (N, 3)
    palm_w = ee_frame.data.target_pos_w[..., 0, :]     # (N, 3)
    jaw_w = ee_frame.data.target_pos_w[..., 1, :]      # (N, 3)

    cube_z = cube_pos_w[:, 2]
    cube_top = cube_z + 0.010
    cube_bottom = cube_z - 0.010

    cond_a = palm_w[:, 2] >= (cube_top + 0.010)        # palm ≥ cube_top + 1 cm
    cond_b = jaw_w[:, 2] <= (cube_top + 0.005)         # jaw ≤ cube_top + 5 mm
    cond_c = jaw_w[:, 2] >= (cube_bottom - 0.005)      # jaw ≥ cube_bottom − 5 mm

    midpoint_xy = 0.5 * (palm_w[:, :2] + jaw_w[:, :2])
    lateral_d = torch.norm(cube_pos_w[:, :2] - midpoint_xy, dim=1)
    cond_d = lateral_d <= 0.015                        # ≤ 1.5 cm

    # Gripper closure check.
    # SO-101 gripper joint range = (-10°, +100°) per leisaac asset cfg.
    # BinaryJointPositionActionCfg in our env commands:
    #   open  -> q_target = 0.5 rad (≈ 29°)
    #   close -> q_target = 0.0 rad
    # i.e. SMALLER q == more closed. The original `q >= 0.7 * q_max`
    # check was reversed (it required q >= 1.22 rad ≈ 70°, which the
    # action manager can never command — gripper joint stays in [0, 0.5]).
    # That bug is why grasp_strict fired ZERO times across 967 iters of
    # V2.18 cold + V2.18b. Fix: check that q is within 30 % of the close
    # command target.
    gripper_ids, _ = robot.find_joints([gripper_joint_name])
    gripper_idx = gripper_ids[0]
    q_gripper = robot.data.joint_pos[:, gripper_idx]
    # close target = 0.0, open target = 0.5 → 70 % of closing travel = 0.15
    cond_e = q_gripper <= 0.15                         # ≥ 70 % closed

    cube_speed_xy = torch.norm(cube_vel_w[:, :2], dim=1)
    cond_f = cube_speed_xy < 0.50                      # < 0.50 m/s

    return cond_a & cond_b & cond_c & cond_d & cond_e & cond_f


def cube_grasped_strict_float(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_joint_name: str = "gripper",
) -> torch.Tensor:
    """Float-cast of cube_grasped_strict for use as a RewTerm.

    Used directly as the V2.18 ``grasp_strict`` reward (weight +2.0)
    and indirectly as the multiplicative gate in lift/goal rewards.
    """
    return cube_grasped_strict(env, cube_cfg, robot_cfg, ee_frame_cfg,
                               gripper_joint_name).float()


def ee_to_cube_distance_clipped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    clip_max: float = 0.30,
) -> torch.Tensor:
    """V2.18 #1 — Linear distance penalty, clipped at 0.30 m.

    Clipping bounds the cold-start dense penalty so the cumulative
    per-episode cost cannot exceed |−0.30 × 300 × weight| = |−90 × weight|.
    With weight=−1.0 this gives a worst-case −90/ep, well below the
    drop penalty −50, eliminating the V2.13 v1 suicide-by-drop trap.

    Use with weight=-1.0.
    """
    cube: RigidObject = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    palm = ee_frame.data.target_pos_w[..., 0, :]
    cube_pos = cube.data.root_pos_w
    d = torch.norm(palm - cube_pos, dim=-1)
    return torch.clamp(d, max=clip_max)


def palm_xy_above_cube(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    std: float = 0.04,
    height_margin: float = 0.005,
) -> torch.Tensor:
    """V2.18 #3 — Precision lateral alignment reward (palm above cube).

    Returns ``1 - tanh(||palm_xy - cube_xy|| / std)`` IF palm is above
    the cube (palm_z > cube_top + height_margin), else 0. The gate
    prevents the policy from getting xy-alignment credit while the
    palm is at table level or below cube top (= sideways snake approach).

    std = 0.04 m = 2 × cube_half (0.010) + 2 × jaw_x_offset (0.021).
    Tight enough that a sideways approach gets ~0.4 reward (not full 0.8).

    Use with weight=+0.8.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    palm = ee_frame.data.target_pos_w[..., 0, :]
    cube_pos = cube.data.root_pos_w

    cube_top = cube_pos[:, 2] + 0.010
    above = (palm[:, 2] > (cube_top + height_margin)).float()

    dxy = torch.norm(palm[:, :2] - cube_pos[:, :2], dim=-1)
    score = 1.0 - torch.tanh(dxy / std)
    return above * score


def hover_height_gaussian(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_height: float = 0.05,
    sigma: float = 0.025,
    gripper_joint_name: str = "gripper",
) -> torch.Tensor:
    """V2.18 #4 — Gaussian hover bonus peaked 5 cm above cube top.

    Returns ``exp(-((h - target_height)/sigma)^2)`` where
    ``h = palm_z - cube_top``, gated multiplicatively by:
      * palm_xy aligned with cube (using palm_xy_above_cube as alignment)
      * NOT cube_grasped_strict (this term turns off once strict grasp
        fires, freeing the policy to descend)

    Target h = 5 cm matches the palm-jaw vertical offset (~5 cm),
    meaning when palm is at hover height, the jaw is exactly at cube_top.

    Use with weight=+0.5.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    palm = ee_frame.data.target_pos_w[..., 0, :]
    cube_pos = cube.data.root_pos_w
    cube_top = cube_pos[:, 2] + 0.010

    h = palm[:, 2] - cube_top
    raw = torch.exp(-((h - target_height) / sigma) ** 2)

    # Alignment gate (only fires when palm is laterally aligned with cube)
    dxy = torch.norm(palm[:, :2] - cube_pos[:, :2], dim=-1)
    align = (1.0 - torch.tanh(dxy / 0.04))
    align_gate = (align > 0.6).float()

    # Not-grasped gate
    grasped = cube_grasped_strict(env, cube_cfg, robot_cfg, ee_frame_cfg,
                                  gripper_joint_name)
    not_grasped = (~grasped).float()

    return raw * align_gate * not_grasped


def lift_height_gated(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    spawn_z: float = 0.041,
    max_lift: float = 0.10,
    gripper_joint_name: str = "gripper",
) -> torch.Tensor:
    """V2.18 #6 — Bounded lift reward multiplicatively gated by strict grasp.

    Returns ``clamp((cube.z - spawn_z) / max_lift, 0, 1) × strict_grasp``.
    Capped at 1.0 by the normalization — even if the cube is somehow
    flung to 1 m, the reward stays bounded. With weight=+1.5 the per-step
    ceiling is +1.5, well within the |r|≤5 budget.

    The multiplicative ``× strict_grasp`` makes "lift without grasp"
    mathematically impossible — addresses V2.7/V2.9 flick exploit AND
    the V2.16/V2.17 "fake lift" from ungated dense lift.

    Use with weight=+1.5.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_z = cube.data.root_pos_w[:, 2]
    height_above = torch.clamp((cube_z - spawn_z) / max_lift, 0.0, 1.0)
    grasped = cube_grasped_strict(env, cube_cfg, robot_cfg, ee_frame_cfg,
                                  gripper_joint_name).float()
    return height_above * grasped


def goal_tracking_gated(
    env: ManagerBasedRLEnv,
    std: float = 0.20,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    command_name: str = "object_pose",
    min_lift_height: float = 0.08,
    gripper_joint_name: str = "gripper",
) -> torch.Tensor:
    """V2.18 #7/#8 — Goal tracking with double gate: strict_grasp × lifted.

    Returns ``(1 - tanh(d / std)) × strict_grasp × (cube_z > min_lift)``.

    Triple condition prevents the policy from collecting goal-tracking
    credit by sliding the cube along the table into the goal projection
    (which would NOT count as success but COULD farm the dense reward).

    Use with weight=+1.0 (coarse, std=0.20) or +0.5 (fine, std=0.04).
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    cube_pos_w = cube.data.root_pos_w
    command = env.command_manager.get_command(command_name)
    goal_pos_b = command[:, :3]
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_b
    )

    d = torch.norm(goal_pos_w - cube_pos_w, dim=-1)
    raw = 1.0 - torch.tanh(d / std)

    grasped = cube_grasped_strict(env, cube_cfg, robot_cfg, ee_frame_cfg,
                                  gripper_joint_name).float()
    lifted = (cube_pos_w[:, 2] > min_lift_height).float()
    return raw * grasped * lifted


def palm_to_jaw_orient_v218(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """V2.18 #9 — Palm-to-jaw direction reward (top-down posture).

    Returns ``-delta_hat[2]`` where ``delta_hat = (jaw - palm) / ||(jaw - palm)||``.

    Semantics (Isaac Sim world frame, +z up):
      * jaw strictly below palm (top-down) → delta_z < 0 → -delta_z > 0
        → returns +1 (rewarded)
      * jaw at same z (horizontal/snake)  → delta_z = 0 → returns 0
      * jaw above palm (gripper-up)        → delta_z > 0 → returns -1

    Use with weight=+0.3 (small residual nudge — palm_xy_above_cube +
    hover_height already encode top-down posture geometrically).

    NB: SIGN-VERIFY before launch via dump_scene_frames.py. The Isaac Sim
    world frame has +z up, so jaw being below palm in a top-down pose
    means delta_z < 0 (negative), hence ``-delta_z > 0`` reward. This
    is OPPOSITE sign from V2.15's broken formulation (which incorrectly
    used `+delta.z` and rewarded gripper-up).
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    palm = ee_frame.data.target_pos_w[..., 0, :]
    jaw = ee_frame.data.target_pos_w[..., 1, :]
    delta = jaw - palm
    delta_norm = delta / (torch.norm(delta, dim=-1, keepdim=True) + 1e-6)
    return torch.clamp(-delta_norm[..., 2], min=-1.0, max=1.0)


def jaw_table_impact_penalty(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    safe_height: float = 0.06,
    gripper_joint_name: str = "gripper",
) -> torch.Tensor:
    """V2.18 #10 — Ramped jaw-near-table penalty, gated by NOT grasped.

    Returns ``clamp((safe_height - jaw_z) / safe_height, 0, 1) × (NOT strict_grasp)``.

    The penalty ramps smoothly from 0 (jaw at safe_height = 6 cm) to 1
    (jaw at 0 m world). With weight=-2.0 this gives at most -2/step when
    jaw is at the floor AND not grasping.

    Once grasp_strict fires, the term turns off — the policy is free to
    descend toward the table to place the cube.

    Use with weight=-2.0.
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    jaw = ee_frame.data.target_pos_w[..., 1, :]
    jaw_z = jaw[:, 2]

    pen = torch.clamp((safe_height - jaw_z) / safe_height, min=0.0, max=1.0)

    grasped = cube_grasped_strict(env, cube_cfg, robot_cfg, ee_frame_cfg,
                                  gripper_joint_name)
    not_grasped = (~grasped).float()
    return pen * not_grasped


def cube_at_goal_with_lift(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.05,
    min_lift_height: float = 0.08,
    command_name: str = "object_pose",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """V2.18 success bonus — cube within ``distance_threshold`` of goal
    AND cube has been lifted above ``min_lift_height``.

    Pairs with the matching DoneTerm. The min_lift_height closes the
    "slide cube into goal projection" exploit class.

    Returns float {0.0, 1.0}.
    """
    from .terminations import cube_reached_goal as _cube_reached_goal
    reached = _cube_reached_goal(env, distance_threshold, command_name,
                                 cube_cfg, robot_cfg)
    cube: RigidObject = env.scene[cube_cfg.name]
    lifted = cube.data.root_pos_w[:, 2] > min_lift_height
    return (reached & lifted).float()


# =============================================================================
# V2.19 — full rewrite from notes/v219_reward_search_claude.md
# =============================================================================
# Design principles applied (each cited in the architecture doc):
#   - Contact-impulse grasp predicate (DexPoint, ManiSkill, Lin et al. 2025)
#   - Multiplicative gating on grasp/lift cascade
#   - Bounded shaping kernels (tanh, exp) only — no raw -d
#   - One-time milestone bonuses on 0->1 transitions (Eureka)
#   - Dual-scale tracking (coarse + fine) gated on lift
#   - DrEureka safety cocktail (torque, work, qlimit, anti-jam)
#   - Positive success terminal dominates cumulative shaping
#
# All terms below are designed to be combined as the V2.19 reward stack.
# Old V2.18b functions above are preserved for archive — V2.19 uses the
# new functions exclusively.


def _get_jaw_force_norm(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Return per-env L2 norm of the latest contact force on `sensor_name`.

    Returns zeros if sensor missing (graceful fallback so training still runs
    if the contact sensor was forgotten in the scene cfg).
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene[sensor_name]
    forces = sensor.data.net_forces_w_history  # (N, history, num_bodies, 3)
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device)
    # Latest frame, summed over bodies, L2 norm.
    latest = forces[:, 0]                        # (N, num_bodies, 3)
    return latest.norm(dim=-1).sum(dim=-1)       # (N,)


def cube_grasped_contact_v219(
    env: ManagerBasedRLEnv,
    gripper_sensor_name: str = "contact_gripper",
    jaw_sensor_name: str = "contact_jaw",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_name: str = "ee_frame",
    force_threshold: float = 1.0,
    cube_speed_threshold: float = 0.5,
    cube_proximity_threshold: float = 0.10,
) -> torch.Tensor:
    """V2.19 contact-impulse grasp predicate (DexPoint / ManiSkill style)
    with cube-proximity gate to eliminate table / self-collision false
    positives.

    Returns ``BoolTensor (num_envs,)`` True when ALL hold:
      (i)   total contact force on `gripper` body > force_threshold
      (ii)  total contact force on `jaw` body > force_threshold
      (iii) cube within `cube_proximity_threshold` of the jaw fingertip
            (ee_frame.target[1], offset (-0.021, -0.070, 0.02) from jaw body)
      (iv)  cube lateral velocity < cube_speed_threshold

    The proximity gate is mandatory: GPU PhysX (cuda:0) silently fails or
    hangs env build when ContactSensorCfg.filter_prim_paths_expr is set,
    so sensors report TOTAL contact (any source). Without the proximity
    gate, top-down approach with both jaws flat on the table fires a
    false grasp every step.

    Why d_jaw only (not d_palm): leisaac's ee_frame.target[0] ("gripper")
    has NO offset -- it points at the wrist plate body origin, ~9 cm
    behind the fingertips in top-down. d_palm would be misleading. The
    jaw fingertip is right next to the palm fingertip during a grasp, so
    a cube near the jaw fingertip is necessarily near the palm fingertip
    too. One gate is sufficient and more robust.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_pos = cube.data.root_pos_w
    cube_speed = cube.data.root_lin_vel_w[:, :2].norm(dim=-1)

    f_gripper = _get_jaw_force_norm(env, gripper_sensor_name)
    f_jaw = _get_jaw_force_norm(env, jaw_sensor_name)

    if ee_frame_name in env.scene.sensors:
        ee_frame = env.scene[ee_frame_name]
        jaw_tip = ee_frame.data.target_pos_w[:, 1, :]      # fingertip-offset target
        d_jaw = (cube_pos - jaw_tip).norm(dim=-1)
        cube_in_grip_zone = d_jaw < cube_proximity_threshold
    else:
        cube_in_grip_zone = torch.ones_like(f_gripper, dtype=torch.bool)

    grasped = (
        (f_gripper > force_threshold)
        & (f_jaw > force_threshold)
        & cube_in_grip_zone
        & (cube_speed < cube_speed_threshold)
    )
    return grasped


def cube_grasped_contact_v219_float(
    env: ManagerBasedRLEnv,
    gripper_sensor_name: str = "contact_gripper",
    jaw_sensor_name: str = "contact_jaw",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_name: str = "ee_frame",
    force_threshold: float = 1.0,
    cube_speed_threshold: float = 0.5,
    cube_proximity_threshold: float = 0.10,
) -> torch.Tensor:
    """Float-cast of cube_grasped_contact_v219 — used as continuous gate."""
    return cube_grasped_contact_v219(
        env, gripper_sensor_name, jaw_sensor_name, cube_cfg, ee_frame_name,
        force_threshold, cube_speed_threshold, cube_proximity_threshold,
    ).float()


def _ensure_extras_buffer(env: ManagerBasedRLEnv, key: str) -> torch.Tensor:
    """Return a per-env BoolTensor stored in env.extras, creating if absent.

    This is how we track "previous step" state for one-time milestone
    bonuses across episode steps. The buffer is reset to False at episode
    reset by hooking into env.episode_length_buf == 0.
    """
    if not hasattr(env, "_v219_state"):
        env._v219_state = {}
    if key not in env._v219_state:
        env._v219_state[key] = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
    return env._v219_state[key]


def grasp_milestone_v219(
    env: ManagerBasedRLEnv,
    gripper_sensor_name: str = "contact_gripper",
    jaw_sensor_name: str = "contact_jaw",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """V2.19 — one-time bonus on the 0->1 transition of contact-grasp.

    Returns ``FloatTensor (num_envs,)`` with value 1.0 only at the step
    where ``cube_grasped_contact_v219`` first becomes True after a False.
    Subsequent True steps yield 0.0 — prevents the "hold-and-don't-move"
    exploit (Eureka reflection logs). Use with weight=+5.0 (size to
    dominate per-step shaping at the moment of grasp acquisition).
    """
    grasped_now = cube_grasped_contact_v219(
        env, gripper_sensor_name, jaw_sensor_name, cube_cfg, force_threshold
    )
    prev = _ensure_extras_buffer(env, "grasp_prev")
    # Reset buffer to False at episode start (episode_length_buf == 0).
    just_reset = env.episode_length_buf == 0
    prev = torch.where(just_reset, torch.zeros_like(prev), prev)
    transition = grasped_now & (~prev)
    # Update for next step.
    env._v219_state["grasp_prev"] = grasped_now.clone()
    return transition.float()


def lift_clipped_gated(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    spawn_z: float = 0.0565,
    max_lift: float = 0.15,
    gripper_sensor_name: str = "contact_gripper",
    jaw_sensor_name: str = "contact_jaw",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """V2.19 — bounded lift reward, gated on contact-grasp.

    Returns ``clamp((cube.z - spawn_z) / max_lift, 0, 1) * is_grasped``.
    Per-step max = 1.0. Use with weight=+5.0 (per V2.11 spec).

    ManiSkill PickCube + DextrAH-G clipped pattern: prevents unbounded
    accumulation as cube rises higher than needed.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_z = cube.data.root_pos_w[:, 2]
    height_above = torch.clamp((cube_z - spawn_z) / max_lift, 0.0, 1.0)
    grasped = cube_grasped_contact_v219_float(
        env, gripper_sensor_name, jaw_sensor_name, cube_cfg, force_threshold
    )
    return height_above * grasped


def lift_milestone_v219(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    spawn_z: float = 0.0565,
    threshold: float = 0.10,
) -> torch.Tensor:
    """V2.19 — one-time bonus on the first lift-above-threshold transition.

    Returns 1.0 at the step where (cube.z - spawn_z) > threshold for the
    first time in the episode. Use with weight=+5.0.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_z = cube.data.root_pos_w[:, 2]
    lifted_now = (cube_z - spawn_z) > threshold
    prev = _ensure_extras_buffer(env, "lift_prev")
    just_reset = env.episode_length_buf == 0
    prev = torch.where(just_reset, torch.zeros_like(prev), prev)
    transition = lifted_now & (~prev)
    env._v219_state["lift_prev"] = lifted_now.clone()
    return transition.float()


def reach_coarse_v219(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    std: float = 0.10,
) -> torch.Tensor:
    """V2.19 — bounded coarse reach kernel (Isaac Lab default).

    ``1 - tanh(d / 0.10)``. Use with weight=+1.0.
    Dual-scale partner of `reach_fine_v219` — coarse term shapes long-range
    approach, fine term provides steep gradient inside last 2 cm.
    """
    cube: RigidObject = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    palm = ee_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(palm - cube.data.root_pos_w, dim=-1)
    return 1.0 - torch.tanh(d / std)


def reach_fine_v219(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    std: float = 0.02,
) -> torch.Tensor:
    """V2.19 — bounded fine reach kernel (Isaac Lab default).

    ``1 - tanh(d / 0.02)``. Use with weight=+0.5.
    """
    cube: RigidObject = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    palm = ee_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(palm - cube.data.root_pos_w, dim=-1)
    return 1.0 - torch.tanh(d / std)


def finger_straddle_v219(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    margin: float = 0.005,
) -> torch.Tensor:
    """V2.19 — finger-straddle geometry bonus (IsaacGymEnvs franka_cabinet).

    Returns 1.0 when palm is above cube top AND jaw is below cube top
    (= the cube is between the two contact points vertically). 0.0
    otherwise. Use with weight=+0.5.

    Adapted to SO-101: palm = top contact, jaw = bottom contact, both
    coming from the same parent body in our case but with independent
    Z positions in top-down pose.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    palm = ee_frame.data.target_pos_w[..., 0, :]
    jaw = ee_frame.data.target_pos_w[..., 1, :]
    cube_z = cube.data.root_pos_w[:, 2]
    palm_above = palm[:, 2] > cube_z + margin
    jaw_below = jaw[:, 2] < cube_z - margin
    return (palm_above & jaw_below).float()


def goal_tracking_lift_gated_dual_scale(
    env: ManagerBasedRLEnv,
    std: float = 0.30,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "object_pose",
    min_lift_height: float = 0.04,
    gripper_sensor_name: str = "contact_gripper",
    jaw_sensor_name: str = "contact_jaw",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """V2.19 — bounded goal-distance kernel, double-gated on lift AND grasp.

    ``(1 - tanh(d_cube_goal / std)) * (cube.z > min_lift) * is_grasped``.

    Per Isaac Lab pattern, called twice with std=0.30 (coarse, w=16) and
    std=0.05 (fine, w=5).

    Both gates are required:
      - lift gate (cube.z > min_lift): closes the "slide cube into goal
        projection on the table" exploit
      - grasp gate (contact-impulse): closes the "wedge cube against
        gripper without grasping, lift via friction, farm goal-tracking"
        exploit (verifier finding 2026-05-12 — without this gate, the
        policy could earn +21/step by faking a lift)
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_pos_w = cube.data.root_pos_w
    command = env.command_manager.get_command(command_name)
    goal_pos_b = command[:, :3]
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_b
    )
    d = torch.norm(goal_pos_w - cube_pos_w, dim=-1)
    raw = 1.0 - torch.tanh(d / std)
    lift_gate = (cube_pos_w[:, 2] > min_lift_height).float()
    grasp_gate = cube_grasped_contact_v219_float(
        env, gripper_sensor_name, jaw_sensor_name, cube_cfg, force_threshold,
    )
    return raw * lift_gate * grasp_gate


def success_terminal_v219(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.05,
    min_lift_height: float = 0.12,
    static_frames_required: int = 5,
    cube_speed_threshold: float = 0.05,
    command_name: str = "object_pose",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_sensor_name: str = "contact_gripper",
    jaw_sensor_name: str = "contact_jaw",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """V2.19 — positive success terminal (replaces -50 drop penalty).

    Triggers 1.0 (one-time) when:
      - cube within `distance_threshold` of goal
      - cube_z > `min_lift_height`
      - cube held in contact-grasp
      - cube has been static for >= `static_frames_required` consecutive frames

    Use with weight=+15.0 (sized to dominate cumulative shaping per
    Skalse et al. NeurIPS 2022, Wang & Lin 2025). The static-frames check
    closes the "throw-grasp" exploit (Lin et al. CoRL 2025).
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_pos_w = cube.data.root_pos_w
    cube_speed = cube.data.root_lin_vel_w.norm(dim=-1)

    # Goal proximity.
    command = env.command_manager.get_command(command_name)
    goal_pos_b = command[:, :3]
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_b
    )
    near_goal = torch.norm(goal_pos_w - cube_pos_w, dim=-1) < distance_threshold
    high_enough = cube_pos_w[:, 2] > min_lift_height

    grasped = cube_grasped_contact_v219(
        env, gripper_sensor_name, jaw_sensor_name, cube_cfg, force_threshold
    )

    # Static-frames counter.
    counter = _ensure_extras_buffer_int(env, "static_count")
    just_reset = env.episode_length_buf == 0
    counter = torch.where(just_reset, torch.zeros_like(counter), counter)
    is_static = cube_speed < cube_speed_threshold
    counter = torch.where(is_static, counter + 1, torch.zeros_like(counter))
    env._v219_state["static_count"] = counter.clone()

    success = near_goal & high_enough & grasped & (counter >= static_frames_required)
    return success.float()


def _ensure_extras_buffer_int(env: ManagerBasedRLEnv, key: str) -> torch.Tensor:
    """Same as _ensure_extras_buffer but for IntTensor (counters)."""
    if not hasattr(env, "_v219_state"):
        env._v219_state = {}
    if key not in env._v219_state:
        env._v219_state[key] = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )
    return env._v219_state[key]


def torque_penalty_v219(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """V2.19 — DrEureka motor torque penalty (sim2real safety).

    ``-1e-3 * sum(tau^2)``. Use with weight=1.0 (sign in the term).
    Returns ``- sum(applied_torque^2)`` — apply -1e-3 weight in env config.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    tau = robot.data.applied_torque
    return -torch.sum(tau ** 2, dim=-1)


def work_penalty_v219(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """V2.19 — mechanical work penalty (Hora et al. CoRL 2022).

    ``-1e-4 * sum(|tau * q_dot|)``. Correlates with motor heating better
    than torque alone — important for SO-101's small servos.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    tau = robot.data.applied_torque
    q_dot = robot.data.joint_vel
    return -torch.sum(torch.abs(tau * q_dot), dim=-1)


def joint_limit_penalty_v219(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    safety_margin: float = 0.05,
) -> torch.Tensor:
    """V2.19 — soft joint-limit penalty (DrEureka safety cocktail).

    Quadratic penalty for each joint outside ``[soft_lower + margin,
    soft_upper - margin]``. Returns ``-sum(relu(violation)^2)``.
    Use with weight=-1e-2 (sign external).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    q = robot.data.joint_pos
    soft_low = robot.data.soft_joint_pos_limits[..., 0]
    soft_high = robot.data.soft_joint_pos_limits[..., 1]
    over_high = torch.relu(q - (soft_high - safety_margin))
    under_low = torch.relu((soft_low + safety_margin) - q)
    return -torch.sum(over_high ** 2 + under_low ** 2, dim=-1)


def close_no_contact_penalty_v219(
    env: ManagerBasedRLEnv,
    gripper_action_idx: int = -1,
    gripper_sensor_name: str = "contact_gripper",
    jaw_sensor_name: str = "contact_jaw",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    force_threshold: float = 1.0,
    close_action_threshold: float = 0.0,
) -> torch.Tensor:
    """V2.19 — anti-jam penalty: commanding close without grasping anything.

    Returns -1.0 when the policy is currently *commanding* the gripper to
    close (raw action[gripper_action_idx] < close_action_threshold) AND
    no cube is being grasped (no contact above force_threshold). Use with
    weight=0.5 (function is already negative).

    Verifier note 2026-05-12 fix: switched from joint-position-based check
    (q_gripper <= 0.15) to action-based (action < 0) per spec — checks
    the policy's *intent* to close rather than the joint's settled state,
    which avoids one-step-lag double-penalty during settling.

    BinaryJointPositionAction layout in V2.19 env:
      action[..., 0:5] = arm joints (shoulder_pan, shoulder_lift,
                        elbow_flex, wrist_flex, wrist_roll)
      action[..., 5]   = gripper binary (negative -> close, positive -> open)
    Default `gripper_action_idx=-1` picks the last entry.

    Closes the "gripper closes on empty space" exploit (RotateIt 2023,
    ByteDance Seed 2025).
    """
    a = env.action_manager.action  # (N, action_dim)
    is_close_cmd = a[..., gripper_action_idx] < close_action_threshold
    grasped = cube_grasped_contact_v219(
        env, gripper_sensor_name, jaw_sensor_name, cube_cfg, force_threshold
    )
    pinching_air = is_close_cmd & (~grasped)
    return -pinching_air.float()


def orientation_quat_tracking_v219(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    decay: float = 3.0,
    desired_quat_w: tuple[float, float, float, float] | None = None,
) -> torch.Tensor:
    """V2.19 — bounded orientation tracking (Human2Sim2Robot, Lum et al. CoRL 2025).

    Rewards the gripper being aligned in the desired top-down grasp
    posture. Returns ``exp(-decay * theta_err)`` ∈ (0, 1].

    Two modes (verifier note 2026-05-12 fix #5):
    1. **Default (axis-angle on approach vector)**: when `desired_quat_w`
       is None, computes theta_err as the angle between the palm->jaw
       approach vector (in world frame) and world -Z. This is
       mathematically equivalent to ``quat_error_magnitude(q_gripper,
       q_desired)`` for the case of aligning a single body axis with a
       world direction — both reduce to the axis-angle between two unit
       vectors. Geometry-driven, no URDF axis-convention assumptions.
    2. **Explicit quaternion**: when `desired_quat_w` is provided
       (w, x, y, z tuple), uses Isaac Lab's
       ``quat_error_magnitude(q_palm_w, q_desired)`` directly. Use this
       when the desired orientation depends on the cube's own orientation
       (long thin objects, antipodal grasps with specific roll, etc.).

    Bounded and smooth — softer than V2.18b's hard prior. Use with weight=+1.0.
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    if desired_quat_w is not None:
        # Mode 2: explicit quaternion error.
        from isaaclab.utils.math import quat_error_magnitude
        palm_quat = ee_frame.data.target_quat_w[..., 0, :]   # (N, 4) in (w,x,y,z)
        target = torch.tensor(
            list(desired_quat_w), device=env.device, dtype=palm_quat.dtype
        ).expand_as(palm_quat)
        theta_err = quat_error_magnitude(palm_quat, target)
        return torch.exp(-decay * theta_err)

    # Mode 1: axis-angle on approach vector.
    palm = ee_frame.data.target_pos_w[..., 0, :]
    jaw = ee_frame.data.target_pos_w[..., 1, :]
    delta = jaw - palm
    delta_norm = delta / (torch.norm(delta, dim=-1, keepdim=True) + 1e-6)
    # Cosine with world -Z; cos=1 when palm directly above jaw (top-down).
    cos_with_down = -delta_norm[..., 2].clamp(-1.0, 1.0)
    theta_err = torch.acos(cos_with_down)  # ∈ [0, pi]
    return torch.exp(-decay * theta_err)
