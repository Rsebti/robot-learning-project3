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
