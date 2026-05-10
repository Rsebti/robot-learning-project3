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
