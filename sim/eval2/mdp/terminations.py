"""Custom termination terms — saner success criterion + failure cutoff.

LeIsaac's default success termination (`cube_height_above_base` with
``height_threshold=0.20``) is misaligned with our reward shaping in two
ways:

1. Our `UniformPoseCommandCfg.object_pose` samples goal z uniformly in
   ``[0.10, 0.20]``. Even a perfectly tracked goal at z=0.12 ends ~12 cm
   above the robot base — below the 0.20 m success threshold. So success
   is statistically impossible on roughly half the rollouts no matter how
   good the policy is. Our Phase A run hit `success=0.01%` while
   `lifting_object` reached 8.4/15 (~55 % of frames lifted): proof that
   the policy was working but the metric couldn't see it.

2. The threshold is one-dimensional (only z). A perfect lift to z=0.21
   over the wrong xy still counts as "success", and a near-miss at
   (goal.x, goal.y, goal.z=0.19) doesn't. That doesn't match the actual
   task ("place the cube AT the goal pose").

`cube_reached_goal` is a drop-in replacement that returns `True` when the
3-D distance between the cube and the commanded goal pose is below a
threshold (default 5 cm). It uses the same goal pose the rewards already
track (``command_name="object_pose"``), transforming it from robot root
frame to world frame on the fly to match the cube's world-frame position.

`cube_dropped` is a failure-side cutoff: when the cube falls off the
table (z below the robot base by more than ``drop_threshold``), there is
no recoverable trajectory left in the episode. Without it, those rollouts
keep accumulating ~120 zero-reward steps after the drop, which dilutes
the advantage signal and the value function fits noise. With it, the
episode ends immediately and the critic learns "trajectories that drop
the cube are short and worthless".

Wire `cube_reached_goal` into `TerminationsCfg.success` and `cube_dropped`
into a new `TerminationsCfg.cube_dropped` term.
"""
from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms


def cube_reached_goal(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.05,
    command_name: str = "object_pose",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Success when the cube is within `distance_threshold` of the
    commanded goal pose.

    Args:
        env: Isaac Lab env.
        distance_threshold: Euclidean distance threshold in meters
            between cube position and goal position. Default 5 cm —
            tight enough to filter out near-misses, lax enough to
            register before the policy is millimeter-perfect.
        command_name: name of the `UniformPoseCommandCfg` term in the
            CommandsCfg whose first three dims encode the target xyz.
        cube_cfg: scene entity config for the cube.
        robot_cfg: scene entity config for the robot (used to resolve
            the goal pose from robot-root frame to world frame).

    Returns:
        ``(num_envs,)`` bool tensor.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Goal is published in robot root frame: shape (B, 7) — take xyz.
    command = env.command_manager.get_command(command_name)
    goal_pos_b = command[:, :3]

    # Transform to world frame so we can compare to cube.root_pos_w.
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_b
    )
    distance = torch.norm(cube.data.root_pos_w - goal_pos_w, dim=-1)
    return distance < distance_threshold


def cube_dropped(
    env: ManagerBasedRLEnv,
    world_z_threshold: float = 0.04,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Failure termination: the cube has fallen off the table.

    Uses the cube's z in the **world frame**. Empirical measurement on
    the LeIsaac scene (``measure_cube_height.py``):

      - cube on table (spawn) :  cube.z = +0.0615 m
      - robot base body       :  base.z = +0.0100 m  (essentially at floor)
      - cube on the floor     :  cube.z ≈ +0.02 m  (cube half-extent above floor)

    Because ``base.z`` is already at the floor, a relative-to-base check
    cannot distinguish "cube on table" from "cube on floor" — there's
    only ~5 cm of margin total. We check the absolute world z directly:
    any value below ``world_z_threshold`` (default 0.04 m) is well below
    the table top (0.0615) and well above what the cube could reach
    while still being on the table, so it's a clean drop signal.

    Why we want this termination: without an early failure cutoff, a
    dropped cube leaves the policy collecting ~120 steps of zero reward
    (all gated rewards are off because the cube isn't
    grasped/lifted/at-goal anymore). Those steps dilute the advantage
    estimate and force the value function to learn a long zero-tail
    prediction. Terminating immediately on drop gives the critic a
    clean "this trajectory ended badly" signal via the bootstrap value
    estimate at truncation.

    Note on giving-up exploit: PPO might in principle learn to drop the
    cube on purpose to escape negative regularizers (``action_rate``,
    ``joint_vel``). In V2.8 those regularizers are -1e-4 per step
    against a baseline reward that is typically positive (reaching +
    intermittent grasping/lifting), so dropping is net negative EV — no
    exploit expected. Watch ``Episode_Termination/cube_dropped`` though:
    if it climbs above ~20 % sustained, that's the signature of the
    give-up policy and we'd add a small ``-1.0`` weight RewTerm tied to
    this same predicate.

    Args:
        world_z_threshold: cube z in world frame below which the cube
            counts as dropped. Default 0.04 m — between table top
            (0.0615) and cube-on-floor (~0.02), no false positive from
            transient grasp dips, no missed drops.
        cube_cfg: scene entity config for the cube.

    Returns:
        ``(num_envs,)`` bool tensor. True = cube dropped, episode ends.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_pos_w[:, 2] < world_z_threshold


def ee_far_from_cube(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.5,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Fail-fast termination: the EE has wandered too far from the cube.

    Cuts off "wandering" episodes where the policy has lost the task —
    the gripper has drifted so far from the cube that there is no
    productive trajectory left in the remaining episode budget.

    Why this saves training time (V2.12 addition): typical EE-cube
    distance at reset is ~24 cm. The SO-101 reach radius is ~30-40 cm.
    If the EE drifts to >50 cm, the bras has gone in the wrong direction
    long enough that recovery is unlikely. Without this termination,
    those "lost" episodes still consume 100+ sim steps doing nothing
    useful. Cutting them early lets the simulator reset and reroll a
    productive trajectory faster, accelerating early-iter learning.

    Why no associated reward penalty (note `cube_dropped_penalty` is a
    separate term): the value function bootstrap at truncation already
    gives PPO the signal "this region is bad" (V(out_of_bounds) is low
    because nothing valuable can happen from there). Adding a negative
    RewTerm risks creating "fear of edges" — policy avoids the cube
    boundary even when approaching legitimately. Termination alone is
    enough.

    Args:
        distance_threshold: ||EE - cube|| in world frame above which the
            episode terminates. Default 0.5 m — generous (2× typical
            reset distance) so the policy has room to maneuver, tight
            enough to fail fast on lost trajectories.
        cube_cfg: scene entity config for the cube.
        ee_frame_cfg: scene entity config for the EE FrameTransformer
            (uses target index 0, same as `lift_mdp.object_ee_distance`).

    Returns:
        ``(num_envs,)`` bool tensor. True = EE too far, episode ends.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(cube.data.root_pos_w - ee_pos_w, dim=-1)
    return distance > distance_threshold
