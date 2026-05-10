"""Phase A smoke test env — SO-101 lift cube with reachable task space.

The default `SoArm101LiftCubeEnvCfg` from `isaac_so_arm101` inherits Franka's
spawn and goal ranges, which include positions up to ~47 cm from the robot
base — physically unreachable for the SO-101 (~33 cm reach). On those
episodes PPO can never succeed, so the success rate is bounded by construction
and the gradient signal is mostly noise.

This subclass keeps every other piece (scene, reward shaping, action space)
identical and only:

1. tightens the spawn and goal ranges so every sampled episode is feasible
2. (in V2) disables the upstream curriculum that ramps action_rate / joint_vel
   penalties to -1e-1 around iter ~400. That ramp was inherited from
   locomotion configs and locks the policy into a "reaching only" local
   optimum on manipulation tasks: once movement is heavily penalized, the
   safest behavior is to approach the cube and stop (see iter 919 run dated
   2026-05-08 where lifting_object went to 0 right after the curriculum
   kicked in).

Spawn range:
    cube spawn       x ∈ [0.15, 0.25], y ∈ [-0.10, 0.10]   (≤ 27 cm)
    lift goal        x ∈ [-0.05, 0.05], y ∈ [-0.20, -0.10],
                     z ∈ [0.10, 0.20]                       (≤ 22 cm)
"""
from isaaclab.utils import configclass

from isaac_so_arm101.tasks.lift.joint_pos_env_cfg import SoArm101LiftCubeEnvCfg


@configclass
class SoArm101LiftCubeRestrictedEnvCfg(SoArm101LiftCubeEnvCfg):
    """Lift task with cube spawn + goal kept inside SO-101 reachable workspace."""

    def __post_init__(self):
        super().__post_init__()

        # Tighten cube spawn around (0.20, 0.00, 0.015) — well inside SO-101 reach.
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.10, 0.10),
            "z": (0.0, 0.0),
        }

        # Tighten lift goal so it stays reachable from the cube pickup zone.
        self.commands.object_pose.ranges.pos_x = (-0.05, 0.05)
        self.commands.object_pose.ranges.pos_y = (-0.20, -0.10)
        self.commands.object_pose.ranges.pos_z = (0.10, 0.20)


@configclass
class SoArm101LiftCubeRestrictedEnvCfg_PLAY(SoArm101LiftCubeRestrictedEnvCfg):
    """Smaller scene with no observation noise — for replaying a checkpoint."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101LiftCubeRestrictedV2EnvCfg(SoArm101LiftCubeRestrictedEnvCfg):
    """V2 — same restricted spawn/goal, but with the curriculum disabled.

    The upstream `LiftEnvCfg.curriculum` ramps `action_rate` and `joint_vel`
    reward weights from -1e-4 to -1e-1 once `env.common_step_counter > 10000`
    (about iter 417 with our 24 num_steps_per_env). That ramp is appropriate
    for legged locomotion (where smooth low-amplitude actions are the goal)
    but actively harmful for manipulation: it punishes the very motions
    needed to grasp and lift the cube. Empirically: in our 2026-05-08 run,
    `lifting_object` went to 0 right after the curriculum kicked in.

    We push the activation threshold far beyond `max_iterations` so it never
    fires.
    """

    def __post_init__(self):
        super().__post_init__()
        # Effectively disable the upstream curriculum. Setting num_steps to
        # ~10**12 keeps the original term in place (and visible in TensorBoard
        # as a constant -1e-4) without triggering the weight bump.
        self.curriculum.action_rate.params["num_steps"] = 10**12
        self.curriculum.joint_vel.params["num_steps"] = 10**12


@configclass
class SoArm101LiftCubeRestrictedV2EnvCfg_PLAY(SoArm101LiftCubeRestrictedV2EnvCfg):
    """Play variant of V2."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
