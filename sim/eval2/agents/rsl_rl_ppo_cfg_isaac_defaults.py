"""Replay of the original isaac_so_arm101 PPO defaults — on a clean env.

Why this exists:

In our first runs we saw `Loss/entropy` explode (10 → 15.5) and the policy
collapse to noise around iter 200. We attributed the failure to the PPO
hyperparameters (entropy_coef=0.006, init_noise_std=1.0,
schedule="adaptive" + tight desired_kl=0.01) and switched to the
lerobot-sim2real-derived V2 config that fixed it.

But those first runs ALSO had two env-side bugs that we only diagnosed
later:

1. `convex_decomposition` collider on the SO-101 gripper, inherited from
   isaac_so_arm101 — the moving jaw self-blocks at +0.36 mid-close, so
   the gripper cannot physically grasp a rigid cube without the
   "magic-attach" workaround the archived branch used.
2. The upstream `LiftEnvCfg.curriculum` ramps `action_rate` and
   `joint_vel` reward weights from -1e-4 to -1e-1 around iter ~417 —
   appropriate for legged locomotion, actively harmful for manipulation
   (it punishes the very motions needed to grasp).

Both bugs are gone on the LeIsaac scaffold (clean USD robot, no
inherited curriculum). So it's worth verifying whether the original PPO
defaults were really at fault or whether they were just a confounding
factor.

This config replays them verbatim. Compare its convergence to the V2 run
on the same env (`Isaac-LeIsaac-SO101-Lift-Visual-v0`).
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgIsaacDefaults(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_isaac_defaults"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
