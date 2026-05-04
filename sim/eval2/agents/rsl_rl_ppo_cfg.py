"""rsl_rl PPO config for Eval 2 (mirrors the upstream Lift defaults)."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Eval2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    # 20 000 iter on RTX 5070 ~= 8-10 hours. Budget for the overnight robust
    # convergence run. Target: std collapses below 0.5 and success rate
    # climbs above 90 % under the FULL randomized task (cluster + bowl).
    # If the policy doesn't converge by 5 k iter, it likely won't converge
    # at all without architectural changes (BC warmstart, smaller action
    # scale, hierarchical policy, etc.) — treat 5 k as the diagnostic point.
    max_iterations = 20000
    save_interval = 500  # 40 checkpoints over 20k iter
    experiment_name = "eval2_pick_in_bowl"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        # Lowered from 1.0 — the previous 1k-iter run drove std UP to ~3.7
        # because the policy couldn't find a good gradient. Starting smaller
        # gives the optimizer less room to escape into "permanent exploration"
        # mode and biases it toward exploitation earlier.
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Default rsl_rl entropy bonus. The action_l2 penalty in the env
        # rewards now does the "shrink std" job directly (by pulling the
        # actor mean toward 0, which lets std collapse with it), so we keep
        # entropy_coef at the standard value here.
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
