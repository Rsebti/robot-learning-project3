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
        # v1.4: lowered to 0.3 (was 0.5 in v1.3, 1.0 in v1.2). The two
        # previous runs both blew std up — start with the smallest value
        # that still allows initial exploration, so the optimizer has less
        # ground to cover if it tries to expand.
        init_noise_std=0.3,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # v1.4: lowered from 0.006 to 0.001. The entropy bonus is the term
        # that *encourages* PPO to keep std large; with 0.006 it overpowered
        # our action_l2 penalty and std climbed from 0.5 to 4.65 in 3 k iter.
        # Cutting entropy_coef 6x removes that upward pressure so the
        # action_l2 penalty (-1e-1, see env config) can do its job.
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
