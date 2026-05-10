"""V2.10b PPO config — identical to V2.10 except for `experiment_name`.

V2.10 (warm-start from V2.9 + cold-start tested) failed because the env
side `joint_acc_l2` weight at -1e-3 dominated the gradient. V2.10b is a
pure env-side fix (joint_acc weight ÷33) — the PPO config is unchanged
on purpose, so any difference in convergence is attributable solely to
the rebalanced reward.

`experiment_name = "lift_v2_10b"` keeps logs separate from the failed
V2.10 runs in `logs/rsl_rl/lift_v2_10/`.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgV210b(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_v2_10b"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.4,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
