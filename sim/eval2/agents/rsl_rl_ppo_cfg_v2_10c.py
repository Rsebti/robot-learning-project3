"""V2.10c PPO config — identical to V2.10b/V2.10 except for `experiment_name`.

V2.10b (joint_acc rebalanced ÷33) succeeded at suppressing smoothness
paralysis but exposed a deeper task-gradient issue: the existing
`reaching_object` reward (tanh, std=0.15) saturates past 60 cm, so the
policy drifts away from the cube and gets stuck at d ≈ 50 cm with no
reward signal pulling it back.

V2.10c adds a linear distance term `-||EE - cube||` (weight -1.0) on the
env side. PPO config is unchanged from V2.10/V2.10b, so any difference
in convergence is attributable solely to the new reward term.

`experiment_name = "lift_v2_10c"` keeps logs separate from V2.10/V2.10b.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgV210c(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_v2_10c"
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
