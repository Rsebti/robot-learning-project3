"""V2.11 PPO config — Isaac Lab Lift canonical + retain V2.10's init_noise.

Aligns the PPO config closer to Isaac Lab Lift defaults (entropy_coef=0.005
instead of V2.10's 0.002). The 0.002 reduction in V2.10 was driven by V2.9
chaos suppression, but research warns that aggressive entropy reduction
causes premature commitment when the task signal is weak. With V2.11's
mild smoothness (-1e-4) and ee_to_cube_distance (-1.0), the policy needs
exploration headroom to find the cube — bumping back to 0.005 keeps that.

Other params: full Isaac Lab canonical (value_loss=1.0, n_epochs=5, etc.).

`experiment_name = "lift_v2_11"` keeps logs separate from V2.10/V2.10b/c.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgV211(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_v2_11"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.4,                                    # V2.10 keep
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,                                    # V2.11: 0.002 → 0.005
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
