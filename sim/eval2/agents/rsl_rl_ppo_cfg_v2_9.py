"""V2.9 PPO config — V2.8 PPO unchanged, only logs separated.

V2.9 keeps the V2.8 / V2.8.5 PPO hyperparams verbatim. The V2.8 → V2.9
delta is purely on the **env side**: ``LeIsaacLiftCubeRLEnvCfgV29``
adds a ``cube_dropped_penalty`` RewTerm (weight=-5.0) paired with the
existing V2.8.5 ``cube_dropped`` DoneTerm to discourage the
~41 % drop rate observed on V2.8.5.

Why a separate config class (not just import V2.8): the
``experiment_name`` is independent so TensorBoard groups runs cleanly
under ``logs/rsl_rl/lift_v2_9/`` next to the existing baselines.

All hyperparameters identical to ``LiftCubePPORunnerCfgV28`` /
``LiftCubePPORunnerCfgV285``.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgV29(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_v2_9"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
