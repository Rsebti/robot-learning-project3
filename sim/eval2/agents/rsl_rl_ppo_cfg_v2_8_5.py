"""V2.8.5 PPO config — V2.8 PPO unchanged, only logs separated.

V2.8.5 keeps the V2.8 PPO hyperparams verbatim (the user reports V2.8
"marché moyennement" — i.e. the canonical Isaac Lab PPO settings are on
the right track for this task). The only V2.8 → V2.8.5 delta is on the
**env side**: ``LeIsaacLiftCubeRLEnvCfgV285`` adds the `cube_dropped`
failure termination so dropped-cube rollouts no longer leave a ~120-step
zero-reward tail polluting the advantage estimate.

We define a separate config class here (not just import V2.8) so the
``experiment_name`` is independent and TensorBoard groups the runs
cleanly under ``logs/rsl_rl/lift_v2_8_5/`` next to the existing
``logs/rsl_rl/lift_v2_8/`` baseline.

All hyperparameters are identical to ``LiftCubePPORunnerCfgV28``:
``init_noise_std=1.0``, ``hidden_dims=[256, 128, 128]``,
``value_loss_coef=1.0``, ``use_clipped_value_loss=True``,
``entropy_coef=0.005``, ``num_learning_epochs=5``,
``num_mini_batches=4``, ``learning_rate=1e-3``,
``schedule="adaptive"``, ``desired_kl=0.01``, ``gamma=0.99``,
``lam=0.95``, ``max_grad_norm=1.0``, ``clip_param=0.2``,
``num_steps_per_env=50``.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgV285(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_v2_8_5"
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
