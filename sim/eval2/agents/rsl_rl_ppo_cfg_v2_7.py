"""V2.7 PPO config — V2.6 + relaxed adaptive KL target.

V2.6 used ``desired_kl=0.01`` (HW4 value), but in our massively parallel
regime (256 envs × 50 steps × 10 epochs / 32 mini-batches = ~4000 grad
steps per rollout vs HW4's ~20 with 1 env), KL drifts faster per rollout
than HW4's hyperparams expected. After 107 iterations on V2.6, the
adaptive schedule had divided the LR by ~15x (3e-4 → ~2e-5), starving
the policy of gradient and locking it in the flick-exploit local optimum.

Change vs V2.6:

+----------------+--------+--------+----------------------------------+
| Parameter      | V2.6   | V2.7   | Why                              |
+----------------+--------+--------+----------------------------------+
| ``desired_kl`` | 0.01   | 0.03   | Lets KL drift up to 0.06 per     |
|                |        |        | rollout before the LR shrinks.   |
|                |        |        | Keeps the adaptive safety net    |
|                |        |        | for true divergence but gives    |
|                |        |        | the policy room to actually      |
|                |        |        | learn during early training.     |
+----------------+--------+--------+----------------------------------+

V2.7 is paired with the V2.7 reward shaping
(``LeIsaacLiftCubeRLEnvCfgV27`` / ``LeIsaacLiftCubeRLVisualEnvCfgV27``)
which closes the flick exploit by gating ``lifting_object`` and the
goal-tracking terms on the grasp condition.

Everything else inherits from V2.6: ``init_noise_std=0.6``,
``hidden_dims=[256, 128, 128]``, ``value_loss_coef=0.01``,
``use_clipped_value_loss=True``, ``entropy_coef=0.005``,
``num_learning_epochs=10``, ``num_mini_batches=32``,
``learning_rate=3e-4``, ``schedule="adaptive"``, ``gamma=0.99``,
``lam=0.95``, ``max_grad_norm=0.5``, ``num_steps_per_env=50``.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgV27(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_v2_7"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.01,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=10,
        num_mini_batches=32,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.03,                     # V2.7: was 0.01 (HW4-too-tight)
        max_grad_norm=0.5,
    )
