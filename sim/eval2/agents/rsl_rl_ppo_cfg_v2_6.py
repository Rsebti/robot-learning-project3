"""V2.6 PPO config — V2.5 + 3 HW4 alignments missed in V2.5.

V2.5 took the four "headline" HW4 changes (``value_loss_coef``,
``num_learning_epochs``, ``gamma``, ``hidden_dims``) but kept three V2
defaults that diverge from the validated HW4 PPO config. V2.6 closes the
gap.

Changes vs V2.5:

+----------------------------+--------+--------+-------------------------+
| Parameter                  | V2.5   | V2.6   | Why                     |
+----------------------------+--------+--------+-------------------------+
| ``entropy_coef``           | 0.002  | 0.005  | HW4 keeps a 2.5x larger |
|                            |        |        | entropy bonus → more    |
|                            |        |        | exploration. Phase B    |
|                            |        |        | plateaued because the   |
|                            |        |        | policy committed too    |
|                            |        |        | early to a sub-optimal  |
|                            |        |        | grasp pose.             |
+----------------------------+--------+--------+-------------------------+
| ``use_clipped_value_loss`` | False  | True   | HW4 ``compute_value_lo``|
|                            |        |        | ``ss`` takes ``max(uncl |
|                            |        |        | ipped, clipped)``. This |
|                            |        |        | bounds value updates    |
|                            |        |        | when the value head is  |
|                            |        |        | poorly trained early.   |
+----------------------------+--------+--------+-------------------------+
| ``schedule``               | fixed  |adaptive| HW4 adapts LR on KL:    |
| ``desired_kl``             | 0.2    | 0.01   | lr/=1.5 if kl>2·target, |
|                            |        |        | lr*=1.5 if kl<target/1.5|
|                            |        |        | Slows updates when the  |
|                            |        |        | policy drifts too far,  |
|                            |        |        | speeds them up when     |
|                            |        |        | conservative — built-in |
|                            |        |        | safety net for the      |
|                            |        |        | early-Phase-B regime.   |
+----------------------------+--------+--------+-------------------------+

Everything else inherits from V2.5: ``init_noise_std=0.6``,
``hidden_dims=[256, 128, 128]``, ``num_learning_epochs=10``,
``num_mini_batches=32``, ``learning_rate=3e-4``, ``clip_param=0.2``,
``max_grad_norm=0.5``, ``num_steps_per_env=50``, ``gamma=0.99``,
``lam=0.95``, ``value_loss_coef=0.01``.

Source: `ETH Robot Learning HW4 ex3_ppo_config.py
<C:\\Users\\user\\Desktop\\MA2\\Robot-Learning-ETHz\\hw4_reinforcement_learning\\exercises\\ex3_ppo_config.py>`_
— task was SO-100 EE tracking; convergence at 500 iter; mean return 54.91.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgV26(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_v2_6"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.01,
        use_clipped_value_loss=True,         # V2.6: was False (HW4-aligned)
        clip_param=0.2,
        entropy_coef=0.005,                  # V2.6: was 0.002 (HW4-aligned)
        num_learning_epochs=10,
        num_mini_batches=32,
        learning_rate=3.0e-4,
        schedule="adaptive",                 # V2.6: was "fixed" (HW4-aligned)
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,                     # V2.6: was 0.2 (HW4-aligned, target_kl)
        max_grad_norm=0.5,
    )
