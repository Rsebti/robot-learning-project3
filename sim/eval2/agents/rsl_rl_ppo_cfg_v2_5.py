"""V2.5 PPO config — V2 + HW4-inspired adjustments.

Built on the V2 baseline (lerobot-sim2real-derived hyperparameters that
fixed the entropy explosion of the original isaac_so_arm101 defaults) and
adds four changes from the **ETH Robot Learning HW4 SO-100 tracking
task** — a publicly validated PPO config that converged in 500
iterations to mean return 54.91 with EE tracking error 0.017 m on the
same robot family (SO-100 = SO-101 minus the gripper variant).

Changes vs V2:

+----------------------+--------+--------+-------------------------------+
| Parameter            | V2     | V2.5   | Why                           |
+----------------------+--------+--------+-------------------------------+
| ``value_loss_coef``  | 0.5    | 0.01   | When the value head is poorly |
|                      |        |        | trained early in PPO, a high  |
|                      |        |        | value coefficient dominates   |
|                      |        |        | the total loss and starves    |
|                      |        |        | the policy gradient. HW4      |
|                      |        |        | dropped it 50x and converged. |
+----------------------+--------+--------+-------------------------------+
| ``num_learning_     `` | 4    | 10     | More gradient updates per     |
| ``epochs``           |        |        | rollout. PPO clipping keeps   |
|                      |        |        | the policy from drifting too  |
|                      |        |        | far per epoch even with 10    |
|                      |        |        | passes — HW4 confirmed.       |
+----------------------+--------+--------+-------------------------------+
| ``gamma``            | 0.95   | 0.99   | Longer reward horizon. The    |
|                      |        |        | pick-place chain (reach →     |
|                      |        |        | grasp → lift → transport →    |
|                      |        |        | drop) spans 30–100 frames at  |
|                      |        |        | 50 Hz, gamma=0.95 underweighs |
|                      |        |        | the terminal success.         |
+----------------------+--------+--------+-------------------------------+
| ``hidden_dims``      | [256,  | [256,  | Slightly more capacity in the |
|                      |  128,  |  128,  | last layer to accommodate the |
|                      |   64]  |  128]  | wider Phase B obs (state +    |
|                      |        |        | 512-D ResNet features).       |
+----------------------+--------+--------+-------------------------------+

Everything else (``init_noise_std=0.6``, ``schedule="fixed"``,
``learning_rate=3e-4``, ``entropy_coef=0.002``, ``clip_param=0.2``,
``max_grad_norm=0.5``, ``num_steps_per_env=50``, ``num_mini_batches=32``)
inherits from V2.

Source: `ETH Robot Learning HW4
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
class LiftCubePPORunnerCfgV25(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_v2_5"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 128, 128],   # V2.5: was [256, 128, 64]
        critic_hidden_dims=[256, 128, 128],  # V2.5: was [256, 128, 64]
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.01,                # V2.5: was 0.5 (HW4-inspired, biggest impact)
        use_clipped_value_loss=False,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=10,              # V2.5: was 4 (HW4-inspired)
        num_mini_batches=32,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,                          # V2.5: was 0.95 (HW4-inspired)
        lam=0.95,
        desired_kl=0.2,                      # irrelevant under fixed schedule
        max_grad_norm=0.5,
    )
