"""V2.10 PPO config — smoothness-focused.

V2.9 reached ~32% success in training but the **deterministic** policy
(used at deploy) saturated joint velocities at 10 rad/s on every step.
The policy learned chaotic high-speed motion that occasionally succeeded
through stochastic luck during PPO sampling. The mean of the policy is
unusable.

Diagnosis from `play_diagnose.py` on V2.9 model_1499:
  - Mean |qdot|max per step ≈ 9.84 (joint limit is 10.00) — saturation
  - Max |qdot| spikes to 78 rad/s (impulsive collisions)
  - Cube velocities up to 5 m/s when the gripper hits it

V2.10 changes (vs V2.9):

+----------------------+--------+--------+----------------------------+
| Param                | V2.9   | V2.10  | Why                        |
+----------------------+--------+--------+----------------------------+
| ``init_noise_std``   | 1.0    | 0.4    | Smaller initial exploration|
|                      |        |        | so the deterministic mean  |
|                      |        |        | is closer to actual policy.|
+----------------------+--------+--------+----------------------------+
| ``entropy_coef``     | 0.005  | 0.002  | Less reward for randomness;|
|                      |        |        | force commitment.          |
+----------------------+--------+--------+----------------------------+

Other PPO params unchanged (Isaac Lab canonical: value_loss_coef=1.0,
n_epochs=5, n_mini=4, LR=1e-3, schedule=adaptive, desired_kl=0.01,
gamma=0.99, max_grad_norm=1.0).

Most of the smoothness fix happens **on the env side** (V2.10 env):
  - ``action_rate_l2`` weight: -1e-4 → -5e-2 (×500)
  - ``joint_vel_l2`` weight: -1e-4 → -1e-2 (×100)
  - ``JointPositionAction.scale``: 0.5 → 0.25 (joint deltas halved)

The combined effect is a strong incentive to move slowly and smoothly.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgV210(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_v2_10"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.4,                              # V2.10: was 1.0
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,                              # V2.10: was 0.005
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
