"""V2.12 PPO config — clone of V2.9 PPO, only experiment_name differs.

V2.12 reverts to the V2.9 PPO hyperparameters. The V2.9 → V2.12 delta
is purely on the **env side**: V2.12 switches the action class from
``JointPositionActionCfg`` (absolute control) to
``RelativeJointPositionActionCfg`` (delta control), which gives a hard
mechanical cap on per-step joint velocity (vmax = scale / dt_ctrl).

V2.10/V2.11 attempts at custom PPO configs (lower init_noise, lower
entropy_coef, etc.) were partly motivated by the chaotic policy V2.9
produced. With the action space now structurally bounded by delta
control, V2.9's hyperparams are appropriate again — they were the ones
that converged on the task.

`experiment_name = "lift_v2_12"` keeps logs separate from V2.9/V2.10/V2.11.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgV212(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_v2_12"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                                    # V2.9 default
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,                                    # V2.9 default
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        # V2.12 fix #4 — desired_kl 0.01 → 0.02 (Phase B safety margin).
        # With 256 envs × 50 steps / 4 minibatches = 3200 transitions per
        # mini-batch, KL estimates have high variance. desired_kl=0.01 is
        # tight enough that random KL spikes from a single bad mini-batch
        # could trigger the adaptive scheduler to halve the LR. V2.10
        # cold-start saw this collapse (LR went to 1e-5). Relaxing to
        # 0.02 gives the adaptive scheduler 2× tolerance for noise.
        desired_kl=0.02,
        max_grad_norm=1.0,
    )
