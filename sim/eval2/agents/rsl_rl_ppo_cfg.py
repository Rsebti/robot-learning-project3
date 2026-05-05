"""rsl_rl PPO config for Eval 2 (mirrors the upstream Lift defaults)."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Eval2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "eval2_pick_in_bowl"
    # Obs normalization stays — helps the critic regardless of reward shape.
    empirical_normalization = True
    # clip_actions stays at 1.0 — caps action magnitude without penalty.
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        # v1.7: 0.5 — the middle ground between v1.5 (0.3, too narrow:
        # release never sampled) and v1.6 (1.0, too wide: std exploded to
        # 4.2 in 1800 iter). With the stage_progress reward providing a
        # MONOTONIC signal, the policy needs less random exploration to
        # find progress — each step closer to the next stage is rewarded.
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # v1.7-A: 0.0005 (was 0.003). With the wide reaching_target kernel
        # providing a strong dense signal, the entropy bonus is no longer
        # needed to drive exploration — and at 0.003 it was inflating the
        # std uncontrollably (1.24 -> 4.89 over 756 iter in v1.7). Cut by
        # 6x to let the policy commit when the reward gradient says so.
        entropy_coef=0.0005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
