"""PPO hyperparameters for the Squint-port Lift and Place tasks.

Squint itself uses SAC+C51 — incompatible with rsl_rl. We use PPO and
borrow Squint's *training spirit* through hyperparameter choices that
matter for short-horizon manipulation:

- **gamma = 0.95**: Squint uses 0.9 because the SO-101 Lift/Place
  episode is ~15-30 control steps. With our 5-7 s episode at
  decimation=2 (60 Hz physics → 30 Hz control), an episode is ~150-210
  steps. Slightly higher gamma gives a longer effective horizon while
  still avoiding the locomotion-style 0.99 trap.
- **fixed LR = 3e-4** (no adaptive KL throttling): matches Squint and
  prevents the LR-collapse cascade observed in V2.18 cold.
- **entropy_coef = 0.0**: Squint relies on SAC's autotuned entropy.
  PPO equivalent is "kill the entropy bonus so std settles
  naturally" — same effect, prevents the "Loss/entropy ↑ → reward ↓"
  divergence observed in our V2.0 run.
- **clip_param = 0.2, num_mini_batches = 32**: lerobot-sim2real defaults
  validated on SO-100 (close cousin of SO-101) cube grasp in <1h.

Network width is intentionally modest (256-128-64) since we use
state-based obs (~25 dims, no image). For a visual variant later,
upgrade actor/critic to mirror Squint's encoder (CNN → 1024 features
→ MLP).
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SquintLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for Squint-port Lift task."""

    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 50
    experiment_name = "squint_lift"
    empirical_normalization = True  # important when obs scales vary

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=False,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=4,
        num_mini_batches=32,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.2,
        max_grad_norm=0.5,
    )


@configclass
class SquintPlacePPORunnerCfg(SquintLiftPPORunnerCfg):
    """PPO config for Squint-port Place task.

    Same hyperparameters as Lift (the task topology is similar) but
    longer rollouts (50 vs 32 steps per env) because Place needs the
    policy to commit to a multi-stage trajectory (lift → translate →
    descend) before any positive reward signal beyond the grasp bonus
    appears.
    """

    num_steps_per_env = 50
    experiment_name = "squint_place"
