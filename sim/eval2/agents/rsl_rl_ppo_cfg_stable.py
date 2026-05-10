"""Stable PPO config for SO-101 manipulation tasks.

Ports the working hyperparameters from `StoneT2000/lerobot-sim2real`'s
PPOArgs (validated to converge on SO-100 cube grasp in ~1h on RTX 4090, with
visual observations, in ManiSkill 3) into the rsl_rl format used by
`isaac_so_arm101`.

Why this differs from `LiftCubePPORunnerCfg` (the upstream default):

The upstream config (`isaac_so_arm101.tasks.lift.agents.rsl_rl_ppo_cfg`) was
copied from rsl_rl's legged-locomotion examples (ANYmal, etc.) where:
- horizons are long (gait cycles ~50+ steps)  -> gamma=0.98 makes sense
- exploration must stay broad indefinitely    -> entropy_coef=0.006 is a bonus
- KL adaptive scheduler stabilizes locomotion -> desired_kl=0.01 OK

For SO-101 manipulation those settings break PPO:
- short reward horizon (5-15 steps for pick-and-place) -> gamma=0.98 dilutes signal
- entropy bonus pushes std up monotonically once the surrogate plateaus
  -> we observed `Loss/entropy` going from 10 to 15.5 in 350 iters in our first
     run, while lifting reward peaked at 0.8 then collapsed to 0.13
- adaptive KL throttles LR from 5e-3 to 1e-3 -> learning paralysis

The lerobot-sim2real settings, ported here, fix all three:
- gamma=0.8 matches the actual reward horizon
- entropy_coef=0.0 removes the entropy bonus (std drops naturally)
- schedule="fixed" with LR=3e-4 keeps gradients alive
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgStable(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50              # was 24 — longer rollouts (matches sim2real)
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_stable"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,             # was 1.0 — start less noisy (~exp(-0.5))
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,            # was 1.0
        use_clipped_value_loss=False,   # was True — sim2real doesn't clip vloss
        clip_param=0.2,
        entropy_coef=0.0,               # KEY: was 0.006 — kill the entropy bonus
        num_learning_epochs=4,          # was 5
        num_mini_batches=32,            # was 4 — much more gradient updates per epoch
        learning_rate=3.0e-4,           # was 1e-4 — 3x higher
        schedule="fixed",               # KEY: was "adaptive" — no LR throttling
        gamma=0.8,                      # KEY: was 0.98 — short manipulation horizon
        lam=0.9,                        # was 0.95
        desired_kl=0.2,                 # was 0.01 (irrelevant under schedule=fixed)
        max_grad_norm=0.5,              # was 1.0
    )
