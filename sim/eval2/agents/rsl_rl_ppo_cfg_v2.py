"""V2 PPO config — fixes the local optimum issue observed with the Stable config.

Run dated 2026-05-08 (lift_stable, iter 919/1500): the Stable config (ported
from lerobot-sim2real) successfully prevented the entropy-explosion divergence
of the original isaac_so_arm101 defaults, but the policy converged to a local
optimum where it only does `reaching_object` (reward 0.43) and never lifts
(`lifting_object` reward stuck at 0 from iter ~400 onwards).

Two factors caused this local optimum:

1. **Curriculum activated at iter ~417** and ramped `action_rate` /
   `joint_vel` penalty weights from -1e-4 to -1e-1 (×1000 stronger).
   With aggressive lift motions now heavily punished, the policy chose
   "approach and freeze" as its safe option.
   -> Fixed in the env config (V2EnvCfg disables the curriculum).

2. **`gamma = 0.8`** discounts the success_bonus too much. With γ⁵ ≈ 0.33,
   the terminal reward at step 5 is worth only 1/3 of its face value, less
   than the dense `reaching_object` reward integrated over the whole episode.
   The policy has no incentive to commit to lifting.
   -> Bump to gamma = 0.95 (γ⁵ ≈ 0.77, terminal reward stays competitive).

3. **`entropy_coef = 0.0`** removes all exploration once the policy starts
   exploiting. Combined with the curriculum kicking in mid-training, the
   policy has no way to escape the local optimum.
   -> Re-introduce a small `entropy_coef = 0.002` (much less than the
   original 0.006 that caused entropy explosion, but enough to keep some
   stochasticity for escape).

Everything else stays as in the Stable config (longer rollouts, fixed LR
schedule, more mini-batches, reduced init noise).
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgV2(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_v2"
    empirical_normalization = False

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
        entropy_coef=0.002,             # was 0.0 — keep some exploration to escape local optima
        num_learning_epochs=4,
        num_mini_batches=32,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.95,                     # was 0.8 — terminal reward must stay competitive vs dense reaching
        lam=0.95,                       # bumped from 0.9 to keep advantage estimates stable with longer horizon
        desired_kl=0.2,
        max_grad_norm=0.5,
    )
