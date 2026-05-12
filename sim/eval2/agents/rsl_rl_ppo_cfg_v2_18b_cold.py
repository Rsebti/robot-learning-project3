"""V2.18 b PPO config — anti-stall + anti-VF-blowup hot-fix.

Diagnosis of the V2.18 cold crash (run 2026-05-11_23-32-42, dead at iter 415):

    iter 50  : entropy 7.28, lr 4.4e-3, mean_reward -0.5, grasp_strict 0
    iter 200 : entropy 4.57, lr 1.3e-3, mean_reward +10, grasp_strict 0
    iter 350 : entropy 3.4,  lr 1e-5,   mean_reward +13.8, grasp_strict 0  (LR floored)
    iter 414 : entropy 3.0,  lr 1e-5,   mean_reward +13.8, vf 0.009         (last sane)
    iter 415 : entropy 3.0,  lr 1e-5,   mean_reward +14.0, vf 68            (×8000 jump)
    iter 416 : vf 1.33e9
    iter 417 : vf 2.27e19
    iter 418 : vf 6.06e29
    iter 419 : vf inf  → run dead

Root causes (per Claude search artifact `notes/v218b_ppo_search_claude.md`):
  1. `empirical_normalization=False` left the critic regressing against
     non-normalised 555-dim obs → unbounded VF gradients on advantage spikes
  2. `desired_kl=0.02` (vs rsl_rl reference 0.01) → adaptive KL controller
     accepted larger updates per iter → faster σ-collapse → more KL
     overshoots → asymmetric throttling crushed LR to 1e-5 (the rsl_rl
     hard-coded floor in ppo.py:282 — `max(1e-5, lr/1.5)`)
  3. `entropy_coef=0.005` (vs rsl_rl default 0.01) → entropy fell from
     7.3 → 3.0 in 365 iters → exploration dead before grasp_strict could
     fire (sparse 6-cond predicate, 0/0 in 414 iters)
  4. `use_clipped_value_loss=True` with rsl_rl's pessimistic
     `torch.max(unclipped², clipped²)` bound — doesn't cap |V̂|, allows
     bad-minibatch detonation cascade
  5. `noise_std_type="scalar"` shared one σ across 5 arm dims + 1 binary
     gripper dim — gripper σ collapsed with arm σ even though gripper
     contributed near-zero gradient signal until grasp_strict fired

V2.18 b changes (paired with hover_height weight 0.5→0.2 in the env cfg):

  Priority 1-3 (CRITICAL — prevent VF cascade):
    - empirical_normalization      : False → True
    - desired_kl                   : 0.02  → 0.01
    - learning_rate (init)         : 1e-3  → 3e-4

  Priority 4-6 (CRITICAL — keep exploration alive):
    - noise_std_type               : "scalar" → "log"  (per-dim σ)
    - entropy_coef                 : 0.005 → 0.01      (rsl_rl default)
    - use_clipped_value_loss       : True  → False     (let critic re-baseline)

  Priority 7-9 (ACCELERATE convergence):
    - rnd_cfg                      : None → weight 0.002, decay 0→800
    - critic_hidden_dims           : [256,128,128] → [512,256,128]
    - num_mini_batches             : 4 → 8 (+60% SGD steps/iter)
    - num_learning_epochs          : 5 → 4
    - num_steps_per_env            : 50 → 48 (divisible by 8)
    - max_iterations               : 1500 → 3000 (extended budget)

UPSTREAM RSL_RL PATCH (applied 2026-05-12):
    The rsl_rl 3.0.1 LR floor was hard-coded at 1e-5 in
    `.venv/Lib/site-packages/rsl_rl/algorithms/ppo.py:282`. We patched
    it to 1e-4 (artifact priority item #2). Combined with init lr 3e-4,
    this gives ~3 throttle halvings before hitting the new floor, after
    which the optimizer stays responsive instead of freezing.

    The patch is NOT in the venv's package manifest — re-apply after any
    venv rebuild via:
        python sim/eval2/scripts/patch_rsl_rl.py
    See notes/rsl_rl_patches.md for the full context.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
)


@configclass
class LiftCubePPORunnerCfgV218BCold(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48                          # V218: 50 → 48 (÷8)
    max_iterations = 3000                           # V218: 1500 → 3000
    save_interval = 50
    experiment_name = "lift_v2_13"
    empirical_normalization = True                  # V218: False → True (#1 fix)

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",                       # V218: "scalar" → "log" (#4)
        actor_hidden_dims=[256, 128, 128],          # unchanged
        critic_hidden_dims=[512, 256, 128],         # V218: [256,128,128] → wider (#8)
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=False,               # V218: True → False (#6)
        clip_param=0.2,
        entropy_coef=0.01,                          # V218: 0.005 → 0.01 (#5)
        num_learning_epochs=4,                      # V218: 5 → 4 (#9)
        num_mini_batches=8,                         # V218: 4 → 8 (#9)
        learning_rate=3.0e-4,                       # V218: 1e-3 → 3e-4 (#2)
        schedule="adaptive",                        # see "KNOWN UNFIXED RISK"
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,                            # V218: 0.02 → 0.01 (#3)
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        rnd_cfg=RslRlRndCfg(                        # V218: None → enabled (#7)
            weight=0.002,
            weight_schedule=RslRlRndCfg.StepWeightScheduleCfg(
                final_value=0.0,
                final_step=800,
            ),
            reward_normalization=True,
            num_outputs=1,
            learning_rate=1.0e-4,
            predictor_hidden_dims=[128, 128],
            target_hidden_dims=[128, 128],
        ),
    )
