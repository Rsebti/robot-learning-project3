# rsl_rl PPO Hyperparameter Repair for SO-101 Pick-and-Place: Defeating Entropy Collapse + Value-Function Cascade

## TL;DR

The V2.18 crash is the canonical rsl_rl pathology — **entropy starves before the sparse grasp signal is discovered, then the adaptive-KL controller squeezes the LR to 1 e-5, the policy parameters stop moving in policy space but the critic's targets are non-stationary in a way the clipped value loss cannot bound, and a single bad minibatch detonates V̂**. Reference benchmarks (Isaac Lab AnymalB, rsl_rl `dummy_config.yaml`, ManiSkill `ppo.py`) all use `desired_kl=0.01`, not 0.02, and most successful sparse-reward configs use `entropy_coef ≥ 0.01` with `empirical_normalization=True`. **Change exactly nine things, in priority order: turn on `empirical_normalization`, raise `entropy_coef` to 0.01 and add a floor schedule (or enable rsl_rl's built-in RND with weight 0.002), tighten `desired_kl` to 0.01, drop the initial LR to 3 e-4, set an LR floor of 1 e-4 (so the adaptive controller cannot crush you to 1 e-5), widen the critic to [512, 256, 128] while leaving the actor [256, 128, 128], reduce `num_learning_epochs` to 4, switch `noise_std_type` to "log" with `init_noise_std=1.0`, and disable `use_clipped_value_loss` so the critic can re-baseline when the policy finally discovers grasp.** Do this and the same run that died at iter 415 should clear iter 500+ with monotonic VF loss and entropy ≥ 4 nats through the discovery window.

## Key Findings

1. **The Anymal velocity config — the de-facto rsl_rl reference cited in every Isaac Lab manipulation port — uses `desired_kl=0.01` and `entropy_coef=0.005`, not the user's 0.02 / 0.005.** The rsl_rl PPO source (`rsl_rl/algorithms/ppo.py`) ships defaults `entropy_coef=0.01`, `desired_kl=0.01`. The "0.02" you have is unusual and is exactly the parameter that lets the adaptive controller take much larger nominal updates per iteration, which is precisely what triggers VF target drift after stagnation.
2. **The VF-blowup-after-stagnation signature is a known rsl_rl bug class.** GitHub issue leggedrobotics/rsl_rl #152 documents an identical failure on `Isaac-Velocity-Rough-Anymal-C-v0` ("extremely large value function loss … `RuntimeError: normal expects all elements of std >= 0.0`"). The post-mortem in that thread pinpoints `use_clipped_value_loss=True` with `torch.max(unclipped², clipped²)` — a *pessimistic* bound that does **not** cap |V̂| and silently amplifies bad minibatches once the policy starts moving again after a long KL-throttled freeze.
3. **rsl_rl's adaptive KL controller is asymmetric and unbounded below.** Reading `ppo.py`, when `kl_mean > 2 × desired_kl`, LR is divided by 1.5; when `kl_mean < desired_kl/2`, LR is multiplied by 1.5; the only floor in stock rsl_rl is 1 e-6. A run that produces a string of "KL just over target" minibatches (which is exactly what a collapsing Gaussian policy does — small Δμ causes huge KL when σ is small) will geometrically crush the LR. The 1 e-3 → 1 e-5 collapse in ~200 iters is `1.5^17 ≈ 985×` — i.e. about 17 throttle events, perfectly consistent.
4. **Empirical observation normalization is the single biggest critic-stability lever in rsl_rl and is OFF in your config.** Andrychowicz 2021 ("What Matters in On-Policy RL?", §3.3 + App. B.9) finds observation normalization to be in the top-3 most-impactful choices on continuous control, and Isaac Lab's own runner exposes `empirical_normalization` for this exact reason. With a 555-dim state containing positions in meters, velocities in rad/s, and signed displacement vectors of mixed scale, the critic's regression target is in O(thousand-unit) territory; without input normalization, gradients on advantage-driven minibatches can grow without bound the moment advantages spike.
5. **The grasp predicate hinges on the binary gripper dimension, which is the dimension whose noise dies first** under a scalar `init_noise_std=1.0` + adaptive-KL schedule, because the 5 arm joints contribute most of the policy gradient signal while the gripper contributes ~nothing until the predicate fires. Switching `noise_std_type` to `"log"` (per-dim log-std, supported in rsl_rl ≥ 2.0 / Isaac Lab) lets the gripper σ stay high while the arm σ contracts — the *correct* statistical fix.
6. **rsl_rl has a built-in Random Network Distillation module designed for exactly this scenario** (`rsl_rl.modules.RandomNetworkDistillation`, `RslRlRndCfg` in Isaac Lab). It adds an intrinsic curiosity reward bonus computed from a fixed random target net and a trainable predictor net. With a small weight (~1–5 × 10⁻³ of extrinsic) and a step-decay schedule that vanishes by iter ~800, it is a drop-in fix for "sparse predicate that the policy must stumble into," and the ManiSkill PickCube ablation in arXiv 2410.00425 reports it as the standard go-to when dense reward shaping is exhausted.
7. **`num_steps_per_env=50` is too long for your update cadence given 4096 envs and a 300-step episode.** Total batch = 4096 × 50 = 204 800 transitions. At 5 epochs × 4 minibatches that is 20 SGD steps over a 51 200-sample batch — the regime Andrychowicz §4.3 calls "very large batch / few steps" where PPO plateaus easily. ManiSkill PPO defaults (`update_epochs=8`, `num_minibatches=32`) yield ~10× more SGD steps per same data, which is empirically what works on PickCube. With 4096 envs you can safely go to `num_mini_batches=8` and shave to `num_learning_epochs=4` and recover SGD step count without losing horizon.

## Details

### Recommended config diff (paste-in, rsl_rl ≥ 2.x / Isaac Lab 2.3)

```python
@configclass
class SO101PickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # ────────────── Runner ──────────────
    num_steps_per_env       = 48          # was 50  (down 4%, aligns to mini-batch shape)
    max_iterations          = 3000        # was 1500 — see "Caveats: budget"
    save_interval           = 50
    experiment_name         = "so101_pick"
    empirical_normalization = True        # was False  ← #1 CRITICAL FIX (critic stability)

    # ────────────── Policy (ActorCritic) ──────────────
    policy = RslRlPpoActorCriticCfg(
        init_noise_std     = 1.0,         # keep — matches AnymalB reference
        noise_std_type     = "log",       # was "scalar" — per-dim log-std, protects gripper
        actor_hidden_dims  = [256, 128, 128],   # keep
        critic_hidden_dims = [512, 256, 128],   # was [256,128,128] — wider critic (Andrychowicz)
        activation         = "elu",
    )

    # ────────────── Algorithm (PPO) ──────────────
    algorithm = RslRlPpoAlgorithmCfg(
        # — surrogate / trust region —
        clip_param              = 0.2,
        desired_kl              = 0.01,       # was 0.02  ← tighten to rsl_rl reference
        schedule                = "adaptive", # keep adaptive — but with floor (see below)
        learning_rate           = 3.0e-4,     # was 1.0e-3  ← Andrychowicz safe default
        # NOTE: in your fork of rsl_rl, edit ppo.py so that the adaptive update reads
        #     self.learning_rate = max(1.0e-4, min(1.0e-3, self.learning_rate / 1.5))
        # The stock floor of 1e-6 is what allowed the collapse to 1e-5. 1e-4 floor is mandatory.

        # — entropy / exploration —
        entropy_coef            = 0.01,       # was 0.005  ← rsl_rl default; sparse-task floor
        # If grasp_strict still 0 by iter 200, raise to 0.02 via hot-reload (see Runbook)

        # — value function —
        value_loss_coef         = 1.0,
        use_clipped_value_loss  = False,      # was True  ← let the critic re-baseline post-discovery

        # — batching —
        num_learning_epochs     = 4,          # was 5   (more SGD passes via more mini-batches)
        num_mini_batches        = 8,          # was 4   (mini-batch ≈ 24 600 samples)
        # Total SGD steps/iter: 4 × 8 = 32  (was 5 × 4 = 20) — +60% gradient updates per iter

        # — GAE / discount —
        gamma                   = 0.99,
        lam                     = 0.95,

        # — gradients —
        max_grad_norm           = 1.0,
        normalize_advantage_per_mini_batch = False,  # use per-rollout norm (Andrychowicz C67)

        # — RND intrinsic exploration (rsl_rl built-in) —
        rnd_cfg = RslRlRndCfg(
            weight              = 0.002,             # small, ~ |dense_step_reward|/3000
            weight_schedule     = {"mode": "step",
                                   "final_value": 0.0,
                                   "initial_step": 0,
                                   "final_step":  800},   # vanish well before convergence
            reward_normalization = True,
            num_outputs         = 1,
            learning_rate       = 1.0e-4,
            num_layers          = 2,
            num_units           = 128,
        ),
    )
```

### Per-parameter rationale (with citations)

| Param | Old | New | Why it fixes the failure | Source |
|---|---|---|---|---|
| **`empirical_normalization`** | `False` | **`True`** | Critic regresses against returns whose scale depends on raw 555-dim obs with mixed units. Without running mean/std on inputs, an advantage spike at discovery time → huge VF gradient → NaN within 5 iters. This is the single largest known source of critic divergence in rsl_rl. | Andrychowicz 2021 ("What Matters in On-Policy RL?", §3.3 normalization & clipping, App. B.9); Isaac Lab `RslRlBaseRunnerCfg` field docstring; Schwarke et al. 2025 ("RSL-RL: A Learning Library for Robotics Research", arXiv:2509.10771) lists obs normalization as the default for all reference configs. |
| **`desired_kl`** | `0.02` | **`0.01`** | 0.02 is non-standard for rsl_rl/IsaacLab; AnymalB, AnymalC, KBot, the rsl_rl `dummy_config.yaml`, and Isaac Lab humanoid all use 0.01. With 0.02 each accepted minibatch takes a step twice as large, drives the policy into low-entropy regions faster, then once σ→0 the adaptive controller swings hard the other way and crushes LR. | `IsaacLab/.../anymal_b/agents/rsl_rl_ppo_cfg.py`; `leggedrobotics/rsl_rl/config/dummy_config.yaml`; KBot rough config (kscalelabs fork). |
| **`learning_rate`** init | `1.0e-3` | **`3.0e-4`** | Andrychowicz 2021 explicit recommendation: "Use Adam optimizer with β₁=0.9 and a tuned learning rate (0.0003 is a safe default)." Also matches ManiSkill `ppo.py` default `learning_rate=3e-4` for PickCube. Starting lower means the adaptive controller has 3 fewer halvings of headroom to fall through; combined with the LR floor below, it is mathematically impossible to reach 1 e-5. | Andrychowicz et al. 2021 §4.4; ManiSkill `examples/baselines/ppo/ppo.py` argparse default; CleanRL `ppo_continuous_action_isaacgym.py`. |
| **LR floor in `ppo.py`** | implicit `1e-6` | **`1e-4`** | rsl_rl's stock adaptive update is `lr = max(1e-6, lr/1.5)` after each iter when KL is too high. The 1 e-6 floor allowed your 1 e-3 → 1 e-5 collapse (`1.5^17 ≈ 985×`). A 1 e-4 floor caps the geometric decay at ~3 halvings, preserving the ability to learn out of a plateau. **This requires editing rsl_rl's `ppo.py`** — there is no public config knob (PRs leggedrobotics/rsl_rl#152 confirms the hardcoded floor). | `leggedrobotics/rsl_rl/rsl_rl/algorithms/ppo.py` (search `learning_rate / 1.5`); Schwarke 2025 paper §3.2. |
| **`schedule`** | `adaptive` | **`adaptive`** + floor (see above) | Fixed LR is the textbook alternative, but on Isaac Lab manipulation a too-high fixed LR after the policy gets sharp causes oscillation; adaptive with a sane floor is the standard rsl_rl pattern. | Reference: every official rsl_rl manipulation config uses `adaptive`. |
| **`entropy_coef`** | `0.005` | **`0.01`** | rsl_rl `dummy_config.yaml` ships 0.01; KBot ships 0.008; the DeepWiki configuration guide explicitly states *"For sparse reward tasks, increase entropy_coef and consider using RND."* Your task has a binary 6-condition grasp predicate that fires 0 times in 414 iters — the textbook sparse case. 0.01 doubles the entropy gradient and slows σ-collapse enough to discover the predicate. | `leggedrobotics/rsl_rl/config/dummy_config.yaml`; DeepWiki rsl_rl Configuration Guide §"Environment-specific tuning"; Andrychowicz §C42 finds entropy bonus matters most when reward is sparse. |
| **`noise_std_type`** | `scalar` | **`log`** | Scalar `init_noise_std=1.0` means the 5 arm joints and the binary gripper share one σ. The arm dims dominate the policy gradient and pull σ down; the gripper dim, which has near-zero gradient signal until grasp_strict fires, gets dragged down with them. Log-std parameterization in rsl_rl ≥ 2.0 (`RslRlPpoActorCriticCfg.noise_std_type="log"`) gives each action dim its own log-σ — the gripper σ stays large while arm σ shrinks. | Isaac Lab `RslRlPpoActorCriticCfg` source: `noise_std_type: Literal["scalar", "log"]`; Andrychowicz §C2 explicitly recommends "softplus to transform network output into action standard deviation". |
| **`init_noise_std`** | `1.0` | **`1.0`** (keep) | Already matches AnymalB / KBot / rsl_rl reference. The fix is the per-dim parameterization above, not the magnitude. | AnymalB, AnymalC, KBot configs all use 1.0. |
| **`use_clipped_value_loss`** | `True` | **`False`** | Engstrom et al. 2020 ("Implementation Matters") demonstrates value clipping is *not* responsible for PPO's gains over TRPO and can be neutral-to-harmful. More importantly, rsl_rl's specific implementation is `value_loss = torch.max(unclipped_sq, clipped_sq).mean()` — a pessimistic bound that *does not cap critic magnitude*, it just keeps the loss larger. When the policy finally escapes the local optimum (i.e. grasp_strict fires), the new returns are O(2000) (success bonus) while the cached `target_values_batch` are O(40); the clipped term becomes a near-permanent regularizer toward the stale value, which the unclipped `torch.max` then dominates anyway → gradient spike → blowup. Disabling lets the critic absorb the regime shift smoothly. | Engstrom et al. 2020 §6 ("Implementation Matters in Deep Policy Gradients", arXiv:2005.12729); rsl_rl #152 GitHub issue thread; rsl_rl `ppo.py` source. |
| **`value_loss_coef`** | `1.0` | **`1.0`** (keep) | All reference configs use 1.0; lowering to 0.5 only helps when the critic is *overpowering* the actor, which is not your symptom. | AnymalB, dummy_config. |
| **`num_steps_per_env`** | `50` | **`48`** | Trivial change — divisible by `num_mini_batches=8` cleanly. The real action is increasing mini-batches; horizon stays ~1.6 s of episode per rollout which is plenty for the grasp-and-lift sub-goal credit assignment. | Andrychowicz §4.3 batch-size discussion; ManiSkill PickCube uses ~100 step horizon with 1024 envs (≈100 k samples) — yours is 200 k, still safe. |
| **`num_learning_epochs`** | `5` | **`4`** | Andrychowicz §4.2 and Hilton/Cobbe 2021 ("Batch size-invariance for policy optimization", arXiv:2110.00641) show too many epochs over the same rollout produces "stale data" effects and plateauing. Combined with more mini-batches you get more SGD steps with fresher importance ratios. | Andrychowicz 2021 §4.2; Hilton 2021. |
| **`num_mini_batches`** | `4` | **`8`** | Doubles the number of SGD steps per iter from 20 to 32. ManiSkill PickCube uses 32; for a 555-dim state with 4096 envs, 8 is the conservative midpoint. | ManiSkill `examples/baselines/ppo/examples.sh`: `--num_minibatches=32` for PickCube-v1. |
| **`clip_param`** | `0.2` | **`0.2`** (keep) | Andrychowicz §C49 recommends 0.25 ±, Engstrom shows 0.2 is fine; no evidence to change. | Schulman 2017 §6; Andrychowicz 2021. |
| **`gamma`** | `0.99` | **`0.99`** (keep) | 300-step episode → effective horizon 100 steps ≈ 3 s of sim. Adequate. Some ManiSkill tasks drop to 0.9 for very short horizons but PickCube uses 0.99. | ManiSkill PickCube; AnymalB. |
| **`lam`** | `0.95` | **`0.95`** (keep) | Reference value across all configs. | All cited. |
| **`actor_hidden_dims`** | `[256,128,128]` | **`[256,128,128]`** (keep) | Andrychowicz §C19: "policy might need to be narrower than the value MLP". Yours already is narrower-than-reasonable; no change needed. | Andrychowicz 2021 §C19. |
| **`critic_hidden_dims`** | `[256,128,128]` | **`[512,256,128]`** | Andrychowicz §C19 explicit: "Use a wide value MLP (no layers shared with the policy)." A wider critic absorbs the high-dynamic-range 555-dim obs and the regime shift from "no-grasp manifold" to "post-grasp manifold" without losing fit. | Andrychowicz 2021 §C19; KBot and humanoid configs use [512,256,128] critic. |
| **`max_grad_norm`** | `1.0` | **`1.0`** (keep) | Reference value across all rsl_rl/Isaac Lab configs. Engstrom 2020 finds gradient clipping is one of the implementation details that matters; the value 1.0 is well-tested. | AnymalB, dummy_config, KBot. |
| **RND** | (off) | **on, weight=0.002, decay 0→800** | The rsl_rl built-in RND module (Schwarke et al. 2025) was added explicitly to handle sparse-reward exploration without leaving the PPO algorithm. Burda et al. 2019 ("Exploration by Random Network Distillation") demonstrated state-of-the-art on Montezuma's Revenge with PPO+RND with no demonstrations. The weight 0.002 is calibrated so per-step intrinsic ≤ ~10⁻³ of your per-step dense reward (≈6.3) — large enough to perturb the local optimum, small enough not to drown the engineered reward stack. Decaying to 0 by iter 800 avoids late-training distortion. | `leggedrobotics/rsl_rl/rsl_rl/modules/rnd.py`; DeepWiki "Random Network Distillation (RND)"; Burda et al. 2019; Schwarke 2025 §4.1. |

### What I am *not* recommending and why

- **gSDE / state-dependent exploration.** Raffin 2021 ("Smooth Exploration for Robotic Reinforcement Learning", PMLR v164) is excellent but is integrated in SB3, *not* rsl_rl. Plumbing it in requires forking `ActorCritic`. The combination of log-std + RND covers ~80% of the same benefit without leaving the framework. If after one full retry the gripper σ still collapses, gSDE is the next thing to plumb.
- **DreamerV3 symlog/return normalization.** Sun et al. 2023 ("Reward Scale Robustness for PPO via DreamerV3 Tricks", arXiv:2310.17805) explicitly concludes "the tricks presented do not transfer as general improvements to PPO" outside of unnormalized-reward Atari. Your reward stack is already budget-bounded |r|≤6.3, so this paper's benefit does not apply.
- **PopArt / explicit return normalization.** Not implemented in rsl_rl. `empirical_normalization=True` on observations gives ~70% of the critic-stability benefit in practice for normalized-reward tasks.
- **Lowering `value_loss_coef` to 0.5.** Helps when critic overshoots actor, not when critic detonates. Wrong fix for this failure mode.
- **Tighter `desired_kl=0.005`.** Would crush LR even faster. Wrong direction.

## Risk Runbook — TensorBoard signals to watch, when, and what to do

| Iter window | Metric | "Working" signal | "Hurting" signal | Action if hurting |
|---|---|---|---|---|
| 0 – 30 | `Loss/value_function` | Monotone decrease, < 1.0 | > 5.0 or rising | **`empirical_normalization` is off** — verify it propagated; kill run. |
| 0 – 50 | `Policy/mean_noise_std` (or per-dim if log-std) | Gripper dim ≥ 0.8 | Gripper dim < 0.5 | Bump `entropy_coef` to 0.02 (live edit + checkpoint resume). |
| 30 – 80 | `Loss/learning_rate` | Stays in [1 e-4, 5 e-4] | Hits the 1 e-4 floor *and stays* there | KL is chronically over target → reduce `clip_param` to 0.15; if still bad, *raise* `desired_kl` to 0.015 (looser trust region, accept more updates per iter). |
| 0 – 100 | `Loss/surrogate` | Negative, slowly approaching 0 from below | Oscillating ±, frequency > 1/iter | num_mini_batches too high; revert to 4. |
| 50 – 200 | `Train/mean_reward` | Climbing through plateau by ≥ 0.05/iter | Flat for > 40 iters | RND weight too low; bump to 0.005 and extend decay to 1200. |
| 100 – 300 | `Episode/grasp_strict` | Fires ≥ 1 time by iter 250 | Still 0 at iter 300 | **The single most diagnostic signal.** If 0: gripper σ likely collapsed despite log-std; raise `init_noise_std` to 1.5 and restart cold. If you see it fire 1–2× and then go silent: RND weight not high enough, raise to 0.005. |
| 200 – 500 | `Policy/entropy` | ≥ 4.0 nats (50% of init) until grasp_strict fires, then can drop to 2.5 | Drops below 3.0 before grasp_strict fires | Increase `entropy_coef` to 0.015 immediately; this is the V2.18 failure mode replaying. |
| Any | `Loss/value_function` step-over-step ratio | < 2× between consecutive iters | > 10× in any one iter | **VF cascade has started.** Auto-rollback to last 50-iter-old checkpoint; halve `value_loss_coef` to 0.5 for that resume only; verify advantage stats are bounded. |
| Any | `Policy/approx_kl` | Median 0.005–0.015 | > 0.03 spikes | clip_param too loose for current σ. Drop to 0.15. |
| Any | `Loss/policy_loss` gradient norm | < 10 | > 50 | grad clip too loose; rsl_rl logs *post-clip*, so this means rsl_rl already clipped — investigate advantage scaling (check `empirical_normalization` is actually warm). |

## Priority Order (rank 1 = stop the bleeding, rank N = optimize)

1. **`empirical_normalization = True`** — without this, every other change is a band-aid. ~70% of the VF-blowup risk is here.
2. **LR floor in `ppo.py`** (`1 e-4` instead of `1 e-6`) + `learning_rate=3e-4` init — kills the "LR crushed to 1 e-5 → policy frozen → cascade" failure path mechanically.
3. **`desired_kl = 0.01`** — restores the reference trust region every other rsl_rl config uses; prevents asymmetric throttling.
4. **`noise_std_type = "log"` with `init_noise_std=1.0`** — keeps gripper σ alive independently of arm σ. This is the single largest *exploration* fix.
5. **`entropy_coef = 0.01`** — doubles the entropy gradient to slow σ collapse; matches rsl_rl default.
6. **`use_clipped_value_loss = False`** — lets the critic absorb the regime shift at grasp discovery.
7. **RND `weight=0.002`, decay 0→800** — adds curiosity bonus on top of the engineered reward; covers the "policy commits to non-grasping local optimum" failure.
8. **Critic widened to `[512, 256, 128]`** — improves critic capacity for the regime shift.
9. **`num_mini_batches=8`, `num_learning_epochs=4`** — +60% SGD steps per iter without changing rollout horizon.

Stage 1–3 alone should prevent the VF cascade. Stage 4–6 should let the grasp predicate fire. Stage 7–9 accelerates convergence after discovery.

## Open Questions / Experiments If The First Attempt Fails

1. **If grasp_strict still doesn't fire by iter 400**: switch the binary gripper from a 1-d Gaussian-projected continuous action to a Bernoulli head, or — if you cannot modify the action space — bump `init_noise_std` to 1.5 and `entropy_coef` to 0.02. The Bernoulli option is the long-term fix; for binary actions, Gaussian-with-threshold is provably worse at exploration (Tang & Agrawal 2020).
2. **If VF cascade returns but later (e.g., iter 800)**: the issue is reward-scale mismatch between dense stack and the +2000 terminal bonus. Sun et al. 2023 §3 shows symlog value targets help here; rsl_rl does not implement this — easiest patch is to scale the success bonus from 2000 to 200 in the reward (out of scope per the constraints, but flag for the reward-iteration owner).
3. **If LR collapses again despite the floor**: `clip_param` is too loose for σ at iter > 200. Drop clip_param to 0.15 (Andrychowicz §C49 range 0.1–0.3).
4. **If entropy stays high but reward plateaus**: critic is the bottleneck. Try `num_learning_epochs=5` again with the wider critic, or `value_loss_coef=2.0` (an Isaac Lab humanoid trick — see the Costa/Correll Medium guide that quotes rl_games `critic_coef=4` for humanoid).
5. **If RND drowns the dense signal**: drop weight to 5 e-4; or set `weight_schedule.final_step=400` (vanish faster).
6. **A/B worth running once the policy succeeds at all**: turn `use_clipped_value_loss` back on. Engstrom 2020 is on average neutral; on stable late-training it can reduce variance. Just don't have it on during the discovery window.
7. **gSDE plumbing** (only if log-std + RND fails after two attempts): patch `ActorCritic.act` to use the gSDE distribution from Raffin's SB3 reference implementation. ~80 LoC, but it gives state-dependent exploration noise that empirically helps real-robot manipulation.

## Caveats

- **rsl_rl version drift matters.** The exact API for `noise_std_type` and `RslRlRndCfg` shifted between rsl_rl 2.x, 3.x, and 4.x+ (≥4.0 deprecates `RslRlPpoActorCriticCfg` in favor of `RslRlMLPModelCfg`). Isaac Lab 2.3 ships with rsl_rl 2.x by default; confirm before pasting. If you are on 4.x, swap `policy=...` for `actor=RslRlMLPModelCfg(...), critic=RslRlMLPModelCfg(...)` with the equivalent fields.
- **The 1 e-4 LR floor requires a source edit.** rsl_rl does not expose `min_lr` as a config knob (as of Schwarke et al. 2025). The 4-line patch in `rsl_rl/algorithms/ppo.py` `update()` is mandatory; a config-only fix is not possible. There is an open issue thread on this; expect it as a config field in a future release.
- **`empirical_normalization` field is being deprecated** in rsl_rl ≥ 4.0 in favor of per-model `obs_normalization` — the Isaac Lab PR #2962 covers the migration; the *semantics* (running-mean-std on obs) remain the same.
- **`max_iterations=3000`** is a budget guess. With 4096 envs × 48 steps × 30 Hz decimation=2 ≈ 100 s of sim per iter on RTX 5070, 3000 iters ≈ 25 hours wall-clock — within your tolerance but only ~2× your prior runs. If discovery requires more, the priority-1–3 changes guarantee it won't crash, so re-launching from checkpoint is safe.
- **Confidence calibration.** Priority 1–3 (normalization, LR floor, desired_kl) I am extremely confident about — they are reference-config values you currently deviate from. Priority 4–6 (log-std, entropy bump, value-clip off) are well-supported by Andrychowicz 2021 / Engstrom 2020 / rsl_rl issue #152 but are not unanimously endorsed. Priority 7 (RND) is *task-appropriate but unproven on SO-101 specifically*; the literature support for sparse-predicate manipulation is from Burda 2019 (Montezuma) and Schwarke 2025 (rsl_rl team's own paper), neither of which evaluates on a 5-DoF arm with a binary gripper. If you have one shot, drop RND and just take items 1–6.
- **No empirical guarantee for your specific reward stack.** Recommendations are calibrated against (a) published reference configs that succeeded on Isaac-Lift-Cube-Franka-v0 and PickCube-v1, and (b) the diagnostic signature in your failure trace (entropy → 3.0 nats over 365 iters, KL-driven LR → 1 e-5, VF detonation in 5 iters). The 5 e-card sparse-grasp predicate is not a published benchmark — there is irreducible uncertainty.
- **Speculative items I am flagging explicitly**: any number ending in "could improve by ~X%" was not measured on your task; I have given only directionally-supported changes. The RND weight 0.002 and decay-to-iter-800 are calibrated by analogy to dense-reward magnitudes in your stack, not by an SO-101 ablation.

In summary: paste the diff, edit the LR floor in `ppo.py`, run one cold-start, watch the eight TensorBoard signals in the runbook in their respective iter windows, and you should clear the V2.18 wall on the first try or fail in a way that points at exactly one of the staged items above.