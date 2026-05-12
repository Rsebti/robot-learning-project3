# V2.18b architecture — reward + PPO + geometry

Living reference for the active variant as of 2026-05-12.

This doc replaces / supersedes:
- `v218_design_claude_search.md` (original V2.18 design — superseded by
  the in-flight calibration fixes documented here)
- `v218b_ppo_search_claude.md` (Claude search PPO recommendations —
  fully integrated below)
- `eval2_v215_architecture.md` (older V2.15 doc — kept for history)

For the variant chronicle V2 → V2.18 → V2.18b and dead-ends, see
`eval2_pipeline.md`.

---

## 1. Why V2.18b exists

### V2.18 cold-start failure (2026-05-11)
- Trained 414 iterations, then VF blowup (0.009 → ∞ in 5 iters)
- 0 grasp_strict events in 414 iters → hover-stall confirmed
- Mean reward plateaued at +14 from iter 350
- Root causes:
  1. `empirical_normalization=False` left critic exposed to non-normalized 555-dim obs → cascade detonation possible
  2. `desired_kl=0.02` (vs rsl_rl reference 0.01) → larger updates, faster σ-collapse, asymmetric LR throttling
  3. `entropy_coef=0.005` (vs default 0.01) → entropy fell 7.3 → 3.0 in 365 iters → exploration dead before grasp could be discovered
  4. `use_clipped_value_loss=True` → pessimistic VF bound that doesn't actually cap |V̂|
  5. Scalar `noise_std` shared across arm (5 dims) and gripper (1 dim) → gripper σ collapsed with arm σ even though gripper had near-zero gradient signal
  6. **Geometric assumption wrong**: design assumed palm-jaw vertical offset = 5 cm in top-down pose. Actual = **9.3 cm**. → "hover sweet spot" placed jaw 2.3 cm BELOW the table.
  7. **Strict grasp predicate condition (e) reversed**: required `q_gripper >= 70% × 100° = 70°`, but `BinaryJointPositionActionCfg` only commands q ∈ {0, 0.5} rad. → cond (e) literally unreachable, strict_grasp could NEVER fire.

### V2.18b fixes
- 9 PPO hyperparam changes (per `v218b_ppo_search_claude.md`)
- 1 reward weight change (`hover_height` 0.5 → 0.2)
- 1 reward calibration (`hover_height` target_height 0.05 → 0.08, recentered on actual grasp window)
- 1 reward calibration (`jaw_below_cube` safe_height 0.06 → 0.045)
- 1 reward predicate fix (strict_grasp cond (e) inverted, see §4)
- 1 upstream rsl_rl patch (LR floor 1e-5 → 1e-4)
- 1 env infrastructure fix (recorder disabled, see §6)

---

## 2. Reward stack (V2.18b)

12 dense terms + 2 terminals. All bounded per-step magnitude.
Net per-step ∈ [−1.5, +6.3], with target ≈ +4.2 in steady state.
Episode value bound |V| ≤ 600 (γ=0.99 × 300 steps).

### Dense terms

| # | Term | Weight | Formula | Gates | Max/step |
|---|------|--------|---------|-------|----------|
| 1 | `ee_to_cube_distance` | −1.0 | `clamp(d_3D, max=0.30)` | none | −0.30 |
| 2 | `reaching_object` | +1.0 | `1 − tanh(d_3D / 0.10)` | none | +1.0 |
| 3 | `palm_xy_above_cube` | +0.8 | `1 − tanh(d_xy / 0.04)` | `palm_z > cube_top + 5mm` | +0.8 |
| 4 | `hover_height` | **+0.2** | `exp(−((h − 0.08)/0.025)²)` where `h = palm_z − cube_top` | align_xy > 0.6 AND NOT grasped | +0.2 |
| 5 | `grasping_cube` | +2.0 | strict 6-cond predicate (binary, see §4) | none (the gates ARE the predicate) | +2.0 |
| 6 | `lifting_object` | +1.5 | `clamp((cube_z − 0.0565)/0.10, 0, 1)` | × strict_grasp | +1.5 |
| 7 | `object_goal_tracking` | +1.0 | `1 − tanh(d_cube_goal / 0.20)` | × strict × (cube_z > 0.08) | +1.0 |
| 8 | `object_goal_tracking_fine_grained` | +0.5 | `1 − tanh(d_cube_goal / 0.04)` | × strict × (cube_z > 0.08) | +0.5 |
| 9 | `gripper_orientation_penalty` | +0.3 | `−delta_hat.z` (palm→jaw direction; +1 = top-down, −1 = gripper-up) | none | +0.3 |
| 10 | `jaw_below_cube_penalty` | −2.0 | `clamp((0.045 − jaw_z)/0.045, 0, 1)` | × NOT grasped | −2.0 |
| 11 | `action_rate` | −0.01 | base_mdp.action_rate_l2 | none | small |
| 12 | `joint_vel` | −0.001 | base_mdp.joint_vel_l2 | none | small |

### Terminal terms

| Term | Weight | Trigger | Notes |
|------|--------|---------|-------|
| `success_bonus` | +2000 | `cube within 5cm of goal AND cube_z > 8cm` | one-shot at episode end |
| `cube_dropped_penalty` | −50 | cube_z < 0.04 (= below table − 1cm) | one-shot, terminates episode |

### Per-step budget breakdown
- Max positive: reach 1.0 + grasp 2.0 + lift 1.5 + goal 1.0 + fine 0.5 + palm_xy 0.8 + hover 0.2 + orient 0.3 = **+7.3**
- Max negative: ee_dist −0.30 + jaw_table −2.0 + action/vel ≈ −0.05 = **−2.35**
- Net target in steady state: ≈ +4.2 (dense terms only; terminal +2000 sparse)

### Multiplicative gating chain
The terms #5–#8 form a **gated cascade**:
```
hover_height       ←── × NOT(grasping_cube)
grasping_cube      ──→ × lifting_object
                   ──→ × goal_tracking (+ × cube_z > 8cm)
                   ──→ × goal_tracking_fine (+ × cube_z > 8cm)
```
Once `grasping_cube` fires, hover turns off (frees descent), and lift +
goal terms become available. This makes "lift without grasp" and "goal
tracking without lift" mathematically impossible — addresses V2.7 / V2.9
flick exploit and V2.16 / V2.17 fake-lift exploit.

---

## 3. Critical geometry (verified 2026-05-12)

### Cube spawn (constant across resets)
- `cube_z = 0.0565 m` (center)
- `cube_top = 0.0665 m`
- `cube_bottom = 0.0465 m`
- Table top ≈ `0.0465 m` (cube sits on table)

### Robot frames
- `palm` body z home = 0.2769 (= ee_frame target[0] with no offset)
- `jaw` body z home = 0.2971 (in horizontal home pose)
- `jaw_target` z home = 0.2761 (= jaw body + offset (-0.021, -0.07, +0.02) in body frame)
- In **horizontal home pose**: palm_target − jaw_target ≈ +0.001 (basically same Z)
- In **top-down pose** (wrist rotated 90°): palm_target − jaw_target = **+0.0934 m**
  (jaw 9.3 cm below palm — the −0.07 body-Y offset rotates fully to world −Z, plus ~2.3 cm of body geometry)

→ Verified empirically via `sim/eval2/scripts/verify_palm_jaw_offset.py`.

### Hover sweet spot calibration (V2.18b)
For a robust grasp, jaw needs to be inside the cube vertically:
`jaw_z ∈ [cube_bottom − 5mm, cube_top + 5mm] = [0.0415, 0.0715]`

That requires palm_z ∈ [0.135, 0.165] = **[cube_top + 6.85cm, cube_top + 9.85cm]**.

Hover Gaussian is centered at `palm_z = cube_top + target_height`:
- V2.18 had `target_height = 0.05` → peak at palm = 0.117 → jaw at 0.024 = **2 cm BELOW table** → grasp impossible
- **V2.18b uses `target_height = 0.08`** → peak at palm = 0.147 → jaw at 0.054 = inside cube center (-1cm from cube_top) → grasp valid

### Visual reference
```
  palm  ----- 0.147 (= cube_top + 8 cm)   <- hover peak V2.18b
         |
         | 9.34 cm offset (top-down)
         |
  jaw   ----- 0.054 (= cube_z − 1 cm)     <- inside cube
        ┌─┐  cube_top    = 0.0665
        │ │  cube_z      = 0.0565
        └─┘  cube_bottom = 0.0465
        ═══  table_top   = 0.0465
```

---

## 4. Strict grasp predicate (the binary "discovery" the policy makes)

Function: `cube_grasped_strict` in `sim/eval2/mdp/rewards.py`.

All 6 conditions must be **simultaneously true** to count as a grasp:

| Cond | Check | Threshold | Meaning |
|------|-------|-----------|---------|
| (a) | `palm_z ≥ cube_top + 1 cm` | 0.0765 | palm is above cube top by safety margin |
| (b) | `jaw_z ≤ cube_top + 5 mm` | 0.0715 | jaw has descended past the top of the cube |
| (c) | `jaw_z ≥ cube_bottom − 5 mm` | 0.0415 | jaw is not under the table (catches scoop) |
| (d) | `‖cube_xy − midpoint(palm,jaw)_xy‖ ≤ 1.5 cm` | 0.015 | cube is laterally between the fingers |
| (e) | **`q_gripper ≤ 0.15`** | 0.15 | gripper is closed (within 70% of closing travel from open=0.5 to close=0) |
| (f) | `‖cube_vel_xy‖ < 0.50 m/s` | 0.50 | cube is not in the middle of being ejected |

**Cond (e) was reversed before 2026-05-12 patch.** Old code:
`q_gripper ≥ 0.7 × q_max` where `q_max ≈ 1.745 rad` (= 100°, full SO-101
range). But `BinaryJointPositionActionCfg` only commands q to {0.0, 0.5} —
the joint never reaches 1.22 rad. → strict_grasp fired **0 times across
967 iterations** of V2.18 + V2.18b before the bug was caught.
Fix: `q_gripper ≤ 0.15` (close to the 0.0 close target).

→ See `sim/eval2/mdp/rewards.py:740` for the implementation comment.

---

## 5. PPO config (V2.18b)

File: `sim/eval2/agents/rsl_rl_ppo_cfg_v2_18b_cold.py`.

### Runner
| Param | V2.18 | **V2.18b** | Reason |
|-------|-------|------------|--------|
| `num_steps_per_env` | 50 | **48** | Divisible by `num_mini_batches=8` |
| `max_iterations` | 1500 | **3000** | Extended budget for the harder problem |
| `empirical_normalization` | False | **True** | #1 critic stability lever (Andrychowicz 2021) |
| `save_interval` | 50 | 50 | unchanged |

### Policy (ActorCritic)
| Param | V2.18 | **V2.18b** | Reason |
|-------|-------|------------|--------|
| `init_noise_std` | 1.0 | 1.0 | unchanged (matches AnymalB reference) |
| `noise_std_type` | "scalar" | **"log"** | Per-dim σ → gripper σ stays alive even when arm σ contracts |
| `actor_hidden_dims` | [256,128,128] | [256,128,128] | unchanged |
| `critic_hidden_dims` | [256,128,128] | **[512,256,128]** | Wider critic absorbs regime shift at grasp discovery |
| `activation` | "elu" | "elu" | unchanged |

### Algorithm
| Param | V2.18 | **V2.18b** | Reason |
|-------|-------|------------|--------|
| `learning_rate` (init) | 1.0e-3 | **3.0e-4** | Andrychowicz default; 3 fewer halvings of headroom before LR floor |
| `schedule` | "adaptive" | "adaptive" | unchanged |
| `desired_kl` | 0.02 | **0.01** | rsl_rl reference value (vs unusual 0.02) |
| `entropy_coef` | 0.005 | **0.01** | rsl_rl default; doubles entropy gradient → slower σ collapse |
| `num_learning_epochs` | 5 | **4** | Fewer epochs over same rollout (Andrychowicz §4.2) |
| `num_mini_batches` | 4 | **8** | +60% SGD steps per iter |
| `value_loss_coef` | 1.0 | 1.0 | unchanged |
| `use_clipped_value_loss` | True | **False** | Lets critic re-baseline post-discovery (Engstrom 2020) |
| `clip_param` | 0.2 | 0.2 | unchanged |
| `gamma` | 0.99 | 0.99 | unchanged |
| `lam` | 0.95 | 0.95 | unchanged |
| `max_grad_norm` | 1.0 | 1.0 | unchanged |
| `normalize_advantage_per_mini_batch` | False | False | per-rollout norm (Andrychowicz C67) |

### Random Network Distillation (NEW in V2.18b)
| Param | Value | Note |
|-------|-------|------|
| `weight` | 0.002 | Small intrinsic bonus (~10⁻³ × dense) |
| `weight_schedule` | StepWeight, final_step=800 | Curiosity vanishes by iter 800 |
| `reward_normalization` | True | RND reward kept stable |
| `learning_rate` | 1e-4 | Predictor net LR |
| `predictor/target_hidden_dims` | [128, 128] | Two MLP layers |

### Upstream rsl_rl patch
- File: `.venv/Lib/site-packages/rsl_rl/algorithms/ppo.py:282`
- Change: LR floor `max(1e-5, ...)` → `max(1e-4, ...)`
- Re-apply after any venv rebuild via: `python sim/eval2/scripts/patch_rsl_rl.py`
- Doc: `notes/rsl_rl_patches.md`

### Why each change matters (priority order)

1. `empirical_normalization=True` — critic stability, ~70% of VF-blowup risk
2. LR floor 1e-4 (in `ppo.py`) + `learning_rate=3e-4` init — kills LR collapse cascade
3. `desired_kl=0.01` — restores standard trust region
4. `noise_std_type="log"` — gripper σ stays alive
5. `entropy_coef=0.01` — slows σ collapse
6. `use_clipped_value_loss=False` — critic absorbs regime shift
7. RND weight=0.002 — adds curiosity for sparse-grasp discovery
8. Critic widened to [512,256,128] — capacity for regime shift
9. `num_mini_batches=8`, `num_learning_epochs=4` — +60% SGD steps

---

## 6. Env infrastructure

### Recorder disabled (V2.18b)
LeIsaac's parent env config enables `RecorderManagerCfg` which writes
episode data to `C:/tmp/isaaclab/logs/dataset.hdf5`. We don't need demos
during PPO training, and a corrupted HDF5 (e.g. from a crashed
concurrent process) detonates the run with `wrong B-tree signature`
(burned us at iter 305 of the first V2.18b run on 2026-05-12).

V2.18b sets `self.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE`
in both state-only and visual env post-init. See
`sim/eval2/leisaac_lift_env_cfg.py:RewardsCfgV218B`.

### Robot color (cosmetic)
Robot recolored to matte dark gray (#0D0D0D) via
`sim/eval2/scripts/recolor_robot_usd.py` (idempotent). Backup at
`leisaac/assets/robots/so101_follower.usd.bak`.

---

## 7. Gym task IDs

Registered in `sim/eval2/__init__.py`. Use these for training / play:

| Task ID | Env cfg | PPO cfg |
|---------|---------|---------|
| `Isaac-LeIsaac-SO101-Lift-RL-V218B-cold-v0` | state-only | V218BCold |
| `Isaac-LeIsaac-SO101-Lift-RL-V218B-cold-Play-v0` | state-only, 50 envs | V218BCold |
| `Isaac-LeIsaac-SO101-Lift-Visual-V218B-cold-v0` | with wrist + front cams | V218BCold |
| `Isaac-LeIsaac-SO101-Lift-Visual-V218B-cold-Play-v0` | visual, 50 envs | V218BCold |

---

## 8. Launch + monitor commands

### Cold-start training
```powershell
cd C:\Users\user\Desktop\MA2\robot-learning-project3
$env:RUST_LOG = "error"
C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
  sim/eval2/scripts/train.py `
  --task Isaac-LeIsaac-SO101-Lift-Visual-V218B-cold-v0 `
  --headless --enable_cameras --num_envs 256
```

### Resume from checkpoint
```powershell
C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
  sim/eval2/scripts/train.py `
  --task Isaac-LeIsaac-SO101-Lift-Visual-V218B-cold-v0 `
  --headless --enable_cameras --num_envs 256 `
  --resume `
  --load_run <YYYY-MM-DD_HH-MM-SS> `
  --checkpoint model_<N>.pt
```
The `--resume` flag is mandatory due to a bug in `isaac_so_arm101/scripts/rsl_rl/cli_args.py:76` that overrides the agent_cfg's resume field with the CLI default (False).

### Monitor (in a second terminal)
```powershell
C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
  sim/eval2/scripts/monitor_training.py --interval 30
```

### Play / replay
```powershell
C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
  sim/eval2/scripts/play_diagnose_v2.py `
  --task Isaac-LeIsaac-SO101-Lift-Visual-V218B-cold-Play-v0 `
  --num_envs 4
```

---

## 9. Risk runbook (what to watch in TensorBoard)

| Iter window | Metric | Healthy | Pathological | Action if pathological |
|-------------|--------|---------|--------------|------------------------|
| 0–30 | `Loss/value_function` | < 1.0 monotone | > 5.0 rising | empirical_norm not on; verify config |
| 0–50 | `Policy/mean_noise_std` per-dim | gripper σ ≥ 0.8 | gripper σ < 0.5 | bump entropy_coef to 0.02 |
| 30–80 | `Loss/learning_rate` | [1e-4, 5e-4] | hits 1e-4 floor and stays | reduce clip_param → 0.15; or raise desired_kl → 0.015 |
| 50–200 | `Train/mean_reward` | climb ≥ 0.05/iter | flat > 40 iters | bump RND weight to 0.005 |
| 100–300 | `Episode_Reward/grasping_cube` | fires ≥ 1× by iter 250 | still 0 at iter 300 | grasp_strict still broken — check geometry, gripper threshold, predicate logic |
| 200–500 | `Loss/entropy` | ≥ 4.0 nats | < 3.0 before grasp fires | entropy_coef → 0.015 |
| Any | `Loss/value_function` step ratio | < 2× iter-over-iter | > 10× in 1 iter | VF cascade — rollback 50 iters, halve value_loss_coef |
| Any | `Policy/approx_kl` | median 0.005–0.015 | > 0.03 spikes | clip_param too loose — drop to 0.15 |

---

## 10. File map

| File | Role |
|------|------|
| `sim/eval2/leisaac_lift_env_cfg.py` | Env configs (RewardsCfgV218B, LeIsaacLiftCubeRLEnvCfgV218B + variants) |
| `sim/eval2/mdp/rewards.py` | All custom reward functions including `cube_grasped_strict` |
| `sim/eval2/mdp/observations.py` | Custom observations (ee_to_cube_vec, cube_to_goal_vec) |
| `sim/eval2/mdp/terminations.py` | Custom terminations (cube_reached_goal, cube_dropped) |
| `sim/eval2/agents/rsl_rl_ppo_cfg_v2_18b_cold.py` | PPO config |
| `sim/eval2/__init__.py` | Gym task registrations |
| `sim/eval2/scripts/train.py` | Training entrypoint |
| `sim/eval2/scripts/monitor_training.py` | Live TensorBoard monitor (V2.18 architecture aware) |
| `sim/eval2/scripts/verify_palm_jaw_offset.py` | Palm-jaw geometry verification (top-down pose dump) |
| `sim/eval2/scripts/patch_rsl_rl.py` | Idempotent re-application of the rsl_rl LR floor patch |
| `sim/eval2/scripts/recolor_robot_usd.py` | Robot black recolor (idempotent) |
| `notes/v218b_architecture.md` | This file |
| `notes/v218_design_claude_search.md` | Original V2.18 reward design (superseded) |
| `notes/v218b_ppo_search_claude.md` | PPO search artifact (fully integrated here) |
| `notes/rsl_rl_patches.md` | rsl_rl upstream patch documentation |
| `notes/eval2_pipeline.md` | Full V2 → V2.18b chronicle and dead-ends |

---

## 11. Don't re-suggest (lessons from dead-ends)

- Adding reward weights > 50 on dense terms (V2.17 went 30 → 150 → VF blowup at iter 343)
- Quaternion-based orientation reward without verifying URDF axis convention (V2.13 v2 sign-bug)
- Loose grasp predicate `jaw < 4cm + closed` (V2.7 → V2.15 false positives, 89% ejection)
- DoneTerm without paired penalty (V2.13 v1 give-up exploit)
- Switching to SAC, BC warmstart, DAPG, magic-attach, scripted IK (already considered & rejected)
- Reverting to Isaac defaults PPO (LR=1e-4 too conservative for cold-start)
- Re-patching ppo.py to a different LR floor without a Claude search (1e-4 is the validated value)
- Modifying `isaac_so_arm101/` directly — project code lives in this repo's `sim/`
