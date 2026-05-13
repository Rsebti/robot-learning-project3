# V2.19 architecture — full reward rewrite per CoRL/RSS/ICLR 2022-2026 synthesis

> ⚠ **PARTIAL IMPLEMENTATION — read before assuming the spec is matched**
>
> Two items from `notes/v219_reward_search_claude.md` are NOT yet
> implemented in the V2.19 code:
>
>   1. **Asymmetric critic** — privileged ~30D vector for the critic only
>      (Lambrechts et al. ICML 2025). Spec item #7. The current PPO
>      config uses symmetric AC (both actor and critic see the same
>      observation group). Requires adding a `privileged` obs group
>      and setting `obs_groups = {"policy": ["policy"], "critic":
>      ["policy", "privileged"]}` in the runner config.
>
>   2. **Auxiliary cube-xyz prediction head** on the ResNet-18 features
>      (Scaffolder, Hu et al. ICLR 2024). Spec item #7. Requires forking
>      rsl_rl to add an auxiliary loss term during PPO updates. The
>      Scaffolder pattern is the prescribed cure for the
>      partial-observability aliasing that asymmetric AC alone cannot
>      close (Cai et al. 2024).
>
> Also partial: **antipodal contact stickers** (spec item #6) are NOT
> implemented — SO-101's single-articulated-jaw geometry doesn't fit
> the parallel-jaw assumption cleanly. Replaced by `finger_straddle_v219`
> (palm above + jaw below cube top), which captures similar intent.
>
> Without items 1 and 2, V2.19 is sim2real-clean (no cube ground-truth
> in the actor obs by design when paired with PickLift) but the value
> function inherits the actor's partial observability — high-variance
> gradients are expected per Cai et al. (2024). If V2.19 plateaus
> despite the new reward stack working, asymmetric AC + aux head is
> the next logical step.

Active variant as of 2026-05-12 implementation. Replaces V2.18b reward
stack while keeping V2.18b PPO hyperparameters and geometry verified
empirically (`verify_palm_jaw_offset.py`).

Source artifact : `notes/v219_reward_search_claude.md` (Claude search,
2026-05-12, synthesizing Text2Reward, DrS, ManiSkill2/3, DexPoint,
Eureka, DrEureka, Lin et al. CoRL 2025, Christmann et al. 2024,
Lambrechts et al. ICML 2025, Skalse et al. NeurIPS 2022).

V2.18b is preserved unchanged in archive (env classes, PPO config, reward
classes all intact). V2.19 is a parallel branch.

---

## Why a full rewrite

V2.18b cold-start trained 967 iterations across two runs without a
single grasp_strict event. After fixing the cond_e bug (q ≤ 0.15 instead
of q ≥ 0.7 × q_max), the geometric predicate became reachable in
principle, but the PickLift variant (cube ground-truth removed)
plateaued at mean_reward ~10 with no grasp progress at iter 548.

The user's call: full rewrite per the doc rather than incremental
tweaks. V2.19 implements every concrete recommendation from the doc that
applies to SO-101's single-articulated-jaw geometry.

---

## Reward stack (V2.19)

12 dense terms + 1 success terminal. NO drop penalty — just early
termination on drop.

| # | Term | Weight | Function | Notes |
|---|------|--------|----------|-------|
| 1 | `reaching_object` | +1.0 | `reach_coarse_v219` | `1 − tanh(d/0.10)` |
| 2 | `reaching_fine` | +0.5 | `reach_fine_v219` | `1 − tanh(d/0.02)` |
| 3 | `finger_straddle` | +0.5 | `finger_straddle_v219` | palm above + jaw below cube |
| 4 | `gripper_orientation` | +1.0 | `orientation_quat_tracking_v219` | `exp(−3·θ)` to top-down |
| 5 | `grasp_milestone` | +2.0 | `grasp_milestone_v219` | one-time on first contact-grasp |
| 6 | `lifting_object` | +5.0 | `lift_clipped_gated` | `clamp(h/0.15, 0, 1) × is_grasped` |
| 7 | `lift_milestone` | +5.0 | `lift_milestone_v219` | one-time on first lift > 10 cm |
| 8 | `object_goal_tracking` | +16.0 | `goal_tracking_lift_gated_dual_scale` | std=0.30 coarse |
| 9 | `object_goal_tracking_fine` | +5.0 | `goal_tracking_lift_gated_dual_scale` | std=0.05 fine |
| 10 | `success_bonus` | +15.0 | `success_terminal_v219` | one-time terminal |
| 11 | `motor_torque_penalty` | 1e-3 | `torque_penalty_v219` | DrEureka safety |
| 12 | `motor_work_penalty` | 1e-4 | `work_penalty_v219` | Hora et al. |
| 13 | `joint_limit_penalty` | 1e-2 | `joint_limit_penalty_v219` | DrEureka |
| 14 | `close_no_contact_penalty` | 0.5 | `close_no_contact_penalty_v219` | anti-jam |
| 15 | `action_rate` | -1e-4 | `action_rate_l2` | Isaac Lab default |
| 16 | `joint_vel` | -1e-4 | `joint_vel_l2` | Isaac Lab default |

**REMOVED vs V2.18b**:
- `cube_dropped_penalty` (-50) → no penalty, just early termination
- `grasping_cube` continuous (+2/step) → replaced by one-time milestone
- `palm_xy_above_cube`, `hover_height`, `jaw_below_cube_penalty`,
  `ee_to_cube_distance` → subsumed by reach (dual-scale) + straddle + orient
- `gripper_orientation_penalty` (positional, +0.3) → replaced by
  bounded quaternion tracking (`exp(-3·θ)`, weight +1.0)
- All 6-condition geometric grasp predicate machinery →
  contact-impulse via two ContactSensors

## Contact-impulse grasp predicate

Replaces V2.18b's 6-AND geometric predicate with the
ManiSkill/DexPoint/Lin et al. style:

```
left_force  = ContactSensor(gripper).net_forces_w_history[:, 0].norm()
right_force = ContactSensor(jaw).net_forces_w_history[:, 0].norm()
is_grasped  = (left_force > 0.5) & (right_force > 0.5) & (cube_speed_xy < 0.5)
```

DexPoint ablation (Qin et al., CoRL 2022): switching from geometric to
contact predicate moved real-world bottle-pick from 0% → 83%. Geometric
predicates also brittle under domain randomization — small object size
variations flip the multi-AND discontinuously.

Adapted to SO-101: the cube is pinched between the FIXED `gripper`
body (top wrist plate) and the MOVING `jaw` body. Both are in contact
with the cube during a successful grasp, hence two sensors filtered to
the cube.

## ContactSensor scene additions

Added in `LeIsaacLiftCubeRLVisualEnvCfgV219.__post_init__`:

```python
# NOTE: filter_prim_paths_expr REMOVED -- see "GPU PhysX trap" below.
self.scene.contact_gripper = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper",
    history_length=3,
    track_pose=False,
)
self.scene.contact_jaw = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/jaw",
    history_length=3,
    track_pose=False,
)
```

### GPU PhysX trap (2026-05-12, empirically verified)

Initial design used `filter_prim_paths_expr=["{ENV_REGEX_NS}/Scene/cube"]`
to restrict the sensor to cube contacts only. Two bugs surfaced:

1. The cube's actual prim path is `/Scene/cube/cube` (LeIsaac wraps the
   rigid body inside a scene-group prim). The original filter pointed
   at the WRAPPER, not the body, so PhysX warned
   `GPU contact filter for collider '/World/envs/env_0/Scene/cube' is
   not supported` and silently reported zero forces.
2. Updating the filter to the correct `/Scene/cube/cube` path made
   GPU PhysX (cuda:0) **silently hang the env build** — 12 s in, no
   traceback, no crash dump, just a frozen process.

GPU PhysX (cuda:0) does not support the per-prim contact filter API
on rigid-body cubes in our Isaac Lab 2.3 / IsaacSim 5.1 build. CPU
PhysX would work but tanks throughput.

**Fix shipped** : drop `filter_prim_paths_expr` entirely. Each sensor now
reports the TOTAL contact force on its body (any source). False-positive
control is now done at the predicate level via a **cube-proximity gate**:

```
is_grasped = (F_gripper > 1.0 N)
           & (F_jaw     > 1.0 N)
           & (d(cube_pos, jaw_fingertip) < 0.10 m)   <-- proximity gate
           & (cube_speed_xy < 0.5 m/s)
```

`jaw_fingertip` is `ee_frame.target_pos_w[:, 1, :]` (the leisaac
FrameTransformer's jaw target with the (-0.021, -0.070, 0.02) offset).
We use jaw fingertip rather than palm fingertip because leisaac's
`ee_frame.target[0]` ("gripper") has no offset and points at the wrist
plate body origin (~9 cm behind the palm fingertip in top-down). One
proximity gate is enough since palm and jaw fingertips sit within
~3 cm of each other in any closed-gripper pose.

Threshold of **10 cm** chosen because the jaw body's collision mesh
extends from the jaw body origin all the way to the fingertip (~7.6 cm
away via the offset). During a real grasp the cube can end up anywhere
along this length depending on grasp geometry, so the gate must cover
it. 10 cm rejects table-only contacts (cube on table 20+ cm from the
flopped fingertips) but accepts every plausible grasp configuration.

Original concern about the user's question "if both jaws hit the table
will it false-trigger?" -> NO. Robot crash flat on table puts the jaw
fingertip at table xy, but the cube is on the table at a DIFFERENT xy
(>= 20 cm away in spawn). d_jaw_fingertip > 10 cm rejects it.

### Empirical verification (2026-05-12, `verify_contact_sensors.py`)

| Scenario                              | F_gripper | F_jaw    | Verdict |
|---------------------------------------|-----------|----------|---------|
| HOME open  (no contact)               |   0.00 N  |   0.00 N | OK      |
| HOME closed (gripper closes in air)   |   0.00 N  |   0.00 N | OK      |
| TOP-DOWN, cube held on palm body      |   8.88 N  |   0.00 N | OK      |
| TOP-DOWN, cube held on jaw  body      |  19.58 N  |  11.87 N | OK      |

Forces well above the 1.0 N predicate threshold. Verify uses kinematic
cube hold (re-pin pose each step) to defeat gravity that otherwise pulls
the cube off the body before any contact frame registers -- PhysX does
not generate contact events for teleport-overlaps, only motion-driven
collisions.

## State buffers for milestone / static tracking

Three V2.19 reward functions need to track state across steps:
- `grasp_milestone_v219` : remember `was_grasped_prev_step`
- `lift_milestone_v219` : remember `was_lifted_prev_step`
- `success_terminal_v219` : count consecutive static frames

These use a per-env state buffer attached to the env via
`env._v219_state` dict. Reset at episode start (`episode_length_buf == 0`)
to False / 0. See `_ensure_extras_buffer` and `_ensure_extras_buffer_int`
helpers.

## PPO config

**Unchanged from V2.18b Cold** — same empirical_norm, log noise_std,
RND, wider critic, desired_kl=0.01, lr_init=3e-4, entropy_coef=0.01,
clipped_value_loss=False. The PPO machinery did not cause V2.18b's
stall; the reward stack did. Reusing the same hyperparameters here
isolates the reward effect.

Symmetric AC by default (no `obs_groups` override). Asymmetric AC +
auxiliary cube-xyz prediction head (Scaffolder, Hu et al. ICLR 2024)
deferred — requires modifying rsl_rl source.

## Gym task IDs

```
Isaac-LeIsaac-SO101-Lift-Visual-V219-cold-v0
Isaac-LeIsaac-SO101-Lift-Visual-V219-cold-Play-v0
```

V2.18b task IDs preserved (V218B and V218B-picklift remain registered).

## Launch

```powershell
cd C:\Users\user\Desktop\MA2\robot-learning-project3
$env:RUST_LOG = "error"
C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
  sim/eval2/scripts/train.py `
  --task Isaac-LeIsaac-SO101-Lift-Visual-V219-cold-v0 `
  --headless --enable_cameras --num_envs 256
```

## What to watch (V2.19-specific signals)

- **Iter 0-30** : `Loss/value_function` should stay <1.0 (empirical_norm
  on, reward magnitudes bounded). If >5 → reward magnitudes too large,
  reduce weights.
- **Iter 30-100** : `Episode_Reward/grasp_milestone` should fire ≥1×
  by iter 100. If still 0 → contact sensors not detecting (wrong cube
  prim path), check `_get_jaw_force_norm` is reading non-zero values.
- **Iter 100-300** : `Episode_Reward/lifting_object` and `lift_milestone`
  start firing. `mean_reward` should climb past V2.18b's plateau (~10).
- **Iter 300-500** : `Episode_Reward/object_goal_tracking` (coarse +
  fine) start firing as cube is moved toward goal.
- **Iter 500+** : `Episode_Termination/success` fires. Run successful.

## What's NOT yet implemented (deferred from doc)

- **Antipodal contact stickers** (Lin et al. 2025) — requires defining
  per-object stickers; SO-101 single jaw doesn't fit the antipodal
  parallel-jaw assumption cleanly. Skipped.
- **Asymmetric critic** with privileged ~30D vector (Lambrechts et al.
  ICML 2025) — requires `obs_groups` setup AND aux head modification
  to rsl_rl. Deferred to V2.20.
- **Auxiliary cube-xyz prediction head** on ResNet features (Scaffolder,
  Hu et al. ICLR 2024) — requires forking rsl_rl. Deferred.
- **Curriculum on initial state** (Petrenko et al. RSS 2023) — Isaac
  Lab provides `CurriculumManager`, can be added once base reward works.
- **Lipschitz / LipsNet smoothness** (Christmann et al. 2024) — requires
  custom policy head. Deferred.
- **Multi-seed validation** (Booth et al. AAAI 2023) — best practice;
  user can run 3-5 seeds in parallel on Brev for variance estimate.

## Files modified / created

- `sim/eval2/mdp/rewards.py` — appended ~400 lines of V2.19 functions
- `sim/eval2/mdp/__init__.py` — exported new functions
- `sim/eval2/leisaac_lift_env_cfg.py` — appended `RewardsCfgV219`,
  `LeIsaacLiftCubeRLVisualEnvCfgV219`, `_PLAY` variants (~250 lines)
- `sim/eval2/agents/rsl_rl_ppo_cfg_v2_19_cold.py` — NEW
- `sim/eval2/__init__.py` — added 2 task registrations
- `notes/v219_reward_search_claude.md` — archived doc
- `notes/v219_architecture.md` — this file

## Don't re-suggest (V2.19 specific)

- "Add cube_dropped_penalty back" — explicitly removed per Skalse et al.
  2022 ; rely on early termination
- "Add per-step `grasping_cube` continuous reward" — replaced by
  one-time milestone per Eureka findings
- "Increase action_rate weight beyond 1e-4" — Christmann 2024 confirms
  ≥1e-3 stalls Isaac Lab manipulation
- "Use raw negative-distance reach penalty" — bounded tanh kernel only
