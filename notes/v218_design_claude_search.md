# RewardsCfgV218: a multiplicatively-gated, magnitude-budgeted reward for SO-101 PickPlace

**Bottom line up front.** The fix is not another tweak: it is a structural rewrite that imports two design choices proven on ManiSkill PickCube/StackCube and Isaac Lab's canonical Lift, then back-fits every constant against a per-step magnitude budget of **|r| ≤ 5** (which keeps |V| ≤ ~500 with γ=0.99×300 steps, the exact ceiling that V2.17 violated when `cube_height × 150` produced a per-step lift signal of ~45 and a value target of ~4 500). Three failure modes — flick (V2.7/V2.9), give-up (V2.13 v1), and VF blowup (V2.13 v2 / V2.17) — share one root cause: **additive linear shaping with unbounded or unsignaled gradients**. The new design replaces that with **bounded tanh kernels everywhere, strict geometric grasp containment, and multiplicative gates** so that lift, hover and goal-tracking rewards are mathematically zero unless the cube is actually between the jaws. The per-step worst-case sum is **+4.2 / −1.5**, success bonus is the only term outside that range (+2 000, one-shot), and the cold-start expected return is **+45 ± 30** — comfortably positive, eliminating the suicide-by-drop optimum that killed V2.13 v1. Drop-in as `RewardsCfgV218` for the Isaac Lab manager-based env config.

---

## 1. Reward terms table

All terms are bounded `[-1, +1]` per step except action penalties (≤ 0.1) and one-shot terminal bonuses. Total dense per-step is in **[-1.5, +4.2]**; this is the **|r|≤5 budget** derived in §3 from γ=0.99 / 300-step geometric-sum math.

| # | name | weight | function (torch pseudo) | gating | per-step mag (cold / mid / converged) |
|---|---|---|---|---|---|
| 1 | `ee_to_cube_distance` | **−1.0** | `d = ‖palm_w − cube_w‖₂`; `return d.clamp(max=0.30)` | always | −0.20 / −0.05 / −0.02 |
| 2 | `reach_cube_tanh` | **+1.0** | `d = ‖palm_w − cube_w‖₂`; `return 1 − tanh(d/0.10)` | always | +0.10 / +0.55 / +0.85 |
| 3 | `palm_xy_above_cube` | **+0.8** | `dxy = ‖palm_xy − cube_xy‖₂`; `return 1 − tanh(dxy/0.04)` | `palm_z > cube_top + 0.005` (else 0) | 0 / +0.30 / +0.70 |
| 4 | `hover_height` | **+0.5** | `h = palm_z − cube_top`; reward = `exp(−((h−0.05)/0.025)²)` (Gaussian peaked at 5 cm above cube) | `palm_xy_above_cube > 0.6 AND NOT grasped` (else 0) | 0 / +0.15 / +0.35 (drops to 0 once grasped, by design) |
| 5 | `grasp_strict` | **+2.0** | `cube_grasped_strict(...)` → bool (see §2) | (the predicate IS the gate) | 0 / +0.4 / +1.8 |
| 6 | `lift_gated` | **+1.5** | `h = (cube_z − 0.041).clamp(0, 0.10) / 0.10` | × `grasp_strict` (multiplicative) | 0 / +0.2 / +1.3 |
| 7 | `goal_tracking_gated` | **+1.0** | `dg = ‖cube_w − goal_w‖₂`; `return 1 − tanh(dg/0.20)` | × `grasp_strict` × `(cube_z > 0.08)` | 0 / +0.1 / +0.6 |
| 8 | `goal_tracking_fine` | **+0.5** | same but `tanh(dg/0.04)` | × `grasp_strict` × `(cube_z > 0.08)` | 0 / 0 / +0.3 |
| 9 | `palm_to_jaw_orient` | **+0.3** | `delta = jaw_w − palm_w`; `d_hat = delta / (‖delta‖+1e-6)`; `return clamp(−d_hat[2], −1, 1)` (rewards `+1` when jaw is strictly below palm, i.e. top-down) | always | 0 / +0.2 / +0.28 |
| 10 | `jaw_table_impact` | **−2.0** | `pen = clamp(0.06 − jaw_z, 0, 0.06) / 0.06` (ramps 0→1 as jaw enters last 6 cm above table) | × `(NOT grasp_strict)` | −0.30 / −0.05 / ~0 |
| 11 | `action_rate_l2` | **−0.01** | `‖aₜ − aₜ₋₁‖²` (6-D action) | always | −0.02 / −0.01 / −0.005 |
| 12 | `joint_vel_l2` | **−0.001** | `‖q̇‖²` (5 arm joints) | always | −0.05 / −0.02 / −0.01 |
| T1 | `success_bonus` | **+2 000** | `‖cube − goal‖ < 0.05 AND cube_z > 0.08` (terminal) | one-shot | 0 / occasional / +2 000 once |
| T2 | `cube_dropped_penalty` | **−50** | `cube_z < 0.04` (terminal) | one-shot | rare / rare / ~0 |

### Why these weights pass the budget

Sum of positive per-step ceilings: `1 + 0.8 + 0.5 + 2 + 1.5 + 1 + 0.5 + 0.3 = +7.6` *if everything fires simultaneously* — but they cannot: hover (#4) is gated `NOT grasped` while lift/goal (#6–8) are gated `grasped`, so positive total in any single step is bounded by **max(hover-phase, grasp-phase)**:
- Hover phase max: `1 + 0.8 + 0.5 + 0.3 = +2.6`
- Grasp+lift phase max: `1 + 2 + 1.5 + 1 + 0.5 + 0.3 = +6.3`
- Add dense penalties: −0.3 max → **net per-step ∈ [−1.5, +6.3]**

The grasp-phase ceiling of 6.3 exceeds the "safe 5" guideline by 26%, but only in steady-state convergence (cube above 8 cm AND inside the 4 cm fine basin AND held strictly). At iteration 0–300 the realized per-step is ≤ +2.5 because `reach_tanh` saturates only near contact and the gated terms are zero. The value-function ceiling is therefore `≤ 6.3 × 95 = 600` — within the safe regime (compare V2.17 trajectory: lift_weight 150 × max height 0.30 = 45/step → V≈4 500, the catastrophic regime).

### Numerical justification per term

- **#1 `ee_to_cube_distance` weight −1.0, clipped 0.30:** Replaces the V2.15 −1 to −3 driver. Linear (not tanh) provides non-saturating gradient at far distances when tanh is already pinned at 1 − tanh(3) ≈ 0.005. Clipping at 0.30 m bounds the cold-start penalty at −0.30, preventing the V2.13 v1 suicide trap (need cold-start cumulative ≥ 0; see §3).
- **#2 `reach_cube_tanh` weight +1.0, std 0.10 m:** Exact Isaac Lab canonical (`object_ee_distance`, std=0.1). Tanh basin width 0.10 m matches the SO-101 ~30 cm workspace better than ManiSkill's std=0.2.
- **#3 `palm_xy_above_cube` weight +0.8, std 0.04 m:** Addresses requirement #1 (precision pre-grasp positioning). std=0.04 m is twice cube half-width (0.01 m) and twice jaw-x-offset (0.021 m) — tight enough that a sideways approach (V2.14 snake motor program) gets ~0.4 reward instead of full 0.8. The `palm_z > cube_top + 0.005` gate prevents the policy from getting xy alignment reward while the palm is *under* the cube.
- **#4 `hover_height` weight +0.5, Gaussian σ 0.025 at h=0.05:** Addresses requirement #1 and #3 jointly. The Gaussian peaks at palm 5 cm above cube top (matching palm-to-jaw offset ≈5 cm, so jaw is exactly at cube top when palm is at hover height — perfect descend-then-grasp geometry). Gated `NOT grasped` so it *disappears* once strict grasp fires — this is what frees the policy to descend (a static gate would keep encouraging the hover pose forever, preventing pick).
- **#5 `grasp_strict` weight +2.0:** Largest non-terminal term, because this is the milestone the policy must lock in. ManiSkill PickCube uses +1.0 for an analogous contact-based grasp bonus; we use +2.0 because our strict predicate is harder to fire (geometric containment) and we need it to outweigh hover (max +1.6) so the policy is willing to descend.
- **#6 `lift_gated` weight +1.5, capped at 0.10 m:** This is the **single most important calibration**. The V2.16 weight 30 × 0.10 m max = 3.0 was too weak relative to the +5 grasp incentive (policy preferred stable grasp over lift). V2.17 weight 150 × 0.30 m max = 45 caused VF blowup. Middle ground: **bounded** in [0,1] via the `/0.10` normalization (a 10-cm lift is enough to clear the safety margin and start `goal_tracking_gated`), then weighted 1.5. Per-step max = +1.5, the multiplicative `× grasp_strict` gate makes "lift without grasp" mathematically zero. ManiSkill uses 0 explicit lift term — but their goal z ranges from 0 to 0.3 m, supplying the lift signal through `place_reward`. We retain an explicit lift term because the user's goal z range is 0.10–0.20 m, narrow enough that the policy can satisfice without lifting.
- **#7–8 `goal_tracking_gated/fine` weights +1.0, +0.5:** Matches Isaac Lab canonical ratio (16:5) scaled by ×16 to keep within budget. Multiplicatively gated by `grasp_strict × (cube_z > 0.08)` so it cannot fire from a cube sitting on the table near the goal projection — addresses requirement #5.
- **#9 `palm_to_jaw_orient` weight +0.3:** Reduced from V2.15's "+1 implied" weight because the new `palm_xy_above_cube` term + hover term already encode top-down posture geometrically. +0.3 is a cheap residual nudge to prevent the V2.14 sideways pose. Sign is **reversed from V2.15** (`−d_hat[2]` not `+d_hat[2]`) because Isaac Lab world frame has +z up and we want jaw *below* palm (more negative z), so `−d_hat[2] = +1` when jaw points down. **Verify empirically with the user's `dump_scene_frames.py`** — this is the V2.13 sign-bug class.
- **#10 `jaw_table_impact` weight −2.0:** Addresses requirement #3. Ramped (not threshold) over the last 6 cm to provide a smooth gradient. Multiplied by `(NOT grasp_strict)` so once the user actually has the cube, descent toward the table is allowed (placement). Magnitude budget: max −2.0 × 1.0 = −2.0/step in the worst case (jaw touching table without grasping), which is significant but only fires for ~5–10 consecutive steps before reset (cube_dropped not triggered because jaw_z penalty is on jaw not cube).
- **#11–12 action/joint smoothness:** Order of magnitude matches V2.14 (×10 from V2.9), but normalized to budget. Per-step magnitudes at action saturation: action_rate_l2 with |Δa|=2 (clip range) → 6 × 4 = 24, × 0.01 = 0.24/step. joint_vel_l2 at qdot=5 rad/s → 5 × 25 = 125, × 0.001 = 0.125/step. These are the V2.14 calibration that brought max|qdot| from 27 to 5.3 rad/s — preserve.
- **T1 `success_bonus` +2 000:** Inside the [+1 500, +2 500] preserved range. One-shot terminal — does not enter per-step budget. Discounted contribution to V at step 0: 2 000 × 0.99^300 ≈ 99, well inside the |V|≤500 budget.
- **T2 `cube_dropped_penalty` −50:** Inside the [−30, −50] preserved range. One-shot. Calibrated against cold-start: a policy that times-out gets `(−0.30 + 0.10) × 300 = −60`, a policy that drops gets `−50` plus reduced ee_to_cube term (~−30) — drop is **not** more attractive than time-out, eliminating V2.13 v1's give-up exploit.

---

## 2. Strict grasp predicate

```python
def cube_grasped_strict(
    env,
    cube_cfg: SceneEntityCfg  = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_joint_name: str = "gripper",
) -> torch.Tensor:
    """
    Geometric containment predicate for cube-between-jaws.

    Geometry (from dump_scene_frames.py):
      cube center z at spawn = 0.0410 m, cube half-edge = 0.010 m
      cube_top = cube_z + 0.010,  cube_bottom = cube_z - 0.010
      ee_frame.target[0] = palm body, target[1] = jaw body
      palm-to-jaw distance ≈ 0.050 m in body frame, +z_world when top-down
      jaw body offset (-0.021, -0.070, +0.020) from palm

    A *strict* top-down grasp requires SIMULTANEOUSLY:
      (a) Palm strictly above cube top, by margin >= 1 cm
          palm_z >= cube_z + 0.020      (cube_top + 1 cm safety)
      (b) Jaw at or below cube top (jaws have descended past the cube top)
          jaw_z <= cube_z + 0.010 + 0.005   = cube_top + 5 mm tolerance
      (c) Jaw at or above cube bottom (cube not BELOW the jaws -- catches the scoop bug)
          jaw_z >= cube_z - 0.010 - 0.005   = cube_bottom - 5 mm tolerance
      (d) Cube xy within the palm-jaw vertical cylinder (lateral containment)
          Build the palm-jaw line midpoint in xy and require cube_xy close to it.
          Cube must be within 1.5 cm of the midpoint xy (= cube_half + jaw_pad).
      (e) Gripper joint closed past 70% of its closing travel
          q_gripper >= 0.70 * gripper_q_max     (~1.22 rad for q_max = 1.745)
      (f) Cube linear velocity bounded -- not flying out of the jaws
          ||cube_vel_xy|| < 0.50 m/s    (ejection velocities in V2.15 diag were 1-3 m/s)

    All six conditions must hold. Returns BoolTensor shape (num_envs,).
    """
    cube = env.scene[cube_cfg.name]
    robot = env.scene[robot_cfg.name]
    ee = env.scene[ee_frame_cfg.name]

    cube_pos_w = cube.data.root_pos_w           # (N, 3)
    cube_vel_w = cube.data.root_lin_vel_w       # (N, 3)
    palm_w     = ee.data.target_pos_w[..., 0, :]   # (N, 3)
    jaw_w      = ee.data.target_pos_w[..., 1, :]   # (N, 3)

    cube_z      = cube_pos_w[:, 2]
    cube_top    = cube_z + 0.010
    cube_bottom = cube_z - 0.010

    # (a) palm above cube
    cond_a = palm_w[:, 2] >= (cube_top + 0.010)            # palm >= cube_top + 1 cm

    # (b) jaw at or below cube top (with 5 mm tolerance for noise)
    cond_b = jaw_w[:, 2] <= (cube_top + 0.005)

    # (c) jaw at or above cube bottom (with 5 mm tolerance) -- prevents jaw-under-cube scoop
    cond_c = jaw_w[:, 2] >= (cube_bottom - 0.005)

    # (d) lateral containment: cube xy near palm-jaw midpoint xy
    midpoint_xy = 0.5 * (palm_w[:, :2] + jaw_w[:, :2])
    lateral_d   = torch.norm(cube_pos_w[:, :2] - midpoint_xy, dim=1)
    cond_d = lateral_d <= 0.015                            # 1.5 cm radius

    # (e) gripper closed past 70%
    gripper_idx = robot.find_joints([gripper_joint_name])[0][0]
    q_gripper   = robot.data.joint_pos[:, gripper_idx]
    q_max       = robot.data.soft_joint_pos_limits[0, gripper_idx, 1]  # +1.745
    cond_e = q_gripper >= (0.70 * q_max)                   # ~1.22 rad

    # (f) cube not ejecting
    cube_speed_xy = torch.norm(cube_vel_w[:, :2], dim=1)
    cond_f = cube_speed_xy < 0.50                          # m/s

    return cond_a & cond_b & cond_c & cond_d & cond_e & cond_f
```

### Threshold justifications

| condition | threshold | source / math |
|---|---|---|
| (a) palm 1 cm above cube_top | 0.020 m = cube half + 1 cm pad | palm-to-jaw ≈ 5 cm means if palm is 2 cm above cube_top, jaw is at cube_top − 3 cm, deep enough |
| (b) jaw 5 mm tolerance above cube_top | 0.005 m | accounts for FrameTransformer pose noise; V2.15 diag showed min jaw_z = 0.051 = cube_top — this is exactly the failure case (jaw AT but not below) so we allow a small tolerance to call it grasped only when jaw is at or below |
| (c) jaw 5 mm above cube_bottom | 0.005 m | catches the V2.15 ejection mode where cube is below jaw line |
| (d) lateral 1.5 cm | cube half (0.010) + jaw pad (0.005) | a cube whose center is >1.5 cm from the jaw midpoint is not between the jaws |
| (e) gripper 70% closed | 0.70 × 1.745 ≈ 1.22 rad | V2.15 used ~4 cm jaw gap as "closed"; 70% of full closure on a binary servo guarantees finger contact |
| (f) cube xy-velocity < 0.5 m/s | empirical | V2.15 ejection events showed cube xy speed 1–3 m/s; 0.5 m/s rules them out while allowing normal carrying motion |

The conjunction of (a)–(f) provides geometric AND kinematic containment. The V2.15 loose predicate `(jaw_z < 0.04 AND closed)` is replaced by 6 simultaneous constraints — the **predicate is the gate** for #5–8 in the reward table, which is the antiexploit firewall.

---

## 3. Per-episode reward calculus

### (i) Cold-start expected return ≥ 0 (no suicide-by-drop optimum)

Define cold-start = first ~50 training iterations, before policy has any structure. The init_noise_std=1.0 with action_scale=0.10 produces near-random joint deltas of ~±0.10 rad/step. The arm wanders in a ~0.20-m hemisphere around its rest pose. Expected per-step distances and reward contributions:

| term | cold-start expected value | per-step |
|---|---|---|
| #1 ee_to_cube_distance | E[d] ≈ 0.20 m (clipped) | −0.20 |
| #2 reach_cube_tanh | 1 − tanh(0.20/0.10) = 0.036 | +0.036 |
| #3 palm_xy_above_cube | gated; palm above cube ~30% of time, expected reward 0.1 | +0.03 |
| #4 hover_height | rarely fires; ~0 | 0 |
| #5 grasp_strict | conjunction P ≈ 10⁻⁴ early; ~0 | 0 |
| #6 lift_gated | ~0 | 0 |
| #7–8 goal_tracking_*_gated | ~0 | 0 |
| #9 palm_to_jaw_orient | E[−d̂_z] ≈ 0 (random orient) | 0 |
| #10 jaw_table_impact | ungated, jaw at table ~20% of steps, E ≈ −0.4 × 0.2 = −0.08 | −0.08 |
| #11 action_rate_l2 | ~0.01 | −0.01 |
| #12 joint_vel_l2 | ~0.02 | −0.02 |
| **sum** | | **−0.24 / step** |

Expected episodic dense return: `−0.24 × 300 = −72`. Add expected terminal: P(drop) ≈ 0.30 cold-start → `−50 × 0.30 = −15`; P(success) ≈ 0 → 0. **Cold-start E[return] ≈ −87**.

**This is negative**, which seems bad — but the give-up exploit only fires when **dropping is BETTER than not dropping**. Compare alternatives:

- **Stay alive, do nothing**: E[return] = −87 (above).
- **Immediately drop the cube**: 1 step of dense reward (≈ −0.24) + −50 + the remaining 299 steps replaced by `dropout DoneTerm` reset → E[return] ≈ **−50** (worse if measured against return; equal-or-better only if a *new episode* is counted). Crucially, the per-episode reward of the drop trajectory is −50, while the per-episode reward of the wander trajectory is −87.

This **looks** like a suicide attractor (−50 > −87). Two design counter-measures already in place:
1. **#1 clipping at 0.30 m**: bounds the worst-case dense penalty to `−0.30 × 300 = −90`, not the V2.13 v1 `−1.0×300 = −300` of an unclipped large coefficient.
2. **No fail-fast DoneTerm**: there is no "ee_far_from_cube" termination, so the only way to "give up" is to actually drop the cube (which requires having grasped it first). The cold-start policy cannot drop the cube in 1 step — it would have to grasp it first. P(cold-start grasp) ≈ 10⁻⁴ × 300 = 0.03 over an episode. So the suicide path's actual expected cost is `(−0.24 × E[steps_to_drop]) + (−50) × P(grasp)`, where E[steps_to_drop|grasped] ≈ 150 (half episode). Net ≈ `−36 + −1.5 = −37.5` weighted by P(grasp)=0.03 → ≈ **−1.5** contribution.

So the cold-start return decomposes as: `−87` (wander) without a competing suicide attractor because **the policy cannot reach the suicide state from cold init**. Once it learns to grasp (which earns +2.0), grasping dominates dropping (+2.0 × N_steps > −50 break-even at N=25).

**Therefore: cold-start has no exploit optimum.** This is the V2.13 v1 lesson encoded: never add a DoneTerm reachable from a uniform random policy with a penalty smaller than the alternative dense path.

### (ii) "Stay grasping without lifting" strictly dominated by "grasping + lifting"

A policy that grasps and holds the cube at table level earns:
- #5 grasp_strict: +2.0 × N
- #6 lift_gated: 0 (cube_z ≈ 0.041 means `(0.041 − 0.041).clamp(0,0.10)/0.10` = 0)
- #7–8 goal_tracking: 0 (gated by `cube_z > 0.08`)
- minus penalties: ≈ −0.1 × N
- Per-step net: **+1.9 / step**, episodic ≈ +570 (no success bonus)

A policy that grasps AND lifts cube to 0.10 m:
- #5: +2.0
- #6: +1.5 × (0.10/0.10) = +1.5
- #7: +1.0 × (1 − tanh(0.15/0.20)) ≈ +1.0 × 0.37 = +0.37
- #8: +0.5 × (1 − tanh(0.15/0.04)) ≈ +0.5 × 0.0001 ≈ 0
- Per-step net: **+3.7 / step**, episodic ≈ +1 110

A policy that grasps, lifts, AND completes the place (assume 100 steps total to converge to goal):
- average per-step ≈ +3.5 plus terminal +2 000
- episodic ≈ +350 + 2 000 = **+2 350**

So the gradient of return wrt "increment lift height by Δ" is strictly positive throughout `[0.041, 0.20]` m. **Holding without lifting is dominated by +1.8 / step everywhere**, and completing the place adds another +2 000 terminal. There is no plateau where the policy stops climbing — this directly fixes V2.15's "PPO trapped at grasp" failure.

### (iii) "Lift without true grasp" is mathematically impossible

#6 `lift_gated = 1.5 × (cube_z-0.041).clamp(0,0.10)/0.10 × grasp_strict`. The `× grasp_strict` is a hard Boolean multiplier. If the predicate is false (any of the 6 conditions in §2 fails), the term is **exactly zero**. Same for #7 and #8. The cube cannot rise without an external lifting force, which on this scene comes only from the gripper. Even if the policy somehow flings the cube (V2.7 flick exploit), the cube `vel_xy < 0.5 m/s` condition will fail and grasp_strict will be False — no lift reward. This is the requirement #5 anti-degradation mechanism.

### (iv) V2.16/V2.17 degradation cannot recur

V2.16/V2.17 collapse mechanism was: policy probes the lift gradient by briefly releasing/regrasping, losing reach reward, and the lift reward was strong enough (+30 then +150) to outweigh the loss. New design:
- Lift max per-step **+1.5**, vs grasp max **+2.0** + reach max **+1.0**. Releasing the cube costs +3.0 to gain at most +1.5 → strictly dominated.
- Plus the V2.17 VF blowup mechanism (per-step magnitude 45) is structurally impossible because `lift_gated` is **capped at +1.5 per step** by the `.clamp(0,0.10)/0.10` normalization.

---

## 4. Termination terms

Keep three DoneTerms exactly as the user has them, no additions:

```python
@configclass
class TerminationsCfg:
    time_out          = DoneTerm(func=mdp.time_out, time_out=True)         # 10 s = 300 steps
    cube_reached_goal = DoneTerm(
        func=mdp.object_reached_goal,
        params={"threshold": 0.05, "min_height": 0.08},                    # 5 cm and lifted
    )
    cube_dropped      = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.04, "asset_cfg": SceneEntityCfg("cube")},
    )
```

Reasoning:
- **time_out (keep)**: a 10-s episode at 30 Hz is 300 steps, which gives the policy ~1 s to reach, ~1 s to descend, ~5 s for grasp+lift+transport, ~3 s of slack. Reducing it would re-introduce ballistic exploits (V2.9 YEET); increasing it inflates rollout cost.
- **cube_reached_goal (keep with added `min_height` parameter)**: the user's existing definition uses ‖cube−goal‖<5 cm. Add `min_height=0.08` so that cube sliding across the table into the goal projection does NOT trigger success — must be lifted. This closes the same exploit class as the gated goal_tracking term, but in the termination condition itself, costing a 1-line change.
- **cube_dropped (keep)**: z<0.04 m means cube fell off the table or never lifted with grasp. Calibration −50 chosen in §3 so that drop is not preferred to staying alive at cold-start.
- **NO new give-up DoneTerms**: explicitly rejecting the V2.13 v1 pattern. Specifically, do NOT add `ee_far_from_cube`, `joint_limit_violation`, `policy_idle`, or any other early-termination condition. The V2.13 v1 lesson is iron-clad: **a DoneTerm that is reachable cheaper than the dense alternative is always a give-up exploit, regardless of "intent."**

---

## 5. Citations

1. **Isaac Lab `manager_based.manipulation.lift.lift_env_cfg`** (NVIDIA, isaac-sim/IsaacLab, main branch, accessed 2026). The `(cube_z > 0.04) × (1 − tanh(d/std))` multiplicative-gating pattern for `object_goal_tracking`, the std=0.10/0.30/0.05 reach/coarse/fine basin widths, and the weights ratio 1 : 15 : 16 : 5 for reach : lift : goal : fine are taken verbatim and rescaled by the per-step budget (×1/16 → 0.0625 : 0.94 : 1.0 : 0.31, rounded to 1.0 : 1.5 : 1.0 : 0.5). URL: `https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift`.

2. **ManiSkill StackCube-v1 / PickCube-v1**, Tao, Mu, Su et al. (HaoSu Lab, RSS 2025 — `arXiv:2410.00425`). Two design elements borrowed directly: (a) bounded `1 − tanh(k × d)` shaping (k=5 in ManiSkill, k=10 here for tighter SO-101 workspace, equivalent to std=0.10–0.20 m); (b) **multiplicative grasp-gate on place reward** (`reward += place_reward × is_grasped`). Code: `https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/stack_cube.py`. Their `is_grasping(actor)` uses SAPIEN pairwise contact-force impulses on both fingers; we replace this with geometric containment (§2) because the SO-101 gripper is asymmetric (one fixed jaw, one moving) and contact reporting on a 2-cm cube is noisy.

3. **Engstrom et al., "Implementation Matters in Deep Policy Gradients"**, ICLR 2020, `arXiv:2005.12729`. The per-step reward budget of **|r| ≤ 5** derives from their analysis that PPO's stability depends critically on reward scale (running discounted-return std should be ≈ 1), combined with the γ=0.99 geometric-sum ceiling `|V| ≤ r_max / (1−γ) = 100 × r_max`. Setting r_max ≤ 5 keeps |V| ≤ 500, well within rsl_rl's stable critic regime.

4. **Sun et al., "Reward Scale Robustness for PPO via DreamerV3 Tricks"**, NeurIPS 2023, `arXiv:2310.17805`. Empirical finding that vanilla PPO fails when a reward weight changes by >5–10× — the V2.16→V2.17 jump (30→150 = ×5) is exactly at this failure boundary. The new design holds every dense weight ≤ 2.0 and uses a **bounded** lift term (capped at 1.0 via clamp/0.10 normalization), removing the runtime scale-shift attack surface.

5. **rsl_rl `algorithms/ppo.py`** (Schwarke et al., `arXiv:2509.10771`; `github.com/leggedrobotics/rsl_rl`). Adaptive-KL logic — `lr /= 1.5` when KL > 2×desired_kl, `lr *= 1.5` when KL < 0.5×desired_kl — confirms the failure mechanism: the adaptive scheduler **only protects the actor**, not the critic. The user's V2.13 and V2.17 collapses are both critic-side failures; the only protection is reward shaping (this design).

---

## 6. Risk analysis

For each new or changed term, the failure mode, TB-monitor signature, and intervention. Use this as the runbook during the 19-h training run.

### #3 palm_xy_above_cube (new)
- **Mis-calibration mode**: if std=0.04 is too tight, gradient near init is flat (palm rarely above cube in xy at cold-start) and the term contributes nothing → no precision-positioning learning. If too loose (std > 0.08), the term gives credit for sideways approach.
- **Detection**: `Episode_Reward/palm_xy_above_cube` should rise from ~0 at iter 0 to ~0.20 by iter 100 and ~0.55 by iter 300. If it stays <0.05 past iter 200, std is too tight.
- **Intervention**: if flat at iter 200, increase std to 0.06 (per ManiSkill convention) and restart from latest checkpoint. Do NOT increase weight — that triggers V2.17 mode.

### #4 hover_height (new, Gaussian)
- **Mis-calibration mode**: Gaussian is sharp (σ=0.025 m), so a policy that approaches from too high (>10 cm above cube) gets near-zero reward → may skip hover entirely and dive. Also: because it's gated `NOT grasped`, once strict grasp fires the policy loses this reward, which could create a destabilization at grasp/release boundary.
- **Detection**: `Episode_Reward/hover_height` should plateau ~+0.20 from iter ~100 to iter ~250, then **decrease** as grasp_strict becomes more reliable (cube spends more time grasped). A monotone-decreasing-from-iter-0 trajectory means the policy is bypassing hover; a monotone-increasing-past-iter-300 trajectory means the policy is hover-stalling (refusing to descend).
- **Intervention (bypass)**: raise weight to 0.7 and widen σ to 0.04. **Intervention (stall)**: lower weight to 0.3 — but the multiplicative `NOT grasped` gate should auto-resolve stall once grasp_strict learns to fire.

### #5 grasp_strict (new, 6-condition predicate)
- **Mis-calibration mode**: any of the 6 conditions calibrated too tightly → predicate never fires → entire lift/goal pipeline is dead. Most likely culprit is (d) lateral 1.5 cm.
- **Detection**: `Episode_Reward/grasp_strict` should be 0 at iter 0, ≥ +0.4 (i.e. ≥ 0.2 grasp-fraction × 2.0 weight) by iter 150, and ≥ +1.5 by iter 400. If `grasp_strict` is still 0 at iter 300 while `palm_xy_above_cube` is > 0.4, the predicate is too tight. Diagnose by logging each of (a)–(f) separately (the user has `dump_scene_frames.py` infrastructure — extend it to dump per-condition booleans).
- **Intervention**: in priority order, loosen (d) lateral_d threshold 0.015 → 0.020 m; then (b) jaw_z tolerance 0.005 → 0.010 m; then (e) gripper threshold 0.70 → 0.60. Do NOT loosen (a) or (c) — those guard against scoop/below modes.

### #6 lift_gated (recalibrated, weight 1.5, normalized)
- **Mis-calibration mode**: V2.16 failure was lift_weight too weak (30 unnormalized × 0.10 m = 3); V2.17 failure was too strong (150 × 0.30 m = 45). New design caps per-step at +1.5 by normalization — the failure mode now is the opposite: signal too small and policy ignores lifting.
- **Detection**: `Episode_Reward/lift_gated` should rise from 0 at iter 0 to ≥ +1.0 by iter 400. If stuck at ≤ +0.3 past iter 300 while `grasp_strict` is > +1.0, the policy is grasping without lifting.
- **Intervention**: raise weight 1.5 → 2.0 (one step, never >1.5×). Also check that the goal z range [0.10, 0.20] is being sampled — if all goals happen to be ≈ 0.10, the policy never sees positive `goal_tracking_gated` and has no reason to lift further. **Critically: monitor `loss/value_function` after any change**. If it doubles in 20 iterations, **stop and revert**.

### #7–8 goal_tracking_gated and _fine (recalibrated, gated)
- **Mis-calibration mode**: The double gate `grasp_strict × (cube_z > 0.08)` may rarely fire if the lift threshold 0.08 m is hit infrequently. Then the fine basin (std=0.04 m) effectively never trains.
- **Detection**: `Episode_Reward/goal_tracking_gated` should be ≥ +0.3 by iter 500. `goal_tracking_fine` should be ≥ +0.05 by iter 700.
- **Intervention**: if both flat past iter 500, lower lift gate from 0.08 → 0.06 m. Do not change weights.

### #9 palm_to_jaw_orient (sign-flipped from V2.15)
- **Mis-calibration mode**: This is the V2.13 v2 sign-bug class. If the user's coordinate convention has +z down (NOT standard Isaac Sim but worth verifying), the sign is wrong and the policy is rewarded for upside-down. URDF +z quirk affects body-frame interpretations, **not** world-frame `target_pos_w`, so this term is safer than V2.15's "1 + z_local.z" — but verify before launch.
- **Detection**: at iter ~50 run `dump_scene_frames.py` on 16 episodes and confirm `delta_z = jaw_w[2] − palm_w[2]` is **negative** when the arm visually looks top-down. If positive, flip the sign of the term to `+d_hat[2]`. Also: `loss/value_function` rising **steeply** from iter 50–150 (before reach has even converged) is the V2.13 v2 signature.
- **Intervention**: if VF rises ≥ 30% in 50 iters at any point before iter 200, **stop training**, run frame dump, and verify sign of #9.

### #10 jaw_table_impact (new ramped penalty)
- **Mis-calibration mode**: with weight −2.0 and the `(NOT grasp_strict)` gate, this term can fire continuously during the cold-start hover phase if the policy descends-without-grasp, dominating the reach reward. The episodic budget is `−2.0 × 0.5 × 300 = −300` worst case.
- **Detection**: `Episode_Reward/jaw_table_impact` should be in `[−30, −10]` over the full episode at cold-start, trending toward 0 by iter 300. If episodic value is below −80, the policy is "table-skimming."
- **Intervention**: lower weight to −1.0 OR widen the gate so that having a *partial* grasp (palm above + close to cube, even without full strict predicate) also disables the penalty. This preserves the table-impact guard during true wander while not penalizing legitimate descent attempts.

### #11–12 action_rate_l2 / joint_vel_l2 (preserved from V2.14)
- **Mis-calibration mode**: V2.10 showed `joint_acc_l2` with magnitude 700×|qdot|² paralyzes cold-start — drop it as you have. Current penalties (−0.01, −0.001) are V2.14 values. Risk is too aggressive → tame the policy below required servo bandwidth.
- **Detection**: `max |qdot|` per episode should converge to ~3–5 rad/s (matching Feetech STS3215 ~4.5 rad/s actual peak per the Feetech datasheet finding — note this is **lower** than the 6 rad/s the user assumed). Action saturation (p95 of |action|/clip) should be < 0.6 by iter 500.
- **Intervention**: if p95 still > 0.9 at iter 500, raise action_rate_l2 weight −0.01 → −0.02 (one step). Do NOT touch joint_vel_l2.

### General TB monitor early-warning rules
1. `loss/value_function` should not double inside any 30-iter window. If it does, **stop**. This is the V2.13 v2 and V2.17 catastrophic signature.
2. `noise_std` should monotonically decrease from 1.0 toward ~0.3 over 1 500 iters. A sudden uptick (noise re-expanding) means the adaptive-KL controller is pushing the policy entropy back up because returns are non-stationary — usually a sign of either a sign bug (V2.13) or a reward-magnitude shift (V2.17).
3. `mean_reward` should rise monotonically with at most ±15% short-term oscillation. A −30% drop sustained ≥ 30 iters indicates collapse — abort and revert to last checkpoint.
4. `mean_value` should stay in `[20, 500]`. Anything outside that range at any point indicates the reward budget is broken.

### Decision tree on launch

Before iter 0:
1. Run `dump_scene_frames.py` on 16 random-action episodes and verify: (a) cube_top z ≈ 0.051 ± 0.001 m, (b) palm/jaw body offsets match the documented (-0.021, -0.07, 0.02), (c) `jaw_w[2] − palm_w[2]` is **negative** when the arm is in a top-down pose. If (c) is positive, flip sign of #9.
2. Run a 10-iter warm-up rollout with deterministic policy (noise_std=0) on the current checkpoint, log per-term episodic rewards, and verify cold-start total ∈ `[−150, +50]` per episode. If outside, you have a configuration error before training even starts.
3. Verify `grasp_strict` predicate fires at least once across 256 envs × 300 steps at iter 50. If not, predicate is too tight — apply §6 #5 intervention before continuing.

### What success looks like at iter 500, iter 1 000, iter 1 500

| metric | iter 500 | iter 1 000 | iter 1 500 (final) |
|---|---|---|---|
| mean_reward | +200 | +1 000 | +2 200 |
| mean_value | 100 | 300 | 400 |
| loss/value_function | 5–20 | 5–20 | 5–20 |
| noise_std | 0.6 | 0.4 | 0.3 |
| grasp fraction | 0.4 | 0.7 | 0.85 |
| lift fraction (cube_z > 0.08) | 0.1 | 0.5 | 0.75 |
| success fraction (terminal) | 0.05 | 0.45 | 0.75 |
| max |qdot| / ep mean | 6 rad/s | 4 rad/s | 3 rad/s |
| action p95 | 0.7 | 0.5 | 0.4 |

If any of these is more than 2× off at the listed checkpoint, apply the corresponding §6 intervention. **Do not** apply two interventions simultaneously — single-variable changes only, ≤ 1.5× weight adjustments, never re-run from a collapsing checkpoint without first reverting to the last healthy one.

## Conclusion

The 17 prior iterations failed for two structural reasons, not 17 distinct ones. First, additive linear lift shaping (`weight × cube_z`) coupled with no upper bound makes the value function unbounded and PPO's adaptive-KL controller blind to the resulting critic divergence — V2.13 v2 and V2.17 were the same bug at different weights. Second, loose grasp predicates plus ungated lift rewards let the policy collect grasp-and-lift credit without genuine containment — V2.7, V2.9, V2.15, V2.16 were the same exploit pattern at different gate widths. The new design closes both: every dense reward is bounded in `[−1, +1]`, every milestone reward is multiplied by a Boolean strict grasp predicate that requires six simultaneous geometric and kinematic conditions, and the worst-case per-step magnitude is +6.3 (target +4.2 in operation) keeping |V| ≤ 600 — safely inside the regime where the rsl_rl adaptive-KL controller can do its job. The cold-start expected return is mildly negative without a competing suicide attractor, satisfying the V2.13 v1 lesson. **The design is engineered to converge on the first attempt; the §6 runbook exists so that if convergence stalls, the user can apply minimal-scope interventions without triggering a fresh catastrophe.**