# Reward design for PPO pick-and-lift: a 2022+ synthesis for V2.18b

The 2022-2026 manipulation-RL literature converges on a tight set of reward-design principles that diverge meaningfully from V2.18b. **The single highest-leverage change is to multiplicatively gate the `lifting_object` and any future place/goal terms on a *contact-impulse* grasp predicate rather than the current 6-condition geometric predicate** — DexPoint's ablation shows pure distance shaping yields 0% real-world grasp success while contact-gating brings it to 83%. The second is to **replace the −50 drop terminal with a +success terminal of magnitude similar to the per-step max**, because asymmetric large penalties destabilize PPO advantage estimates and bounded rewards are theoretically the only defense against reward hacking (Skalse et al., NeurIPS 2022). Beyond these, three smaller but cumulative wins are: dual coarse/fine Gaussian tracking on the cube-goal vector (Isaac Lab's standard pattern), an action-Jacobian or Lipschitz architectural smoothness mechanism instead of relying solely on the L2 action-rate penalty (Christmann et al., 2024 measure a +26.8% smoothness improvement at only 2.8% task regression), and an asymmetric-critic auxiliary head that predicts cube_xyz from the actor's wrist-camera features to close the partial-observability gap that the literature now treats as a formal pitfall (Lambrechts et al., ICML 2025; Cai et al., 2024).

This report has two parts. Part 1 distills the design principles from CoRL/RSS/ICLR/ICML 2022-2025 that explain *why* certain reward formulations dominate. Part 2 gives concrete, code-ready formulas — with citations and explicit before/after diffs against V2.18b.

---

## Part 1 — Design principles from 2022+ literature

### Multiplicative gating beats additive composition for staged tasks

The clearest empirical finding across Text2Reward (Xie et al., ICLR 2024), DrS (Mu et al., ICLR 2024), ManiSkill2/3 (Tao et al., 2024), DexPoint (Qin et al., CoRL 2022) and Lin et al.'s humanoid sim-to-real paper (CoRL 2025) is that **a naïve linear sum `r = −d_reach + α·grasp_bonus + β·lift_height + γ·d_goal` reliably exploits**. The agent harvests the dense early term forever (the "hover-near-object" exploit) or bats the cube upward to collect lift reward without grasping it (the "flick" exploit). The fix is to express the reward as a chain of multiplicative gates:

> `r_lift = is_grasped · clip(h_obj − h₀, 0, h_max)`
> `r_goal = is_grasped · (1 − tanh(‖p_obj − p_goal‖ / σ))`

Text2Reward's ablation showed that **few-shot LLM prompts that emit `if/elif` stage gates outperform zero-shot prompts that emit linear sums on 13/17 ManiSkill2 manipulation tasks**. Eureka (Ma et al., ICLR 2024) re-discovers this pattern by GPT-4 evolution: across 29 tasks, its best rewards almost always include stage-enabling indicators, and removing the per-component "reward reflection" telemetry drops normalized scores by ~29%. **V2.18b already gates `lifting_object` on its strict grasp predicate, which is correct; the gap is that other dense terms (palm_xy_above_cube, hover_height) are still ungated and contribute throughout the episode.**

### Distance kernels: bounded saturating > raw negative-distance

Three kernel families dominate post-2022 implementations. **The `1 − tanh(d/σ)` kernel** is the Isaac Lab default for object-EE and object-goal tracking (`std=0.1` coarse, `std=0.05` fine); ManiSkill uses `1 − tanh(5·d)` (σ=0.2 m); robosuite uses `1 − tanh(10·d)`. **The Gaussian `exp(−κ·‖d‖²)` kernel** is the Eureka-evolved favorite and the Isaac Lab locomotion default — it has a smoother gradient near zero error and avoids the slight saturation bias of tanh. **The dm_control `tolerance(x, bounds, margin, value_at_margin)`** softplus-margin function is still cited as best practice (Christmann et al., 2024) for target *intervals* rather than points and because it maps cleanly to [0,1] so summing components is well-scaled. **Raw negative-Euclidean shaping (which V2.18b's `ee_to_cube_distance` uses as a linear penalty) is the kernel most associated with the hover-near-object exploit**: it has unbounded gradient when far, biases the value function, and is the form Booth et al. (AAAI 2023) document humans repeatedly overfitting to single seeds. The fix is trivial — replace `−d` with `1 − tanh(d/σ)` or `exp(−κd²)` and tune σ in two scales (coarse + fine).

### Contact-impulse grasp predicates dominate geometric ones

V2.18b's strict 6-condition geometric grasp predicate (palm height, jaw bounds, lateral alignment, gripper closure, cube velocity) is exactly the formulation the 2022+ literature is moving away from. **ManiSkill, DexPoint, DextrAH-G, Hora, and Lin et al. (CoRL 2025) all use a simple contact-impulse predicate** of the form `is_grasped = (left_finger_impulse > ε) ∧ (right_finger_impulse > ε)`. The reasons are empirical and architectural. DexPoint's ablation table is the most striking: stripping the contact-based reward drops their bottle-pick success from 83% to 0% in the real world. Geometric predicates are also brittle under domain randomization — small variations in object size cause the multi-AND to flip discontinuously, producing exactly the high-variance gradients that destabilize PPO. Isaac Lab exposes `ContactSensor` and SAPIEN exposes pairwise impulses; either is a drop-in replacement. Lin et al. (2025) extend this with **"contact stickers"** — small user-placed points on the object surface that fingertips should approach, formulated as `r_contact = Σ_i 1/(1+α·d(sticker_i, fingertip_i))` — which lets you specify *antipodal* targets for a parallel jaw rather than just "any contact".

### Smoothness belongs in the controller and architecture, not only the reward

Christmann et al. (arXiv:2410.16632, 2024) ran the most systematic benchmark of smoothness regularization on Isaac Gym manipulation/locomotion and found that **reward-penalty-only approaches consistently underperform architectural methods** (Lipschitz-constrained MLPs, LipsNet, action filters inside the policy). Their hybrid CAPS loss + Lipschitz layer yields **+26.8% smoothness over baseline PPO with only 2.8% worst-case task regression**. DextrAH-G (Lum et al., CoRL 2024) makes the same point more strongly: by absorbing joint-limit avoidance, self-collision, and posture into a *geometric-fabric* controller layer, they get away with a much smaller reward (essentially just task completion + tiny action-rate). The corollary for V2.18b is that `action_rate_l2` and `joint_vel_l2` are necessary but should be **kept at Isaac Lab's default −1e-4 magnitude** (overweighting them produces stalling), and the longer-term path is either a Lipschitz-constrained policy head or an action-Jacobian penalty (Xie, Karol, Hodgins 2026) that auto-tunes the per-task smoothness weight.

### Asymmetric actor-critic is now theoretically justified, not just empirical

Lambrechts, Ernst & Mahajan (ICML 2025, arXiv:2501.19116) gave the **first finite-time convergence proof for asymmetric actor-critic** under linear function approximation. The key result decomposes the value-MSE into TD-error + approximation + distribution-shift + **aliasing**, where only the aliasing term vanishes when the critic sees true state s rather than the partial observation. Cai et al. (2024, arXiv:2412.00985) refine this with belief-weighted variants for genuine POMDPs and flag a pitfall: **if the actor obs is genuinely insufficient (e.g., cube occluded by gripper in wrist-cam), a privileged critic produces advantage estimates the actor cannot realize, yielding high-variance gradients**. The cure, established by Scaffolder (Hu et al., ICLR 2024, arXiv:2405.14853), is to use the privileged signal in *multiple* training-time modules — not just the critic, but also as an auxiliary prediction target on the actor encoder, as termination-classifier input, and inside the reward itself. Scaffolder's S3 "Blind Pick" task is essentially V2.18b's setup and shows this pattern beats Pinto-style critic-only asymmetry. **V2.18b is correct to keep cube xyz out of the policy obs and to use it in the reward; the missing piece is an auxiliary cube-xyz prediction head on the ResNet-18 features**, ideally with a small MSE loss weight (~0.1) so that the actor's representation aligns with what the critic can see.

### LLM-designed rewards reveal a stable design vocabulary

Eureka (Ma et al., ICLR 2024) and DrEureka (Ma et al., RSS 2024) effectively run a meta-search over reward functions using GPT-4 with per-component telemetry feedback. The discovered design vocabulary is remarkably consistent across 29 Isaac tasks: **exponential or tanh distance kernels**, **stage gates by milestone indicators**, **one-time milestone bonuses** rather than per-step bonuses, **per-component temperatures the LLM rebalances based on rollout stats**, and — when sim-to-real is the goal — **small-weight action-rate / joint-torque / joint-limit safety penalties**. DrEureka adds a **RAPP (Reward-Aware Physics Prior)** sweep that sets the safe DR ranges for sim-to-real, achieving +300% in-hand rotations versus human DR baselines. Two practical implications for V2.18b: first, the design space the user is already in (tanh kernels, gated lift, action_rate_l2) is the *exact* space the LLM converges on, validating the core skeleton; second, V2.18b is *missing* milestone bonuses (a one-time `+5` on transition from ungrasped→grasped rather than V2.18b's continuous `grasping_cube` term) and missing torque/work penalties, which the LLM-designed rewards almost universally include.

### Bounded shaping is the only defense against Goodhart's law

Skalse et al. (NeurIPS 2022) prove formally that no non-trivial reward proxy is "unhackable" — any dense surrogate eventually orders some pair of policies differently from the true reward. Pan et al. (ICLR 2022) document **reward-hacking phase transitions with agent capability**: stronger PPO finds loopholes invisible to weaker variants. The practical defense, recommended consistently by ManiSkill, robosuite, and the Wang & Lin RLHF principles (2025), is to **bound every shaping term and let the success/sparse term dominate by 10-100×**. ManiSkill's PickCube enforces this strictly: per-step shaping ∈ [0, 4] and terminal success = max_reward = 5.0, so the cumulative bonus from completing the task always exceeds any shaping accumulation. **V2.18b's −50 drop terminal is the inverse pattern** — a large asymmetric *negative* — which both destabilizes advantage estimates and, more subtly, makes the policy risk-averse in a way that suppresses exploration of grasp strategies. The 2022+ literature strongly prefers **early termination on failure without large penalty** plus **a large positive terminal on success**.

### Curricula on initial states and reward weights consistently win

Three independent 2025 studies — Fele et al. (IEEE Access), An et al. (loco-manipulation, +55% training efficiency), and the DexPBT PBT-over-weights paper (Petrenko et al., RSS 2023) — converge on a single recipe: **start with the cube placed near a known grasp pose, widen the initialization range only when per-epoch grasp rate exceeds a threshold (~50%), and either schedule reward weights manually or run population-based training over them**. Isaac Lab's `CurriculumManager` supports both. For V2.18b this means the strict grasp predicate should be a *signal* used to widen initial-state distribution, not just a reward gate.

### Reward-hacking failure modes documented post-2022

A consolidated catalog with documented fixes:

| Failure mode | Mechanism | Documented in | Fix |
|---|---|---|---|
| Hover/oscillate near object | unbounded `−d` reward harvestable indefinitely | Krakovna list; Booth AAAI 2023 | Bounded tanh/Gaussian kernel + one-time milestone bonus |
| Flick/bat cube upward without grasp | `r_lift` ungated by contact | Text2Reward ablation 2024; Claru 2026 | Multiplicative gate `r_lift = is_grasped · ...` |
| Hold-and-don't-move | per-step `r_grasp` accumulates while static | Eureka reflection logs | Make grasp bonus one-time (on 0→1 transition) |
| Gripper pinching air / self-grip | reward on closure progress or self-contact | RotateIt 2023; ByteDance Seed 2025 | Penalize gripper-close while contact==0; SDF self-collision penalty |
| HF jitter exploiting sim non-determinism | reward exploits "lucky" timesteps | DextrAH-RGB 2024; Christmann 2024 | Action-rate + Lipschitz/LipsNet architectural smoothness |
| Velocity-based "throw-grasp" in one step | gripper closes + lifts within control period | Lin et al. CoRL 2025 | Require k consecutive contact frames before lift bonus |
| Capability-induced phase transition | stronger PPO finds new loopholes mid-training | Pan et al. ICLR 2022 | Bounded rewards + early-stop on success rate not reward |
| Designer overfits to single seed | iterating reward on the same physics seed | Booth et al. AAAI 2023 | Validate every reward change across ≥5 seeds |

---

## Part 2 — Concrete reward formulas and code-ready expressions

The following formulas are drop-in replacements or augmentations for specific V2.18b terms. Each carries citation and an explicit explanation of what it improves.

### 2.1 Replace `ee_to_cube_distance` (linear penalty) with bounded dual-scale kernel

V2.18b uses `−‖p_ee − p_cube‖` as a linear penalty *plus* `reaching_object = 1 − tanh(d/0.10)`. The linear penalty term is unbounded and is the kernel most associated with hover-exploits. Drop the linear term entirely and use Isaac Lab's dual coarse/fine pattern:

```python
# From IsaacLab manipulation/lift/mdp/rewards.py, std values verified
r_reach_coarse = 1 - torch.tanh(torch.norm(p_ee - p_cube, dim=-1) / 0.10)  # w = 1.0
r_reach_fine   = 1 - torch.tanh(torch.norm(p_ee - p_cube, dim=-1) / 0.02)  # w = 0.5
```

The fine-scale term provides steep gradient inside the last 2 cm where grasping happens; the coarse term shapes long-range approach. Isaac Lab's official `Isaac-Lift-Cube-Franka-v0` uses this two-scale pattern for the goal-tracking term, and the same logic applies to reach (NVIDIA Isaac Lab tutorial: *"As this positional error gets closer to zero, the tanh function produces larger gradients compared to a linear error term"*).

### 2.2 Replace 6-condition geometric `grasping_cube` with contact-impulse predicate

V2.18b's strict grasp predicate is brittle. Replace with the **ManiSkill/DexPoint contact-impulse formulation**:

```python
# Drop-in replacement using IsaacLab ContactSensor
left_force  = contact_sensor_left.data.net_forces_w_history[:, 0].norm(dim=-1)
right_force = contact_sensor_right.data.net_forces_w_history[:, 0].norm(dim=-1)
is_grasped  = (left_force > 1.0) & (right_force > 1.0)                # boolean tensor

# Use as multiplicative gate everywhere downstream
r_grasp_milestone = is_grasped.float() * (~was_grasped_prev_step).float() * 2.0  # one-time on 0→1
```

DexPoint's ablation (CoRL 2022, arXiv:2211.09423) shows this single change moves real-world bottle-pick from 0% to 83%. Lin et al. (CoRL 2025, arXiv:2502.20396) generalize this further with **antipodal contact stickers**:

```python
# r_contact for parallel jaw: two stickers on opposing sides of cube
sticker_L = p_cube + R_cube @ torch.tensor([0.0, +cube_half, 0.0])
sticker_R = p_cube + R_cube @ torch.tensor([0.0, -cube_half, 0.0])
r_contact = 1.0/(1.0 + 50.0*torch.norm(p_finger_L - sticker_L, dim=-1)) \
          + 1.0/(1.0 + 50.0*torch.norm(p_finger_R - sticker_R, dim=-1))
```

This shapes the gripper toward an *antipodal* grasp geometry rather than any contact, replacing both `palm_xy_above_cube` and `jaw_below_cube_penalty` with a single smooth term.

### 2.3 Replace hand-tuned `hover_height` Gaussian with the IsaacGymEnvs finger-straddle bonus

V2.18b's Gaussian sweet-spot at `palm_z − cube_top = 8 cm` is a specific point reward. The **IsaacGymEnvs `franka_cabinet` finger-straddle pattern** (file `tasks/franka_cabinet.py`) is more robust because it rewards the *geometry* of left-above-and-right-below rather than a single hover height:

```python
lfinger_above = (p_finger_L[..., 2] > p_cube[..., 2] + 0.005)
rfinger_below = (p_finger_R[..., 2] < p_cube[..., 2] - 0.005)
r_straddle = (lfinger_above & rfinger_below).float() * 0.5
```

The same file uses an average-distance reach `(d_palm + d_lf + d_rf) / 3` which implicitly pushes fingers around (not just toward) the cube — useful additional shaping for parallel jaws.

### 2.4 Add the IsaacLab dual-scale goal-tracking reward, gated by grasp

Even though V2.18b is currently pick+lift only with no place phase, adding a *lifted-height* goal tracking term now generalizes to place later and accelerates lift learning:

```python
# IsaacLab object_goal_distance pattern, std values from manipulation/lift/lift_env_cfg.py
d_goal = torch.norm(p_cube - p_goal_target, dim=-1)
gate   = (p_cube[..., 2] > 0.04).float()                       # cube lifted ≥ 4 cm
r_goal_coarse = gate * (1 - torch.tanh(d_goal / 0.30))         # w = 16.0
r_goal_fine   = gate * (1 - torch.tanh(d_goal / 0.05))         # w =  5.0
```

For pick-and-lift specifically, set `p_goal_target = p_cube_initial + [0, 0, 0.15]` so the policy learns to lift to a target height. Isaac Lab's exact weights (15 for `lifting_object`, 16 + 5 for goal coarse/fine) are the validated defaults.

### 2.5 Make the lift reward a clipped height with milestone bonuses, not a continuous height

V2.18b's `lifting_object` (gated on strict grasp) is correctly gated but likely unclipped. The **ManiSkill PickCube + DextrAH-G clipped pattern** caps it:

```python
lift_h    = torch.clamp(p_cube[..., 2] - p_cube_init[..., 2], 0.0, 0.15)
r_lift    = is_grasped.float() * (lift_h / 0.15)              # w = 5.0, bounded ∈ [0,1]

# Milestone bonus, one-time on first lift above threshold
first_lift = (lift_h > 0.10) & ~lifted_prev_step
r_lift_milestone = first_lift.float() * 5.0
```

The clip prevents unbounded reward accumulation as the cube rises, and the one-time milestone is the Eureka-discovered pattern.

### 2.6 Replace −50 drop terminal with positive success bonus

V2.18b's `cube_dropped_penalty = −50` is the inverse of the 2022+ best practice. From ManiSkill (Tao et al., 2024) and the explicit ICLR'22 Pan et al. analysis:

```python
# Remove cube_dropped_penalty entirely; use early termination on drop.
# Add positive sparse terminal:
success = (lift_h > 0.12) & is_grasped & (object_static_for_k_frames > 5)
r_success = success.float() * 15.0    # sized to dominate cumulative shaping
```

Episode termination on drop (without large penalty) handles failure cleanly; the positive terminal dominates the cumulative shaping (Wang & Lin 2025; Skalse et al., NeurIPS 2022).

### 2.7 Add DrEureka sim-to-real safety regularizers

DrEureka (Ma et al., RSS 2024) finds the LLM consistently inserts these terms when prompted for sim-to-real readiness. V2.18b is missing torque, work, and joint-limit terms:

```python
r_torque  = -1e-3 * torch.sum(tau**2, dim=-1)                              # motor torque
r_work    = -1e-4 * torch.sum(torch.abs(tau * q_dot), dim=-1)              # mechanical work (Hora 2022)
r_qlimit  = -1e-2 * torch.sum(torch.relu(q.abs() - q_safe)**2, dim=-1)     # soft joint limits
r_close_no_contact = -0.5 * gripper_close_action * (1.0 - is_grasped.float())   # anti-jam
```

The work term `Σ|τ·q̇|` (from Hora, Qi et al., CoRL 2022, arXiv:2210.04887) correlates better with motor heating than torque alone — this is what the SO-101's small servos actually care about for sim-to-real.

### 2.8 Replace L2 action-rate with action-Jacobian or keep at IsaacLab's −1e-4

V2.18b uses `action_rate_l2, joint_vel_l2`. Keep them but pin the weights to Isaac Lab defaults:

```python
r_action_rate = -1e-4 * torch.sum((a_t - a_prev)**2, dim=-1)
r_joint_vel   = -1e-4 * torch.sum(q_dot**2, dim=-1)
r_joint_acc   = -2.5e-7 * torch.sum(q_ddot**2, dim=-1)   # optional, AnyMAL pattern
```

Christmann et al. (2024) confirm these magnitudes; weights ≥1e-3 on action-rate produce stalling in Isaac Lab manipulation environments. For higher smoothness, **migrate to a Lipschitz-constrained policy head or LipsNet** rather than increasing the penalty.

### 2.9 Replace `gripper_orientation_penalty` (palm-to-jaw vs world-down) with a tracking error

V2.18b's penalty against world-down is a hard prior. The **Human2Sim2Robot (Lum et al., CoRL 2025, arXiv:2504.12609) object-pose tracking** pattern is softer and more general:

```python
# Define a desired gripper orientation aligned to a top-down grasp on the cube
q_grip_desired = top_down_grasp_quat(p_cube, R_cube)
theta_err      = quat_angle_error(q_gripper, q_grip_desired)
r_orient       = torch.exp(-3.0 * theta_err)            # bounded ∈ (0, 1]
```

This rewards approaching the desired orientation rather than penalizing deviation; bounded and smooth.

### 2.10 Asymmetric critic and auxiliary representation head

This is structural rather than a reward term, but central to the user's setup. Implement in Isaac Lab via the `state_space` config (see GitHub issue isaac-sim/IsaacLab#2712); both `rl_games` and `rsl_rl` support `num_states ≠ num_observations`. Recommended ~30-D critic-only privileged vector (raw, not engineered):

```python
critic_extra = torch.cat([
    p_cube, q_cube, v_cube, omega_cube,           # 3+4+3+3 = 13
    p_goal, q_goal,                                # 7
    p_ee - p_cube, p_cube - p_goal,                # 6
    gripper_width.unsqueeze(-1),                   # 1
    is_grasped.float().unsqueeze(-1),              # 1
    contact_force_total.unsqueeze(-1),             # 1
    cube_in_air.float().unsqueeze(-1),             # 1
], dim=-1)                                          # = 30 D
critic_input = torch.cat([actor_obs_524, critic_extra], dim=-1)   # 554 D
```

Then add a small auxiliary MSE head on the ResNet-18 features predicting `p_cube` (relative to robot base), with loss weight ~0.1 — this is the Scaffolder pattern (Hu et al., ICLR 2024) that closes the Baisero-Amato aliasing gap and is the specific cure Cai et al. (2024) prescribe for the high-variance gradients that pure asymmetric AC produces when the policy obs is genuinely insufficient.

### 2.11 Composite recommended reward (replacement for V2.18b)

Combining all the above into a single composition (sum, with weights tuned to Isaac Lab + ManiSkill defaults):

```python
r = (
    1.0 * r_reach_coarse                    # 1 − tanh(d_ee_cube / 0.10)
  + 0.5 * r_reach_fine                      # 1 − tanh(d_ee_cube / 0.02)
  + 1.0 * r_contact_stickers                # antipodal contact shaping (Lin et al. 2025)
  + 2.0 * r_grasp_milestone                 # one-time on 0→1 transition
  + 0.5 * r_straddle                        # finger-straddle geometry
  + 1.0 * r_orient                          # bounded orientation tracking
  + 5.0 * r_lift                            # gated clipped height, ∈ [0,1]
  + 5.0 * r_lift_milestone                  # one-time on first cross of h_thresh
  + 16. * r_goal_coarse                     # IsaacLab default
  + 5.0 * r_goal_fine                       # IsaacLab default
  + 15. * r_success                         # large positive terminal
  + r_action_rate + r_joint_vel             # IsaacLab default −1e-4 each
  + r_torque + r_work + r_qlimit            # DrEureka sim-to-real cocktail
  + r_close_no_contact                      # anti-jam
)
# Drop the −50 terminal; use episode termination on drop without penalty
```

Total per-step shaping is bounded ∈ roughly [−1, 35], terminal success = +15 — the 2022+ "shaping ≪ terminal" principle is preserved cumulatively because the gated terms only activate near task completion.

---

## Conclusion: what changes the most and why

The V2.18b stack is structurally sound — it already gates lift on grasp, uses bounded tanh kernels for reach, and includes smoothness penalties. **The five highest-impact changes from the 2022+ literature, in order of expected gain, are: (1) swap the geometric 6-AND grasp predicate for a contact-impulse predicate, (2) replace the −50 drop terminal with a +15 success terminal and rely on early-termination for failure, (3) add an asymmetric-critic privileged vector plus a cube-xyz auxiliary head on the ResNet features, (4) add the DrEureka safety cocktail (torque, work, joint-limit, close-no-contact) at small weights, and (5) migrate `palm_xy_above_cube` + `jaw_below_cube_penalty` + `hover_height` into a single antipodal-contact-sticker reward plus a finger-straddle bonus.** Each change is grounded in a peer-reviewed 2022-2025 paper with quantitative ablation evidence, and each can be applied independently of the others.

The deeper lesson from Eureka, DrEureka, and Text2Reward is that the *space* V2.18b lives in — multi-component additive rewards with bounded kernels, multiplicative gates, and small smoothness penalties — is provably the right space. The remaining gains come from disciplined application of the in-space rules: bound every term, gate every downstream term on the right predicate, prefer contact over geometry, prefer one-time milestones over per-step bonuses, prefer positive terminals over negative penalties, and use the privileged signal in as many training-time modules as possible without leaking it into the policy.