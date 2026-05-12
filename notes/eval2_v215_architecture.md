# V2.15 — Full architecture (rewards + PPO + ResNet pipeline)

Complete reference for the V2.15 RL pipeline : data flow, networks,
observation/action spaces, all 13 reward terms with formulas + weights,
and PPO hyperparameters.

> **Variant** : `Isaac-LeIsaac-SO101-Lift-{RL,Visual}-V215-{,Play-}v0`
> **PPO config** : `sim/eval2/agents/rsl_rl_ppo_cfg_v2_13.py`
> **Env config** : `sim/eval2/leisaac_lift_env_cfg.py::LeIsaacLiftCubeRLVisualEnvCfgV215`

---

## 1. Global pipeline

```
   ┌────────────────────────────────────────────────────────────────┐
   │                  Isaac Lab — Manager-Based Env                  │
   │                                                                 │
   │   ┌─────────┐    ┌───────────┐    ┌─────────────────────────┐  │
   │   │  Wrist  │──▶ │ ResNet-18 │──▶ │  wrist_features (512D)  │  │
   │   │  RGB    │    │  frozen   │    └─────────────────────────┘  │
   │   │ 224x224 │    │  eval()   │              │                   │
   │   └─────────┘    └───────────┘              │                   │
   │   ┌─────────────────────────────────────────┘                   │
   │   │  joint_pos (6) + joint_vel (6) + object_pos (3) + ...       │
   │   │  + actions (6) + placeholders (9) + relative_vecs (6) = 43D  │
   │   └────────────────────┬────────────────────────────────────────┘
   │                        ▼
   │            ┌────────────────────────┐                            │
   │            │  Observation = 555D    │                            │
   │            └───────────┬────────────┘                            │
   │                        │                                          │
   │       ┌────────────────┴────────────────┐                        │
   │       ▼                                  ▼                        │
   │  ┌──────────┐                       ┌──────────┐                  │
   │  │  Actor   │                       │  Critic  │                  │
   │  │   MLP    │                       │   MLP    │                  │
   │  │ [256,    │                       │ [256,    │                  │
   │  │  128,    │                       │  128,    │                  │
   │  │  128]    │                       │  128]    │                  │
   │  │   ELU    │                       │   ELU    │                  │
   │  └─────┬────┘                       └─────┬────┘                  │
   │        ▼                                  ▼                        │
   │   μ (6D), σ                           V(s) ∈ ℝ                    │
   │        │                                                          │
   │        ▼                                                          │
   │   action ~ N(μ, σ²)                                               │
   │        │                                                          │
   │        ▼                                                          │
   │  ┌─────────────────────────┐                                      │
   │  │  Action processing      │                                      │
   │  │  clip = [-1, +1]        │                                      │
   │  │  scale = 0.10 (DELTA)   │                                      │
   │  │  → delta joint pos      │                                      │
   │  └─────────────────────────┘                                      │
   │        │                                                          │
   │        ▼                                                          │
   │  ┌─────────────────────────┐                                      │
   │  │  SO-101 robot (PhysX)   │                                      │
   │  │  + cube + table + goal  │                                      │
   │  └─────────────────────────┘                                      │
   │        │                                                          │
   │        ▼                                                          │
   │   rewards (13 terms) + dones (3 terms)                            │
   └────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  rsl_rl PPO (rollout 50 steps × 256 envs)
                            │
                            ▼
                      gradient step
```

256 parallel environments per training iter. Each iter = 50 rollout
steps × 256 envs = 12 800 timesteps of experience.

---

## 2. ResNet-18 encoder (frozen)

Defined in [`sim/eval2/policy/visual_encoder.py`](../sim/eval2/policy/visual_encoder.py).

| Property | Value |
|---|---|
| Architecture | ResNet-18 (torchvision, ImageNet-pretrained) |
| Mode | `eval()` — BatchNorm stats frozen |
| Trainable | **No** — gradients NOT flowing back to ResNet |
| Input | RGB 224 × 224, uint8 from wrist `TiledCamera` |
| Input normalization | ImageNet mean = `[0.485, 0.456, 0.406]`, std = `[0.229, 0.224, 0.225]` |
| Output | 512-D feature vector (final avg-pool output) |
| Computed where | inside the observation term `wrist_image_features` |
| Computed when | once per env step, before the actor sees it |

**Why frozen** : (1) PPO struggles to learn good vision features from
reward gradient alone — IL/BC is much more sample-efficient for vision.
(2) Pre-trained ImageNet features are good enough for distinguishing
"cube on table" patterns. (3) Freezing keeps the actor's parameter
count small (no ~11 M ResNet params in the policy).

**Trade-off** : ResNet features are not domain-specific. If sim-to-real
domain gap is large, the policy may overfit ResNet feature quirks. To
be re-evaluated after real-robot deployment.

---

## 3. Observation space (555-D, policy group)

Defined in [`leisaac_lift_env_cfg.py::ObservationsCfgVisualV2`](../sim/eval2/leisaac_lift_env_cfg.py) and inherited up to V2.15.

| idx | term | dim | source | semantic |
|---|---|---|---|---|
| 0 | `joint_pos` | 6 | robot joints | absolute joint angles (rad) |
| 1 | `joint_vel` | 6 | robot joints | joint velocities (rad/s) |
| 2 | `object_position` | 3 | cube | cube position in robot base frame |
| 3 | `target_object_position` | 7 | command | goal pose (xyz + quat) in robot base frame |
| 4 | `actions` | 6 | action manager | last action commanded |
| 5 | `wrist_features` | **512** | ResNet-18 | frozen visual features |
| 6 | `target_color_placeholder` | 6 | zeros | reserved for Eval 2 multi-color (Phase C) |
| 7 | `bowl_xyz_placeholder` | 3 | zeros | reserved for Eval 2 bowl position (Phase C) |
| 8 | `ee_to_cube_vec` | 3 | computed | `cube_pos_w − palm_pos_w` (world frame) |
| 9 | `cube_to_goal_vec` | 3 | computed | `goal_pos_w − cube_pos_w` (world frame) |
| | **total** | **555** | | |

The Phase-C placeholders are zeros now but already in the obs to keep
the policy network shape forward-compatible with the multi-cube + bowl
Eval 2 task without rebuilding from scratch.

The relative vectors `ee_to_cube_vec` and `cube_to_goal_vec` are
explicitly given to the network so the MLP doesn't have to learn the
subtraction operation from the absolute positions.

---

## 4. Action space (6-D)

| idx | name | type | range |
|---|---|---|---|
| 0–4 | arm joints (5×) | `RelativeJointPositionActionCfg` | raw `clip = [-1, +1]`, then `scale = 0.10` → joint delta capped at ±0.10 rad/step |
| 5 | gripper (1×) | `BinaryJointPositionActionCfg` | binarized : action > 0 → open (0.5 rad), action < 0 → close (0.0 rad) |

**Velocity cap (V2.14+)** : `vmax = scale / dt_ctrl = 0.10 / (1/30 s) = 3 rad/s`,
i.e., 50 % of the Feetech STS3215 motor limit (6 rad/s). Hard cap by
construction — the policy cannot demand faster motion.

**Why DELTA + clip + scale = 0.10** :
- DELTA control (vs absolute) decouples the action range from the joint
  angle range — `scale` becomes a velocity cap, not a position cap.
- `clip = [-1, +1]` ensures `scale` actually bounds the per-step delta
  regardless of the policy's raw output (which can technically be
  unbounded since `init_noise_std=1.0`).
- `scale = 0.10` is the V2.14 "Slow & Precise" calibration to kill the
  Bang-Bang vertical smash exploit observed in V2.13 v3.

---

## 5. Reward architecture (13 terms)

Defined in [`leisaac_lift_env_cfg.py::RewardsCfgV215`](../sim/eval2/leisaac_lift_env_cfg.py) (inheritance chain V2 → V2.7 → V2.8.5 → V2.12 → V2.13v3 → V2.14 → V2.15). Custom MDP functions in [`sim/eval2/mdp/rewards.py`](../sim/eval2/mdp/rewards.py).

### 5.1 Positive terms (drive)

| name | weight | formula | gating | role |
|---|---|---|---|---|
| **`reaching_object`** | +3.0 | `tanh(d_ee_cube / 0.15)` where `d_ee_cube = ‖palm_pos − cube_pos‖` (then `1 − tanh(...)`) | none | dense, saturates at d > 60 cm. Anti-suicide insurance + bonus when policy is near cube. |
| **`grasping_cube`** | +5.0 | `(d_jaw_cube < 0.04) ∧ (gripper_angle < 0.26 rad)` | none | binary intermediate signal — fires when the jaw is close to the cube AND the gripper is partially closed. |
| **`lifting_object`** | +10.0 | `cube_z_rel > 0.08 m` above robot base | **grasp ∧ lift** (V2.7 anti-flick gate) | binary — cube is sustained above table. |
| **`object_goal_tracking`** | +16.0 | `tanh(d_cube_goal / 0.3)` (then `1 − tanh(...)`) | grasp ∧ lift | dense, large-radius pull toward goal. |
| **`object_goal_tracking_fine_grained`** | +5.0 | `tanh(d_cube_goal / 0.05)` (then `1 − tanh(...)`) | grasp ∧ lift | dense, small-radius precision. |
| **`success_bonus`** | **+2500** | `‖cube_pos − goal_pos‖ < 0.05 m` | terminal | sparse — massive one-shot reward when goal is reached. |

### 5.2 Negative terms (smoothness + posture)

| name | weight | formula | gating | role |
|---|---|---|---|---|
| **`action_rate_l2`** | -1e-3 | `‖aₜ − aₜ₋₁‖²` summed across joints | n/a | smoothness — penalize action saccades (V2.14 ×10 vs V2.13 default −1e-4). |
| **`joint_vel_l2`** | -1e-3 | `‖q̇‖²` summed across arm joints | n/a | smoothness — penalize absolute joint velocity (V2.14 ×10). |
| **`ee_to_cube_distance`** | -3.0 | `‖palm_pos − cube_pos‖` (raw linear) | none | non-saturating driver toward the cube — keeps a non-zero gradient even after `reaching_object`'s tanh has saturated. |
| **`cube_dropped_penalty`** | -30.0 | `cube_z < 0.04 m` (binary) | n/a | discourages dropping the cube off the table (real fall) AND suicide-by-drop exploit. |
| **`gripper_orientation_penalty`** | **-5.0** | `(1.0 + δ̂_z)` where `δ̂ = (jaw_pos − palm_pos) / ‖jaw_pos − palm_pos‖` (V2.15 palm→jaw direction) | none | forces top-down grasp posture. **0** when jaw is directly below palm (down), **1** when horizontal (snake), **2** when jaw above palm (gripper-up). |
| **`jaw_below_cube_penalty`** | **-50.0** | `clamp((cube_bottom − jaw_z), min=0)` where `cube_bottom = cube_pos_z − 0.011` | n/a | NEW V2.15. Physics-breach prevention — jaw penetrating below the table top (= below the cube's bottom) is punished. |
| `scoop_grasp_penalty` | 0 (dropped) | `clamp((palm_z − wrist_z), min=0)` | none | DEAD (V2.13 v3+). Was meant to detect twisted-wrist scoop pose but redundant with the new orientation penalty. |

### 5.3 Per-step calculation example (cold-start, d_ee_cube ≈ 0.24 m, random orientation ≈ 0.5)

```
+0.24    reaching_object   (+3 × (1 − tanh(0.24/0.15)) = +3 × 0.08)
+0.00    grasping_cube     (no grasp yet)
+0.00    lifting_object    (gated off)
+0.00    object_goal_*     (gated off)
+0.00    success_bonus     (terminal only)
-0.001   action_rate
-0.025   joint_vel         (random commands × 5 joints)
-0.72    ee_to_cube_dist   (-3 × 0.24)
0        cube_dropped      (not dropped)
-2.5     gripper_orient    (-5 × 0.5, palm-jaw direction ~ horizontal)
0        jaw_below_cube    (not below)
─────
≈ -2.99 per step → -897 per 300-step episode (timeout)

Suicide-by-drop @ step 20 :  20 × (-3.0) + (-30) = -90    [from drop term]
                                                    +60 from reach   = -30 net

Stay-near at saturated reach (d=0.05, top-down, orient=0) :
+2.04   reach   (+3 × (1 − tanh(0.05/0.15)) = +3 × 0.68)
-0.15   ee_to_cube
0       orient (top-down)
─────
+1.89 per step → +567 per episode → MUCH better than suicide → policy seeks this.
```

The intent of the V2.15 reward shaping :
1. Linear `ee_to_cube_distance` drives the EE toward the cube globally.
2. Saturating `reaching_object` rewards proximity (anti-suicide).
3. `gripper_orientation_penalty` (V2.15 palm→jaw direction) makes
   top-down the only zero-cost approach posture.
4. `jaw_below_cube_penalty` forces the jaw to stop AT the cube level,
   not under it (= no table penetration).
5. `grasping_cube` and `lifting_object` are sparse positive signals
   gated on each other to prevent flick exploits.
6. `success_bonus` is the huge terminal incentive that propagates back
   through the value function.

---

## 6. Termination terms

| name | type | trigger | reward |
|---|---|---|---|
| `time_out` | `TimeOut` | episode length 10 s = 300 steps reached | nothing |
| `success` | DoneTerm | `‖cube_pos − goal_pos‖ < 0.05 m` | already gives the `success_bonus = +2500` via the reward manager (paired) |
| `cube_dropped` | DoneTerm | `cube.z < 0.04 m` (world frame) | already gives `cube_dropped_penalty = -30` via the reward manager (paired) |

The `ee_far_from_cube` DoneTerm (V2.12 fail-fast) is **disabled** since
V2.13 v2 (caused the give-up exploit when paired with high-magnitude
per-step penalties).

---

## 7. PPO hyperparameters

Defined in [`sim/eval2/agents/rsl_rl_ppo_cfg_v2_13.py::LiftCubePPORunnerCfgV213`](../sim/eval2/agents/rsl_rl_ppo_cfg_v2_13.py) (shared by V2.13 → V2.15, env-side changes only).

### 7.1 Network

| Param | Value | Notes |
|---|---|---|
| `actor_hidden_dims` | `[256, 128, 128]` | 4-layer MLP (input 555 → 256 → 128 → 128 → 6) |
| `critic_hidden_dims` | `[256, 128, 128]` | same shape as actor, outputs scalar V(s) |
| `activation` | `elu` | smooth activation, helps with gradient flow |
| `init_noise_std` | 1.0 | initial standard deviation of the Gaussian action policy; large = exploratory |

The actor outputs the mean of a Gaussian policy `N(μ, σ²)`, where `σ`
is a separately-learned parameter (not output by the MLP). Actions are
sampled from this Gaussian at each step.

### 7.2 PPO algorithm

| Param | Value | Role |
|---|---|---|
| `gamma` | 0.99 | discount factor — at γ=0.99 with 300-step episodes, the reward at step 300 has weight 0.99³⁰⁰ ≈ 0.049 in the return at step 0. |
| `lam` (GAE λ) | 0.95 | Generalized Advantage Estimation discount — controls bias/variance trade-off in advantage estimates. |
| `clip_param` | 0.2 | PPO surrogate-loss clipping range (`±20 %` change in policy ratio per update). |
| `value_loss_coef` | 1.0 | weight of the value-function loss in the total loss (V2.8 fix vs V2 default 0.5 which caused VF starvation). |
| `use_clipped_value_loss` | True | value loss also uses a PPO-style clipping. |
| `entropy_coef` | 0.005 | bonus for policy entropy — encourages exploration. |
| `num_learning_epochs` | 5 | passes over each rollout batch (V2.8 fix vs V2 default 4). |
| `num_mini_batches` | 4 | each rollout split into 4 minibatches (V2.8 fix vs V2 default 32). |
| `learning_rate` | 1e-3 | base learning rate (V2.8 increase vs V2 default 3e-4). |
| `schedule` | `adaptive` | LR adapts to keep KL divergence near `desired_kl`. |
| `desired_kl` | 0.02 | target KL per gradient step (V2.12 fix vs default 0.01 → too tight for the V2.7+ shaping). |
| `max_grad_norm` | 1.0 | gradient clipping (V2.8 fix vs V2 default 0.5). |

### 7.3 Rollout / training loop

| Param | Value | Notes |
|---|---|---|
| `num_steps_per_env` | 50 | per-env rollout length per iter |
| `num_envs` | 256 | parallel environments |
| `total per iter` | 50 × 256 = 12 800 | training samples per gradient step |
| `max_iterations` | 1500 | total iters → ~19 M samples total |
| `save_interval` | 50 | checkpoint every 50 iters → `model_0.pt`, `model_50.pt`, … |
| `empirical_normalization` | False | observations are NOT auto-normalized; the env provides them in reasonable ranges already. |
| `experiment_name` | `lift_v2_13` | logs go to `logs/rsl_rl/lift_v2_13/<timestamp>/` |

The KL-adaptive LR schedule works like this each iter :
- Compute KL divergence between old and new policy across the batch.
- If KL > 2 × `desired_kl` → halve LR.
- If KL < 0.5 × `desired_kl` → double LR.
- Otherwise keep LR.

This is critical with our shaping : if the policy makes too large an
update (e.g., when grasp first starts firing and the gradient spikes),
the adaptive LR throttles to keep training stable.

---

## 8. Episode structure

| Property | Value |
|---|---|
| `episode_length_s` | **10.0 s** (V2.14+, was 5 s) |
| Control rate | 30 Hz |
| `decimation` | 2 (physics runs at 60 Hz, control reads/writes every 2 physics steps) |
| Total steps per episode | 10 / (1/30) = **300** |
| Reset events | `reset_all` (LeIsaac stock reset) + `reset_cube_position` (cube xyz/yaw randomization) |

At reset :
- Robot joints set to default home pose (all zero).
- Cube spawned at a random position in a 15×15 cm box around the
  natural cube spawn point in front of the robot, with ±30° yaw.
- Goal pose drawn uniformly from x ∈ [-0.1, +0.1] m, y ∈ [-0.2, -0.1] m,
  z ∈ [+0.1, +0.2] m in the robot base frame.
- Cube root_pos_w.z = **0.0410 m** (constant — cube center).
- Cube bottom = 0.031 m = table top level.

---

## 9. Putting it together — one PPO iter

```
1.  For each of 256 envs:
    a.  Sample 50 consecutive env steps:
        - At each step, encode wrist RGB through ResNet → 512 features.
        - Build 555-D observation.
        - Actor MLP outputs μ; sample action ~ N(μ, σ²).
        - Clip raw to [-1, +1], scale by 0.10 → joint delta.
        - Apply via PhysX, step world by decimation=2 physics steps.
        - Receive 13 reward terms summed + dones.
        - Store (obs, action, reward, done, V(s)) in rollout buffer.

2.  Compute GAE advantages and returns over the rollout (γ=0.99, λ=0.95).

3.  Loop 5 epochs over the rollout:
    a.  Split into 4 minibatches of 12800/4 = 3200 samples each.
    b.  For each minibatch:
        - Compute new policy log_probs and V(s) on the same obs.
        - Compute PPO clipped surrogate loss + value loss (clipped, ×1.0)
          + entropy bonus (×0.005).
        - Backprop, gradient-clip to max_grad_norm=1.0.
        - Optimizer step (Adam, LR throttled by adaptive KL schedule).

4.  Log to TensorBoard: Episode_Reward/*, Episode_Termination/*,
    Loss/value_function, Loss/surrogate, Loss/entropy,
    Policy/mean_noise_std, Train/mean_reward,
    Train/mean_episode_length.

5.  If iter % 50 == 0: save model_<iter>.pt checkpoint.
```

ETA on RTX 5070 (V2.15) :
- ~35–50 s/iter (varies with GPU contention from other apps).
- 1500 iters × ~45 s ≈ **19 h** total.
- First lifts expected around iter 200–400, first successes around iter
  500–800, depending on convergence quality.
