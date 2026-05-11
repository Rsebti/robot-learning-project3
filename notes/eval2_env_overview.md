# Eval 2 — Environment overview

Single-page reference for the Eval 2 RL environment : scene, dimensions,
axes, action / observation / reward spaces. All dimensions verified
empirically via [`sim/eval2/scripts/dump_scene_frames.py`](../sim/eval2/scripts/dump_scene_frames.py) on 2026-05-11.

## Scene composition

| Asset | Type | Source |
|---|---|---|
| **`robot`** (SO-101) | Articulation | LeIsaac USD scene (with USD recolor table + cube 2 cm) |
| **`cube`** | RigidObject | LeIsaac scene, resized 3 → 2 cm via `resize_cube_usd.py` |
| **table** | scene mesh | LeIsaac base scene, recolored #B8ADA9 via `recolor_table_usd.py` |
| **`ee_frame`** | FrameTransformer sensor | tracks `gripper` body (target[0]) and `jaw` body (target[1]) |
| **`wrist`** | TiledCamera sensor | RGB camera mounted on robot wrist, 224×224 |

## Coordinate frames

All positions reported by Isaac Lab are in **world frame** unless
explicitly suffixed with `_b` (robot base frame).

- **World origin** = scene origin (Isaac Sim default).
- **Robot base** spawns at `(0.3500, -0.6400, 0.0100)` in world frame.
- **z-axis points UP**; gravity is `(0, 0, -9.81)`.

## Robot SO-101 (6-DoF arm + 1-DoF binary gripper)

Body chain (verified, in order) :

```
base → shoulder → upper_arm → lower_arm → wrist → gripper → jaw
```

Body positions at the home pose (joints all at zero) :

| Body | (x, y, z) world | Notes |
|---|---|---|
| `base` | (+0.35, -0.64, +0.010) | almost on the floor |
| `shoulder` | (+0.33, -0.62, +0.105) | |
| `upper_arm` | (+0.35, -0.59, +0.159) | |
| `lower_arm` | (+0.35, -0.56, +0.272) | |
| `wrist` | (+0.35, -0.42, +0.277) | rotational joint |
| `gripper` | (+0.33, -0.36, +0.277) | palm |
| `jaw` | (+0.31, -0.34, +0.297) | fingertips |

**Important** : at home, the gripper extends **forward** in robot frame
(world +y direction). The gripper local `+z` axis points **backward**
(into the robot base, world `-y`). This caused a sign-bug in the
quaternion-based `gripper_orientation_penalty` of V2.13 v1/v2 — V2.15
uses a position-based formulation instead.

Joint limits (verified) :

| idx | joint | min (rad) | max (rad) | range |
|---|---|---|---|---|
| 0 | `shoulder_pan` | -1.920 | +1.920 | 3.840 |
| 1 | `shoulder_lift` | -1.745 | +1.745 | 3.491 |
| 2 | `elbow_flex` | -1.745 | +1.571 | 3.316 |
| 3 | `wrist_flex` | -1.658 | +1.658 | 3.316 |
| 4 | `wrist_roll` | -2.793 | +2.793 | 5.585 |
| 5 | `gripper` | -0.175 | +1.745 | 1.920 (binary in action) |

## Cube

| Property | Value | Source |
|---|---|---|
| Side length | 2 cm | USD bbox |
| `cube.root_pos_w.z` at reset | **0.0410 m** | center of mass (constant) |
| Cube bottom | 0.0310 m (= table top) | center − half size |
| Cube top | 0.0510 m | center + half size |
| Spawn randomization | x ± 7.5 cm, y ± 7.5 cm, yaw ± 30° | LeIsaac stock + USD edits |
| Color | red (default LeIsaac material) | — |

## EE frame (FrameTransformer)

| target | body | offset (local) | live pos (home) |
|---|---|---|---|
| `[0]` | `gripper` (palm) | (0,0,0), rot identity | (0.33, -0.36, **0.277**) |
| `[1]` | `jaw` (fingertips) | (-0.021, -0.07, 0.02), rot identity | (0.33, -0.27, **0.276**) |

Used by reward functions to read palm / jaw world positions without
diving into the URDF body indices.

## Wrist camera

| Property | Value |
|---|---|
| Mount | on robot wrist body |
| Type | RGB TiledCamera |
| Resolution | 224 × 224 |
| Focal length | ~36.5 mm |
| Encoder | ResNet-18 ImageNet, frozen `eval()` mode |
| Feature dim | 512 (flattened into the 555-D policy observation) |

The encoder pre-computes features inside the observation function
(`mdp.observations.wrist_image_features`), so the policy sees ResNet
features directly — the CNN is **not** trained end-to-end with PPO.

## Action space (6-D)

| idx | name | type | range / convention |
|---|---|---|---|
| 0-4 | arm joints (5×) | `RelativeJointPositionActionCfg` | raw `clip = [-1, +1]`, `scale = 0.10` (V2.14+) → joint delta capped at ±0.10 rad/step → vmax = 3 rad/s |
| 5 | gripper (1×) | `BinaryJointPositionActionCfg` | open = 0.5 rad if action > 0, close = 0.0 rad if action < 0 |

Episode length : **10.0 s** (V2.14+) = 300 control steps at 30 Hz.

Why DELTA (not absolute) + clip + scale = 0.10 : aligned with Feetech
STS3215 servo limit (6 rad/s); 50 % safety margin makes sim-to-real
robust. See pipeline doc § "Stack V2.15" for full rationale.

## Observation space (555-D, policy group)

| idx | name | dim | source |
|---|---|---|---|
| 0 | `joint_pos` | 6 | robot joints |
| 1 | `joint_vel` | 6 | robot joints |
| 2 | `object_position` | 3 | cube position (robot frame) |
| 3 | `target_object_position` | 7 | goal pose (robot frame) — xyz + quat |
| 4 | `actions` | 6 | last action commanded |
| 5 | `wrist_features` | 512 | ResNet-18 features (frozen) |
| 6 | `target_color_placeholder` | 6 | one-hot (Phase C reserve, currently zeros) |
| 7 | `bowl_xyz_placeholder` | 3 | bowl position (Phase C reserve, currently zeros) |
| 8 | `ee_to_cube_vec` | 3 | `cube_pos_w − palm_pos_w` (world frame) |
| 9 | `cube_to_goal_vec` | 3 | `goal_pos_w − cube_pos_w` (world frame) |
| | **total** | **555** | |

The Phase-C placeholders are zeros today but already in the obs to keep
the policy network shape forward-compatible with the multi-cube + bowl
Eval 2 task without rebuilding.

## Reward terms (V2.15 weights)

| term | weight | gating | notes |
|---|---|---|---|
| `reaching_object` (tanh, std=0.15) | **+3.0** | none | anti-suicide insurance |
| `grasping_cube` (binary) | +5.0 | none | jaw < 4 cm + grip < 0.26 rad |
| `lifting_object` | +10.0 | grasp ∧ lift (z\_rel > 0.08) | V2.7 anti-flick gate |
| `object_goal_tracking` (std=0.3) | +16.0 | grasp ∧ lift | anti-flick |
| `object_goal_tracking_fine_grained` (std=0.05) | +5.0 | grasp ∧ lift | precision |
| `success_bonus` (sparse) | +2500 | terminal | distance < 5 cm |
| `ee_to_cube_distance` (linear `-‖EE-cube‖`) | **-3.0** | none | non-saturating driver |
| `action_rate_l2` | -1e-3 | n/a | smoothness (V2.14 ×10) |
| `joint_vel_l2` | -1e-3 | n/a | smoothness (V2.14 ×10) |
| `cube_dropped_penalty` (binary, z<0.04) | -30.0 | n/a | discourage suicide |
| `gripper_orientation_penalty` (palm→jaw direction) | **-5.0** | none | V2.15 : returns 0 (top-down), 1 (horiz/snake), 2 (gripper-up) |
| `jaw_below_cube_penalty` (linear, `cube_bottom-jaw_z` clamped ≥0) | **-50.0** | n/a | V2.15 : physics breach prevention |
| `scoop_grasp_penalty` | 0 (dropped) | none | redundant since V2.13 v3 |

## Termination terms

| term | type | trigger |
|---|---|---|
| `time_out` | `TimeOut` | 300 steps (10 s) reached |
| `success` | DoneTerm | cube within 5 cm of goal |
| `cube_dropped` | DoneTerm | `cube.z < 0.04 m` |

`ee_far_from_cube` (V2.12 fail-fast DoneTerm) is **disabled** since
V2.13 v2 (give-up exploit channel).

## Goal (command)

| Property | Value |
|---|---|
| Command type | `UniformPoseCommand` |
| Frame | robot base |
| x range | -0.10 .. +0.10 m |
| y range | -0.20 .. -0.10 m (forward of base) |
| z range | +0.10 .. +0.20 m (lifted) |
| quat | identity (no orientation goal) |

## PPO config (V2.15, inherited from V2.9 baseline)

| Param | Value | Notes |
|---|---|---|
| `init_noise_std` | 1.0 | V2.9 default (large exploration) |
| `entropy_coef` | 0.005 | |
| `value_loss_coef` | 1.0 | |
| `n_epochs` | 5 | |
| `n_mini_batches` | 4 | |
| `learning_rate` | 1e-3 | |
| `schedule` | adaptive | |
| `desired_kl` | 0.02 | V2.12 fix (was 0.01) |
| `gamma` | 0.99 | |
| `lam` (GAE) | 0.95 | |
| `max_grad_norm` | 1.0 | |
| `actor_hidden_dims` | [256, 128, 128] | ELU activations |
| `num_steps_per_env` | 50 | rollout length per iter |
| `max_iterations` | 1500 | |
| `experiment_name` | `lift_v2_13` | all V2.13–V2.15 share same exp dir |
