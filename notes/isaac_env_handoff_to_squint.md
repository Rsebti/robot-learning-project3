# Isaac Squint-Native env — handoff to the Squint-side Claude

Mirror of `notes/squint_friend_dumps/env_diff_for_isaacsim.md` but written
from the Isaac side. Read alongside `sim/eval2/envs/squint_native/*.py`
and the latest audit logs in `notes/`.

---

## 0. Files

| Path | Purpose |
|---|---|
| `sim/eval2/envs/squint_native/squint_robot.py` | ArticulationCfg (USD path, home qpos, actuators) |
| `sim/eval2/envs/squint_native/squint_scene.py` | InteractiveSceneCfg (cube, distractor, bin, cam, table, lights) |
| `sim/eval2/envs/squint_native/squint_actions.py` | Custom `DeltaTargetJointPositionAction` (= Squint's `pd_joint_target_delta_pos`) |
| `sim/eval2/envs/squint_native/squint_observations.py` | qpos / target_qpos / wrist_rgb_16 / goal_color_one_hot |
| `sim/eval2/envs/squint_native/squint_events.py` | reset_robot_to_home / reset_scene_squint / make_robot_matte_black |
| `sim/eval2/envs/squint_native/squint_env_cfg.py` | ManagerBasedRLEnvCfg (sim, physx, decim, episode length) |
| `sim/eval2/envs/squint_native/squint_env.py` | `SquintNativePlaceEnv` (subclass that tracks wrist cam to gripper each step) |
| `squint/deploy_isaac_native.py` | Deploy a checkpoint on the Isaac env |
| `squint/converted_usd/so101_squint_renamed.usd` | USD asset converted from Squint's URDF (link names stripped of `_link`) |
| `sim/eval2/scripts/rename_squint_urdf.py` | Pre-rename URDF before conversion |
| `sim/eval2/scripts/convert_squint_urdf.py` | URDF → USD via `isaaclab.sim.converters.UrdfConverter` |
| `sim/eval2/scripts/debug_full_audit.py` | Replays the canonical Squint audit on the Isaac side |
| `sim/eval2/scripts/dump_obs.py` | Dump obs + GT joint state |
| `sim/eval2/scripts/view_squint_native.py` | GUI viewer (no policy) |
| `sim/eval2/scripts/replay_squint_trajectory.py` | Replay a Squint qpos trajectory in Isaac |
| `notes/squint_friend_dumps/` | All the audit data from the Squint side |
| `notes/isaac_native_audit_v5.txt` | Latest debug_full_audit output |

---

## 1. Sim / physics parameters

| Param | Squint target | Isaac current | Status |
|---|---|---|---|
| `sim_freq` | 100 Hz | 100 Hz (sim.dt = 1/100) | ✅ |
| `control_freq` | 10 Hz | 10 Hz (decimation = 10) | ✅ |
| `solver_position_iterations` | 15 | 15 (per-articulation) | ✅ |
| `solver_velocity_iterations` | 1 | 1 (per-articulation) | ✅ |
| `contact_offset` | 0.02 | (PhysX default ≈ 0.02) | ≈ |
| `rest_offset` | 0 | (default) | ≈ |
| `bounce_threshold_velocity` | 2.0 | 2.0 | ✅ |
| `enable_pcm` | True | (default True) | ≈ |
| `enable_tgs` | True | (configurable) | ≈ |
| `enable_ccd` | False | False (default) | ✅ |
| Gravity (everywhere) | `(0, 0, -9.81)` (re-trained with full gravity) | `(0, 0, -9.81)` | ✅ |
| Robot gravity disabled | NO (cube + bin + robot all feel gravity) | NO (`disable_gravity=False`) | ✅ |

Note on settle behavior: the earlier `settle_behavior.txt` dump showed
qvel = 0 across all steps, which we (wrongly) interpreted as the robot
being gravity-free. **The new training run has full gravity on every
body**, so the qpos drift / non-zero qvel we see in PhysX is expected
and the policy is trained against it.

---

## 2. Robot — converted Squint URDF

- USD source: `squint/converted_usd/so101_squint_renamed.usd`
- Link rename map applied before URDF→USD:

```python
"base_link"                  → "base"
"shoulder_link"              → "shoulder"
"upper_arm_link"             → "upper_arm"
"lower_arm_link"             → "lower_arm"
"wrist_link"                 → "wrist"
"gripper_link"               → "gripper"
"moving_jaw_so101_v1_link"   → "jaw"
# finger1_tip, finger2_tip, gripper_frame_link kept as-is
```

- Actuator config (ImplicitActuator on all 6 joints):
  - `stiffness = 1000`
  - `damping = 100`
  - `effort_limit = 100`
  - `velocity_limit = 100`

- Home qpos (Squint `start` keyframe):
  - `[0, 0, 0, π/2, -π/2, 60° × π/180]`
  - `=  [0.0000, 0.0000, 0.0000, 1.5708, -1.5708, 1.0472]`

- Reset noise: `N(0, σ=0.02 rad)` per joint, independent.

---

## 3. Action term — `DeltaTargetJointPositionAction`

Custom term to replicate Squint's `pd_joint_target_delta_pos`:

```python
# On every step:
delta = clamp(action_normalized, -1, +1) * scale_per_joint
target_qpos = clamp(target_qpos + delta, soft_limits)
asset.set_joint_position_target(target_qpos)

# On reset:
target_qpos = current qpos (refreshed via asset.update(dt=0.0))
```

- `bounds = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2]` (arm × 5, gripper)
- joint order: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`

The integration is identical to Squint's controller per
`action_replay.txt` (+0.5 action over 10 steps integrates to +0.5 target
with the same PD-lag pattern).

---

## 4. Scene

| Object | Geometry | Color | Physics | Notes |
|---|---|---|---|---|
| Robot | converted Squint USD | matte black `(0.02, 0.02, 0.02)` via OmniPBR override (startup event) | full gravity | |
| Cube (target) | `(0.020, 0.020, 0.020)` cuboid | red `(1, 0, 0)` | density 700, friction 0.3, restitution 0 | spawns in 20×20cm box around (0.3, 0) |
| Cube (distractor) | `(0.020, 0.020, 0.020)` cuboid | blue `(0, 0, 1)` | same as target | placed face-to-face with target: dir ∈ [0, 2π) random, dist = `2 * half_size + uniform(0, 0.005)` |
| Bin (5 parts) | floor `0.08×0.10×0.005`, walls `0.005 thick × 0.030 high` | white | kinematic | placed coherently with shared yaw |
| Table | `(1.5, 1.0, 0.10)` cuboid | matte grey `#B8ADA9 = (0.722, 0.678, 0.663)` | static | top at z=0 |
| DomeLight | | `(1, 1, 1)` intensity 300 | | |
| DistantLight | | `(1, 1, 1)` intensity 300, angle 10° | | |

No greenscreen, no segmentation overlay (matches the flattable_woodcube
training: `apply_overlay=False`, `obs_mode='rgb'`).

---

## 5. Wrist camera

- Mount: kinematic, **explicitly tracked each step** in
  `SquintNativePlaceEnv.step()`. Isaac's `CameraCfg.OffsetCfg` mount
  inheritance under `/Robot/gripper/wrist_cam` produced ~10 cm world
  position drift; tracking it manually via `set_world_poses()` matches
  Squint to within ~1.5 cm.

- Offset relative to gripper body (matches `WRIST_CAMERA_BASE_POS` /
  `WRIST_CAMERA_BASE_ROT_RAD` in Squint):
  - position: `(-0.0049, 0.0498, -0.0591)` (gripper local frame)
  - rotation: `R_z(yaw=-35.31°) · R_y(pitch=91°) · R_x(roll=-90°)` 
    (Hamilton scalar-first, applied as `q_yaw * q_pitch * q_roll`)
  - convention in Isaac: `"world"` (forward = local +X, SAPIEN-style)

- Cam params: 128×128, FOV 71°, clipping (0.01, 100), `update_period=0`,
  `update_latest_camera_pose=True`.

- Image obs: 128×128 → 16×16 via `F.interpolate(mode='area')` inside
  `squint_observations.wrist_rgb_16`. Identical method to Squint deploy
  pipeline (verified in audit §9).

---

## 6. Observations

`obs['policy']` shape `(N, 18)`:

| dim | content |
|---|---|
| `[0:6]` | `qpos` (no noise during deploy; std=0 set in `joint_pos_with_noise`) |
| `[6:12]` | `target_qpos` (integrated by `DeltaTargetJointPositionAction`) |
| `[12:18]` | `goal_color_one_hot` (deploy: idx 0 = red, fixed; never re-sampled per episode) |

`obs['rgb']['rgb']` shape `(N, 16, 16, 3)` uint8.

---

## 7. Verified alignment points

| Aspect | Squint reference | Isaac measured | Source |
|---|---|---|---|
| Joint count | 6 | 6 | audit |
| Joint names + order | `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]` | same | audit |
| Hard joint limits (rad) | `[-1.92, 1.92], [-1.745, 1.745], [-1.69, 1.69], [-1.658, 1.658], [-2.744, 2.841], [-0.175, 2.094]` | identical | audit |
| Joint-axis probe (Δ tip for +0.1×10) | see `action_replay.txt` | matches within ~1 mm | audit |
| Cam world pos at home | `(+0.286, -0.006, +0.123)` | `(+0.288, +0.007, +0.109)` | audit |
| Cam optical axis | `(-0.54, +0.07, -0.84)` | `(-0.59, +0.03, -0.81)` | audit |
| Cam ground projection | `(+0.21, +0.00)` | `(+0.21, +0.01)` | audit |
| RGB mean at home (no overlay) | R:181 G:166 B:163 | R:181 G:178 B:178 | obs_dump |
| Controller integration | +0.5 over 10 steps → qpos +0.471 | reproduces same lag | action_replay |
| State shape | `(N, 18)` | `(N, 18)` | dump_obs |
| RGB image shape | `(N, 16, 16, 3) uint8` | `(N, 16, 16, 3) uint8` | dump_obs |

Settle drift under full gravity is expected on both sides; we no longer
try to force qvel == 0 in Isaac.

---

## 8. Open questions for the Squint side

1. **Distractor color sampling**: how is the distractor color picked?
   Random from palette excluding the goal? We currently fix it to blue.

2. **Goal color sampling at reset**: in your training, is the goal color
   re-sampled every reset? Our env currently hard-codes red (idx 0). For
   eval, we'd want to re-sample to match training distribution.

3. **`PlaceRandomizationConfig.initial_qpos_noise_scale`**: `0.02` rad per
   joint? Per-joint independent or correlated? We use per-joint
   independent `randn() * 0.02`.

4. **Soft joint limits**: Squint's audit prints HARD limits. Does the
   training/controller clip targets to soft limits (slightly inside
   hard), and if so by how much?

5. **Sim2real shaping that affects deploy obs**: any RGB normalization /
   color-jitter applied to the wrist RGB before it enters the encoder
   that isn't documented in `train_squint.py`?

6. **Cube/distractor color tinting on rendering**: are the cube colors
   passed to SAPIEN's `RenderMaterial(base_color=...)` exactly as listed
   in `COLOR_PALETTE`, or is there gamma / sRGB conversion in between?

---

## 9. What we want from the Squint side next

| Want | Why |
|---|---|
| The new 18-d checkpoint file `runs/placecube_flattable_woodcube_run1/ckpt.pt` | All ckpts currently on Isaac side are 12-d (old `b8ada9` run). The new policy can't be tested yet. |
| One numerical reference action for `goal=0 (red)` + the exact 18-d state our env produces (we can send the state) | Bit-identical inference verification — narrows obs-gap vs physics-gap |
| Confirmation of the new training's render pipeline (lights, materials, post-processing) | Aligns visual obs between sims |

---

## 10. How to reproduce on the Isaac side

```powershell
cd C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101

# Full scene audit (mirrors squint_audit_canonical.txt format)
C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
    -m sim.eval2.scripts.debug_full_audit `
    --task Isaac-SquintNative-Place-Play-v0 --num_envs 1 --enable_cameras --headless

# Obs dump (state shape, RGB stats, settle behavior)
C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
    -m sim.eval2.scripts.dump_obs `
    --task Isaac-SquintNative-Place-Play-v0 --settle_steps 20 --per_step_dump

# Deploy a checkpoint
C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
    C:/Users/user/Desktop/MA2/robot-learning-project3/squint/deploy_isaac_native.py `
    --ckpt <path>.pt --n_episodes 5 --max_steps 200 --headless

# GUI viewer (no policy)
C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
    -m sim.eval2.scripts.view_squint_native
```
