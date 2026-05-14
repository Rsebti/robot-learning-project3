# Squint (fedecomi04) vs Isaac Lab sim2sim audit

Audit date: 2026-05-13.
Checkpoint under test: `C:/Users/user/Downloads/ckpt (7).pt` (SAC+C51, trained on `SO101PlaceCube-v1` in `squint_fedecomi04`).
Isaac task under test: `Isaac-SquintNative-Place-Play-v0`.

---

## TL;DR — Top suspects ranked

1. **SHOWSTOPPER — action mis-scaling (factor up to 20×).** Our Isaac action term feeds the policy's normalized `[-1, +1]` output directly into the integrated target *without multiplying by the per-joint bound*. In Squint, the controller's `_clip_and_scale_action` maps `[-1, +1]` → `±bound` rad. Net effect: when the policy outputs something modest like `+0.02`, Isaac applies a `+0.02` rad delta (saturated arm-joint), while Squint applies `+0.001` rad (0.02 × 0.05). The deltas only agree when the policy saturates at `±1`. This alone explains a 0 % success rate even with otherwise-perfect obs. Concrete fix in §1 and §9.1.
2. **SHOWSTOPPER — wrist RGB is all-black.** `notes/obs_dump.txt` shows `obs['rgb']['rgb']` with min/mean/max = `0.000`. The policy was trained on textured 16×16 frames; feeding zeros is pure OOD. Likely caused by the matte_black + emissive-only event chain disabling everything visible to the wrist cam at deploy time. Concrete fix in §2 and §9.2.
3. **Possible — wrist camera orientation convention.** Squint uses SAPIEN's "camera looks along local +X" convention applied through `wrist_camera_mount.pose * local_offset`. Isaac `set_world_poses(..., convention="world")` documents forward = +X, up = +Z, which *should* match — but SAPIEN's render-camera frame is actually "look down local -Z" (debug audit explicitly notes `local -Z (OpenGL/SAPIEN render cam)`). If Squint's mount handed SAPIEN a `look-down-+X` quaternion, then SAPIEN's renderer compositing internally rotated forward into local +X. Reproducing that exact same quaternion under Isaac `world` → wired into the camera frame the same way should be equivalent, but this needs a 1-pixel diff test on a known-pose canonical to confirm. See §2 and §9.3.
4. **Possible — table mass / spawn box mismatch.** Squint's spawn box is `pos=[0.3, 0]`, `half_size=0.25/2 = 0.125` (a 25×25 cm box) for the Place task, but our env uses `SPAWN_BOX_HALF = 0.1` (a 20×20 cm box). With a smaller spawn region the cube/bowl distribution at deploy is *inside* the Squint training distribution, so this on its own should not drop success to 0 — but it shifts the canonical-replay positions away from anywhere the policy was trained on edges. See §6.
5. **Possible — `is_grasping` proximity gate sensitivity / contact forces.** Squint's grasp check is purely contact-force directionality based; our port substitutes that with `F_g ≥ 0.5 & F_j ≥ 0.5 & cube_to_jaw < 5 cm & cube lifted`. Reward shape only matters during *training*; for deploy this doesn't directly change policy behaviour but it *does* change the success counters reported by terminations/rewards and may mask a true success while you're debugging.

---

## 1. Action space & controller — **SHOWSTOPPER MISMATCH**

| Aspect | Squint | Isaac (ours) | Verdict |
|---|---|---|---|
| Joint count | 6 | 6 | MATCH |
| Joint name order to controller | `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]` (URDF kinematic chain, SAPIEN `active_joints` order) — confirmed by `obs_dump.txt` qpos starting at `[0, 0, 0, π/2, -π/2, π/3]` matching the `start` keyframe | Same — `squint_env_cfg.py:48-57` and `squint_actions.py:60` with `preserve_order=True` | MATCH |
| Per-joint bounds (arm) | ±0.05 rad — `envs/robot/so101.py:102-103` | ±0.05 rad — `squint_env_cfg.py:59` | MATCH |
| Per-joint bound (gripper) | ±0.2 rad | ±0.2 rad | MATCH |
| Controller class | `PDJointPosController` with `use_delta=True, use_target=True, normalize_action=True` — `pd_joint_pos.py` | `DeltaTargetJointPositionAction` — `squint_actions.py` | MATCH (semantics) |
| Stiffness / damping / force_limit | 1000 / 100 / 100 — `envs/robot/so101.py:104-106` | 1000 / 100 / 100 — `squint_robot.py:89-92` | MATCH |
| `balance_passive_force` (gravity comp) | False (explicit) — `so101.py:131` | Gravity on (`disable_gravity=False`) — `squint_robot.py:55` | MATCH |
| **Action range emitted by policy** | `[-1, +1]` (normalized — `_clip_and_scale_action_space` rewrites `single_action_space` to `Box(-1, 1)`) | Same — `actor.action_scale = 1.0`, `bias = 0` (since saved buffer == `(high - low)/2`) | MATCH |
| **Action → rad-delta scaling** | `physical_delta = clip(action, -1, 1) × bound` (per joint) → applied in `_target_qpos += physical_delta` — `pd_joint_pos.py:82` after `_preprocess_action` → `_clip_and_scale_action` (`gym_utils.py:104`: `0 + bound × action`) | `delta = clamp(action, -bound, +bound)` — `squint_actions.py:166`. **Does NOT multiply by `bound`.** Comment at line 159-165 explicitly (and incorrectly) assumes the policy already outputs in radians. | **MISMATCH — SHOWSTOPPER** |
| Target integration | `_target_qpos += physical_delta`, no per-step clamp to qlimits | `_target = clamp(_target + delta, joint_lo, joint_hi)` (soft limit clamp) | SUSPECT — Squint does NOT clamp the target to joint limits; only the URDF/PhysX joint limit hard-clamps the actual qpos. Our extra clamp probably matters less than the scaling bug but worth removing later. |
| Target initialization on reset | Target ← current qpos (after reset_mask) — `pd_joint_pos.py:67-69` | Same — `squint_actions.py:121-138` reads `joint_pos` after `update(dt=0)` | MATCH |
| URDF joint limits | shoulder_pan ±1.92, shoulder_lift ±1.745, elbow_flex ±1.69, wrist_flex ±1.658, wrist_roll [-2.744, 2.841], gripper [-0.175, 2.094] | Same (preserved through URDF→USD conversion) | MATCH |

**Concrete fix (the only change needed for §1):** `sim/eval2/envs/squint_native/squint_actions.py:152-172` — replace the `delta = clamp(actions, -self._scale, +self._scale)` line with

```python
# Policy outputs in [-1, +1] (Squint sets normalize_action=True, so
# single_action_space is Box(-1, 1) and the saved action_scale=1.0).
# Map to rad-delta by multiplying by per-joint bound.
delta = torch.clamp(actions, -1.0, +1.0) * self._scale
```

This is THE bug that produces a non-functional rollout regardless of obs correctness.

---

## 2. Observation space — **SHOWSTOPPER MISMATCH on RGB**

### 2a. State vector (`obs['state']`)

| Aspect | Squint | Isaac (ours) | Verdict |
|---|---|---|---|
| Layout | `[noisy_qpos(6), controller.target_qpos(6), goal_color_one_hot(6)]` = **18 floats**. Traced through `BaseEnv._get_obs_state_dict` → `Place._get_obs_agent` (`place.py:807-823`) which returns `dict(noisy_qpos, controller, goal_color)`; `FlattenRGBDObservationWrapper` then runs `flatten_state_dict` over `{"agent": _get_obs_agent(), "extra": {}}` (extra is `{}` because training uses `obs_mode="rgb"`, so `obs_mode_struct.state == False`). Python ≥3.7 dict iteration is insertion order. | `policy` group: `qpos | target_qpos | goal_color_one_hot` (18) — `squint_observations.py` | MATCH |
| qpos noise std (DR) | `np.deg2rad(5) ≈ 0.0873` rad | 0.0873 (`squint_env_cfg.py:84`) | MATCH |
| Source for `target_qpos` (12-d slice) | `controller._target_qpos` — i.e. the integrated PD target, NOT the actual qpos | `DeltaTargetJointPositionAction._target` — same semantics | MATCH |
| `goal_color_one_hot` palette order | red(0), blue(1), green(2), yellow(3), purple(4), orange(5) — `place.py:29-39` | Same — `squint_observations.py:51-58`, `squint_events.py:443-450` | MATCH |
| dtype | float32 | float32 | MATCH |

### 2b. RGB observation (`obs['rgb']`)

| Aspect | Squint | Isaac (ours) | Verdict |
|---|---|---|---|
| Raw cam resolution | 128 × 128 (training default) | 128 × 128 (`squint_scene.py:122-123`) | MATCH |
| Downsample target | 16 × 16, `F.interpolate(..., mode='area')` — `utils.py:39` `DownsampleObsWrapper` | Same — `squint_observations.py:213-215` | MATCH |
| dtype / range | uint8, `[0, 255]` | uint8 (`squint_observations.py:215`) | MATCH |
| Channels | 3 (RGB; `FlattenRGBDObservationWrapper` reads `cam_data["rgb"]`) | 3 (`rgb = rgb[..., :3]` line 175) | MATCH |
| Color jitter | torchvision `ColorJitter(0.3, 0.3, 0.3, 0.05)` always applied at train **and** eval (`train_squint.py:666-668`) | Custom port `_apply_color_jitter(0.3, 0.3, 0.3, 0.05)` — `squint_observations.py:95-147`, enabled via `apply_jitter=True` in cfg | MATCH (approximate — our hue rotation is a cheap RGB rotation, theirs is full HSV; the visual diff after 16×16 area downsample is tiny) |
| Greenscreen / overlay | **OFF** in training (`train_squint.py:87` defaults `apply_overlay=False`) | OFF (`squint_observations.py:103-105` `greenscreen_bg_rgb=None`) | MATCH |
| **Actual frame content at deploy** | Textured render of cube + bowl + (matte black) robot, on a `#B8ADA9`-painted table | **min/mean/max = 0.000 / 0.000 / 0.000** per `notes/obs_dump.txt` — the wrist cam is producing a fully black frame | **MISMATCH — SHOWSTOPPER** |
| Wrist cam FOV | `fovy = deg2rad(71)` — `base_random_env.py:540` `WRIST_CAMERA_FOV` | Horizontal FOV 71° via `horizontal_aperture=20.955, focal_length=20.955 / (2·tan(35.5°))` — at 1:1 aspect ratio, h-FOV == v-FOV | MATCH |
| Wrist cam local position (relative to gripper body) | `(-0.0049, 0.0498, -0.0591)` — `base_random_env.py:538` | Same — `squint_scene.py:67` | MATCH |
| Wrist cam local rotation (extrinsic / intrinsic) | Euler RPY `(-90°, 91°, -35.31°)` composed as `q = q_pitch · q_yaw · q_roll` — `base_random_env.py:601-612` | Same composition reproduced in `_euler_rpy_zyx_to_quat_wxyz` — `squint_scene.py:42-64`, with the docstring explicitly explaining that variable-name `q_py` is actually `q_pitch · q_yaw` not `q_yaw · q_pitch` | MATCH (formula-identical) |
| Apply per-substep `mount.pose * local_offset` | Yes — `_after_control_step` runs `_update_wrist_camera_pose` (`base_random_env.py:633-639`) | Yes — `squint_env.py:118-138` runs `_update_wrist_cam_pose` after every sim substep, plus once at the end of `step()` (line 186) | MATCH |
| Convention used to write world pose | SAPIEN scene-add (mount=...), camera looks along local +X (SAPIEN render-cam) | `set_world_poses(..., convention="world")` (Isaac: forward +X, up +Z) | SUSPECT — the conventions are documented as equivalent (both `+X forward`), but ROS/OpenGL discrepancies under the hood plus Isaac's `convert_camera_frame_orientation_convention(..., origin=convention, target="opengl")` (`camera.py:330`) may introduce a 90° hidden rotation we haven't validated against a known pose. The "all-black" obs in §2b above also smells like a camera pointing into the floor or the inside of the gripper body. |

**Hypothesis for the black-frame bug:** the `make_robot_matte_black` and `make_scene_emissive` startup events repaint nearly every shader in the scene to flat-dark colours, plus the dome/distant lights are at intensity 300 only — but `make_scene_emissive` *also* mirrors every diffuse into emissive at intensity 1.0, which for `(0.04, 0.04, 0.04)` matte black means barely-visible self-illumination. If the table got correctly recolored to `#B8ADA9` and the cube has emissive_color=RGB but the wrist cam looks down into the gripper jaw (which now emits ~0), you can plausibly get a near-black frame. Recommended check: `python -m sim.eval2.scripts.dump_wrist_image` to actually save the rendered frame; if it's truly all-black, disable `make_scene_emissive`/`make_robot_matte_black` one at a time and compare.

---

## 3. Reward & termination — VALIDATION ONLY

| Aspect | Squint | Isaac (ours) | Verdict |
|---|---|---|---|
| Termination on success | NO — `ManiSkillVectorEnv(ignore_terminations=not partial_reset)` with `partial_reset=False` → all terminations forced off; only the env-level timeout (`max_episode_steps=75`) ends episodes | NO — only `time_out` (`squint_env_cfg.py:242`); `success()` is computed in `squint_terminations.py` but never wired into `DoneTerm` | MATCH |
| Episode length | 75 control steps @ 10 Hz = 7.5 s (`@register_env("SO101PlaceCube-v1", max_episode_steps=75)`, `place.py:950`) | 50 steps @ 10 Hz = 5.0 s (`squint_env_cfg.py:290`) | **MISMATCH** — our episodes are 33 % shorter than Squint training. The policy likely emits a placement plan that needs more steps to settle. Suspect, secondary. |
| Dense reward shape | `compute_dense_reward` returns the same state-machine logic | `squint_rewards.squint_dense_reward` mirrors the formula | MATCH (close port; minor differences in proximity-gate thresholds, but reward isn't queried at deploy time so this is moot for the 0 % success problem) |

---

## 4. Domain randomization — mostly matches

| Aspect | Squint | Isaac (ours) | Verdict |
|---|---|---|---|
| Cube friction range | `(0.05, 0.6)` — `place.py:144` `item_friction_range` (NOT (0.4, 0.6)) | `(0.4, 0.6)` — `squint_env_cfg.py:184` | **MISMATCH** — our friction sampling is biased high. Squint goes much lower (very slippery painted wood). |
| Cube mass range | `(0.003, 0.006)` kg | `(0.003, 0.006)` kg | MATCH |
| Cube half-size range | `(0.009, 0.011)` m — i.e. cube side 18-22 mm | Fixed `size=(0.020, 0.020, 0.020)` (`squint_scene.py:150`) — 2 cm cube, exactly the mid of Squint's range, no DR | SUSPECT — at deploy time the policy isn't seeing the same size distribution it trained on, but the mean matches so this is small-effect. |
| Table friction | `(0.05, 0.4)` range — `place.py:151` (Squint **DOES** randomize the table) | Fixed 0.5 (`squint_scene.py:272-273`) | **MISMATCH** — we've pinned the table friction *above* Squint's max. The CLAUDE.md comment in our cfg ("Squint pins the table friction at 0.5") is FACTUALLY INCORRECT — Squint pins the range to `(0.05, 0.4)` and samples per-env. This makes sliding the cube on the table much harder in our env than Squint's training set. |
| Distractor friction | Same range as cube | `(0.4, 0.6)` — same bias as cube | Same MISMATCH as cube |
| Lighting | `ambient ∈ [0.2, 0.5]` grey-only DR + 2 fixed directional lights — `base_random_env.py:166-179` | Dome 300 + distant 300, intensity hardcoded (`squint_scene.py:292-298`); no lighting DR. Plus all-emissive-on-base-color override. | **MISMATCH** — completely different look from training. Compounds with §2b. |
| Robot color | Black `(0, 0, 0)` (`base_random_env.py:67`) when DR off; random per-link if `robot_color="random"` and DR on | Forced matte `(0.04, 0.04, 0.04)` via `make_robot_matte_black` | MATCH (close enough — both deploys are black) |
| Greenscreen overlay | OFF in training run (`train_squint.py:87` default `apply_overlay=False`) | OFF (no compositing) | MATCH |
| Goal color sampling | Uniform over 6 colors per episode | Uniform over 6, sampled in `reset_goal_and_distractor_colors` event | MATCH |
| Distractor color sampling | Uniform permutation excluding goal idx, then `[:n_distractors]` | `offset = randint(NUM_COLORS-1); distractor_idx = offset + (offset >= goal_idx)` → uniform over the 5 non-goal colors | MATCH |
| Bowl xy range | Spawn box centered `[0.3, 0]`, half-size `0.25/2 = 0.125` (= 25 × 25 cm) — `place.py:188-189` | Spawn box half-size 0.1 (= 20 × 20 cm) — `squint_events.py:39` | **MISMATCH** — our env's spawn region is smaller. Inside Squint's distribution but biased to centre. |
| Bowl placement | Sampler enforces non-overlap with item cluster (target + distractors); cluster radius = 3 × cube_half_size + 0.01 with face-to-face distractors — `place.py:716-720` | Rejection sampling with `ITEM_BOWL_MIN_DIST = √(0.075² + 0.075²) + 0.01 + 0.01 ≈ 0.116 m` between cube and bowl centres | SUSPECT — different geometry but spawn box is small enough that either works in practice |
| Distractor placement | Face-to-face with target on 1 of 4 cardinal directions (rotated by target yaw) — `place.py:730-776` | Face-to-face: angle uniform on `[0, 2π)`, gap = `2 × half + uniform(0, 0.005)` — `squint_events.py:691-694` | **MISMATCH** — Squint constrains distractor to a discrete cardinal direction (in the target's frame); we use a continuous angle. Distractors are also placed touching with NO gap in Squint; we add 0–5 mm random gap. Probably small effect since the goal cube position is in the obs vector via target_qpos (the policy can target without needing to differentiate which cube is which from the constraint pattern). |
| Distractor count | Default `n_distractors=1` (passed via `--n_distractors`) — `train_squint.py:90` | 1 distractor (`squint_scene.py:178`) | MATCH |
| Gripper stiffness DR | `(500, 2000)` range — `base_random_env.py:63` | Not implemented (we leave it at the 1000 baseline). The cfg writes nothing into `gripper_stiffness_range`. | **MISMATCH** — small effect; the policy was trained over a wide gripper-stiffness distribution but happens to fall inside our pin at training mean. Doesn't matter for deploy IF the policy doesn't rely on the gripper-params privileged obs (it doesn't: state is only `[qpos, target_qpos, goal_color]`). |

---

## 5. Physics setup — mostly matches

| Aspect | Squint | Isaac (ours) | Verdict |
|---|---|---|---|
| `sim_freq` / `control_freq` | 100 / 10 Hz — `base_random_env.py:148` | 100 / 10 Hz — `squint_env_cfg.py:287-288` | MATCH |
| `bounce_threshold_velocity` | 2.0 — `place.py:_default_sim_config` defers to SAPIEN default (which is 2.0); confirmed via `notes/squint_friend_dumps/physics_params.txt` ref | 2.0 — `squint_env_cfg.py:319` | MATCH |
| Solver iterations per body | pos=15, vel=1 (per-articulation override, set globally in `Place._default_sim_config`) | pos=15, vel=1 on robot (`squint_robot.py:64-65`), cube (line 168), distractor (line 201), bowl (line 227) | MATCH |
| `contact_offset` / `rest_offset` (cube) | `contact_offset = 0.02` / `rest_offset = 0` — SAPIEN defaults (PxScene `contactOffset=0.02`, `restOffset=0`) | 0.02 / 0 — `squint_scene.py:165-166` | MATCH |
| `contact_offset` / `rest_offset` (bowl) | Same defaults | 0.02 / 0 — `squint_scene.py:233` | MATCH |
| `friction_combine_mode` | SAPIEN default = `MIN` (`PxMaterial::CombineMode::eMIN`) — confirmed in handoff notes | `min` set on cube/distractor/table materials and forced via `align_squint_materials` for everything else | MATCH |
| Friction correlation distance | ManiSkill default = 0.025 m | 0.025 — `squint_env_cfg.py:330` | MATCH |
| PCM / TGS / friction-every-iter | All ON — `notes/v219_architecture.md` cites `enable_pcm=True, enable_tgs=True, enable_friction_every_iter=True` | All ON — `squint_env_cfg.py:349-360` | MATCH |
| CCD | OFF | OFF — `squint_env_cfg.py:362` | MATCH |
| `gpu_total_aggregate_pairs_capacity` | bumped for many envs | 64K (`squint_env_cfg.py:327`) | MATCH for `num_envs=1` deploy |
| Sleep threshold | SAPIEN default 0.005 | 0.005 — `squint_env_cfg.py:321` | MATCH |

Physics is in excellent shape — no actionable mismatches here.

---

## 6. Scene — small geometry mismatches

| Aspect | Squint | Isaac (ours) | Verdict |
|---|---|---|---|
| Cube spawn box centre | `[0.3, 0]` (robot frame xy) | `(0.3, 0.0)` — `squint_events.py:38` | MATCH |
| Cube spawn box half-size | `0.25 / 2 = 0.125` — `place.py:189` | `0.1` — `squint_events.py:39` | **MISMATCH** — smaller. See §4. |
| Bowl spawn box | Same as cube (single sampler used for both) | Same as cube | MATCH (relative) |
| Cube size (fixed during deploy) | Mid `0.020` m side; DR range `0.018–0.022` | `0.020` (`squint_scene.py:150`) | MATCH-ish |
| Bowl mesh | `envs/meshes/bowl.obj` with CoACD `(threshold=0.3, max_convex_hull=8)`, density=500, friction 0.5 — `place.py:493-499` | `squint/meshes/bowl.usd` baked from same `.obj` via Isaac `convex_decomp`, density 500, friction 0.5 — `squint_scene.py:222-244` | MATCH (geometric — minor differences in convex decomposition will cause slightly different bowl-rim contacts but the AABB is identical) |
| Table dimensions / pose | Half `(2.418/2, 1.209/2, 0.9196429/2)`, init pose `(-0.12, 0, -0.9196429)` then `yaw=π/2` applied → world centre `(0.617, 0, -0.46)` (after yaw swap of x↔y half-extents) | Centre `(0.617, 0, -0.46)`, size `(1.209, 2.418, 0.9196429)` — `squint_scene.py:260-262` | MATCH (we pre-applied the yaw swap so size dims are flipped to match Squint's world extent) |
| Table colour | `#B8ADA9` (0.722, 0.678, 0.663) | Same — `squint_scene.py:264` | MATCH |
| Distractor: face-to-face neighbor | Discrete cardinal direction in target frame, ZERO gap | Continuous angle, 0–5 mm gap | MISMATCH — see §4 |
| Distractor count | 1 (default) | 1 | MATCH |

---

## 7. Robot — matches after URDF audit fixes

| Aspect | Squint | Isaac (ours) | Verdict |
|---|---|---|---|
| URDF link order in chain | base → shoulder → upper_arm → lower_arm → wrist → gripper → moving_jaw (active joint = gripper) + fixed children `finger1_tip`, `finger2_tip`, `gripper_frame_link` | Same — `notes/robot_prim_tree.txt` confirms `base, shoulder, upper_arm, lower_arm, wrist, gripper, finger1_tip, jaw, finger2_tip, gripper_frame_link` | MATCH |
| Renamed link in our converted USD | `moving_jaw_so101_v1_link` | `jaw` (after `rename_squint_urdf.py`) | MATCH-by-rename — our `is_grasping` and reward code uses `body_names.index("jaw")` |
| Link masses | base 0.147, shoulder 0.100006, upper_arm 0.103, lower_arm 0.104, wrist 0.079, gripper 0.087, gripper_frame_link 1e-9, moving_jaw 0.012 — sum ≈ 0.632 kg | Same masses preserved through URDF→USD; `gripper_frame_link` is 1e-9 in URDF, but our `finger1_tip` / `finger2_tip` were converter-defaulted to mass=1.0 (these links are empty in URDF) → we zero them via `zero_finger_tip_masses` (`squint_events.py:106-155`) | MATCH (after applied fix) |
| Link COMs | Various (URDF inertial origin) | Patched via `fix_link_coms` (`squint_events.py:57-103`) using exact URDF values from handoff | MATCH (after applied fix) |
| Joint limits | shoulder_pan ±1.92, shoulder_lift ±1.745, elbow_flex ±1.69, wrist_flex ±1.658, wrist_roll [-2.744, 2.841], gripper [-0.175, 2.094] — `so101.urdf:343-443` | Preserved through URDF→USD; also exposed as `data.soft_joint_pos_limits` | MATCH |
| Joint friction / damping (URDF) | NONE (URDF has no `<dynamics damping=...>` tags) → all 0 | Same (URDF→USD doesn't inject damping) | MATCH |
| PD drive stiffness / damping | 1000 / 100 | 1000 / 100 | MATCH |
| Episode length | 75 control steps (PlaceCube) or 50 (Lift; PlaceCan also 50) — `place.py:950`, `lift.py`, `place.py:956` | 50 — `squint_env_cfg.py:290` | **MISMATCH** — see §3 (chose Lift's value; PlaceCube needs 75). |
| Robot colour | matte black (DR off) | matte black via `make_robot_matte_black` | MATCH |
| Per-body friction overrides (gripper jaws) | 2.0 / 2.0 + `patch_radius=0.1, min_patch_radius=0.1` on `gripper_link`, `moving_jaw`, `finger1_tip`, `finger2_tip` — `so101.py:28-46` | Replicated in `align_squint_materials` (`squint_events.py:158-282`) with `torsional_patch_radius=0.1` | MATCH |
| Bowl friction | 0.5 / 0.5 | 0.5 / 0.5 via `align_squint_materials` | MATCH |

---

## 8. Policy interface (training-time vs deploy-time)

| Aspect | Squint | Isaac (ours) | Verdict |
|---|---|---|---|
| Actor formula | `action = tanh(mean) × action_scale + action_bias`, with `action_scale = (env.single_action_space.high - low)/2`, `action_bias = (high + low)/2`. After `_clip_and_scale_action_space` rewrites `single_action_space` to `Box(-1, 1)`, this yields `action_scale = 1.0`, `action_bias = 0.0` per joint. | Identical — `SquintActor` in `sim/eval2/policy/squint_sac.py:194-196` does `tanh(mean) × action_scale + action_bias`, with buffers loaded from the ckpt. | MATCH |
| Saved `action_scale` in ckpt | `[1, 1, 1, 1, 1, 1]` (all ones; comes from the normalized action space) | Loaded directly from ckpt | MATCH |
| Obs dict shape into actor | `obs['rgb']` shape `(B, 16, 16, 3)` uint8, `obs['state']` shape `(B, 18)` float32 | Same shapes — see §2 | MATCH |
| Encoder input normalization | `obs = obs / 255.0 - 0.5` (inside `CNNEncoder.forward`) — `train_squint.py:315` | Same — `sim/eval2/policy/squint_sac.py` (CNNEncoder.forward, by inspection of our port file) | MATCH |
| RGB layout into encoder | `(B, H, W, C) → permute → (B, C, H, W)` channels_last | Same | MATCH |
| State input | Flat 18-d float32, exactly the concatenation `[qpos, target_qpos, goal_color]` | Same — derived from `obs['policy']` (which contains the 18 floats produced by our PolicyCfg group with `concatenate_terms=True`) | MATCH |

**Probe script `probe_policy_actions.py`** is correctly set up — it instantiates a `SquintActor` with `n_state` inferred from `proj.state_proj.0.weight.shape[-1]` (this will read 18 from ckpt 7) and feeds the env's `obs["policy"]` slice + `obs["rgb"]` slice. The action output of the actor (in `[-1, +1]`) is then handed straight back to `env.step(action)`, where the action manager invokes `DeltaTargetJointPositionAction.process_actions` — and *this* is where the §1 SHOWSTOPPER bug bites. Probe is correct, downstream is wrong.

---

## 9. Specific bug suspicions (with concrete fix proposals)

### 9.1. **[SHOWSTOPPER] Action mis-scaling in `DeltaTargetJointPositionAction.process_actions`**

**File:** `C:/Users/user/Desktop/MA2/robot-learning-project3/sim/eval2/envs/squint_native/squint_actions.py`
**Lines:** 152–172

**Current buggy code:**
```python
delta = torch.clamp(actions, -self._scale, +self._scale)
self._raw_actions[:] = actions
self._processed_actions[:] = delta
self._target = torch.clamp(self._target + delta, self._joint_lo, self._joint_hi)
```

**Issue:** the policy outputs in `[-1, +1]` (Squint sets `normalize_action=True` so its `single_action_space` is `Box(-1, 1)` and the saved `action_scale` buffer = 1.0). Squint's controller then maps `[-1, +1] × bound → rad delta`. Our port skips the multiply, so policy outputs `±0.5` produce `±0.05` rad delta (saturated) where Squint would produce `±0.025` rad — a `1 ÷ bound = 20×` mis-scaling at small magnitudes.

**Fix:**
```python
delta = torch.clamp(actions, -1.0, +1.0) * self._scale
self._raw_actions[:] = actions
self._processed_actions[:] = delta
self._target = self._target + delta  # remove our extra soft-joint-limit clamp; Squint doesn't do this
```

Verify after fix with `probe_policy_actions.py`: arm deltas should now look like small fractions of 0.05 rad (e.g. ±0.005 to ±0.05) rather than always near `±0.05`.

### 9.2. **[SHOWSTOPPER] Wrist RGB is all-black at deploy**

**Evidence:** `notes/obs_dump.txt` line 6 — `min/mean/max = 0.000 / 0.000 / 0.000`.

**Likely cause:** the chain `make_robot_matte_black` (RGB 0.04) + `make_scene_emissive` (mirror diffuse → emissive at intensity 1.0) + dome/distant lights at intensity 300 produces a scene where everything outside the cube's emissive material is barely visible. Combined with the wrist cam pointing into the gripper-jaw cavity, you get near-zero pixels.

**Diagnostic next step:** run `python -m sim.eval2.scripts.dump_wrist_image --task Isaac-SquintNative-Place-Play-v0` and inspect the saved PNG. If it's truly black, the bug is in the rendering pipeline (lights / shader emissive); if it has visible content but the obs dump is zero, the bug is in `wrist_rgb_16` (e.g. wrong sensor key, wrong dtype after downsample, jitter producing 0s).

**Fixes to try in order:**
1. **Disable `make_scene_emissive` and `make_robot_matte_black`** (comment them out in `squint_env_cfg.py:125-152`), bump dome light intensity to 1000, and re-run. Squint's actual training scene was not emissive-only (Squint also disabled overlay/greenscreen for this run); the policy was trained with `ambient ∈ [0.2, 0.5]` + 2 directional lights producing fully-shaded views.
2. If the frame is now plausible, switch back the robot colour event only (`make_robot_matte_black` at intensity 0 — Squint's training also renders the robot as flat black).

### 9.3. Wrist camera convention sanity check

**File:** `C:/Users/user/Desktop/MA2/robot-learning-project3/sim/eval2/envs/squint_native/squint_env.py:103-104`

The call is `cam.set_world_poses(positions=..., orientations=..., convention="world")`. Isaac then does (internally, at `camera.py:330`) `convert_camera_frame_orientation_convention(orientations, origin="world", target="opengl")` because the underlying USD camera always renders in OpenGL convention (`-Z` forward). The composition `gripper_pose × local_offset` produces the quaternion of a frame whose `+X` points along Squint's camera optical axis (SAPIEN's `add_camera(mount=, pose=)` uses the convention "camera looks along the mount's local +X").

So Isaac's `"world"` = "camera looks along world-frame-attached +X" is *semantically* what we want, and `convert_camera_frame_orientation_convention` will rotate forward `+X → -Z` internally to feed USD. **This should be correct.** But it's worth a one-pixel sanity test: place a brightly-coloured cube at a known position, render in both sims with the same gripper pose, and compare the cube's pixel coordinates. The `squint_rgb_128.npy` dumped by `debug_squint_audit.py` (SAPIEN side, line 403) and an equivalent dump from Isaac at the same gripper qpos should match to within a few pixels.

### 9.4. Episode length mismatch — secondary but worth fixing

**File:** `squint_env_cfg.py:290` — change `self.episode_length_s = 5.0` → `7.5` to match `max_episode_steps=75` for `PlaceCube-v1`. Important if you ever resume training in Isaac; deploy is unaffected (deploy is open-loop until truncation).

### 9.5. Friction range mismatches

**File:** `squint_env_cfg.py:180-227` and `squint_scene.py:158-159, 190-191, 272-273`.

- Cube/distractor friction DR: change ranges from `(0.4, 0.6)` to `(0.05, 0.6)` to match `place.py:144`.
- Table friction: re-enable DR with range `(0.05, 0.4)` (per-env material) instead of pinned 0.5. Squint does NOT pin the table; the CLAUDE.md comment that says it does is wrong.

### 9.6. Spawn box half-size

**File:** `squint_events.py:39` — change `SPAWN_BOX_HALF = 0.1` → `0.125` to match `place.py:189`.

### 9.7. `is_grasping` thresholds (verify only)

Squint's `is_grasping` (`so101.py:166-188`) is `lforce >= 0.5 N AND rforce >= 0.5 N AND angle(force, gripper_axis) <= 110°` per jaw. Our port (`squint_rewards.py:64-67, 152-158`) uses `F_g ≥ 0.5 & F_j ≥ 0.5 & cube_to_jaw < 5 cm & cube lifted`. The proximity gate is our workaround for the unfiltered contact sensor — it should be roughly correct but doesn't match Squint's angle constraint. Since rewards aren't used at deploy time (we only need success/timeout), this is informational.

### 9.8. Joint name / order parity — verified MATCH

The probe `probe_policy_actions.py` and the action term BOTH iterate joints in the order `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`, which is also the order Squint's `active_joints` iterator yields (URDF kinematic chain order). Our `obs_dump.txt` confirms qpos[0..5] = `[0, 0, 0, 1.57, -1.57, 1.04]` matching the home keyframe `[0, 0, 0, π/2, -π/2, π/3]`. No issue here.

### 9.9. Goal color palette — verified MATCH

Both code paths use palette index `0=red, 1=blue, 2=green, 3=yellow, 4=purple, 5=orange` (Squint `place.py:29-39`, our `squint_observations.py:51-58` AND `squint_events.py:443-450`). No issue here.

---

## Minimum viable fix set

To get to a non-zero success rate as fast as possible:

1. **Apply §9.1** (action scaling) — *required, single-line edit.*
2. **Apply §9.2 step-1** (disable the matte_black/emissive event chain, bump dome to 1000) — *required if the wrist frame is genuinely black; otherwise diagnose first.*
3. **Apply §9.4** (episode length 5 s → 7.5 s) — only matters if you continue training; deploy doesn't care.
4. **Apply §9.5–9.6** (DR ranges + spawn box) — only matters if you continue training; deploy works inside the distribution either way.

After 1+2 the policy should produce coherent (non-saturated, non-blind) actions on the wrist-cam image stream; that alone is enough to recover most of the gap. If still 0 %, focus next on §9.3 (camera convention quantitative sanity check on a canonical replay seed).
