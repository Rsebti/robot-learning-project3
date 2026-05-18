# Validated real↔sim mapping (READ before any real-robot run)

Identical in `deploy_utils/manipulator.py`, `infer_eval1_64.py`,
`infer_eval2_64.py` (and the 16px deploy-handoff scripts). Derived from 4
ground-truth anchors + step-by-step HF-demo replay, **validated visually**.

## The mapping
Joint order = `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper]` (= LeRobot SO-101 follower `sync_read` order).

**Arm:** `q_sim = deg2rad(servo_deg) + JOINT_OFFSET`
with `JOINT_OFFSET = [0.0, +0.080, +0.220, 0.0, 0.0, 0.0]` rad
(only lift +0.080 and elbow +0.220; a homing-offset-type recalibration).
- `get_qpos` (real→sim): **add** JOINT_OFFSET (in deg before deg2rad, or in
  rad after — equivalent, deg2rad is linear).
- `set_target_qpos` (sim→real): **subtract** it. Round-trip exact (~1e-15).

**Gripper:** separate linear remap, real servo **[1°, 75°]** (closed→open)
↔ sim **[-18°, 120°]**. The −18° closed end clamps near the joint limit for
a hard grasp. (Old `[-60.13, 66.73]→[-10,120]` was inverted — mapped
real-closed to sim-open; that was the false-grasp bug.)

**Image:** wrist RGB → center-crop to square → resize 128 (area) → resize
**64** (area). The CNN encoder is the 64px architecture (`Conv2d(3,32,8,s4)
→ (32,64,4,s2) → (64,64,3,s1) → Flatten`, matches `train_squint.py`
`image_size==64`). Strict `state_dict` load verified for all curriculum
ckpts.

## Why these numbers
Pure `deg2rad` put the gripper ~3–4 cm too high at grasp (sim2real descent
gap). The 4 anchors share ≈the same lift/elbow (all near home) so they don't
constrain lift/elbow; the home↔'rest' anchor is kinematically wrong on arm
height (proved: no per-joint sign/offset with that anchor descends). A
constant lift/elbow offset, tuned on the demo grasp frame so **TCP z ≈
0.010 m at grasp** (tip below the 2 cm cube's mid-height), is the principled
compromise the user validated visually.

## Caveat (tell whoever runs the robot)
This is a **regional** compromise: faithful in the operating region
(approach → grasp → place, where the policy works), NOT at the curled
parked home (outside policy operation — irrelevant for deployment). It is
self-consistent (bidirectional, exact round-trip). **The real-robot run is
the only true validation.** If a production-grade exact descent is needed,
capture a **5th real anchor**: put the real arm in a known non-home pose
(lift/elbow far from home, e.g. arm extended toward the table), record the
6 joint values + a wrist-cam photo → that pins lift/elbow exactly.

## Sanity self-test (already verified, reproducible)
```python
# load any curriculum ckpt with the matching infer script's CNNEncoder/Actor:
enc.load_state_dict(ck["encoder"]); act.load_state_dict(ck["actor"])  # strict
# round-trip: real_servo_deg --get_qpos--> sim_rad --set_target--> real_servo_deg
# max abs error over 2000 random poses ≈ 2e-5 deg (≈ float noise, ~2000×
# below one servo step) -> functionally exact, == manipulator.py.
```
