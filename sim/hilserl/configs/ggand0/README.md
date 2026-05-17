# ggand0/hil-serl-so101 configs — yellow v1

This folder holds the configs that drive the **ggand0 stack** (lerobot 0.3.2
fork + their `hil-serl-so101` repo). Schema is ggand0-flavored — flat
top-level `robot`/`teleop`/`wrapper` keys — different from our lerobot 0.5.2
configs in `../env_config_so101.json` and `../train_config_hilserl_so101.json`.

See `notes/ggand0_evaluation.md` for the decision rationale.

## Files

- `yellow_v1_record.json` — record-mode config. Uses `lerobot.scripts.rl.gym_manipulator` (note: 0.3.2 path; in 0.5.2 it would be `lerobot.rl.gym_manipulator`). Drives 10 fps leader-mode recording with the SO-101 wrist camera. Default `num_episodes: 2` and `repo_id: ...-ggand0smoke` for a safe smoke test.
- `yellow_v1_train.json` — SAC actor/learner config. Spawns ResNet-10 vision encoder + 18-D state. Online buffer 200k, offline buffer 5k. Runs on Mac MPS by default.

## Hard-coded paths (Mac-specific)

Both configs reference absolute Mac paths:

| Field | Mac value |
|---|---|
| `robot.port` | `/dev/tty.usbmodem5B141129871` |
| `teleop.port` | `/dev/tty.usbmodem5B141128171` |
| `robot.cameras.gripper_cam.index_or_path` | `0` |
| `robot.calibration_dir` | `/Users/admin/.cache/huggingface/lerobot/calibration/robots/so101_follower` |
| `teleop.calibration_dir` | `/Users/admin/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader` |
| `robot.mujoco_model_path` | `/Users/admin/Documents/ETH/M4/Robot Learning /Project S101/robot-learning-project3/sim/hilserl/assets/so101_mjcf/so101_new_calib.xml` |
| `device` (env + policy) | `mps` |

To run on the 5090 workstation (Linux + CUDA), copy each file as
`*.linux.json` and edit the six entries above. Don't add a template engine
yet — manual fork-and-edit is faster for two machines.

## Locked decisions baked in

From `notes/hilserl_eval2_plan.md` plan-eng-review v2 (do NOT relitigate):

- **D1 single yellow** — `task: hilserl_pick_yellow_v1`.
- **D2 fixed bowl pose** — handled implicitly via `wrapper.ik_reset_ee_pos` (currently ggand0's default `[0.25, 0.0, 0.07]`; **TODO**: replace with our measured bowl pose `(-15.5, 29.5)` cm + the desired hover height once we settle on it).
- **D3 manual keyboard reward** — `reward_classifier_pretrained_path: null` for v1. Keyboard `9` triggers success during record; trained classifier comes in v1.5.

## Action / observation shape

- 4-D action: `[delta_x, delta_y, delta_z, gripper]`, all in `[-1, 1]`, scaled by `action_scale: 0.02` (m per unit). Wrist_flex (joint 3) and wrist_roll (joint 4) are **locked at 90°** by ggand0's IK — that's the trick that turns the 5-DoF arm into a 3-DoF reacher and is why the action is exactly 4-D.
- 18-D state: leader + follower proprioception auto-computed by `wrapper.add_full_proprioception: true`. We do not hand-construct this vector.
- 128×128 RGB image, center-y cropped from 480×480 then resized.
- 10 Hz control loop.

## Workspace bounds

`end_effector_bounds = {min: [0.057, -0.244, -0.035], max: [0.430, 0.286, 0.248]}` (meters in robot frame). Measured today with `lerobot-find-joint-limits` + a 1 cm safety margin. Caveat: these were computed under lerobot 0.5.2's placo+URDF FK; ggand0's MuJoCo FK uses the same CAD source so they should agree to mesh-import precision. Verify with a probe before trusting any narrow bound.

## Things still TODO before recording

1. Verify the locked wrist_flex/wrist_roll target (`{"3": 90.0, "4": 90.0}`) physically points the gripper downward toward the table on our calibration. Our home pose has wrist_flex=57.7° and wrist_roll=−9.275°, so 90°/90° may NOT correspond to a clean vertical gripper on this calibration. If it doesn't, override `locked_joint_positions` to our home values.
2. Replace `ik_reset_ee_pos: [0.25, 0.0, 0.07]` with the EE position above our actual bowl (D2). We need to FK our `fixed_reset_joint_positions` through the MJCF to get this.
3. Pick a smoke-test `repo_id` we actually own write access for. Current `osammotg1/projet3-hilserl-yellow-v1-ggand0smoke` matches the convention.
