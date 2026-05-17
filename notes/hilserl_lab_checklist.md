# HIL-SERL — at-the-robot checklist (single-color v1, yellow)

> Use this when you have the SO-101 follower + leader + cubes + bowl in front
> of you. Companion to `notes/hilserl_eval2_plan.md` (the design doc) and
> `sim/hilserl/README.md` (the config-by-config reference).
>
> Target color = **yellow** (Option C — single-color v1). Change everywhere
> if you pivot.

## Pre-flight — before the arm moves (≤5 min)

```bash
PY=/Users/admin/miniforge3/bin/python3.13
cd "/Users/admin/Documents/ETH/M4/Robot Learning /Project S101/robot-learning-project3"

# 1. URDF loads + FK is sensible (~0.39, 0.0, 0.23 m home pose)
$PY -c "from lerobot.model.kinematics import RobotKinematics
import numpy as np
rk = RobotKinematics('sim/hilserl/assets/so101/urdf/so_arm101.urdf',
                     target_frame_name='gripper_frame_link')
print('joints:', rk.joint_names)
print('FK home XYZ:', rk.forward_kinematics(np.zeros(6))[:3, 3])"

# 2. SmolVLA still imports (we upgraded transformers 4.57.6 -> 5.3.0 with [hilserl]).
#    If this breaks, our parallel BC baseline is at risk — investigate before
#    proceeding.
$PY -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
print('SmolVLA OK')"

# 3. HF auth as osammotg1
hf auth whoami

# 4. Both arms enumerated by macOS
ls /dev/tty.usbmodem* | head -5
# Should show two ports. If they have moved, update env_config_so101.json
# (env.robot.port + env.teleop.port) OR override at runtime via env vars.
```

## Step 1 — End-effector workspace bounds (~10 min)

Replaces the placeholder bounds in
`sim/hilserl/configs/env_config_so101.json` →
`processor.inverse_kinematics.end_effector_bounds`.

```bash
$PY -m lerobot.find_joint_limits \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B141129871 \
  --robot.id=so101_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5B141128171 \
  --teleop.id=so101_leader
```

Move the leader through the workspace where the yellow + distractor cubes
and the bowl will sit. Stay generous — the bounds clip exploration, so
under-shooting risks blocking the policy from reaching corners.

Paste the printed `Max ee position` / `Min ee position` values into both
JSONs (env_config AND train_config — they share the env block) at:

```json
"end_effector_bounds": {
  "min": [<min_x>, <min_y>, <min_z>],
  "max": [<max_x>, <max_y>, <max_z>]
}
```

## Step 2 — Record 15 EE-space demos (~25 min)

Scene per episode:
- **One yellow cube** + **one distractor cube** (any color from the trained
  palette — blue, green, violet, red, or orange) placed adjacent.
- Bowl at the fixed (-15.5, 29.5) cm pose (or wherever you want — v1
  trains at one bowl pose).

```bash
bash teleop/record_hilserl_demos.sh
# Override per session if needed:
#   FOLLOWER_PORT=/dev/tty.usbmodem... LEADER_PORT=/dev/tty.usbmodem... \
#     bash teleop/record_hilserl_demos.sh
```

During each episode, drive the leader to:
1. Approach yellow cube
2. Grasp it (close jaw)
3. Lift + carry to bowl
4. Release into bowl
5. Press `s` for success, then `esc`-to-fail any episode you want to redo.

The recorder pushes to `osammotg1/projet3-hilserl-yellow-v1` on HF.

## Step 3 — Replay sanity check (~2 min, optional)

Edit `sim/hilserl/configs/env_config_so101.json`:
- `mode: "replay"`
- `dataset.replay_episode: 0` (then 7, then 14 — sample evenly)

```bash
$PY -m lerobot.rl.gym_manipulator \
  --config_path sim/hilserl/configs/env_config_so101.json
```

The follower should reproduce your demo. If it doesn't, calibration drifted
between recording and replay → re-run calibration before training.

Reset `mode` back to `"record"` (or whatever) when done.

## Step 4 — Determine the wrist-image ROI (~5 min)

```bash
$PY -m lerobot.rl.crop_dataset_roi \
  --repo-id osammotg1/projet3-hilserl-yellow-v1
```

Draw a rectangle around the workspace area (cubes + bowl visible, gripper
visible during reach). Press `c` to confirm. The script writes a new dataset
`osammotg1/projet3-hilserl-yellow-v1-cropped` and prints the crop tuple.

Paste the printed crop into both JSONs at:

```json
"crop_params_dict": {
  "observation.images.wrist": [<top>, <left>, <height>, <width>]
}
```

(Both env_config_so101.json AND train_config_hilserl_so101.json — they share
the env block.)

Also update `train_config_hilserl_so101.json` → `dataset.repo_id` to the
**cropped** dataset name.

## Step 5 — Network probe for the learner location (~2 min)

We need to decide where the SAC learner runs.

```bash
# From the Mac, probe the 5090 over SSH
ping -c 5 <5090-hostname-or-ip>
# Median RTT under 5 ms → gRPC will be fine at 10 Hz
# Median RTT over 50 ms → strongly consider running learner on the Mac
```

If 5090 is reachable cleanly, leave `train_config_hilserl_so101.json`
`policy.storage_device: "cuda"`. If we have to fall back to the Mac,
change it to `"mps"` and `policy.device: "mps"`.

## Step 6 — Start the learner and actor (~5 min spin-up, then ~2.5 h training)

**Learner** (on the 5090, via SSH):

```bash
# Sync the project repo to the 5090 first (rsync, git pull, etc.)
$PY -m lerobot.rl.learner \
  --config_path sim/hilserl/configs/train_config_hilserl_so101.json
```

**Actor** (on the Mac, separate terminal, robot connected):

```bash
$PY -m lerobot.rl.actor \
  --config_path sim/hilserl/configs/train_config_hilserl_so101.json
```

What to watch:
- The intervention rate plot in W&B `projet3-hilserl` should decay over the
  first ~30 minutes. If it's still flat after an hour, stop and re-evaluate.
- Episodic return should trend upward. Flat → policy not learning.
- The actor's "actor_lag" metric should stay under a few hundred ms. If it
  spikes, gRPC is congested.

Intervention strategy (per HF doc):
- Let the policy explore the first few episodes unimpeded.
- Intervene only when it's about to do something destructive (knock cube
  off table, miss the bowl entirely). Don't intervene to "help it find" the
  cube — that defeats the exploration.
- As the policy improves, your interventions should become rarer and
  shorter (just the final grasp / final placement).

## Step 7 — Day-3 evaluation (~45 min)

Stop training when intervention rate has been flat-low for ~15 min.

```bash
# 5-rollout PRIMARY eval (yellow cube at trained bowl pose)
$PY -m lerobot.rl.gym_manipulator \
  --config_path sim/hilserl/configs/env_config_so101.json \
  --mode=eval \
  --policy.path=osammotg1/projet3-hilserl-yellow-v1-policy \
  --dataset.num_episodes_to_record=5

# 5-rollout BOWL-OOD probe (yellow cube, bowl moved +3 cm in x or y)
# This is the silent-failure-on-eval-day detector — required per plan §7.
# Manually move the bowl ~3 cm and re-run the same command above.
```

Record per-rollout: success Y/N + qualitative failure mode (missed
grasp / wrong cube / dropped en route / missed bowl).

## Decision points to flag to the team

- **If the URDF self-collision warnings cause IK errors during recording:** the
  bounds may be too tight, or the URDF collision geometry needs tweaking.
  Loosen `end_effector_bounds` first.
- **If `record_hilserl_demos.sh` errors with "extra/missing input feature":**
  the env-config's `processor.observation.add_*_to_observation` flags don't
  match what the SAC policy's `input_features` expects. Re-derive shapes.
- **If SmolVLA inference is now broken** (transformers 5.3.0 incompatibility):
  pin `transformers==4.57.6` in a separate venv for SmolVLA inference only.
  Don't downgrade in the lerobot-hilserl env — that breaks HIL-SERL.

## After the run

- Update `notes/sanity_results.md` or create `notes/hilserl_yellow_v1_results.md`
  with per-rollout outcomes (5 primary + 5 bowl-OOD).
- Decide v1.5 direction based on results:
  - ≥3/5 success: v1.5 = retrain with bowl randomization OR pivot to per-color
  - <3/5: investigate; consider Option B (SmolVLA-seeded demos) per plan §4.
