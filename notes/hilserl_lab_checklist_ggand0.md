# HIL-SERL at-the-robot checklist — ggand0 stack (single-color v1, yellow)

> Companion to `notes/hilserl_eval2_plan.md` (design doc) and
> `sim/hilserl/configs/ggand0/README.md` (config reference).
> **Schema:** ggand0/lerobot @ feat/hil-serl (lerobot 0.3.2 fork).
> **Old checklist** at `notes/hilserl_lab_checklist.md` is for the
> lerobot 0.5.2 path — kept around but NOT what we run for v1.

Convention used below: `GGAND0` = `~/Desktop/eval-ggand0/hil-serl-so101`,
`REPO` = `/Users/admin/Documents/ETH/M4/Robot Learning /Project S101/robot-learning-project3`,
`UVRUN` = `cd $GGAND0 && uv run`.

## Pre-flight — before the arm moves (≤5 min)

```bash
# 0. Setup is done? (one-time, see notes/ggand0_setup_mac.md)
cd ~/Desktop/eval-ggand0/hil-serl-so101 && uv run python -c "import lerobot; print(lerobot.__version__)"
# Expected: 0.3.2

# 1. HF auth as osammotg1
uv run huggingface-cli whoami

# 2. Both arms enumerated
ls /dev/tty.usbmodem* | head -5
# Should show two entries. If ports moved, update both ggand0 configs.

# 3. Camera is visible at OpenCV index 0
uv run lerobot-find-cameras opencv | head

# 4. Verify MuJoCo model loads with our vendored XML
uv run python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('$REPO/sim/hilserl/assets/so101_mjcf/so101_new_calib.xml')
print('joints:', m.njnt, '  bodies:', m.nbody)
print('gripperframe site id:', mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe'))
"
# gripperframe site id should be >= 0 (not -1)
```

## Step 1 — Verify locked-wrist orientation (~10 min)

ggand0's IK locks `wrist_flex` (joint 3) and `wrist_roll` (joint 4) at
fixed angles. We left their defaults `{"3": 90.0, "4": 90.0}` in
`yellow_v1_record.json`. On our calibration, our home pose has
`wrist_flex=57.7°` and `wrist_roll=−9.275°` — so 90°/90° may NOT
point the gripper vertically downward on this Mac.

```bash
# Move the follower to (wrist_flex=90, wrist_roll=90) and visually inspect.
uv run python <<'EOF'
import time
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
cfg = SO101FollowerConfig(
    port="/dev/tty.usbmodem5B141129871",
    id="so101_follower",
    use_degrees=True,
    calibration_dir="/Users/admin/.cache/huggingface/lerobot/calibration/robots/so101_follower",
    cameras={},
)
r = SO101Follower(cfg)
r.connect()
home = {"shoulder_pan.pos": -2.593, "shoulder_lift.pos": -95.429,
        "elbow_flex.pos": 97.670, "wrist_flex.pos": 90.0,
        "wrist_roll.pos": 90.0, "gripper.pos": 50.0}
r.send_action(home)
time.sleep(2)
input("Look at the gripper. Is it pointing straight down? (enter to disconnect)")
r.disconnect()
EOF
```

If yes → leave configs as-is.
If no → override `locked_joint_positions` in BOTH `yellow_v1_record.json`
and `yellow_v1_train.json` to the angles where the gripper is vertical.
A common substitute: `{"3": 90.0, "4": 0.0}` (only flex points down,
roll at calibration zero). Test iteratively.

## Step 2 — IK reset pose calibration (~5 min)

`yellow_v1_record.json:wrapper.ik_reset_ee_pos = [0.25, 0.0, 0.07]`
is ggand0's default. We want this to land above our actual bowl pose
**(-15.5, 29.5) cm** (D2 locked decision) at a safe hover height
(~12 cm above table = +0.12 m).

Edit both ggand0 configs:
```json
"ik_reset_ee_pos": [-0.155, 0.295, 0.12]
```

Then dry-run with the follower powered but cubes/bowl removed:

```bash
$UVRUN python -m lerobot.scripts.rl.gym_manipulator \
    --config_path $REPO/sim/hilserl/configs/ggand0/yellow_v1_record.json \
    --num_episodes=1 --mode=record
# Watch the arm move to the IK reset pose. Confirm it's hovering over
# where the bowl will go. Press ESC immediately to abort the record.
```

## Step 3 — Record 2-episode smoke test (~10 min)

Stay at `num_episodes: 2` and `repo_id: ...-ggand0smoke` until smoke
passes — this avoids polluting the real v1 dataset.

```bash
$UVRUN python -m lerobot.scripts.rl.gym_manipulator \
    --config_path $REPO/sim/hilserl/configs/ggand0/yellow_v1_record.json
```

Per-episode workflow:
- Drive the leader to teleop the follower.
- Approach yellow → grasp → lift → carry to bowl → release.
- Press `9` for success, then `ESC` to end the episode.
- Press left-arrow to re-record an episode if you mess it up.
- Press `q` to quit recording entirely.

Verify on HF that `osammotg1/projet3-hilserl-yellow-v1-ggand0smoke` got
2 episodes. Spot-check the videos — gripper-cam should be 640×480 RGB,
~10 fps for the parquet rows. Action column should be (4,) shape.

If smoke passes → move on. If it fails → debug here, NOT later.

## Step 4 — Record real 15-episode dataset (~30 min)

Edit `yellow_v1_record.json`:
- `num_episodes: 15`
- `repo_id: "osammotg1/projet3-hilserl-yellow-v1"`
- `push_to_hub: true`

Then:
```bash
$UVRUN python -m lerobot.scripts.rl.gym_manipulator \
    --config_path $REPO/sim/hilserl/configs/ggand0/yellow_v1_record.json
```

Scene per episode:
- **One yellow cube** + **one distractor cube** (any color) placed adjacent.
- Bowl at fixed (-15.5, 29.5) cm.

## Step 5 — Crop the wrist image ROI (~5 min)

```bash
$UVRUN python -m lerobot.scripts.rl.crop_dataset_roi \
    --repo-id osammotg1/projet3-hilserl-yellow-v1
```

Drag a rectangle covering the workspace (cubes + bowl + gripper visible
during reach). Press `c` to confirm. Outputs a new dataset
`osammotg1/projet3-hilserl-yellow-v1-cropped` and prints the crop tuple.

Paste the crop tuple into `yellow_v1_train.json` →
`env.wrapper.crop_params_dict.observation.images.gripper_cam`. Also
update `dataset.repo_id` to the cropped variant.

## Step 6 — Build the offline buffer (~2 min)

We need to merge the cropped dataset into an offline buffer that the
SAC learner can warm-start from. ggand0's pattern uses a JSON config
under `data/labels/`.

For v1, the simplest approach is to point the train config's `dataset.repo_id`
directly at the cropped dataset — that's the offline buffer.

If that's not enough (e.g. you want to filter only successful frames),
adapt `scripts/create_grasponly_offline_dataset.py` from ggand0:

```bash
# Optional path — only if naive offline buffer underperforms
$UVRUN python $GGAND0/scripts/create_grasponly_offline_dataset.py
# Edit data/labels/offline_v1.json first to point at our cropped dataset
```

## Step 7 — Start the learner and actor (~5 min spin-up, ~2-3 h training)

**Learner** (Terminal 1, on Mac MPS for v1):

```bash
$UVRUN python -m lerobot.scripts.rl.learner \
    --config_path $REPO/sim/hilserl/configs/ggand0/yellow_v1_train.json
```

**Actor** (Terminal 2, on Mac with robot connected):

```bash
$UVRUN python -m lerobot.scripts.rl.actor \
    --config_path $REPO/sim/hilserl/configs/ggand0/yellow_v1_train.json
```

What to watch:
- `episode_count` should increase steadily (every ~10 s at 10 Hz × 10 s control_time).
- After ~30 min, the SAC actor should start completing more episodes
  without intervention. If intervention rate is still ~100% after 1 h,
  stop and investigate (likely IK or reward issues).
- `actor_lag` should stay low; if it spikes, the MPS learner is too slow
  and we need to migrate it to the 5090.

Intervention strategy (during HIL phase):
- Let the policy explore unimpeded for the first few episodes.
- Press `7` to take over the leader during destructive behaviors only
  (off-table, knocking distractor cube, etc.).
- Don't help it "find" the cube — that defeats exploration.
- For successful demonstrations during HIL: press `9` to mark success at
  the moment of release in the bowl.

## Step 8 — Stop training + evaluate (~30 min)

Stop when intervention rate has been < 30 % for 15 consecutive minutes.

```bash
$UVRUN python $GGAND0/scripts/hilserl_inference.py \
    --config_path $REPO/sim/hilserl/configs/ggand0/yellow_v1_train.json \
    --checkpoint $REPO/outputs/hilserl_yellow_v1/checkpoints/<latest>/pretrained_model \
    --num_episodes 5 \
    --record_video --video_dir $REPO/outputs/hilserl_yellow_v1/eval/
```

Record per-rollout: success Y/N + qualitative failure mode.

Then BOWL-OOD probe: manually shift the bowl ~3 cm in x or y and re-run
the 5 rollouts. This catches policies that memorize the absolute bowl
position rather than visually grounding it.

## Decision points to flag to the team

- **Locked wrist angles don't point gripper down** → see Step 1 override.
- **MPS too slow for SAC learner** → migrate learner to 5090, change
  `policy.device: "cuda"`, edit `actor_learner_config.learner_host` to
  the 5090's reachable address.
- **Smoke records 0 episodes / crashes immediately** → likely
  `mujoco_model_path` not resolvable, or wrist_flex hits a software
  joint limit when ggand0 IK pushes it during init.

## After the run

- Update `notes/sanity_results.md` or create
  `notes/hilserl_yellow_v1_results.md` with per-rollout outcomes
  (5 primary + 5 bowl-OOD).
- Decide v1.5 direction based on results:
  - ≥3/5 success: v1.5 = multi-color (the 5090-converted dataset is
    waiting at `osammotg1/projet3-hilserl-multicolor-v1` per
    `notes/HANDOFF_5090_HILSERL_DATASET_CONVERSION.md`).
  - <3/5 success: investigate failure modes; potentially add reward
    classifier (v1.5 was going to do this anyway, per D3 → trained
    ResNet-10 classifier).
