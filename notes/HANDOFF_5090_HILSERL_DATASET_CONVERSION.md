# Handoff: HIL-SERL multi-color dataset conversion (run on the 5090)

> Paste this into a fresh Claude Code session running **directly on
> `ethrc-rl-ws1`** (the 5090). The conversion is too slow / awkward to
> drive over SSH from the Mac; iterating locally on the 5090 is faster.

## TL;DR

Convert `osammotg1/projet3-eval2-v1-tom-hugo` (101 episodes, 6 colors,
joint-space action space) → a HIL-SERL-loadable LeRobot v3 dataset in
EE-space, push to HF as `osammotg1/projet3-hilserl-multicolor-v1`.

We've already validated the FK conversion math on the Mac for the yellow
subset (16 episodes). The remaining work is (a) generalize to all 6 colors,
and (b) wrap the converted frames as a proper v3 dataset (videos + meta).

## Why we need this

We're running Option C on the robot side today: re-record ~15 fresh
yellow-target demos in EE space at the robot, train a single-color
HIL-SERL policy. That handles **v1**.

For **v1.5** (multi-color goal-conditioned policy), we don't have time to
re-record ~90 more demos for the other 5 colors. The 101-episode
`projet3-eval2-v1-tom-hugo` already has those trajectories — we just need
to convert them from joint-space actions (what was recorded for ACT/SmolVLA
training) into EE-space actions + extended state (what HIL-SERL's SAC actor
expects).

## State of the 5090 when this handoff starts

- User: **`tommaso`** (NOT `ethrc` — many old scripts assume `ethrc`; paths
  under `/home/ethrc/...` need to be changed to `/home/tommaso/...`).
- Hostname: `ethrc-rl-ws1`. GPU: NVIDIA GeForce RTX 5090.
- OS: Ubuntu 22.04 / 24.04 (kernel `6.17.0-23-generic`).
- Active project clone: `/home/tommaso/Desktop/robot-learning-project3`.
- This clone is on a **different branch / commit** than the Mac side. Last
  observed commit (2026-05-17): `e17384d launch_act.sh: default to
  goal-conditioned eval-2 dataset`.
- A second clone exists at `/home/tommaso/RobotLearningClassProject/...`
  (older, ignore).
- `uv` is installed; lerobot lives under `~/Desktop/lerobot-edit-scripts/`
  (with `uv.lock`). The `.venv/` may or may not exist — run `uv sync`
  there first if it's missing.

## First — get our HIL-SERL scaffolding onto the 5090

The Mac side built today (branch `tom-main`):

- `sim/hilserl/assets/so101/urdf/so_arm101.urdf` + `assets/*.stl` (URDF
  for placo IK; ~15 MB of meshes, gitignored — re-fetch instructions in
  `sim/hilserl/assets/so101/README.md`)
- `sim/hilserl/scripts/convert_eval2_to_hilserl_yellow.py` ← **the script
  you'll generalize.** Already validated end-to-end on the yellow subset.
- `sim/hilserl/configs/env_config_so101.json` and
  `sim/hilserl/configs/train_config_hilserl_so101.json`
- `sim/hilserl/README.md` (run order + per-config rationale)
- `notes/hilserl_eval2_plan.md` (design doc)

```bash
cd /home/tommaso/Desktop/robot-learning-project3
git fetch origin
git checkout tom-main      # or whatever branch holds the hilserl scaffolding
git pull
ls sim/hilserl/scripts/    # should show convert_eval2_to_hilserl_yellow.py
```

If the URDF mesh files aren't present (gitignored), re-fetch per
`sim/hilserl/assets/so101/README.md`:

```bash
TMP=/tmp/isaac_so_arm101_for_urdf
DEST=sim/hilserl/assets/so101/urdf/assets
rm -rf "$TMP"
git clone --depth=1 https://github.com/MuammerBay/isaac_so_arm101.git "$TMP"
mkdir -p "$DEST"
cp "$TMP/src/isaac_so_arm101/robots/trs_so101/urdf/assets/"*.stl "$DEST/"
rm -rf "$TMP"
```

## What "convert" actually means — the schema delta

The source dataset (`projet3-eval2-v1-tom-hugo`) was recorded for ACT /
SmolVLA training. HIL-SERL needs different inputs:

| Column | Source (`tom-hugo`) | HIL-SERL target |
|---|---|---|
| `observation.state` | 6 joint positions in degrees | **12 floats**: 6 joint positions + 6 joint velocities |
| `observation.images.wrist` | 480×640 video at 30 fps | same (or 128×128 cropped — done at training time via `crop_dataset_roi`, **don't crop in the converter**) |
| `action` | 6 joint position targets in degrees | **4 floats**: 3 EE deltas (dx, dy, dz in meters) + 1 gripper delta |
| FPS | 30 | **keep at 30** (see Insight below — do NOT downsample to 10) |
| Task strings | "Pick {color} block and place in bowl at (-15.5,29.5) cm" | same; HIL-SERL's SAC actor doesn't read them but we keep for goal-conditioning v1.5 |

### Insight — why we keep 30 fps

At 30 fps the max per-frame EE delta in the recorded trajectories is
~17 mm. If we downsample to 10 fps (take every 3rd frame), each new
step's delta = sum of 3 consecutive original deltas ≈ 3× larger →
~43 mm/step. That **exceeds** the `end_effector_step_sizes: 0.02 m`
clamp in `env_config_so101.json`, so SAC would be cloning saturated
actions. Fix: keep dataset at 30 fps; change `env.fps` from `10` to `30`
in both `env_config_so101.json` and `train_config_hilserl_so101.json`.
SO-101 servos handle 30 Hz fine — SmolVLA inference already runs there.

## What's already validated (on the Mac) — proof the math works

We ran `convert_eval2_to_hilserl_yellow.py` on the Mac for the yellow
subset only (16 episodes, 7036 frames). The diagnostic report at
`outputs/datasets/projet3-hilserl-yellow-v1-converted/conversion_report.md`
showed healthy numbers:

```
EE pose range (meters, robot frame)
  X: +0.074 → +0.364   (29 cm reach forward)
  Y: -0.138 → +0.166   (30 cm sweep left/right)
  Z: -0.030 → +0.209   (24 cm up/down)

EE deltas at 30 fps
  mean |Δxyz|: ~0.8 mm/frame
  max  |Δxyz|: 14-17 mm/frame
  frames exceeding 20 mm: 0.0%      ← under the SAC clamp ✓

Gripper deltas
  min -4.04, max +6.43, mean +0.05  ← non-trivial signal ✓
```

Two bugs we already hit and fixed in the script; **don't re-introduce them**:

1. **`lerobot.model.kinematics.RobotKinematics.forward_kinematics` expects
   joint angles in DEGREES**, not radians. The wrapper does the
   `np.deg2rad()` conversion internally (line ~67 of
   `lerobot/model/kinematics.py`). Pass dataset values straight through —
   they're already in degrees.
2. **placo's C++ binding needs `float64`**, not `float32`. The
   `obs_state.astype(np.float64)` cast in the existing script handles this.

## What's still TODO on the 5090

### Step 1 — Generalize the conversion script (~20 min)

Take `sim/hilserl/scripts/convert_eval2_to_hilserl_yellow.py` and make it
process **all 6 task indices**, not just `task_index == 0`. Suggested
new name: `convert_eval2_to_hilserl_multicolor.py`.

Changes:
- Remove the `TARGET_TASK_INDEX = 0` filter; iterate over all unique
  task_indices found in the source parquets.
- Preserve `task_index` in the output so the dataset stays goal-conditioned.
- Output directory: `outputs/datasets/projet3-hilserl-multicolor-v1-converted/`.

Sanity check: should process ~43,289 frames across 101 episodes; report
should show EE ranges similar to the yellow subset (the workspace is the
same; only the cube colors differ).

### Step 2 — Wrap as a LeRobot v3 dataset (~60 min, the hard part)

The current script outputs a flat parquet — that's **not** a loadable
dataset. Need to build the full v3 directory structure:

```
outputs/datasets/projet3-hilserl-multicolor-v1/
├── meta/
│   ├── info.json                 # schema with the new feature shapes
│   ├── tasks.parquet              # same 6 task strings as source, indexed
│   ├── stats.json                 # per-feature mean/std/min/max
│   └── episodes/                  # per-episode metadata (frame counts, etc.)
│       └── chunk-000/file-000.parquet
├── data/
│   └── chunk-000/
│       └── file-{000..NNN}.parquet   # converted frames, grouped by episode
└── videos/
    └── observation.images.wrist/
        └── chunk-000/
            └── file-{000..NNN}.mp4   # video files (see Step 3)
```

Mirror the source's `meta/info.json` structure. Critical changes:
- `features.observation.state.shape: [12]` (was `[6]`)
- `features.observation.state.names`: extend with `joint_velocity` entries
- `features.action.shape: [4]` (was `[6]`)
- `features.action.names`: `["ee_dx", "ee_dy", "ee_dz", "gripper_delta"]`
- `features.observation.images.wrist`: unchanged (video, 480×640, 30 fps)
- `total_frames`: drop by 101 (we lose one frame per episode to finite-diff)
- `total_episodes`: 101 (same)

Probe the source `info.json` for the exact field names you need to mirror:

```bash
cat ~/.cache/huggingface/hub/datasets--osammotg1--projet3-eval2-v1-tom-hugo/snapshots/*/meta/info.json
```

### Step 3 — Handle the videos

Two viable approaches:

**3a. Reuse source mp4 files as-is** (recommended — fastest, no encoding):
- The source packs ~3-5 episodes per video file at 30 fps with `av1` codec.
- We're keeping every frame and not changing fps, so the video data is
  identical. Just copy / symlink the source mp4 files into the new
  dataset's `videos/observation.images.wrist/chunk-000/` directory.
- The `frame_index` in our converted parquets points to the same frames
  in the videos as it did in the source. Should "just work" for any
  episodes we keep at their original parquet→video file alignment.

**3b. Re-encode per episode** (cleaner, more work):
- Use `ffmpeg` (likely already installed on the 5090) to split source
  videos into per-episode segments, then re-pack.
- Only needed if you change the episode→file packing.

Recommend 3a unless 3b becomes necessary for some schema reason.

### Step 4 — Stats + meta finalization

LeRobot's `LeRobotDataset.__init__` reads `meta/stats.json` for input
normalization. Compute:
- For each numeric feature: `mean`, `std`, `min`, `max`, `q01`, `q99`
- For images: per-channel mean/std (use ImageNet stats `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` as a fallback)

Look at the source `meta/stats.json` for the exact JSON shape.

### Step 5 — Push to HF

```bash
hf auth login          # paste write token for osammotg1
hf upload osammotg1/projet3-hilserl-multicolor-v1 \
    outputs/datasets/projet3-hilserl-multicolor-v1 . \
    --repo-type=dataset --private
```

### Step 6 — Smoke test the resulting dataset

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("osammotg1/projet3-hilserl-multicolor-v1")
print("episodes:", ds.meta.total_episodes)
print("features:", list(ds.meta.features.keys()))
print("obs.state shape:", ds.meta.features["observation.state"]["shape"])
print("action shape:", ds.meta.features["action"]["shape"])
# Should print: 101 / [...12] / [...4]
```

If that loads cleanly, the dataset is good for HIL-SERL training.

## Acceptance criteria

- ✅ `LeRobotDataset("osammotg1/projet3-hilserl-multicolor-v1")` loads
  without errors
- ✅ 101 episodes, ~43,188 frames (101 frames lost to finite-diff)
- ✅ `observation.state` is 12-dim; `action` is 4-dim
- ✅ Sanity probe: random frame's EE delta magnitude is < 20 mm
- ✅ Task strings preserved (`tasks.parquet` has same 6 entries as source)

## What NOT to do

- ❌ **Don't modify the lerobot library** under
  `~/Desktop/lerobot-edit-scripts/` (editable install — that rule from
  the Mac side carries over).
- ❌ **Don't downsample to 10 fps** — the EE delta clamp issue above.
- ❌ **Don't crop the wrist image in the converter** — let `crop_dataset_roi`
  do it later, after we've decided the ROI at the robot.
- ❌ **Don't push as `osammotg1/projet3-hilserl-yellow-*`** — that namespace
  is reserved for the freshly-recorded single-color v1 dataset we're
  building at the robot today.
- ❌ **Don't push under `tommaso/...`** — match the existing
  `osammotg1/*` convention so the Mac side can pull it without auth
  changes.

## Tasks for the 5090 Claude (suggested TodoList)

```
1. git pull tom-main into ~/Desktop/robot-learning-project3
2. Fetch URDF meshes per sim/hilserl/assets/so101/README.md
3. Sanity-test FK on the 5090 (run the URDF load + FK smoke test)
4. Sanity-test the existing yellow-only conversion script
   (validate it produces the same numbers we saw on the Mac)
5. Write convert_eval2_to_hilserl_multicolor.py (generalize from yellow)
6. Write build_v3_dataset.py (info.json + tasks.parquet + episodes + data + videos)
7. Compute stats.json
8. Local smoke test: LeRobotDataset(local_path) loads
9. Push to HF as osammotg1/projet3-hilserl-multicolor-v1
10. Remote smoke test: LeRobotDataset(hf_repo_id) loads
11. Write notes/hilserl_multicolor_conversion_results.md with EE-range
    + delta-distribution numbers (mirror the yellow-only report format)
```

## When this is done

Drop a note in Slack / WhatsApp / wherever the team coordinates,
something like:

> v1.5 multi-color HIL-SERL dataset ready:
> `osammotg1/projet3-hilserl-multicolor-v1`. 101 episodes, all 6 colors,
> EE-space actions at 30 fps. Load with `LeRobotDataset(...)`.
> Conversion report: `notes/hilserl_multicolor_conversion_results.md`.

The Mac side will pick this up once v1 (single-color yellow, fresh demos)
is trained and we move to v1.5 (color-conditioned).

## Provenance

- Source dataset: `osammotg1/projet3-eval2-v1-tom-hugo` (101 ep, 43,289
  frames, 6 colors, av1 video @ 30 fps, codebase_version v3.0)
- Mac-side scaffolding commit: tom-main HEAD as of 2026-05-17
- Conversion logic source: `sim/hilserl/scripts/convert_eval2_to_hilserl_yellow.py`
- Validation reference: the yellow-only report at
  `outputs/datasets/projet3-hilserl-yellow-v1-converted/conversion_report.md`
- HF doc: `huggingface.co/docs/lerobot/hilserl` (the target HIL-SERL recipe)
