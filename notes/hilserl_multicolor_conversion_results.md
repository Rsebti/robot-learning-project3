# HIL-SERL multi-color dataset conversion — results

Ran on `ethrc-rl-ws1` (RTX 5090) on 2026-05-17. Output:
`outputs/datasets/projet3-hilserl-multicolor-v1/` (4.4 MB meta+data,
128 MB videos symlinked from source).

## Source

- `osammotg1/projet3-eval2-v1-tom-hugo` (snapshot `acf7fe673333`)
- 101 episodes, 43,289 frames, 6 colors @ 30 fps
- Source schema: joint-space (6 pos / 6 actions), av1 wrist video @ 480×640

## Output

- **Local**: `outputs/datasets/projet3-hilserl-multicolor-v1/`
- **Target HF repo**: `osammotg1/projet3-hilserl-multicolor-v1` (NOT YET PUSHED)
- 101 episodes, **43,188 frames** (lost 101 = one per episode to finite-diff)
- Schema: `observation.state[12]` (6 pos + 6 vel), `action[4]` (3 EE deltas + 1 gripper delta)
- FPS unchanged at 30 (videos reused as-is)

## Per-task frame counts (converted)

| task_index | color  | src frames | converted |
|---|---|---|---|
| 0 | yellow | 7052  | 7036  |
| 1 | blue   | 8697  | 8676  |
| 2 | green  | 7187  | 7171  |
| 3 | violet | 7012  | 6996  |
| 4 | red    | 7128  | 7112  |
| 5 | orange | 6213  | 6197  |
| **total** |     | 43,289 | **43,188** |

## EE pose range (meters, robot frame, all colors)

- X: +0.063 → +0.400  (34 cm forward reach)
- Y: −0.230 → +0.200  (43 cm left/right sweep)
- Z: −0.038 → +0.245  (28 cm vertical span)

Slightly larger workspace than yellow-only (X: 0.074→0.364, Y: −0.138→0.166,
Z: −0.030→0.209) — expected: the other colors are placed across the full
table area, yellow alone clusters tighter.

## EE deltas (meters / 30-fps frame)

| axis | mean |Δ| | max |Δ| |
|---|---|---|
| dx | 0.0008 | 0.0192 |
| dy | 0.0008 | 0.0223 |
| dz | 0.0009 | 0.0190 |

Frames where max-axis delta > 0.02 m (the `end_effector_step_sizes` clamp): **6 / 43,188 = 0.014%**
— well within tolerance.

Yellow-only had 0.00%; the 6 outliers come from blue/green episodes
that include a couple of fast hand movements. Negligible.

## Gripper delta (raw units / frame)

- min: −4.107
- max: +6.434
- mean: +0.0548

Non-trivial signal. Same magnitudes as the Mac-side yellow-only validation.

## Joint state (12-dim observation.state)

- joint pos range: −126.4° → +100.5°
- joint vel range: −211.0°/s → +269.0°/s

Velocities computed as forward-difference: `(q[t+1] − q[t]) * 30 fps`.

## Smoke test

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("test/local", root="outputs/datasets/projet3-hilserl-multicolor-v1")
# episodes: 101
# total frames: 43188
# obs.state shape: (12,)
# action shape: (4,)
# observation.images.wrist: (3, 480, 640) float32  ← loads from video correctly
```

## What changed vs source

- `meta/info.json`: `action.shape: [6]→[4]`, `observation.state.shape: [6]→[12]`,
  names updated (ee_dx/dy/dz/gripper_delta + 6 .pos / 6 .vel)
- `meta/episodes/*.parquet`: per-episode `length -= 1`, `to_timestamp -= 1/30`,
  `dataset_from/to_index` recomputed, per-episode action/state stats recomputed
- `meta/tasks.parquet`: unchanged (same 6 color strings)
- `meta/stats.json`: action+state stats recomputed for new shapes; image stats
  lifted verbatim from source (pixels unchanged)
- `data/chunk-000/*.parquet`: rebuilt with new feature dtypes/shapes,
  episode packing preserved (same N episodes per data file), global `index`
  column recomputed
- `videos/observation.images.wrist/chunk-000/*.mp4`: **symlinks** to source
  files (every frame referenced still exists; the 1 dropped frame per
  episode is just unreferenced in the new parquet)

## Two bugs not re-introduced

1. ✅ Passed joint angles to `RobotKinematics.forward_kinematics` in **degrees**
   (the placo wrapper does `np.deg2rad` internally).
2. ✅ Cast `observation.state` to `float64` before FK (placo's C++ binding
   refuses float32).

## Files

- Conversion script: `sim/hilserl/scripts/convert_eval2_to_hilserl_multicolor.py`
- v3 builder: `sim/hilserl/scripts/build_v3_dataset.py`
- Output dataset: `outputs/datasets/projet3-hilserl-multicolor-v1/`
- Intermediate parquet: `outputs/datasets/projet3-hilserl-multicolor-v1-converted/all_ee_frames.parquet`
- Yellow-only validation (re-run on 5090, numbers match Mac):
  `outputs/datasets/projet3-hilserl-yellow-v1-converted/conversion_report.md`

## HF push — done

Pushed to `osammotg1/projet3-hilserl-multicolor-v1` (private), commit
`b0e20e7f1fcd4a2d755f516e39117abbac49bade`. Created `v3.0` git tag on
the dataset (LeRobotDataset looks up that tag by codebase_version).

138 MB upload, ~5 sec at 31 MB/s.

Remote smoke test:

```python
ds = LeRobotDataset("osammotg1/projet3-hilserl-multicolor-v1")
# episodes: 101
# total frames: 43188
# obs.state shape: (12,)  names: 6 .pos + 6 .vel
# action shape: (4,)      names: ee_dx, ee_dy, ee_dz, gripper_delta
# task at idx 0: "Pick yellow block..."
# task at idx 5000: "Pick blue block..."
```

All acceptance criteria from the handoff met.
