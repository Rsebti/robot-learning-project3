# Eval 2 — Perception module

Maps a wrist-camera RGB image to the (x, y, z) positions of the red and
blue colored blocks in the robot base frame, in meters.

This is the visual front-end used at deploy time on the real SO-101: the
RL policy stays state-based (24-D obs vector), but the block xyz it
receives come from this CNN instead of from the simulator's PhysX
ground-truth.

## Files

| File | Purpose |
|---|---|
| `model.py`            | `ColorBlockCNN` — small CNN (~120 k params), regresses 6 floats (xyz_red + xyz_blue, in meters, robot frame). |
| `dataset.py`          | `BlockPositionDataset` — wraps the captured `.pt` file as a PyTorch `Dataset`. |
| `capture_dataset.py`  | One-off Isaac Sim script. Renders N images of the v2 env from a fixed scan pose with randomized block / bowl positions, writes `data.pt`. |
| `train.py`            | Training loop. MSE on 6 outputs, Adam, train/val 80/20 split, early stop. Reports per-coordinate MAE in cm. |
| `inference.py`        | `PerceptionPipeline` — loads a trained checkpoint, takes an image (np or torch), returns `(red_xyz, blue_xyz)`. |

## Approach (B): direct 3D regression

The CNN outputs world-frame **xyz directly** (not pixel coordinates).
Pros:
- No table-plane calibration at deploy time.
- Robust to small table-height variations.
- The network learns depth from the apparent block size in the image
  (blocks have fixed 2×2×2 cm physical dimensions).

The single restriction: the camera must be in a similar pose as the
one used to capture training data. We hold the robot in a fixed
"scan" pose during data capture, and at deploy time the policy will
issue actions starting from that same scan pose, so the camera
viewpoint stays in distribution.

## Workflow

```
+----------------------+        +-----------------+        +----------------+
| capture_dataset.py   |  --->  |   data.pt       |  --->  |   train.py     |
|  (Isaac Sim 5k img)  |        | (5000 samples)  |        |  (~5 min on    |
+----------------------+        +-----------------+        |   RTX 5070)    |
                                                           +-------+--------+
                                                                   |
                                                                   v
                                                           +----------------+
                                                           |  checkpoint.pt |
                                                           +-------+--------+
                                                                   |
                                                                   v
                                                       +-----------+-----------+
                                                       |  inference.py         |
                                                       |  PerceptionPipeline   |
                                                       |  image -> red_xyz,    |
                                                       |           blue_xyz    |
                                                       +-----------+-----------+
                                                                   |
                                                                   v
                                                       +-----------+-----------+
                                                       | deploy script feeds   |
                                                       | red_xyz, blue_xyz to  |
                                                       | the goal-cond policy  |
                                                       +-----------------------+
```

## How to run

### 1. Capture training data (~10 min, requires Isaac Sim)

```powershell
cd C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101
uv run python -m sim.eval2.perception.capture_dataset `
    --num_samples 5000 `
    --enable_cameras
```

Output: `sim/eval2/perception/data.pt` (~ 100 MB).

### 2. Train the CNN (~5 min on RTX 5070, no Isaac Sim needed)

```powershell
uv run python -m sim.eval2.perception.train `
    --epochs 30
```

Watch the `val_mae` line — target is **< 1 cm** mean per-coordinate
error.

### 3. Smoke test inference

```powershell
uv run python -m sim.eval2.perception.inference
```

(Runs the checkpoint on random noise — just checks the pipeline plumbs
through. Real validation against sim ground-truth comes next.)

### 4. (TODO) Validate against sim ground-truth

A `validate.py` script (not yet written) will:
- Load the v2 env + the trained CNN.
- For each of N reset configurations, capture the wrist image, run the
  CNN, and compare the predicted xyz to the simulator's ground-truth
  block positions.
- Report mean / max error in cm.

This is the final sim-side acceptance test before we trust the
perception on the real robot.

## Known limitations / next steps

- **No domain randomization yet.** The training images are all rendered
  with the env's default lighting and the default Isaac Sim materials.
  When we deploy on the real robot, lighting and color mismatches will
  hurt accuracy. To fix this, add lighting/texture/HSV randomization to
  the env (planned for v3).
- **Fixed scan pose.** The CNN sees the camera from a single robot pose.
  Once the policy starts moving the arm, the camera will drift and the
  CNN may produce noisier outputs. Two options for v2.x:
  1. Capture data across a small range of poses around the scan pose
     (covers small policy-induced viewpoint drift).
  2. Run the perception only at episode start (when the robot is in
     the scan pose) and feed the resulting xyz as a constant goal to
     the policy.
- **Calibration vs real robot.** When we get camera-real-robot access,
  the camera intrinsics + extrinsics in sim (`CameraCfg.OffsetCfg`)
  must be re-tuned to match the physical mount. Tools: ChArUco against
  the robot, or a quick visual-eyeball pass against the real wrist cam.
