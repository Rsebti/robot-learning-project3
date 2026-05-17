# ggand0/hil-serl-so101 — Mac setup (one-time, ~15 min)

The handoff verdict in `notes/ggand0_evaluation.md` chose to use ggand0's
stack for HIL-SERL v1. This walks through the one-time install on this
Mac. Output is a working `uv` venv at
`~/Desktop/eval-ggand0/hil-serl-so101/.venv` plus a verified `lerobot`
import from the fork.

Run these steps **outside** the project repo. None of the artifacts of
this install live inside `robot-learning-project3/`; only the configs
and assets we already vendored under `sim/hilserl/` do.

## 0. Prereqs (already done by Claude earlier in this session)

```bash
ls ~/Desktop/eval-ggand0/
# Expected:
#   hil-serl-so101/    (cloned from https://github.com/ggand0/hil-serl-so101)
#   lerobot/           (cloned from https://github.com/ggand0/lerobot, on feat/hil-serl)
#   pick-101/          (cloned from https://github.com/ggand0/pick-101, for the MuJoCo XML)
```

If any of these are missing:

```bash
cd ~/Desktop/eval-ggand0
git clone https://github.com/ggand0/hil-serl-so101
git clone https://github.com/ggand0/lerobot && cd lerobot && git checkout feat/hil-serl && cd ..
git clone https://github.com/ggand0/pick-101
cd pick-101 && git lfs install --local && git lfs pull && cd ..  # ← required for STL meshes
```

Then re-copy the MJCF meshes into our gitignored slot:

```bash
REPO="/Users/admin/Documents/ETH/M4/Robot Learning /Project S101/robot-learning-project3"
cp ~/Desktop/eval-ggand0/pick-101/models/so101/assets/*.stl \
   "$REPO/sim/hilserl/assets/so101_mjcf/assets/"
file "$REPO/sim/hilserl/assets/so101_mjcf/assets/moving_jaw_so101_v1.stl"
# Expected: "data" (binary STL). NOT "ASCII text" (LFS pointer).
```

## 1a. Patch the lerobot fork's gym_manipulator.py — hardcoded calibration path (DONE on this Mac, 2026-05-17)

ggand0's `feat/hil-serl` branch contains a literal-string hardcode of his
own home-dir calibration path at
`src/lerobot/scripts/rl/gym_manipulator.py:85`:

```python
_IK_CALIBRATION_PATH = "/home/gota/.cache/huggingface/lerobot/calibration/robots/so101_follower/ggando_so101_follower.json"
```

The function `_load_ik_calibration()` (lines 90-97) reads `range_min` /
`range_max` from this JSON to clamp IK-driven motor commands. If the
path doesn't exist, you get `FileNotFoundError` AFTER the IK reset
starts moving the arm — partial-init, very annoying.

Fix applied on this Mac:

```python
_IK_CALIBRATION_PATH = "/Users/admin/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json"
```

(Note: `so_follower/so101_follower.json`, not `so101_follower/...`.
The lerobot convention is directory by robot family, file by ID.)

Re-apply this manually after any `git pull` of the fork.

## 1b. Patch ggand0's pyproject.toml — kill the CUDA / ROCm pins (DONE on this Mac, 2026-05-17)

ggand0 pins torch to CUDA-130 wheels in `hil-serl-so101/pyproject.toml`,
AND the sibling lerobot fork pins to ROCm-6.4 wheels in
`lerobot/pyproject.toml`. Neither works on Mac arm64. We need PyPI's
default wheels (Mac MPS-capable).

We don't touch the lerobot fork's pyproject (it's external untrusted
code per the safety classifier). Instead, override at the root project
level — uv's `override-dependencies` neutralizes the sibling pin.

Current state of `~/Desktop/eval-ggand0/hil-serl-so101/pyproject.toml`
(backup at `.toml.bak`):

```toml
[tool.uv.sources]
lerobot = { path = "../lerobot", editable = true }

[tool.uv]
override-dependencies = [
    "opencv-python-headless>=4.8.0 ; sys_platform == 'never'",
    "torch>=2.4.0,<3",
    "torchvision>=0.19.0,<1",
    "torchaudio>=2.4.0,<3",
    "pytorch-triton-rocm ; sys_platform == 'never'",
]
```

The original `[tool.uv.sources]` torch/torchvision/torchaudio entries
and the `[[tool.uv.index]] name = "pytorch-cuda"` block were removed.
The `pytorch-triton-rocm` override neutralizes the lerobot fork's
ROCm-only triton dep on darwin.

If you need to redo this on a fresh clone of `hil-serl-so101`, apply
the same edit by hand.

## 2. Sync the venv (DONE on this Mac, 2026-05-17 — verified)

```bash
cd ~/Desktop/eval-ggand0/hil-serl-so101
uv sync
```

Verified on this Mac: 186 packages resolved, torch 2.12.0 installed
(PyPI default arm64 build with MPS), lerobot 0.3.2 installed from the
sibling editable fork. Sync took ~30 s on a warm pip cache. `.venv/`
lives at `~/Desktop/eval-ggand0/hil-serl-so101/.venv`.

If you need to redo this on a different machine:
- **`No solution found`** — `lerobot[hilserl]` extras may require pins
  we don't have. Try `uv sync --extra hilserl` explicitly.
- **`Couldn't find Python 3.10+`** — `uv python install 3.13` first
  (we resolved to CPython 3.13.12 on this Mac).
- **`Distribution torch==X+rocm` can't be installed on darwin** — the
  override-dependencies block in step 1 wasn't applied. Re-check the
  pyproject.

## 3. Smoke-test the install (DONE on this Mac, 2026-05-17 — all passed)

```bash
cd ~/Desktop/eval-ggand0/hil-serl-so101 && uv run python -c "
import lerobot, torch, mujoco, numpy as np
print('lerobot:', lerobot.__version__)
print('torch:', torch.__version__, 'MPS:', torch.backends.mps.is_available())
m = mujoco.MjModel.from_xml_path('/Users/admin/Documents/ETH/M4/Robot Learning /Project S101/robot-learning-project3/sim/hilserl/assets/so101_mjcf/so101_new_calib.xml')
print('MuJoCo:', m.njnt, 'joints,', m.nbody, 'bodies')
ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')
data = mujoco.MjData(m); mujoco.mj_forward(m, data)
print('FK at qpos=0:', data.site_xpos[ee].round(4), 'm')
from lerobot.robots.so101_follower.so101_follower_end_effector import SO101FollowerEndEffector
import lerobot.scripts.rl.gym_manipulator
print('All imports OK')
"
```

Recorded output on this Mac:
```
lerobot: 0.3.2
torch: 2.12.0 MPS: True
MuJoCo: 6 joints, 8 bodies
FK at qpos=0: [ 0.3914 -0.0002  0.2344] m
All imports OK
```

The FK position at all-zero joints (~0.39, 0.0, 0.23 m) is the
"arm-vertical-stretched-up" pose for the SO-101 — sanity-checked. If
the numbers differ wildly on your machine, the meshes weren't pulled
via LFS (step 0).

## 4. Verify access to our configs

```bash
ls "/Users/admin/Documents/ETH/M4/Robot Learning /Project S101/robot-learning-project3/sim/hilserl/configs/ggand0/"
# Expected:
#   README.md
#   yellow_v1_record.json
#   yellow_v1_train.json
```

Then dry-load the record config (no robot needed yet):

```bash
cd ~/Desktop/eval-ggand0/hil-serl-so101
uv run python -c "
from lerobot.scripts.rl.gym_manipulator import EnvConfig
import draccus
cfg_path = '/Users/admin/Documents/ETH/M4/Robot Learning /Project S101/robot-learning-project3/sim/hilserl/configs/ggand0/yellow_v1_record.json'
cfg = draccus.parse(EnvConfig, args=['--config_path', cfg_path])
print('record cfg loaded:', cfg.task, cfg.fps, 'Hz,', cfg.num_episodes, 'episodes')
"
```

If draccus complains about extra keys (like `_comment`), strip them from
the JSON or rename the field — that file is the one we wrote.

## 5. HF auth (should already be set up)

```bash
uv run huggingface-cli whoami
# Expected: osammotg1
```

## Known gotchas

- **lerobot 0.3.2 dataset format**: this fork's HF dataset loader uses
  the v2.x format (`data/chunk-000/episode_NNNNNN.parquet`). The 0.5.2
  format we used for SmolVLA datasets (`episodes/chunk-000/file-000.parquet`)
  may not load cleanly here. Our v1 path is to **record fresh** in the
  0.3.2 venv rather than convert.
- **MPS limitations**: SAC's `utd_ratio: 20` does 20 gradient steps per env step. On Mac MPS this may be slow for the learner. If actor lag spikes during training, run the learner on the 5090 over gRPC (see `actor_learner_config` in `yellow_v1_train.json`).
- **Don't activate the venv from the project repo's PowerShell-y shells.** Always `cd ~/Desktop/eval-ggand0/hil-serl-so101 && uv run ...`.

## What's next after this is done

→ `notes/hilserl_lab_checklist_ggand0.md` (Step 1: pre-flight at the robot).
