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
```

## 1. Patch ggand0's pyproject.toml — drop the cu130 torch pin

ggand0's `hil-serl-so101/pyproject.toml` pins torch to cu130 wheels via
`[tool.uv.sources]` + a `pytorch-cuda` index. CUDA wheels don't run on
Mac arm64. We need PyPI's default wheels (which give Mac MPS-capable
torch).

```bash
cp ~/Desktop/eval-ggand0/hil-serl-so101/pyproject.toml \
   ~/Desktop/eval-ggand0/hil-serl-so101/pyproject.toml.bak
```

Then edit `~/Desktop/eval-ggand0/hil-serl-so101/pyproject.toml`:

1. **Delete** the existing `[tool.uv.sources]` block (lines ~42-52) and
   the `[[tool.uv.index]]` block at the bottom.
2. **Replace** with just the lerobot path-editable source:
   ```toml
   [tool.uv.sources]
   lerobot = { path = "../lerobot", editable = true }
   ```

(This keeps the lerobot fork as a sibling-editable install while letting
torch resolve from PyPI defaults.)

## 2. Sync the venv

```bash
cd ~/Desktop/eval-ggand0/hil-serl-so101
uv sync
```

Expected: ~1.5 GB download (torch, torchvision, opencv, mujoco, placo,
timm, scipy, etc.), ~5–10 min depending on network. Creates `.venv/`
inside `hil-serl-so101/`.

If it fails:
- **`No solution found`** — ggand0's `lerobot[hilserl]` extras may require
  pins we don't have. Try `uv sync --extra hilserl` explicitly.
- **`Couldn't find Python 3.10+`** — `uv python install 3.10` first.
- **mujoco wheel mismatch** on arm64 — should not happen, but if it does,
  `uv pip install mujoco==3.x.x` separately into the venv.

## 3. Smoke-test the install

```bash
cd ~/Desktop/eval-ggand0/hil-serl-so101

# Lerobot loads, from the fork (should say 0.3.2, NOT 0.5.2)
uv run python -c "import lerobot; print('lerobot:', lerobot.__version__)"
# Expected: lerobot: 0.3.2 (or 0.3.2.dev — anything starting with 0.3)

# The custom robot class loads
uv run python -c "from lerobot.robots.so101_follower.so101_follower_end_effector import SO101FollowerEndEffector; print('so101_follower_end_effector OK')"

# MuJoCo can read our vendored model
uv run python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('/Users/admin/Documents/ETH/M4/Robot Learning /Project S101/robot-learning-project3/sim/hilserl/assets/so101_mjcf/so101_new_calib.xml')
print('MuJoCo OK,', m.njnt, 'joints')
"

# Gym manipulator entrypoint exists
uv run python -c "import lerobot.scripts.rl.gym_manipulator; print('gym_manipulator OK')"

# Torch can target MPS
uv run python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

If all four print, the venv is ready.

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
