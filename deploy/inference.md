# Inference — running a trained ACT policy on the SO-101

This document is the **operational reference** for running a trained policy on
the real robot. If you are an AI assistant (Claude Code) on the laptop next to
the SO-101, read this first.

## TL;DR

```bash
FOLLOWER_PORT=/dev/ttyACM0 \
POLICY_PATH=Rsebti/projet3-act-sanity \
  bash deploy/infer.sh
```

The model lives on Hugging Face Hub. `lerobot-record` pulls it directly — no
local copy step. PowerShell equivalent at the bottom of this file.

**First time on a new machine?** You need an HF token with read access to
`Rsebti/projet3-act-sanity` (the repo is private):

```bash
hf auth login   # paste a read-scoped token, or a write token if you'll also push
```

Or set `HF_TOKEN=hf_xxx` in the environment before running `infer.sh`.

## What gets called

Two files in this repo do the work:

| file                         | purpose                                                                 |
| ---------------------------- | ----------------------------------------------------------------------- |
| `deploy/infer.sh`            | bash launcher that calls `lerobot-record` with the right flags          |
| `train/launch_act.sh`        | training launcher (only used to **produce** the checkpoint elsewhere)   |

`infer.sh` calls `lerobot-record` in **autonomous mode** (no `--teleop.*`
flags). The policy drives the arm and each rollout is recorded as an episode
under the dataset repo id given by `EVAL_REPO_ID` (locally, unless you set
`PUSH_TO_HUB=true`).

## What the policy needs

A directory called `pretrained_model/` containing exactly these 7 files:

```
pretrained_model/
├── config.json                                                    ← policy hyperparams
├── model.safetensors                                              ← weights (~66 MB)
├── policy_preprocessor.json
├── policy_preprocessor_step_3_normalizer_processor.safetensors    ← input normalizer
├── policy_postprocessor.json
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors ← output unnormalizer
└── train_config.json                                              ← reference (full training cfg)
```

Pass the **directory** (`pretrained_model/`) to `--policy.path`, **not** any
single file inside it. `lerobot-record` will load all 7 from that dir.

## Where the checkpoint lives

After training on the 5090 box, the run produces:

```
/home/ethrc/Desktop/training/checkpoints/projet3/projet3_act_v1bis_<timestamp>/
└── checkpoints/
    ├── 002000/pretrained_model/    ← intermediate
    ├── 004000/pretrained_model/
    ├── ...
    └── last → 020000               ← symlink to most recent
```

The convention for **this repo** is to copy the chosen checkpoint into:

```
train/checkpoints/projet3_act_v1bis/last/pretrained_model/
```

…and commit it. Everything in `train/checkpoints/**` except `last/pretrained_model/`
should be `.gitignore`d to keep repo size manageable (see Maintenance below).

### Three ways to get the checkpoint to the laptop

1. **Hugging Face Hub (default — already set up)**

   The current sanity checkpoint already lives at
   `Rsebti/projet3-act-sanity` (private repo, ~68 MB, 7 files).
   On the laptop, after `hf auth login`:

   ```bash
   POLICY_PATH=Rsebti/projet3-act-sanity bash deploy/infer.sh
   ```

   No git pull, no local copy. `lerobot-record` resolves the repo id and
   downloads on first use; subsequent runs hit the local cache at
   `~/.cache/huggingface/hub/`.

   **To upload a new checkpoint** (from any machine where `hf auth login` was
   done with a write-scoped Rsebti token):

   ```bash
   hf upload Rsebti/projet3-act-sanity \
     <path-to-run>/checkpoints/last/pretrained_model . \
     --repo-type model \
     --commit-message "ACT step 20000"
   ```

   Or just rerun `train/launch_act.sh` with `PUSH_TO_HUB=true` (default).

2. **scp from training box to laptop** — when you can't reach HF or don't
   want the weights in the cache:

   ```bash
   # on the laptop, into a path of your choice
   scp -r ethrc@<5090-host>:/home/ethrc/Desktop/training/checkpoints/projet3/projet3_act_v1bis_<TS>/checkpoints/last/pretrained_model \
        ./local-checkpoint
   POLICY_PATH=./local-checkpoint bash deploy/infer.sh
   ```

3. **Git commit (sanity-only fallback)** — works because `model.safetensors`
   is 66 MB, under GitHub's 100 MB per-file cap. **Avoid for repeated runs**:
   committing every checkpoint will balloon repo size.

   ```bash
   cd /home/tommaso/RobotLearningClassProject/robot-learning-project3
   mkdir -p train/checkpoints/projet3_act_v1bis/last
   cp -r /home/ethrc/Desktop/training/checkpoints/projet3/projet3_act_v1bis_<TS>/checkpoints/last/pretrained_model \
         train/checkpoints/projet3_act_v1bis/last/
   git add train/checkpoints/projet3_act_v1bis/last/pretrained_model
   git commit -m "Add ACT sanity-check checkpoint"
   git push
   ```

## Running inference (full reference)

### Required env var

| variable         | example          | notes                                                  |
| ---------------- | ---------------- | ------------------------------------------------------ |
| `FOLLOWER_PORT`  | `/dev/ttyACM0`   | Linux: `ls /dev/tty*` ; Windows: Device Manager → COM  |

### Common overrides

| variable          | default                                                          |
| ----------------- | ---------------------------------------------------------------- |
| `POLICY_PATH`     | `Rsebti/projet3-act-sanity` (HF repo id) — or any local pretrained_model dir |
| `POLICY_DEVICE`   | `cuda` (set `cpu` if no GPU on the laptop)                       |
| `CAMERA_INDEX`    | `0`                                                              |
| `CAMERA_WIDTH/HEIGHT/FPS` | `640 / 480 / 30`                                         |
| `NUM_EPISODES`    | `5`                                                              |
| `EPISODE_TIME_S`  | `15`                                                             |
| `RESET_TIME_S`    | `10`                                                             |
| `SINGLE_TASK`     | `Pick block and place in bowl`                                   |
| `EVAL_REPO_ID`    | `Rsebti/projet3-eval-sanity`                                     |
| `PUSH_TO_HUB`     | `false`                                                          |
| `DISPLAY_DATA`    | `true` (opens an OpenCV window with the wrist cam + state)       |

### Examples

```bash
# Default sanity-check run (Linux laptop)
FOLLOWER_PORT=/dev/ttyACM0 bash deploy/infer.sh

# CPU fallback
FOLLOWER_PORT=/dev/ttyACM0 POLICY_DEVICE=cpu bash deploy/infer.sh

# Pull policy from HF instead of repo
FOLLOWER_PORT=/dev/ttyACM0 POLICY_PATH=Rsebti/projet3-act-sanity bash deploy/infer.sh

# 10 longer episodes
FOLLOWER_PORT=/dev/ttyACM0 NUM_EPISODES=10 EPISODE_TIME_S=20 bash deploy/infer.sh
```

### Windows / PowerShell

```powershell
$env:FOLLOWER_PORT = "COM5"
$env:POLICY_PATH = "train\checkpoints\projet3_act_v1bis\last\pretrained_model"

lerobot-record `
  --robot.type=so101_follower `
  --robot.port=$env:FOLLOWER_PORT `
  --robot.id=so101_follower `
  --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' `
  --display_data=true `
  --policy.path=$env:POLICY_PATH `
  --policy.device=cuda `
  --dataset.repo_id=Rsebti/projet3-eval-sanity `
  --dataset.num_episodes=5 `
  --dataset.fps=30 `
  --dataset.episode_time_s=15 `
  --dataset.reset_time_s=10 `
  --dataset.single_task="Pick block and place in bowl" `
  --dataset.push_to_hub=false
```

(Or run `infer.sh` directly under Git Bash on Windows — it works.)

## Pre-flight checklist

See `deploy/deploy_policy.md` for the physical-setup checklist (block + bowl
on tape, lighting, e-stop reachable). Re-use that checklist verbatim before
every rollout — the policy was trained on quasi-identical demos and is
sensitive to scene drift.

## Pass criteria (sanity)

- Robot moves smoothly to the block (no jitter, no servo-overcurrent error).
- Closes gripper on the block.
- Lifts and places into the bowl.
- **3 / 5 episodes succeed** → pipeline validated → move to Eval 1.

## If it fails

| symptom                                  | likely cause                                                        |
| ---------------------------------------- | ------------------------------------------------------------------- |
| Policy outputs constant action / freezes | Undertrained — re-run `train/launch_act.sh` with `STEPS=30000`      |
| Robot drifts off-target immediately      | Camera index wrong, or block/bowl moved off the tape marks          |
| Servo over-current shutoff               | Demos were too jerky; re-record more smoothly or lower `temporal_ensemble_coeff` |
| `FileNotFoundError` on a `.safetensors`  | `POLICY_PATH` points one level too high — pass `pretrained_model/`  |
| `RuntimeError: CUDA error` on a 5070/5080 (sm_120) | PyTorch wheel doesn't support Blackwell yet — set `POLICY_DEVICE=cpu` for the smoke test |

## Maintenance — keeping the repo small

Recommended `.gitignore` additions when checkpoints live under `train/checkpoints/`:

```
# Track only the "last" checkpoint; ignore intermediate ones and per-run dirs
train/checkpoints/**
!train/checkpoints/projet3_act_v1bis/
!train/checkpoints/projet3_act_v1bis/last/
!train/checkpoints/projet3_act_v1bis/last/pretrained_model/
!train/checkpoints/projet3_act_v1bis/last/pretrained_model/**
```

Files to **never** commit: `wandb/`, intermediate `checkpoints/0xxxxx/`, and
the optimizer state inside training-only checkpoints (`training_state.safetensors`,
`optimizer_state.safetensors`, `scheduler_state.safetensors`, `rng_state.safetensors`)
— `pretrained_model/` excludes these by design.
