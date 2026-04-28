# launch_act.sh — quick reference

Wraps `lerobot-train` with sensible defaults for **ACT on a small SO-101 dataset**
(`Rsebti/projet3-demos-v1bis`: 19 episodes, 1 wrist cam, 7 DOF). Mirrors the
ETHRC `lerobot-edit-scripts` pattern: `uv run accelerate launch` with bf16 +
torch-inductor, W&B online (key read from `~/.netrc`), checkpoints under
`/home/ethrc/Desktop/training/checkpoints/projet3/<run-name>/`.

## Why these defaults (small-data regime)

- **Smaller ACT** — `dim_model=256, n_encoder_layers=2, n_decoder_layers=1,
  dim_feedforward=1024` (~17 M params). The default 512-wide ACT used for the
  bimanual YAMS data overfits 19 episodes hard.
- **Image augmentations on** — brightness/contrast/saturation/hue ColorJitter
  matching the `5080_workstation` config. Cheap, big win on small data.
- **Frozen-backbone-friendly LR split** — `optimizer_lr=3e-5`, backbone LR
  `1e-5` (don't fine-tune ResNet18 hard on 6 k frames).
- **VAE on** — acts as a regularizer (`use_vae=true, kl_weight=10`).
- **20 k steps × batch 32** — overfits comfortably on a 5090, ~30–45 min.
  Watch the W&B loss curve: stop earlier if it plateaus.

## Run

```bash
# default: 5090 ETHRC box, Rsebti/projet3-demos-v1bis, 20k steps
bash train/launch_act.sh

# inside a tmux session (recommended for long runs)
tmux new -s projet3_act
bash train/launch_act.sh
# Ctrl-b d to detach; tmux attach -t projet3_act to come back
```

## Override knobs

All env vars, all optional:

| variable                  | default                                                  | notes |
| ------------------------- | -------------------------------------------------------- | ----- |
| `DATASET_REPO_ID`         | `Rsebti/projet3-demos-v1bis`                             | any LeRobot v3 dataset on HF |
| `JOB_NAME`                | `projet3_act_v1bis`                                      | also seeds the run-name prefix |
| `RUN_NAME`                | `<JOB_NAME>_<timestamp>`                                 | full run name and output subdir |
| `BATCH_SIZE`              | `32`                                                     | 5090 has headroom for 48 if VRAM allows |
| `STEPS`                   | `20000`                                                  | bump to 30 k if loss still trending down |
| `SAVE_FREQ` / `LOG_FREQ`  | `2000` / `100`                                           | 10 checkpoints across the run |
| `NUM_WORKERS`             | `12`                                                     | dataloader workers |
| `OPTIMIZER_LR`            | `3e-5`                                                   | main LR |
| `OPTIMIZER_LR_BACKBONE`   | `1e-5`                                                   | ResNet18 LR (keep low) |
| `WANDB_PROJECT`           | `projet3-act`                                            | |
| `WANDB_ENTITY`            | `tom-gazzini-ethrc`                                      | personal entity |
| `WANDB_MODE`              | `online`                                                 | `offline` to defer sync |
| `LEROBOT_SCRIPTS_DIR`     | `/home/ethrc/Desktop/lerobot-edit-scripts`               | dir containing the uv-managed `.venv` with `lerobot-train` |
| `OUTPUT_BASE`             | `/home/ethrc/Desktop/training/checkpoints/projet3`       | parent dir for run output |
| `NETRC_PATH`              | `$HOME/.netrc`                                           | source of `WANDB_API_KEY` |
| `WANDB_API_KEY`           | (read from `NETRC_PATH`)                                 | set directly to skip netrc lookup |

## Output

```
$OUTPUT_BASE/<RUN_NAME>/
├── checkpoints/
│   ├── 002000/pretrained_model/
│   ├── 004000/pretrained_model/
│   └── last → 020000/pretrained_model/
└── wandb/
```

Use `last/pretrained_model` as `--policy.path` for deployment.

## Different host?

Set `LEROBOT_SCRIPTS_DIR` to any directory containing a uv-managed `.venv`
with `lerobot` installed (the `.venv/bin/lerobot-train` entry point must be
executable). Set `OUTPUT_BASE` to wherever you want checkpoints.
