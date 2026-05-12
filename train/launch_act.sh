#!/usr/bin/env bash
# launch_act.sh — train ACT on a LeRobot v3 dataset (Rsebti/projet3-demos-v1bis by default).
#
# Designed for the ETHRC 5090 workstation (single GPU). Mirrors the
# lerobot-edit-scripts pattern: uv venv + accelerate launch with bf16 + inductor,
# W&B online (key read from ~/.netrc), checkpoints under $OUTPUT_BASE.
#
# Sized for the small-data regime: 19 episodes, 1 wrist cam, 7-DOF SO-101.
# Smaller ACT (dim_model=256), image augmentations on, lr=3e-5, 20k steps.
#
# Usage:
#   bash train/launch_act.sh                    # use defaults below
#   DATASET_REPO_ID=Rsebti/projet3-demos-v2 bash train/launch_act.sh
#   STEPS=30000 BATCH_SIZE=48 bash train/launch_act.sh
#
# Host-specific overrides (defaults match the ETHRC 5090 box):
#   LEROBOT_SCRIPTS_DIR  directory containing the uv-managed .venv with lerobot installed
#   OUTPUT_BASE          where to write checkpoints (a timestamped subdir is created)
#   NETRC_PATH           ~/.netrc with an api.wandb.ai entry; set WANDB_API_KEY directly to skip
set -euo pipefail

LEROBOT_SCRIPTS_DIR="${LEROBOT_SCRIPTS_DIR:-/home/ethrc/Desktop/lerobot-edit-scripts}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/ethrc/Desktop/training/checkpoints/projet3}"
NETRC_PATH="${NETRC_PATH:-$HOME/.netrc}"

LEROBOT_TRAIN_BIN="$LEROBOT_SCRIPTS_DIR/.venv/bin/lerobot-train"

DATASET_REPO_ID="${DATASET_REPO_ID:-hudela390/projet3-eval1-bowl1-v1-trimmed}"
JOB_NAME="${JOB_NAME:-projet3_act_eval1_v1}"
RUN_NAME="${RUN_NAME:-${JOB_NAME}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/$RUN_NAME}"

BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-30000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-12}"
OPTIMIZER_LR="${OPTIMIZER_LR:-3e-5}"
OPTIMIZER_LR_BACKBONE="${OPTIMIZER_LR_BACKBONE:-1e-5}"

WANDB_PROJECT="${WANDB_PROJECT:-projet3-act}"
WANDB_ENTITY="${WANDB_ENTITY:-tom-gazzini-ethrc}"
WANDB_MODE="${WANDB_MODE:-online}"

# Pull W&B key from ~/.netrc unless caller already set WANDB_API_KEY.
if [[ -z "${WANDB_API_KEY:-}" && -r "$NETRC_PATH" ]]; then
  WANDB_API_KEY="$(awk '/api\.wandb\.ai/{f=1} f && /password/{print $2; exit}' "$NETRC_PATH")"
fi
export WANDB_API_KEY WANDB_MODE

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS="${TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS:-1}"

if [[ ! -x "$LEROBOT_TRAIN_BIN" ]]; then
  printf '[launch_act] ERROR: lerobot-train not found at %s\n' "$LEROBOT_TRAIN_BIN" >&2
  printf '[launch_act] Set LEROBOT_SCRIPTS_DIR to a directory whose .venv has lerobot installed.\n' >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"
cd "$LEROBOT_SCRIPTS_DIR"

printf '[launch_act] dataset=%s\n' "$DATASET_REPO_ID"
printf '[launch_act] output=%s\n'  "$OUTPUT_DIR"
printf '[launch_act] wandb entity=%s project=%s run=%s\n' "$WANDB_ENTITY" "$WANDB_PROJECT" "$RUN_NAME"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

# Image augmentations (matches the ETHRC 5080_workstation default style).
IMAGE_TFS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":0.5,"type":"ColorJitter","kwargs":{"saturation":[0.7,1.3]}},"hue":{"weight":0.2,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}}}'

uv run accelerate launch \
  --num_machines=1 \
  --num_processes=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=inductor \
  "$LEROBOT_TRAIN_BIN" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.revision=main \
  --dataset.video_backend=torchcodec \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=2 \
  --dataset.image_transforms.random_order=false \
  --dataset.image_transforms.tfs="$IMAGE_TFS" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub="${PUSH_TO_HUB:-true}" \
  --policy.repo_id="${POLICY_REPO_ID:-hudela390/projet3-act-eval1-v1}" \
  --policy.dim_model=256 \
  --policy.n_heads=8 \
  --policy.dim_feedforward=1024 \
  --policy.n_encoder_layers=2 \
  --policy.n_decoder_layers=1 \
  --policy.use_vae=true \
  --policy.kl_weight=10.0 \
  --policy.dropout=0.1 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.optimizer_lr="$OPTIMIZER_LR" \
  --policy.optimizer_lr_backbone="$OPTIMIZER_LR_BACKBONE" \
  --policy.optimizer_weight_decay=1e-4 \
  --output_dir="$OUTPUT_DIR" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq="$LOG_FREQ" \
  --eval_freq=0 \
  --num_workers="$NUM_WORKERS" \
  --optimizer.lr="$OPTIMIZER_LR" \
  --job_name="$JOB_NAME" \
  --wandb.enable=true \
  --wandb.project="$WANDB_PROJECT" \
  --wandb.entity="$WANDB_ENTITY"
