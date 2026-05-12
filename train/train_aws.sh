#!/usr/bin/env bash
# train_aws.sh — train ACT on osammotg1/projet3-eval1-smoketest on a Kubernetes pod (NVIDIA L4).
#
# Usage on the pod:
#   bash train_aws.sh                   # use defaults: 40k steps, batch 16
#   STEPS=10 bash train_aws.sh          # smoke test
#   BATCH_SIZE=24 bash train_aws.sh     # bigger batch (if VRAM allows)
#   PUSH_TO_HUB=false bash train_aws.sh # skip final HF push
#
# Prereqs (one-time, on the pod):
#   1. uv venv + lerobot installed (see README at top of this file)
#   2. HF authenticated:  hf auth login  (paste token at prompt)
#
# This script is self-contained — no git clone or external repo access required.

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/eval1}"
DATASET_REPO_ID="${DATASET_REPO_ID:-osammotg1/projet3-eval1-smoketest}"
POLICY_REPO_ID="${POLICY_REPO_ID:-osammotg1/projet3-act-eval1-v1}"
JOB_NAME="${JOB_NAME:-projet3_act_eval1_v1}"
RUN_NAME="${RUN_NAME:-${JOB_NAME}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKDIR/checkpoints/$RUN_NAME}"

STEPS="${STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OPTIMIZER_LR="${OPTIMIZER_LR:-3e-5}"
OPTIMIZER_LR_BACKBONE="${OPTIMIZER_LR_BACKBONE:-1e-5}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"

WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-projet3-act}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

VENV_BIN="$WORKDIR/.venv/bin"
if [[ ! -x "$VENV_BIN/lerobot-train" ]]; then
  printf '[train_aws] ERROR: lerobot-train not found at %s\n' "$VENV_BIN/lerobot-train" >&2
  printf '[train_aws] Run setup first:\n' >&2
  printf '[train_aws]   mkdir -p %s && cd %s\n' "$WORKDIR" "$WORKDIR" >&2
  printf '[train_aws]   uv venv --python 3.12 --system-site-packages .venv\n' >&2
  printf '[train_aws]   source .venv/bin/activate\n' >&2
  printf '[train_aws]   uv pip install lerobot accelerate huggingface_hub wandb torchcodec\n' >&2
  printf '[train_aws]   hf auth login\n' >&2
  exit 1
fi

source "$WORKDIR/.venv/bin/activate"
mkdir -p "$(dirname "$OUTPUT_DIR")"

printf '[train_aws] dataset=%s\n' "$DATASET_REPO_ID"
printf '[train_aws] output=%s\n'  "$OUTPUT_DIR"
printf '[train_aws] steps=%s batch_size=%s\n' "$STEPS" "$BATCH_SIZE"
printf '[train_aws] push_to_hub=%s policy_repo=%s\n' "$PUSH_TO_HUB" "$POLICY_REPO_ID"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

IMAGE_TFS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":0.5,"type":"ColorJitter","kwargs":{"saturation":[0.7,1.3]}},"hue":{"weight":0.2,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}}}'

WANDB_ARGS=(--wandb.enable="$WANDB_ENABLE")
if [[ "$WANDB_ENABLE" == "true" ]]; then
  WANDB_ARGS+=(--wandb.project="$WANDB_PROJECT")
  if [[ -n "$WANDB_ENTITY" ]]; then
    WANDB_ARGS+=(--wandb.entity="$WANDB_ENTITY")
  fi
fi

accelerate launch \
  --num_machines=1 \
  --num_processes=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=inductor \
  "$VENV_BIN/lerobot-train" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.revision=main \
  --dataset.video_backend=torchcodec \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=2 \
  --dataset.image_transforms.random_order=false \
  --dataset.image_transforms.tfs="$IMAGE_TFS" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub="$PUSH_TO_HUB" \
  --policy.repo_id="$POLICY_REPO_ID" \
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
  "${WANDB_ARGS[@]}"
