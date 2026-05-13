#!/usr/bin/env bash
# launch_smolvla.sh — finetune lerobot/smolvla_base on a LeRobot v3 dataset.
#
# Designed for the ETHRC 5090 workstation (single GPU). Mirrors launch_act.sh:
# uv venv + accelerate launch with bf16, W&B online (key read from ~/.netrc),
# checkpoints under $OUTPUT_BASE.
#
# Camera-key remap is the load-bearing detail for SmolVLA finetunes. Our
# dataset has a single `observation.images.wrist` view; pretrained smolvla_base
# expects `observation.images.camera{1,2,3}`. We rename wrist -> camera1 at
# runtime so the pretrained visual encoder weights apply; camera2/camera3 are
# absent from the batch and skipped (empty_cameras=0).
#
# Usage:
#   bash train/launch_smolvla.sh                                       # defaults
#   STEPS=20000 BATCH_SIZE=32 bash train/launch_smolvla.sh             # quick run
#   IMAGE_TRANSFORMS_ENABLE=true IMAGE_TFS='{"brightness":...}' bash …  # aug variant
set -euo pipefail

LEROBOT_SCRIPTS_DIR="${LEROBOT_SCRIPTS_DIR:-/home/ethrc/Desktop/lerobot-edit-scripts}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/ethrc/Desktop/training/checkpoints/projet3}"
NETRC_PATH="${NETRC_PATH:-$HOME/.netrc}"

LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-$LEROBOT_SCRIPTS_DIR/.venv/bin/lerobot-train}"

DATASET_REPO_ID="${DATASET_REPO_ID:-osammotg1/projet3-eval2-v1-tom-hugo}"
JOB_NAME="${JOB_NAME:-projet3_smolvla_eval2_v1}"
RUN_NAME="${RUN_NAME:-${JOB_NAME}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/$RUN_NAME}"

POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
POLICY_REPO_ID="${POLICY_REPO_ID:-osammotg1/projet3-smolvla-eval2-v1}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"

# Our dataset only has the wrist view; smolvla_base was pretrained with
# camera{1,2,3}. Rename wrist -> camera1 at load time so the pretrained
# visual encoder is actually applied to our images. Override RENAME_MAP if
# you bring in a top camera in the future.
RENAME_MAP="${RENAME_MAP:-{\"observation.images.wrist\": \"observation.images.camera1\"}}"

BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-50000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"

WANDB_PROJECT="${WANDB_PROJECT:-projet3-smolvla}"
WANDB_ENTITY="${WANDB_ENTITY:-tom-gazzini-ethrc}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ -z "${WANDB_API_KEY:-}" && -r "$NETRC_PATH" ]]; then
  WANDB_API_KEY="$(awk '/api\.wandb\.ai/{f=1} f && /password/{print $2; exit}' "$NETRC_PATH")"
fi
export WANDB_API_KEY WANDB_MODE

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

if [[ ! -x "$LEROBOT_TRAIN_BIN" ]]; then
  printf '[launch_smolvla] ERROR: lerobot-train not found at %s\n' "$LEROBOT_TRAIN_BIN" >&2
  printf '[launch_smolvla] Set LEROBOT_SCRIPTS_DIR to a directory whose .venv has lerobot installed.\n' >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"
cd "$LEROBOT_SCRIPTS_DIR"

printf '[launch_smolvla] dataset=%s\n' "$DATASET_REPO_ID"
printf '[launch_smolvla] policy_path=%s -> hub=%s\n' "$POLICY_PATH" "$POLICY_REPO_ID"
printf '[launch_smolvla] rename_map=%s\n' "$RENAME_MAP"
printf '[launch_smolvla] steps=%s batch=%s\n' "$STEPS" "$BATCH_SIZE"
printf '[launch_smolvla] output=%s\n'  "$OUTPUT_DIR"
printf '[launch_smolvla] wandb entity=%s project=%s run=%s\n' "$WANDB_ENTITY" "$WANDB_PROJECT" "$RUN_NAME"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

# Image augmentations: disabled by default for the first apples-to-apples
# SmolVLA recipe. Pass IMAGE_TRANSFORMS_ENABLE=true and the same IMAGE_TFS /
# MAX_NUM_TRANSFORMS / RANDOM_ORDER as launch_act.sh (e.g. the dark_noise
# profile) to train a second variant for comparison with our ACT models.
IMAGE_TRANSFORMS_ENABLE="${IMAGE_TRANSFORMS_ENABLE:-false}"
if [[ -z "${IMAGE_TFS:-}" ]]; then
  IMAGE_TFS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":0.5,"type":"ColorJitter","kwargs":{"saturation":[0.7,1.3]}},"hue":{"weight":0.2,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}}}'
fi
MAX_NUM_TRANSFORMS="${MAX_NUM_TRANSFORMS:-2}"
RANDOM_ORDER="${RANDOM_ORDER:-false}"

uv run accelerate launch \
  --num_machines=1 \
  --num_processes=1 \
  --mixed_precision=bf16 \
  "$LEROBOT_TRAIN_BIN" \
  --policy.path="$POLICY_PATH" \
  --policy.device=cuda \
  --policy.push_to_hub="$PUSH_TO_HUB" \
  --policy.repo_id="$POLICY_REPO_ID" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.revision=main \
  --dataset.video_backend=torchcodec \
  --dataset.image_transforms.enable="$IMAGE_TRANSFORMS_ENABLE" \
  --dataset.image_transforms.max_num_transforms="$MAX_NUM_TRANSFORMS" \
  --dataset.image_transforms.random_order="$RANDOM_ORDER" \
  --dataset.image_transforms.tfs="$IMAGE_TFS" \
  --rename_map="$RENAME_MAP" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq="$LOG_FREQ" \
  --eval_freq=0 \
  --num_workers="$NUM_WORKERS" \
  --job_name="$JOB_NAME" \
  --wandb.enable=true \
  --wandb.project="$WANDB_PROJECT" \
  --wandb.entity="$WANDB_ENTITY"
