#!/usr/bin/env bash
# push_smolvla_checkpoint.sh — push a single SmolVLA training checkpoint to HF.
#
# RUNS ON THE 5090 BOX (where training writes /home/ethrc/Desktop/training/...).
# Lets you grab an in-flight intermediate checkpoint without waiting for the
# 50k-step finish line. Pushes pretrained_model/ to a step-tagged repo so it
# never collides with the final `osammotg1/projet3-smolvla-eval2-v1`.
#
# Usage:
#   bash deploy/push_smolvla_checkpoint.sh                     # push latest ("last/")
#   STEP=20000 bash deploy/push_smolvla_checkpoint.sh          # push step 20k
#   RUN_DIR=/path/to/run STEP=30000 bash deploy/push_smolvla_checkpoint.sh
#
# The repo name is derived from STEP: e.g. step=20000 -> "...-v1-step20k".
# Override REPO_ID to set it explicitly.
set -euo pipefail

RUN_DIR="${RUN_DIR:-/home/ethrc/Desktop/training/checkpoints/projet3/projet3_smolvla_eval2_v1_20260513_190321}"
STEP="${STEP:-last}"

HF_BIN="${HF_BIN:-/home/ethrc/Desktop/lerobot-edit-scripts/.venv/bin/hf}"

if [[ ! -x "$HF_BIN" ]]; then
  printf '[push_smolvla] ERROR: hf CLI not found at %s\n' "$HF_BIN" >&2
  exit 1
fi

# Resolve checkpoint dir: STEP=last uses the `last` symlink; otherwise zero-pad
# the step to 6 digits (lerobot convention: "020000").
if [[ "$STEP" == "last" ]]; then
  CKPT_DIR="$RUN_DIR/checkpoints/last/pretrained_model"
  # Resolve to the real step for a meaningful repo name.
  REAL=$(basename "$(readlink -f "$RUN_DIR/checkpoints/last")")
  STEP_NUM=$((10#$REAL))
else
  STEP_NUM=$((10#$STEP))
  PADDED=$(printf '%06d' "$STEP_NUM")
  CKPT_DIR="$RUN_DIR/checkpoints/$PADDED/pretrained_model"
fi

if [[ ! -d "$CKPT_DIR" ]]; then
  printf '[push_smolvla] ERROR: checkpoint dir not found: %s\n' "$CKPT_DIR" >&2
  printf '[push_smolvla] Available checkpoints under %s/checkpoints/:\n' "$RUN_DIR" >&2
  ls -1 "$RUN_DIR/checkpoints" 2>&1 | sed 's/^/  /' >&2
  exit 1
fi

for f in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
  if [[ ! -f "$CKPT_DIR/$f" ]]; then
    printf '[push_smolvla] ERROR: %s missing in %s (checkpoint may still be writing)\n' "$f" "$CKPT_DIR" >&2
    exit 1
  fi
done

# Repo name: step20k for 20000, step47k500 would round; keep it simple — push to
# the nearest thousand only (we never save at non-thousand boundaries anyway).
STEP_K=$(( STEP_NUM / 1000 ))
REPO_ID="${REPO_ID:-osammotg1/projet3-smolvla-eval2-v1-step${STEP_K}k}"

printf '[push_smolvla] source=%s\n' "$CKPT_DIR"
printf '[push_smolvla] target=%s\n' "$REPO_ID"
du -sh "$CKPT_DIR" | awk '{print "[push_smolvla] size="$1}'

# `hf upload` creates the repo if absent (with --create-pr=false by default).
# Private by default since the rest of the project uses private repos.
"$HF_BIN" upload --repo-type model --private "$REPO_ID" "$CKPT_DIR" .

printf '[push_smolvla] done — pull on the robot PC with:\n'
printf '  POLICY_PATH=%s bash deploy/infer_smolvla.sh\n' "$REPO_ID"
