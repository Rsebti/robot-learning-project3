#!/usr/bin/env bash
# download_eval1_checkpoints.sh — pull selected eval1 ACT checkpoints
# from the training workstation to this Mac for local-robot inference.
#
# Source: ssh host robot_learning_teleop_eduroam_tommaso
#         /home/tommaso/training/checkpoints/projet3/projet3_act_eval1_v1_20260506_190843/checkpoints/<step>/pretrained_model
#
# Destination: train/checkpoints/projet3_act_eval1_v1/<step>/pretrained_model
#
# Steps default to "every ~5k from the 2k-spaced saves":
#   006000 010000 016000 020000 026000 030000 036000 040000
# Override by exporting STEPS as a space-separated list:
#   STEPS="010000 020000 030000 040000" bash tools/download_eval1_checkpoints.sh
set -euo pipefail

SSH_HOST="${SSH_HOST:-robot_learning_teleop_eduroam_tommaso}"
REMOTE_RUN="${REMOTE_RUN:-/home/tommaso/training/checkpoints/projet3/projet3_act_eval1_v1_20260506_190843}"
LOCAL_BASE="${LOCAL_BASE:-train/checkpoints/projet3_act_eval1_v1}"
STEPS="${STEPS:-006000 010000 016000 020000 026000 030000 036000 040000}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$LOCAL_BASE"

echo "[download] host=$SSH_HOST"
echo "[download] remote_run=$REMOTE_RUN"
echo "[download] local_base=$REPO_ROOT/$LOCAL_BASE"
echo "[download] steps: $STEPS"
echo

for step in $STEPS; do
  remote_dir="$REMOTE_RUN/checkpoints/$step/pretrained_model/"
  local_dir="$LOCAL_BASE/$step/pretrained_model/"
  mkdir -p "$local_dir"
  printf '[download] step=%s -> %s\n' "$step" "$local_dir"
  rsync -avh --progress -e ssh "$SSH_HOST:$remote_dir" "$local_dir"
done

echo
echo "[download] done. Local layout:"
find "$LOCAL_BASE" -mindepth 2 -maxdepth 2 -type d | sort
