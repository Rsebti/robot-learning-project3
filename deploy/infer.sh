#!/usr/bin/env bash
# infer.sh — run a trained ACT policy on the real SO-101 follower.
#
# Wraps `lerobot-record` in autonomous mode (no teleop) so the policy drives
# the arm and each rollout is logged as an episode for later review.
#
# Usage (env-var driven, all optional except FOLLOWER_PORT on the actual robot):
#
#   FOLLOWER_PORT=/dev/ttyACM0 \
#   POLICY_PATH=train/checkpoints/projet3_act_v1bis/last/pretrained_model \
#     bash deploy/infer.sh
#
#   # or pull from HF directly (no local copy needed):
#   FOLLOWER_PORT=/dev/ttyACM0 POLICY_PATH=Rsebti/projet3-act-sanity \
#     bash deploy/infer.sh
#
# Run on the laptop next to the robot. On Windows use Git Bash or translate
# the env-var lines to the PowerShell equivalent (see deploy/inference.md).
set -euo pipefail

POLICY_PATH="${POLICY_PATH:-train/checkpoints/projet3_act_v1bis/last/pretrained_model}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

FOLLOWER_PORT="${FOLLOWER_PORT:-}"
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"

CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"

EVAL_REPO_ID="${EVAL_REPO_ID:-Rsebti/projet3-eval-sanity}"
NUM_EPISODES="${NUM_EPISODES:-5}"
EPISODE_TIME_S="${EPISODE_TIME_S:-15}"
RESET_TIME_S="${RESET_TIME_S:-10}"
SINGLE_TASK="${SINGLE_TASK:-Pick block and place in bowl}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"

if [[ -z "$FOLLOWER_PORT" ]]; then
  printf '[infer] ERROR: FOLLOWER_PORT is required (e.g. /dev/ttyACM0 on Linux, COM5 on Windows)\n' >&2
  printf '[infer] Linux: ls /dev/tty*  |  Windows: Device Manager → Ports (COM & LPT)\n' >&2
  exit 1
fi

# If POLICY_PATH looks like a local path, validate it; otherwise assume HF repo id.
if [[ "$POLICY_PATH" == /* || "$POLICY_PATH" == ./* || "$POLICY_PATH" == ../* || -e "$POLICY_PATH" ]]; then
  if [[ ! -d "$POLICY_PATH" ]]; then
    printf '[infer] ERROR: POLICY_PATH=%s is not a directory.\n' "$POLICY_PATH" >&2
    printf '[infer] Expected a pretrained_model/ directory containing config.json + model.safetensors.\n' >&2
    exit 1
  fi
  for f in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
    if [[ ! -f "$POLICY_PATH/$f" ]]; then
      printf '[infer] ERROR: %s missing from %s\n' "$f" "$POLICY_PATH" >&2
      exit 1
    fi
  done
fi

CAMERAS_JSON=$(printf '{"wrist": {"type": "opencv", "index_or_path": %s, "width": %s, "height": %s, "fps": %s}}' \
  "$CAMERA_INDEX" "$CAMERA_WIDTH" "$CAMERA_HEIGHT" "$CAMERA_FPS")

printf '[infer] policy=%s\n'  "$POLICY_PATH"
printf '[infer] robot=%s on %s\n' "$ROBOT_TYPE" "$FOLLOWER_PORT"
printf '[infer] camera=opencv idx=%s %sx%s@%s\n' "$CAMERA_INDEX" "$CAMERA_WIDTH" "$CAMERA_HEIGHT" "$CAMERA_FPS"
printf '[infer] %d episodes × %ss (reset %ss)\n' "$NUM_EPISODES" "$EPISODE_TIME_S" "$RESET_TIME_S"

lerobot-record \
  --robot.type="$ROBOT_TYPE" \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.cameras="$CAMERAS_JSON" \
  --display_data="$DISPLAY_DATA" \
  --policy.path="$POLICY_PATH" \
  --policy.device="$POLICY_DEVICE" \
  --dataset.repo_id="$EVAL_REPO_ID" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.fps="$CAMERA_FPS" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  --dataset.single_task="$SINGLE_TASK" \
  --dataset.private=true \
  --dataset.push_to_hub="$PUSH_TO_HUB"
