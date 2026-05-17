#!/usr/bin/env bash
# infer_eval1.sh — run a trained Eval 1 ACT policy on the real SO-101 follower.
#
# Eval 1 = single block + bowl, randomized positions, no target color, no
# goal-conditioning. Defaults below point at the dark-shadow v1 policy
# (osammotg1/projet3-act-eval1-v1-dark-shadow). Mac follower/leader ports
# match the lab Mac.
#
# Most common usage on the lab Mac:
#   bash deploy/infer_eval1.sh
#
# Swap the policy variant without editing the script:
#   POLICY_PATH=osammotg1/projet3-act-eval1-v2-... bash deploy/infer_eval1.sh
set -euo pipefail

POLICY_PATH="${POLICY_PATH:-osammotg1/projet3-act-eval1-v1-dark-shadow}"
POLICY_DEVICE="${POLICY_DEVICE:-mps}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/tty.usbmodem5B141129871}"
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"

# Leader is plugged in only to back-drive the follower during reset between
# episodes (without it, the follower's torque stays on and you can't move it
# back to home pose by hand). Set LEADER_PORT="" to disable.
LEADER_PORT="${LEADER_PORT:-/dev/tty.usbmodem5B141128171}"
LEADER_TYPE="${LEADER_TYPE:-so101_leader}"
LEADER_ID="${LEADER_ID:-so101_leader}"

CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"

NUM_EPISODES="${NUM_EPISODES:-5}"
EPISODE_TIME_S="${EPISODE_TIME_S:-20}"
RESET_TIME_S="${RESET_TIME_S:-10}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"

SINGLE_TASK="${SINGLE_TASK:-Pick block and place in bowl}"

# lerobot enforces the `eval_` prefix on dataset repo_id when --policy.path
# is set. Derive a default from the policy variant so concurrent runs of
# different variants don't collide.
POLICY_SLUG="$(basename "$POLICY_PATH")"
EVAL_REPO_ID="${EVAL_REPO_ID:-osammotg1/eval_${POLICY_SLUG}}"

if [[ -z "$FOLLOWER_PORT" ]]; then
  printf '[infer-eval1] ERROR: FOLLOWER_PORT is required (e.g. /dev/tty.usbmodem... on macOS)\n' >&2
  exit 1
fi

# Local-path validation (only when POLICY_PATH looks like a path on disk).
if [[ "$POLICY_PATH" == /* || "$POLICY_PATH" == ./* || "$POLICY_PATH" == ../* || -e "$POLICY_PATH" ]]; then
  if [[ ! -d "$POLICY_PATH" ]]; then
    printf '[infer-eval1] ERROR: POLICY_PATH=%s is not a directory.\n' "$POLICY_PATH" >&2
    exit 1
  fi
  for f in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
    if [[ ! -f "$POLICY_PATH/$f" ]]; then
      printf '[infer-eval1] ERROR: %s missing from %s\n' "$f" "$POLICY_PATH" >&2
      exit 1
    fi
  done
fi

# lerobot-record refuses to start if its local cache dir for this repo_id
# already exists (FileExistsError from an aborted previous run).
LEROBOT_CACHE_DIR="${HOME}/.cache/huggingface/lerobot/${EVAL_REPO_ID}"
if [[ -d "$LEROBOT_CACHE_DIR" ]]; then
  printf '[infer-eval1] removing stale lerobot cache dir: %s\n' "$LEROBOT_CACHE_DIR"
  rm -rf "$LEROBOT_CACHE_DIR"
fi

CAMERAS_JSON=$(printf '{"wrist": {"type": "opencv", "index_or_path": %s, "width": %s, "height": %s, "fps": %s}}' \
  "$CAMERA_INDEX" "$CAMERA_WIDTH" "$CAMERA_HEIGHT" "$CAMERA_FPS")

TELEOP_FLAGS=()
if [[ -n "$LEADER_PORT" ]]; then
  TELEOP_FLAGS=(
    --teleop.type="$LEADER_TYPE"
    --teleop.port="$LEADER_PORT"
    --teleop.id="$LEADER_ID"
  )
fi

printf '[infer-eval1] policy=%s\n'  "$POLICY_PATH"
printf '[infer-eval1] robot=%s on %s\n' "$ROBOT_TYPE" "$FOLLOWER_PORT"
if [[ -n "$LEADER_PORT" ]]; then
  printf '[infer-eval1] leader=%s on %s (homing during reset)\n' "$LEADER_TYPE" "$LEADER_PORT"
else
  printf '[infer-eval1] leader=<none> (follower will freeze in place during reset)\n'
fi
printf '[infer-eval1] camera=opencv idx=%s %sx%s@%s\n' "$CAMERA_INDEX" "$CAMERA_WIDTH" "$CAMERA_HEIGHT" "$CAMERA_FPS"
printf '[infer-eval1] %d episodes × %ss (reset %ss)\n' "$NUM_EPISODES" "$EPISODE_TIME_S" "$RESET_TIME_S"
printf '[infer-eval1] eval_repo_id=%s (push_to_hub=%s)\n' "$EVAL_REPO_ID" "$PUSH_TO_HUB"

lerobot-record \
  --robot.type="$ROBOT_TYPE" \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.cameras="$CAMERAS_JSON" \
  --display_data="$DISPLAY_DATA" \
  --policy.path="$POLICY_PATH" \
  --policy.device="$POLICY_DEVICE" \
  ${TELEOP_FLAGS[@]+"${TELEOP_FLAGS[@]}"} \
  --dataset.repo_id="$EVAL_REPO_ID" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.fps="$CAMERA_FPS" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  --dataset.single_task="$SINGLE_TASK" \
  --dataset.private=true \
  --dataset.push_to_hub="$PUSH_TO_HUB"
