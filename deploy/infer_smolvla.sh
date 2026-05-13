#!/usr/bin/env bash
# infer_smolvla.sh — run a trained SmolVLA policy on the real SO-101 follower.
#
# Mirrors deploy/infer.sh but with two SmolVLA-specific deltas:
#   1. POLICY_PATH default points at the SmolVLA repo.
#   2. --dataset.rename_map renames the live wrist camera key to camera1,
#      which is the key the model was trained against (the dataset has
#      'observation.images.wrist' but smolvla_base expects 'camera1' etc.,
#      so launch_smolvla.sh aliased wrist -> camera1 at train time).
#      Without this rename the wrist frame would not reach the model.
#
# Usage:
#   bash deploy/infer_smolvla.sh
#   → prompts: "Which cube should the policy try to pick up?"
#
#   TARGET_COLOR=yellow bash deploy/infer_smolvla.sh
#   POLICY_PATH=osammotg1/projet3-smolvla-eval2-v1-step20k bash deploy/infer_smolvla.sh
set -euo pipefail

POLICY_PATH="${POLICY_PATH:-osammotg1/projet3-smolvla-eval2-v1}"
POLICY_DEVICE="${POLICY_DEVICE:-mps}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/tty.usbmodem5B141129871}"
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"

# Leader for back-driving the follower during reset; set LEADER_PORT="" to skip.
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

# Camera-key rename: live frames arrive as 'observation.images.wrist'; the
# SmolVLA checkpoint's config.input_features keys this view as
# 'observation.images.camera1' (set by launch_smolvla.sh at train time).
# Override RENAME_MAP if you move to a different camera labeling at deploy.
RENAME_MAP="${RENAME_MAP:-{\"observation.images.wrist\": \"observation.images.camera1\"}}"

# --- Interactive: pick the target cube color ---
VALID_COLORS=(yellow blue green violet red orange)

if [[ -z "${TARGET_COLOR:-}" ]]; then
  printf '\n[infer-smolvla] Which cube should the policy try to pick up?\n'
  PS3='  > '
  select choice in "${VALID_COLORS[@]}"; do
    if [[ -n "${choice:-}" ]]; then
      TARGET_COLOR="$choice"
      break
    fi
    printf '[infer-smolvla] not a valid option, try again\n'
  done
fi

case " ${VALID_COLORS[*]} " in
  *" $TARGET_COLOR "*) : ;;
  *)
    printf '[infer-smolvla] ERROR: TARGET_COLOR=%s is not in the trained palette\n' "$TARGET_COLOR" >&2
    printf '[infer-smolvla] valid: %s\n' "${VALID_COLORS[*]}" >&2
    exit 1
    ;;
esac

# Same task phrasing the dataset was labeled with — the SmolVLA text encoder
# is the channel that delivers the color instruction to the model, so this
# string is load-bearing (vs. ACT, where it was metadata only).
SINGLE_TASK="${SINGLE_TASK:-Pick ${TARGET_COLOR} block and place in bowl at (-15.5,29.5) cm}"

# Tag the eval dataset by the policy short-name + color so re-runs stay separated.
POLICY_SHORT=$(basename "$POLICY_PATH" | sed 's|/|_|g')
EVAL_REPO_ID="${EVAL_REPO_ID:-osammotg1/eval_${POLICY_SHORT}-${TARGET_COLOR}}"

# lerobot-record refuses to start if its local cache dir for this repo_id already
# exists (FileExistsError). Auto-clean stale empty skeleton from aborted runs.
LEROBOT_CACHE_DIR="${HOME}/.cache/huggingface/lerobot/${EVAL_REPO_ID}"
if [[ -d "$LEROBOT_CACHE_DIR" ]]; then
  printf '[infer-smolvla] removing stale lerobot cache dir: %s\n' "$LEROBOT_CACHE_DIR"
  rm -rf "$LEROBOT_CACHE_DIR"
fi

if [[ -z "$FOLLOWER_PORT" ]]; then
  printf '[infer-smolvla] ERROR: FOLLOWER_PORT is required (e.g. /dev/ttyACM0 on Linux, COM5 on Windows)\n' >&2
  printf '[infer-smolvla] Linux: ls /dev/tty*  |  Windows: Device Manager → Ports (COM & LPT)\n' >&2
  exit 1
fi

# If POLICY_PATH looks like a local path, validate it; otherwise assume HF repo id.
if [[ "$POLICY_PATH" == /* || "$POLICY_PATH" == ./* || "$POLICY_PATH" == ../* || -e "$POLICY_PATH" ]]; then
  if [[ ! -d "$POLICY_PATH" ]]; then
    printf '[infer-smolvla] ERROR: POLICY_PATH=%s is not a directory.\n' "$POLICY_PATH" >&2
    printf '[infer-smolvla] Expected a pretrained_model/ directory containing config.json + model.safetensors.\n' >&2
    exit 1
  fi
  for f in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
    if [[ ! -f "$POLICY_PATH/$f" ]]; then
      printf '[infer-smolvla] ERROR: %s missing from %s\n' "$f" "$POLICY_PATH" >&2
      exit 1
    fi
  done
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

printf '[infer-smolvla] policy=%s\n'  "$POLICY_PATH"
printf '[infer-smolvla] task="%s"\n'   "$SINGLE_TASK"
printf '[infer-smolvla] rename_map=%s\n' "$RENAME_MAP"
printf '[infer-smolvla] robot=%s on %s\n' "$ROBOT_TYPE" "$FOLLOWER_PORT"
if [[ -n "$LEADER_PORT" ]]; then
  printf '[infer-smolvla] leader=%s on %s (homing during reset)\n' "$LEADER_TYPE" "$LEADER_PORT"
else
  printf '[infer-smolvla] leader=<none> (follower will freeze in place during reset)\n'
fi
printf '[infer-smolvla] camera=opencv idx=%s %sx%s@%s\n' "$CAMERA_INDEX" "$CAMERA_WIDTH" "$CAMERA_HEIGHT" "$CAMERA_FPS"
printf '[infer-smolvla] %d episodes × %ss (reset %ss)\n' "$NUM_EPISODES" "$EPISODE_TIME_S" "$RESET_TIME_S"

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
  --dataset.rename_map="$RENAME_MAP" \
  --dataset.private=true \
  --dataset.push_to_hub="$PUSH_TO_HUB"
