#!/usr/bin/env bash
# infer.sh — run a trained ACT policy on the real SO-101 follower.
#
# Eval2 defaults baked in (dark-noise-100k policy, fixed bowl at
# (-15.5, 29.5) cm, macOS follower port). The only thing you choose at
# launch is which cube color the policy should try to grab.
#
# Most common usage on the lab Mac:
#   bash deploy/infer.sh
#   → prompts: "Which cube should the policy try to pick up?"
#
# Skip the prompt by pre-setting TARGET_COLOR:
#   TARGET_COLOR=yellow bash deploy/infer.sh
#
# Override any baked-in default via env var, e.g. swap policy variant:
#   POLICY_PATH=osammotg1/projet3-act-eval2-dark-shadow bash deploy/infer.sh
set -euo pipefail

POLICY_PATH="${POLICY_PATH:-osammotg1/projet3-act-eval2-dark-noise-100k}"
POLICY_DEVICE="${POLICY_DEVICE:-mps}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/tty.usbmodem5B141129871}"
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"

# Leader for back-driving the follower during reset between episodes. Without
# the leader, the follower's torque stays on after each rollout and you can't
# move it back to home pose by hand. Plug the leader in or override with
# LEADER_PORT="" to disable.
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

# --- Interactive: pick the target cube color ---
VALID_COLORS=(yellow blue green violet red orange)

if [[ -z "${TARGET_COLOR:-}" ]]; then
  printf '\n[infer] Which cube should the policy try to pick up?\n'
  PS3='  > '
  select choice in "${VALID_COLORS[@]}"; do
    if [[ -n "${choice:-}" ]]; then
      TARGET_COLOR="$choice"
      break
    fi
    printf '[infer] not a valid option, try again\n'
  done
fi

case " ${VALID_COLORS[*]} " in
  *" $TARGET_COLOR "*) : ;;
  *)
    printf '[infer] ERROR: TARGET_COLOR=%s is not in the trained palette\n' "$TARGET_COLOR" >&2
    printf '[infer] valid: %s\n' "${VALID_COLORS[*]}" >&2
    exit 1
    ;;
esac

SINGLE_TASK="${SINGLE_TASK:-Pick ${TARGET_COLOR} block and place in bowl at (-15.5,29.5) cm}"
EVAL_REPO_ID="${EVAL_REPO_ID:-osammotg1/eval_projet3-eval2-dark-noise-100k-${TARGET_COLOR}}"

# lerobot-record refuses to start if its local cache dir for this repo_id
# already exists (FileExistsError). An aborted previous run leaves an empty
# meta/ skeleton behind; auto-remove it so re-runs work.
LEROBOT_CACHE_DIR="${HOME}/.cache/huggingface/lerobot/${EVAL_REPO_ID}"
if [[ -d "$LEROBOT_CACHE_DIR" ]]; then
  printf '[infer] removing stale lerobot cache dir: %s\n' "$LEROBOT_CACHE_DIR"
  rm -rf "$LEROBOT_CACHE_DIR"
fi

# TODO(human): add a pre-flight confirmation step here, before lerobot-record runs.
#
# Show the user a summary of what's about to happen (policy, color, num_episodes
# × episode_time_s, eval_repo_id) and require an explicit [y/N] confirmation
# before launching the real arm. This is the safety net that catches "wrong
# policy" / "wrong color" / "way too many episodes" before the robot moves.
#
# Constraints to design around:
#   - Pressing just Enter should NOT launch (default N, fail-safe).
#   - `CONFIRM=yes bash deploy/infer.sh` should skip the prompt entirely
#     (so scripted A/B loops aren't blocked).
#   - On a non-y answer, print "[infer] cancelled" and exit 0 (not 1 — user-cancel
#     isn't an error; exit 1 would break for-loop drivers).

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

TELEOP_FLAGS=()
if [[ -n "$LEADER_PORT" ]]; then
  TELEOP_FLAGS=(
    --teleop.type="$LEADER_TYPE"
    --teleop.port="$LEADER_PORT"
    --teleop.id="$LEADER_ID"
  )
fi

printf '[infer] policy=%s\n'  "$POLICY_PATH"
printf '[infer] robot=%s on %s\n' "$ROBOT_TYPE" "$FOLLOWER_PORT"
if [[ -n "$LEADER_PORT" ]]; then
  printf '[infer] leader=%s on %s (homing during reset)\n' "$LEADER_TYPE" "$LEADER_PORT"
else
  printf '[infer] leader=<none> (follower will freeze in place during reset)\n'
fi
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
  ${TELEOP_FLAGS[@]+"${TELEOP_FLAGS[@]}"} \
  --dataset.repo_id="$EVAL_REPO_ID" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.fps="$CAMERA_FPS" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  --dataset.single_task="$SINGLE_TASK" \
  --dataset.private=true \
  --dataset.push_to_hub="$PUSH_TO_HUB"


