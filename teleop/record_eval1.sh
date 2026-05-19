#!/usr/bin/env bash
# Eval 1 — record 10 demos × 6 colors = 60 demos, single dataset, pushed to HF after each color.
#
# Usage:
#   bash teleop/record_eval1.sh
#
# Override defaults via env vars, e.g.
#   EPISODES_PER_COLOR=2 bash teleop/record_eval1.sh    # quick smoke test

set -euo pipefail

REPO_ID="${REPO_ID:-osammotg1/eval1-eth-hg-smoketest}"
EPISODES_PER_COLOR="${EPISODES_PER_COLOR:-10}"
EPISODE_TIME_S="${EPISODE_TIME_S:-55}"
RESET_TIME_S="${RESET_TIME_S:-8}"
FPS="${FPS:-30}"

# Local data root for the dataset. Required because lerobot's resume() refuses
# to write into the HF snapshot cache (root=None). Must be the SAME path across
# all colors so resume can find the previous episodes.
DATASET_ROOT="${DATASET_ROOT:-${HOME}/.lerobot-data/${REPO_ID}}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/tty.usbmodem5B141129871}"
LEADER_PORT="${LEADER_PORT:-/dev/tty.usbmodem5B141128171}"
CAM_INDEX="${CAM_INDEX:-0}"

# Streaming encoding offloads video encoding to a separate worker so the camera
# read thread doesn't starve during episode recording (which is what causes
# the rerun viewer to freeze mid-episode while flowing fine during reset).
STREAMING_ENCODING="${STREAMING_ENCODING:-true}"
ENCODER_THREADS="${ENCODER_THREADS:-2}"
# Keep libsvtav1 to match the existing 20 episodes already in this dataset.
# Override with VCODEC=auto for a fresh dataset to let lerobot pick a faster codec.
VCODEC="${VCODEC:-libsvtav1}"

# START_FROM: skip colors before this one when resuming. Empty = start at yellow.
# Examples:  START_FROM=green  START_FROM=violet  START_FROM=red
START_FROM="${START_FROM:-}"

# MONITOR=true pipes lerobot stdout/stderr through teleop/monitor_fps.py to
# surface slow-frame warnings (and an audible alert) live in the console.
MONITOR="${MONITOR:-false}"
PIPE_THROUGH=(cat)
if [ "${MONITOR}" = "true" ]; then
  PIPE_THROUGH=(python3 "${PWD}/teleop/monitor_fps.py")
fi

# Optional: route Rerun stream to a network gRPC server (e.g. one started with
#   rerun --serve-web --bind 127.0.0.1
# Then open http://127.0.0.1:9090 in your browser to watch live.
# Leave both empty to use the default native Rerun viewer window.
DISPLAY_IP="${DISPLAY_IP:-}"
DISPLAY_PORT="${DISPLAY_PORT:-}"

DISPLAY_FLAGS=""
if [ -n "${DISPLAY_IP}" ]; then
  DISPLAY_FLAGS="${DISPLAY_FLAGS} --display_ip=${DISPLAY_IP}"
fi
if [ -n "${DISPLAY_PORT}" ]; then
  DISPLAY_FLAGS="${DISPLAY_FLAGS} --display_port=${DISPLAY_PORT}"
fi

COLORS=(yellow blue green violet red orange)

export RUST_LOG=error  # silence wgpu/rerun spam

for i in "${!COLORS[@]}"; do
  color="${COLORS[$i]}"
  step=$((i + 1))
  COLOR_UPPER=$(echo "$color" | tr '[:lower:]' '[:upper:]')

  # Skip colors before START_FROM (used to resume after a partial run).
  if [ -n "${START_FROM}" ]; then
    SKIP=true
    for c in "${COLORS[@]}"; do
      if [ "$c" = "${START_FROM}" ]; then SKIP=false; break; fi
      if [ "$c" = "$color" ]; then break; fi
    done
    if [ "$SKIP" = "true" ]; then
      echo "Skipping ${color} (START_FROM=${START_FROM})"
      continue
    fi
  fi

  echo ""
  echo "########################################################"
  printf "##  STEP %d/6 — COLOR: %s  (%d episodes)\n" "$step" "$COLOR_UPPER" "$EPISODES_PER_COLOR"
  echo "##  Place the $COLOR_UPPER cube in the workspace."
  echo "##  Position the bowl somewhere reachable."
  echo "##  Reset positions between each episode (during the ${RESET_TIME_S}s window)."
  echo "########################################################"
  echo ""
  read -rp "Press ENTER when scene is ready..."

  # Resume if not the very first color of this run, OR if we're resuming a previous run.
  RESUME_FLAG=""
  if [ "$i" -gt 0 ] || [ -n "${START_FROM}" ]; then
    RESUME_FLAG="--resume=true"
  else
    # First color of a fresh run: wipe any leftover data from a previous aborted run.
    if [ -d "${DATASET_ROOT}" ]; then
      echo "Removing stale dataset root: ${DATASET_ROOT}"
      rm -rf "${DATASET_ROOT}"
    fi
    LEROBOT_HF_CACHE="${HOME}/.cache/huggingface/lerobot/${REPO_ID}"
    if [ -d "${LEROBOT_HF_CACHE}" ]; then
      echo "Removing stale HF snapshot cache: ${LEROBOT_HF_CACHE}"
      rm -rf "${LEROBOT_HF_CACHE}"
    fi
  fi

  # Retry up to 3 times — macOS sometimes fails to release the USB camera
  # handle quickly enough between subprocess restarts (TimeoutError on connect).
  attempt=1
  max_attempts=3
  while true; do
    set +e
    lerobot-record \
      --robot.type=so101_follower \
      --robot.port="${FOLLOWER_PORT}" \
      --robot.id=so101_follower \
      --robot.cameras="{\"wrist\": {\"type\": \"opencv\", \"index_or_path\": ${CAM_INDEX}, \"width\": 640, \"height\": 480, \"fps\": ${FPS}, \"warmup_s\": 3}}" \
      --teleop.type=so101_leader \
      --teleop.port="${LEADER_PORT}" \
      --teleop.id=so101_leader \
      --display_data=true \
      --dataset.repo_id="${REPO_ID}" \
      --dataset.root="${DATASET_ROOT}" \
      --dataset.num_episodes="${EPISODES_PER_COLOR}" \
      --dataset.fps="${FPS}" \
      --dataset.episode_time_s="${EPISODE_TIME_S}" \
      --dataset.reset_time_s="${RESET_TIME_S}" \
      --dataset.single_task="Pick ${color} block and place in bowl" \
      --dataset.private=false \
      --dataset.push_to_hub=true \
      --dataset.streaming_encoding="${STREAMING_ENCODING}" \
      --dataset.encoder_threads="${ENCODER_THREADS}" \
      --dataset.vcodec="${VCODEC}" \
      ${DISPLAY_FLAGS} \
      ${RESUME_FLAG} 2>&1 | "${PIPE_THROUGH[@]}"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -eq 0 ]; then
      break
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "❌ lerobot-record failed for color ${color} after ${max_attempts} attempts (exit ${rc})."
      exit "$rc"
    fi
    echo "⚠️  lerobot-record failed (exit ${rc}). Retrying in 10 s — attempt $((attempt + 1))/${max_attempts}..."
    sleep 10
    attempt=$((attempt + 1))
  done

  # Brief pause so macOS releases the camera + serial port handles before the next color.
  if [ "$step" -lt "${#COLORS[@]}" ]; then
    echo "Waiting 5 s for camera/serial release before next color..."
    sleep 5
  fi
done

echo ""
echo "############################################################"
echo "##  ✅ All $((${#COLORS[@]} * EPISODES_PER_COLOR)) episodes recorded and pushed."
echo "##     https://huggingface.co/datasets/${REPO_ID}"
echo "############################################################"
