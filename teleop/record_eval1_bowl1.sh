#!/usr/bin/env bash
# Eval 1 — 40 demos at BOWL POSITION 1, varying cube color + cube position.
#
# Experimental design:
#   - Bowl: FIXED at position 1, see teleop/eval1_bowl_positions.json
#       (x_cm=16, y_cm=32 in robot base frame, +x right, +y forward).
#   - Cube color: 5 colors × 8 episodes = 40 demos. Override with env vars.
#   - Cube position: RANDOMIZED every episode by the operator during reset.
#       Place the cube at a roughly different (x, y) on the table each time.
#
# Heads up: with one bowl position, the policy will learn that bowl. Plan to
# record matching sessions at additional bowl positions later — interleave
# across positions if possible (see record_eval1.md for the full strategy).
#
# Usage:
#   bash teleop/record_eval1_bowl1.sh
#
# Useful overrides:
#   EPISODES_PER_COLOR=2 bash teleop/record_eval1_bowl1.sh    # smoke test (10 demos)
#   COLORS=yellow,blue,green,red EPISODES_PER_COLOR=10 bash teleop/record_eval1_bowl1.sh
#   MONITOR=true bash teleop/record_eval1_bowl1.sh            # live FPS health alert
#
# Appending to an existing dataset (do NOT wipe the local root, always resume):
#   RESUME=true EPISODES_PER_COLOR=20 EPISODES_ORANGE=10 bash teleop/record_eval1_bowl1.sh
#
# Per-color episode override: set EPISODES_<COLOR> (uppercase) for any color in
# the COLORS list. Falls back to EPISODES_PER_COLOR. Example: EPISODES_ORANGE=10
# records 10 orange episodes while everything else uses EPISODES_PER_COLOR.

set -euo pipefail

REPO_ID="${REPO_ID:-osammotg1/projet3-eval1-bowl1-v1}"
EPISODES_PER_COLOR="${EPISODES_PER_COLOR:-8}"
# RESUME=true: append to an existing dataset. Never wipes the local root or HF
# cache, and forces --resume=true on every color batch.
RESUME="${RESUME:-false}"
EPISODE_TIME_S="${EPISODE_TIME_S:-55}"
RESET_TIME_S="${RESET_TIME_S:-5}"
FPS="${FPS:-30}"

# Bowl position 1 (from teleop/eval1_bowl_positions.json) — purely informational,
# baked into the dataset's task string so downstream tooling can identify the
# bowl placement that produced each episode batch.
BOWL_POS_ID="${BOWL_POS_ID:-1}"
BOWL_X_CM="${BOWL_X_CM:-16}"
BOWL_Y_CM="${BOWL_Y_CM:-32}"

# Local dataset root — resume requires this to persist between color batches.
DATASET_ROOT="${DATASET_ROOT:-${HOME}/.lerobot-data/${REPO_ID}}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/tty.usbmodem5B141129871}"
LEADER_PORT="${LEADER_PORT:-/dev/tty.usbmodem5B141128171}"
CAM_INDEX="${CAM_INDEX:-0}"

STREAMING_ENCODING="${STREAMING_ENCODING:-true}"
ENCODER_THREADS="${ENCODER_THREADS:-2}"
VCODEC="${VCODEC:-libsvtav1}"

# COLORS as a comma-separated env var → bash array. Defaults to 5 colors so
# the math works out cleanly: 5 × 8 = 40 demos.
if [ -n "${COLORS:-}" ]; then
  IFS=',' read -ra COLORS <<< "$COLORS"
else
  COLORS=(yellow blue green violet red)
fi

# START_FROM: skip colors before this one when resuming after a partial run.
START_FROM="${START_FROM:-}"

# MONITOR=true pipes lerobot through teleop/monitor_fps.py for live FPS alerts.
MONITOR="${MONITOR:-false}"
PIPE_THROUGH=(cat)
if [ "${MONITOR}" = "true" ]; then
  PIPE_THROUGH=(python3 "${PWD}/teleop/monitor_fps.py")
fi

# Optional Rerun gRPC routing (leave empty for local viewer).
DISPLAY_IP="${DISPLAY_IP:-}"
DISPLAY_PORT="${DISPLAY_PORT:-}"
DISPLAY_FLAGS=""
[ -n "${DISPLAY_IP}" ]   && DISPLAY_FLAGS="${DISPLAY_FLAGS} --display_ip=${DISPLAY_IP}"
[ -n "${DISPLAY_PORT}" ] && DISPLAY_FLAGS="${DISPLAY_FLAGS} --display_port=${DISPLAY_PORT}"

export RUST_LOG=error  # silence wgpu/rerun spam

# Per-color totals respect EPISODES_<COLOR> overrides; fall back to EPISODES_PER_COLOR.
TOTAL_DEMOS=0
for _c in "${COLORS[@]}"; do
  _CU=$(echo "$_c" | tr '[:lower:]' '[:upper:]')
  _VN="EPISODES_${_CU}"
  TOTAL_DEMOS=$(( TOTAL_DEMOS + ${!_VN:-${EPISODES_PER_COLOR}} ))
done
unset _c _CU _VN

# Build a "yellow=20 blue=20 ... orange=10" string for the banner.
PER_COLOR_SUMMARY=""
for _c in "${COLORS[@]}"; do
  _CU=$(echo "$_c" | tr '[:lower:]' '[:upper:]')
  _VN="EPISODES_${_CU}"
  PER_COLOR_SUMMARY="${PER_COLOR_SUMMARY}${_c}=${!_VN:-${EPISODES_PER_COLOR}} "
done
unset _c _CU _VN

MODE_STR="FRESH (will wipe local root on first color)"
if [ "${RESUME}" = "true" ]; then
  MODE_STR="RESUME (appending to existing dataset, no wipe)"
fi

echo ""
echo "############################################################"
echo "##  EVAL 1 — BOWL POSITION ${BOWL_POS_ID}  (x=${BOWL_X_CM} cm, y=${BOWL_Y_CM} cm)"
echo "##  Mode:     ${MODE_STR}"
echo "##  Colors:   ${COLORS[*]}"
echo "##  Per color: ${PER_COLOR_SUMMARY}"
echo "##  TOTAL:    ${TOTAL_DEMOS} demos"
echo "##  Dataset:  ${REPO_ID}"
echo "############################################################"
echo ""
echo "Pre-flight checklist:"
echo "  [ ] Bowl placed at position ${BOWL_POS_ID} (${BOWL_X_CM} cm right, ${BOWL_Y_CM} cm forward from robot base)."
echo "  [ ] All ${#COLORS[@]} cube colors available."
echo "  [ ] Wrist cam framing checked (lerobot-find-cameras opencv if unsure)."
echo "  [ ] HF auth OK (hf auth whoami)."
echo ""
read -rp "Press ENTER to begin..."

for i in "${!COLORS[@]}"; do
  color="${COLORS[$i]}"
  step=$((i + 1))
  COLOR_UPPER=$(echo "$color" | tr '[:lower:]' '[:upper:]')

  # Resume from a specific color, if requested.
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

  # Per-color episode count: EPISODES_<COLOR> overrides EPISODES_PER_COLOR.
  VARNAME="EPISODES_${COLOR_UPPER}"
  EPISODES_THIS_COLOR="${!VARNAME:-${EPISODES_PER_COLOR}}"

  echo ""
  echo "########################################################"
  printf "##  STEP %d/%d — COLOR: %s  (%d episodes)\n" "$step" "${#COLORS[@]}" "$COLOR_UPPER" "$EPISODES_THIS_COLOR"
  echo "##  Place the $COLOR_UPPER cube on the table."
  echo "##  Bowl stays at position ${BOWL_POS_ID} (${BOWL_X_CM} cm right, ${BOWL_Y_CM} cm forward). Do NOT move it."
  echo "##  During each ${RESET_TIME_S}s reset window, MOVE the cube to a new (x, y)."
  echo "########################################################"
  echo ""
  read -rp "Press ENTER when scene is ready..."

  # Resume if: (a) any color past the first of a fresh run, (b) START_FROM
  # was set (partial-run resume), or (c) RESUME=true (appending to existing).
  RESUME_FLAG=""
  if [ "$i" -gt 0 ] || [ -n "${START_FROM}" ] || [ "${RESUME}" = "true" ]; then
    RESUME_FLAG="--resume=true"
  else
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
      --dataset.num_episodes="${EPISODES_THIS_COLOR}" \
      --dataset.fps="${FPS}" \
      --dataset.episode_time_s="${EPISODE_TIME_S}" \
      --dataset.reset_time_s="${RESET_TIME_S}" \
      --dataset.single_task="Pick ${color} block and place in bowl at (${BOWL_X_CM},${BOWL_Y_CM}) cm [bowl_pos=${BOWL_POS_ID}]" \
      --dataset.private=true \
      --dataset.push_to_hub=true \
      --dataset.streaming_encoding="${STREAMING_ENCODING}" \
      --dataset.encoder_threads="${ENCODER_THREADS}" \
      --dataset.vcodec="${VCODEC}" \
      ${DISPLAY_FLAGS} \
      ${RESUME_FLAG} 2>&1 | "${PIPE_THROUGH[@]}"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -eq 0 ]; then break; fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "❌ lerobot-record failed for color ${color} after ${max_attempts} attempts (exit ${rc})."
      exit "$rc"
    fi
    echo "⚠️  lerobot-record failed (exit ${rc}). Retrying in 10 s — attempt $((attempt + 1))/${max_attempts}..."
    sleep 10
    attempt=$((attempt + 1))
  done

  if [ "$step" -lt "${#COLORS[@]}" ]; then
    echo "Waiting 5 s for camera/serial release before next color..."
    sleep 5
  fi
done

echo ""
echo "############################################################"
echo "##  ✅ ${TOTAL_DEMOS} episodes recorded at bowl position ${BOWL_POS_ID}."
echo "##     https://huggingface.co/datasets/${REPO_ID}"
echo "############################################################"
