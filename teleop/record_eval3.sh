#!/usr/bin/env bash
# Eval 3 — record teleop demos for the sequential 3-step pick-and-place task.
#
# Scene: FOUR cubes of distinct colors + one bowl at a fixed position.
# Demo: pick THREE cubes in a given color order and place each in the bowl;
#       the 4th color is a distractor that stays in the workspace.
# Task string carries the goal-conditioning input:
#   "Pick c1, c2, c3 blocks in this order and place each in bowl at (x,y) cm"
#
# Recording plan:
#   SCENARIOS scenarios × EPISODES_PER_SCENARIO episodes.
#   Default: 20 scenarios × 5 = 100 demos.
#   Each scenario fixes the 4 colors and the 3-color pick sequence; the bowl
#   stays at one fixed position; cube layout is re-randomized per episode
#   during the reset window.
#
# Override with SCENARIOS / EPISODES_PER_SCENARIO / BOWL_POS_X / BOWL_POS_Y / etc.

set -euo pipefail

REPO_ID="${REPO_ID:-kenzy17/projet3-eval3-v1}"
EPISODES_PER_SCENARIO="${EPISODES_PER_SCENARIO:-5}"
EPISODE_TIME_S="${EPISODE_TIME_S:-90}"
RESET_TIME_S="${RESET_TIME_S:-15}"
FPS="${FPS:-30}"

# Local data root. Required because lerobot's resume() refuses to write into the
# HF snapshot cache (root=None). Must be the SAME path across all scenarios so
# resume can find the previously recorded episodes.
DATASET_ROOT="${DATASET_ROOT:-${HOME}/.lerobot-data/${REPO_ID}}"

# Defaults set for Kenzy's Linux laptop (Eval 3 recording machine).
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM1}"
# CAM_INDEX accepts a numeric OpenCV index (e.g. 0, macOS) OR a V4L2 device
# path (e.g. /dev/video4, Linux — robust across replugs). A path is quoted in
# the camera JSON; a bare index is left unquoted.
CAM_INDEX="${CAM_INDEX:-/dev/video4}"
if [[ "${CAM_INDEX}" == /* ]]; then
  CAM_JSON_VALUE="\"${CAM_INDEX}\""
else
  CAM_JSON_VALUE="${CAM_INDEX}"
fi

STREAMING_ENCODING="${STREAMING_ENCODING:-true}"
ENCODER_THREADS="${ENCODER_THREADS:-2}"
VCODEC="${VCODEC:-libsvtav1}"

DISPLAY_DATA="${DISPLAY_DATA:-true}"
DISPLAY_IP="${DISPLAY_IP:-}"
DISPLAY_PORT="${DISPLAY_PORT:-}"
DISPLAY_FLAGS=""
if [ -n "${DISPLAY_IP}" ]; then DISPLAY_FLAGS="${DISPLAY_FLAGS} --display_ip=${DISPLAY_IP}"; fi
if [ -n "${DISPLAY_PORT}" ]; then DISPLAY_FLAGS="${DISPLAY_FLAGS} --display_port=${DISPLAY_PORT}"; fi

# Each scenario: 4 colors. The FIRST THREE are the pick sequence (in order);
# the FOURTH is the distractor cube that stays on the table the whole episode.
DEFAULT_SCENARIOS="\
yellow blue green violet|\
blue green violet red|\
green violet red orange|\
violet red orange yellow|\
red orange yellow blue|\
orange yellow blue green|\
yellow green red blue|\
blue violet orange green|\
green yellow violet red|\
violet blue yellow orange|\
red green blue violet|\
orange violet green yellow|\
yellow red orange violet|\
blue orange red green|\
green blue orange violet|\
violet yellow red blue|\
red violet green orange|\
orange green yellow red|\
yellow violet blue red|\
blue red green orange"

# Bowl position (cm, robot frame). FIXED for the whole Eval 3 dataset — the
# bowl stays at one marked spot on the table. Embedded in every task string
# as the goal-conditioning input. Override with BOWL_POS_X / BOWL_POS_Y.
BOWL_POS_X="${BOWL_POS_X:-30}"
BOWL_POS_Y="${BOWL_POS_Y:-20}"

SCENARIOS_STR="${SCENARIOS:-${DEFAULT_SCENARIOS}}"

IFS='|' read -r -a SCENARIO_ARR <<< "${SCENARIOS_STR}"

START_FROM_SCENARIO="${START_FROM_SCENARIO:-1}"  # 1-indexed; useful for resuming

export RUST_LOG=error  # silence wgpu/rerun spam

num_scenarios=${#SCENARIO_ARR[@]}
total_eps=$(( num_scenarios * EPISODES_PER_SCENARIO ))

echo "############################################################"
echo "##  Eval 3 recording plan — sequential 3-step pick-and-place"
echo "##  ${num_scenarios} scenarios × ${EPISODES_PER_SCENARIO} episodes = ${total_eps} demos"
echo "##  Bowl: fixed at (${BOWL_POS_X}, ${BOWL_POS_Y}) cm"
echo "##  Episode: ${EPISODE_TIME_S}s record + ${RESET_TIME_S}s reset"
echo "##  Dataset: ${REPO_ID}"
echo "############################################################"

scenario_idx=0  # number of completed lerobot-record invocations (controls --resume)

for s_i in "${!SCENARIO_ARR[@]}"; do
  scenario="${SCENARIO_ARR[$s_i]}"
  read -r c1 c2 c3 c4 <<< "${scenario}"
  scenario_num=$((s_i + 1))

  C1_UP=$(echo "${c1}" | tr '[:lower:]' '[:upper:]')
  C2_UP=$(echo "${c2}" | tr '[:lower:]' '[:upper:]')
  C3_UP=$(echo "${c3}" | tr '[:lower:]' '[:upper:]')
  C4_UP=$(echo "${c4}" | tr '[:lower:]' '[:upper:]')

  if [ "${scenario_num}" -lt "${START_FROM_SCENARIO}" ]; then
    echo "[skip] scenario ${scenario_num}/${num_scenarios} (${C1_UP}>${C2_UP}>${C3_UP}, distractor ${C4_UP}) — already done"
    scenario_idx=$((scenario_idx + 1))
    continue
  fi

  echo ""
  echo "############################################################"
  echo "##  SCENARIO ${scenario_num}/${num_scenarios}    ${EPISODES_PER_SCENARIO} episodes"
  echo "##  Place FOUR cubes in the workspace:"
  echo "##    ${C1_UP}, ${C2_UP}, ${C3_UP}, ${C4_UP}"
  echo "##  Pick ORDER each episode:  ${C1_UP}  ->  ${C2_UP}  ->  ${C3_UP}"
  echo "##  ${C4_UP} is the DISTRACTOR — leave it on the table."
  echo "##  Place all three in the bowl at (${BOWL_POS_X}, ${BOWL_POS_Y}) cm."
  echo "##  Re-randomize the 4 cube positions during each reset window."
  echo "############################################################"
  echo ""
  read -rp "Press ENTER when scene is ready..."

  RESUME_FLAG=""
  if [ "${scenario_idx}" -gt 0 ] || [ "${START_FROM_SCENARIO}" -gt 1 ]; then
    RESUME_FLAG="--resume=true"
  else
    # First scenario of a fresh run — wipe stale local state.
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

  TASK_STR="Pick ${c1}, ${c2}, ${c3} blocks in this order and place each in bowl at (${BOWL_POS_X},${BOWL_POS_Y}) cm [distractor=${c4}]"

  attempt=1
  max_attempts=3
  BATCH_LOG=$(mktemp -t eval3_scenario.XXXXXX)
  while true; do
    set +e
    lerobot-record \
      --robot.type=so101_follower \
      --robot.port="${FOLLOWER_PORT}" \
      --robot.id=so101_follower \
      --robot.cameras="{\"wrist\": {\"type\": \"opencv\", \"index_or_path\": ${CAM_JSON_VALUE}, \"width\": 640, \"height\": 480, \"fps\": ${FPS}, \"warmup_s\": 3}}" \
      --teleop.type=so101_leader \
      --teleop.port="${LEADER_PORT}" \
      --teleop.id=so101_leader \
      --display_data="${DISPLAY_DATA}" \
      --dataset.repo_id="${REPO_ID}" \
      --dataset.root="${DATASET_ROOT}" \
      --dataset.num_episodes="${EPISODES_PER_SCENARIO}" \
      --dataset.fps="${FPS}" \
      --dataset.episode_time_s="${EPISODE_TIME_S}" \
      --dataset.reset_time_s="${RESET_TIME_S}" \
      --dataset.single_task="${TASK_STR}" \
      --dataset.private=true \
      --dataset.push_to_hub=true \
      --dataset.streaming_encoding="${STREAMING_ENCODING}" \
      --dataset.encoder_threads="${ENCODER_THREADS}" \
      --dataset.vcodec="${VCODEC}" \
      ${DISPLAY_FLAGS} \
      ${RESUME_FLAG} 2>&1 | tee "${BATCH_LOG}"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "${rc}" -eq 0 ]; then break; fi
    if [ "${attempt}" -ge "${max_attempts}" ]; then
      echo "❌ lerobot-record failed for scenario ${scenario_num} after ${max_attempts} attempts (exit ${rc})."
      rm -f "${BATCH_LOG}"
      exit "${rc}"
    fi
    echo "⚠️  lerobot-record failed (exit ${rc}). Retrying in 10 s — attempt $((attempt + 1))/${max_attempts}..."
    sleep 10
    attempt=$((attempt + 1))
  done

  # ── FPS summary for this scenario ──────────────────────────────────────────
  hz_values=$(grep -oE 'running slower \([0-9.]+ Hz' "${BATCH_LOG}" 2>/dev/null | grep -oE '[0-9.]+' || true)
  slow_count=$(echo "${hz_values}" | grep -c . 2>/dev/null || echo 0)
  slow_count=$(echo "${slow_count}" | tr -d '[:space:]')
  slow_count=${slow_count:-0}
  echo ""
  echo "============================================================"
  printf "  SCENARIO %d/%d FPS REPORT — target %s Hz\n" "${scenario_num}" "${num_scenarios}" "${FPS}"
  if [ "${slow_count}" -gt 0 ]; then
    min_hz=$(echo "${hz_values}" | sort -n | head -1)
    max_hz=$(echo "${hz_values}" | sort -n | tail -1)
    echo "  ⚠️  BELOW TARGET — slowdowns logged ${slow_count}× this scenario"
    printf "      slowest: %s Hz   highest reported slow: %s Hz\n" "${min_hz}" "${max_hz}"
    echo "      → frames may have been dropped"
  else
    echo "  ✅ ON TARGET — no slowdown warnings detected"
  fi
  echo "============================================================"
  rm -f "${BATCH_LOG}"

  scenario_idx=$((scenario_idx + 1))
  sleep 5  # let the OS release camera/serial handles
done

echo ""
echo "############################################################"
echo "##  ✅ All ${total_eps} episodes recorded and pushed."
echo "##     https://huggingface.co/datasets/${REPO_ID}"
echo "############################################################"
