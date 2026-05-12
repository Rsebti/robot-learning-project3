#!/usr/bin/env bash
# Eval 2 — record teleop demos for the 2-cube targeted pick task.
#
# Scene: two adjacent cubes of different colors + one bowl at a randomized position.
# Demo: pick the TARGET color cube (ignore the distractor) and place in the bowl.
# Task string: "Pick {target_color} block and place in bowl"
#
# Two-phase recording:
#   Phase 1 (main coverage):    6 rotation pairs × 2 targets × EPISODES_PER_TARGET = 60
#   Phase 2 (missing coverage): 9 remaining pairs × 2 targets × EPISODES_PER_TARGET_EXTRA = 36
#   Default total: 60 + 36 = 96 demos.
#
# Override with PAIRS / EXTRA_PAIRS / EPISODES_PER_TARGET / EPISODES_PER_TARGET_EXTRA.

set -euo pipefail

REPO_ID="${REPO_ID:-osammotg1/projet3-eval2-v1}"
EPISODES_PER_TARGET="${EPISODES_PER_TARGET:-5}"
EPISODES_PER_TARGET_EXTRA="${EPISODES_PER_TARGET_EXTRA:-2}"
EPISODE_TIME_S="${EPISODE_TIME_S:-30}"
RESET_TIME_S="${RESET_TIME_S:-5}"
FPS="${FPS:-30}"

DATASET_ROOT="${DATASET_ROOT:-${HOME}/.lerobot-data/${REPO_ID}}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/tty.usbmodem5B141129871}"
LEADER_PORT="${LEADER_PORT:-/dev/tty.usbmodem5B141128171}"
CAM_INDEX="${CAM_INDEX:-0}"

STREAMING_ENCODING="${STREAMING_ENCODING:-true}"
ENCODER_THREADS="${ENCODER_THREADS:-2}"
VCODEC="${VCODEC:-libsvtav1}"

DISPLAY_DATA="${DISPLAY_DATA:-true}"
DISPLAY_IP="${DISPLAY_IP:-}"
DISPLAY_PORT="${DISPLAY_PORT:-}"
DISPLAY_FLAGS=""
if [ -n "${DISPLAY_IP}" ]; then DISPLAY_FLAGS="${DISPLAY_FLAGS} --display_ip=${DISPLAY_IP}"; fi
if [ -n "${DISPLAY_PORT}" ]; then DISPLAY_FLAGS="${DISPLAY_FLAGS} --display_port=${DISPLAY_PORT}"; fi

# Phase 1: 6 rotation pairs (forms a cycle around the color list).
DEFAULT_PAIRS="yellow blue|blue green|green violet|violet red|red orange|orange yellow"
# Phase 2: the remaining 9 pairs needed to cover all 15 unordered color combinations.
DEFAULT_EXTRA_PAIRS="yellow green|yellow violet|yellow red|blue violet|blue red|blue orange|green red|green orange|violet orange"

PAIRS_STR="${PAIRS:-${DEFAULT_PAIRS}}"
EXTRA_PAIRS_STR="${EXTRA_PAIRS:-${DEFAULT_EXTRA_PAIRS}}"

IFS='|' read -r -a MAIN_PAIRS_ARR <<< "${PAIRS_STR}"
IFS='|' read -r -a EXTRA_PAIRS_ARR <<< "${EXTRA_PAIRS_STR}"

START_FROM_BATCH="${START_FROM_BATCH:-1}"  # 1-indexed batch number to start at; useful for resuming

export RUST_LOG=error

main_pairs=${#MAIN_PAIRS_ARR[@]}
extra_pairs=${#EXTRA_PAIRS_ARR[@]}
total_batches=$(( (main_pairs + extra_pairs) * 2 ))
total_eps=$(( main_pairs * 2 * EPISODES_PER_TARGET + extra_pairs * 2 * EPISODES_PER_TARGET_EXTRA ))

echo "############################################################"
echo "##  Eval 2 recording plan"
echo "##  Phase 1 (main):    ${main_pairs} pairs × 2 targets × ${EPISODES_PER_TARGET} demos = $(( main_pairs * 2 * EPISODES_PER_TARGET ))"
echo "##  Phase 2 (missing): ${extra_pairs} pairs × 2 targets × ${EPISODES_PER_TARGET_EXTRA} demos = $(( extra_pairs * 2 * EPISODES_PER_TARGET_EXTRA ))"
echo "##  Total batches:  ${total_batches}    Total episodes: ${total_eps}"
echo "##  Dataset: ${REPO_ID}"
echo "############################################################"

batch_idx=0  # number of completed lerobot-record invocations (controls --resume flag)

# record_phase <phase_name> <episodes_per_target> <pair...>
record_phase() {
  local phase_name="$1"; shift
  local eps_per_target="$1"; shift
  local pairs_arr=("$@")
  local total_pairs_in_phase=${#pairs_arr[@]}

  for pair_i in "${!pairs_arr[@]}"; do
    pair="${pairs_arr[$pair_i]}"
    read -r color_a color_b <<< "${pair}"
    COLOR_A_UP=$(echo "${color_a}" | tr '[:lower:]' '[:upper:]')
    COLOR_B_UP=$(echo "${color_b}" | tr '[:lower:]' '[:upper:]')
    pair_num=$((pair_i + 1))

    # Skip the scene-setup prompt entirely if both batches of this pair are already done.
    pair_last_batch=$((batch_idx + 2))
    if [ "${pair_last_batch}" -lt "${START_FROM_BATCH}" ]; then
      echo "[skip] ${phase_name} pair ${pair_num}/${total_pairs_in_phase} (${COLOR_A_UP}+${COLOR_B_UP}) — both batches already done"
      batch_idx=$((batch_idx + 2))
      continue
    fi

    echo ""
    echo "############################################################"
    echo "##  ${phase_name} pair ${pair_num}/${total_pairs_in_phase}:  ${COLOR_A_UP}  +  ${COLOR_B_UP}"
    echo "##  ${eps_per_target} demos per target."
    echo "##  Place ONE ${COLOR_A_UP} cube AND ONE ${COLOR_B_UP} cube"
    echo "##  ADJACENT to each other (flat cluster) in the workspace."
    echo "##  Place the bowl at a reachable position."
    echo "############################################################"
    echo ""
    read -rp "Press ENTER when scene is ready..."

    for target_color in "${color_a}" "${color_b}"; do
      TARGET_UP=$(echo "${target_color}" | tr '[:lower:]' '[:upper:]')
      if [ "${target_color}" = "${color_a}" ]; then
        DISTRACTOR="${color_b}"
      else
        DISTRACTOR="${color_a}"
      fi
      DISTRACTOR_UP=$(echo "${DISTRACTOR}" | tr '[:lower:]' '[:upper:]')

      batch_num=$((batch_idx + 1))
      if [ "${batch_num}" -lt "${START_FROM_BATCH}" ]; then
        echo "Skipping batch ${batch_num} (target ${TARGET_UP}, START_FROM_BATCH=${START_FROM_BATCH})"
        batch_idx=$((batch_idx + 1))
        continue
      fi

      echo ""
      echo "------------------------------------------------------------"
      echo "  BATCH ${batch_num}/${total_batches}: TARGET ${TARGET_UP} (distractor ${DISTRACTOR_UP})"
      echo "  ${eps_per_target} episodes — pick the ${TARGET_UP} cube each time."
      echo "------------------------------------------------------------"
      read -rp "Press ENTER to start this batch..."

      RESUME_FLAG=""
      if [ "${batch_idx}" -gt 0 ] || [ "${START_FROM_BATCH}" -gt 1 ]; then
        RESUME_FLAG="--resume=true"
      else
        # First batch ever — wipe stale local state.
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
      BATCH_LOG=$(mktemp -t eval2_batch.XXXXXX)
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
          --display_data="${DISPLAY_DATA}" \
          --dataset.repo_id="${REPO_ID}" \
          --dataset.root="${DATASET_ROOT}" \
          --dataset.num_episodes="${eps_per_target}" \
          --dataset.fps="${FPS}" \
          --dataset.episode_time_s="${EPISODE_TIME_S}" \
          --dataset.reset_time_s="${RESET_TIME_S}" \
          --dataset.single_task="Pick ${target_color} block and place in bowl" \
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
          echo "❌ lerobot-record failed for target=${target_color} (${phase_name} pair ${pair_num}) after ${max_attempts} attempts (exit ${rc})."
          rm -f "${BATCH_LOG}"
          exit "${rc}"
        fi
        echo "⚠️  lerobot-record failed (exit ${rc}). Retrying in 10 s — attempt $((attempt + 1))/${max_attempts}..."
        sleep 10
        attempt=$((attempt + 1))
      done

      # ── FPS summary for this batch ─────────────────────────────────────────
      hz_values=$(grep -oE 'running slower \([0-9.]+ Hz' "${BATCH_LOG}" 2>/dev/null | grep -oE '[0-9.]+' || true)
      slow_count=$(echo "${hz_values}" | grep -c . 2>/dev/null || echo 0)
      slow_count=$(echo "${slow_count}" | tr -d '[:space:]')
      slow_count=${slow_count:-0}
      echo ""
      echo "============================================================"
      printf "  BATCH %d/%d FPS REPORT — target %s Hz\n" "${batch_num}" "${total_batches}" "${FPS}"
      if [ "${slow_count}" -gt 0 ]; then
        min_hz=$(echo "${hz_values}" | sort -n | head -1)
        max_hz=$(echo "${hz_values}" | sort -n | tail -1)
        echo "  ⚠️  BELOW TARGET — slowdowns logged ${slow_count}× this batch"
        printf "      slowest: %s Hz   highest reported slow: %s Hz\n" "${min_hz}" "${max_hz}"
        echo "      → frames may have been dropped"
      else
        echo "  ✅ ON TARGET — no slowdown warnings detected"
      fi
      echo "============================================================"
      rm -f "${BATCH_LOG}"

      batch_idx=$((batch_idx + 1))
      sleep 5  # let macOS release camera/serial handles
    done
  done
}

record_phase "PHASE 1 (main coverage)" "${EPISODES_PER_TARGET}" "${MAIN_PAIRS_ARR[@]}"
record_phase "PHASE 2 (missing coverage)" "${EPISODES_PER_TARGET_EXTRA}" "${EXTRA_PAIRS_ARR[@]}"

echo ""
echo "############################################################"
echo "##  ✅ All ${total_eps} episodes recorded and pushed."
echo "##     https://huggingface.co/datasets/${REPO_ID}"
echo "############################################################"
