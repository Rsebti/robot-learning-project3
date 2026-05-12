#!/usr/bin/env bash
# test_eval1_checkpoints.sh — run one inference episode per checkpoint
# on the real SO-101, sequentially.
#
# Per-checkpoint flow:
#   1. Loads the local pretrained_model dir as the policy
#   2. Calls deploy/infer.sh with NUM_EPISODES=1 (one rollout)
#   3. Captures the run log under outputs/eval1_checkpoint_test/<step>.log
#   4. Reports wall-clock time + slow-FPS warning count
#   5. Pauses for Enter before moving on (Ctrl+C to abort)
#
# During each rollout, use lerobot-record's keys:
#   right arrow -> save and end this episode (= "next checkpoint")
#   left arrow  -> cancel + redo
#   esc         -> stop (will exit this checkpoint, loop continues)
#
# Required env:
#   FOLLOWER_PORT   USB serial of the SO-101 follower
#                   (mac: /dev/tty.usbmodem...; ls /dev/tty.usbmodem*)
#
# Common overrides:
#   POLICY_DEVICE   default: mps (Apple Silicon GPU; fall back to `cpu` if ops error)
#   CAMERA_INDEX    default: 0 (built-in webcam is usually 0; wrist may be 1)
#   EPISODE_TIME_S  default: 20
#   STEPS           same default set as download_eval1_checkpoints.sh
#
# Example:
#   FOLLOWER_PORT=/dev/tty.usbmodem58FA0826891 \
#   CAMERA_INDEX=1 \
#     bash tools/test_eval1_checkpoints.sh
set -euo pipefail

if [[ -z "${FOLLOWER_PORT:-}" ]]; then
  echo "[test] ERROR: FOLLOWER_PORT is required (e.g. /dev/tty.usbmodem...)" >&2
  echo "[test] mac: ls /dev/tty.usbmodem*" >&2
  exit 1
fi

LOCAL_BASE="${LOCAL_BASE:-train/checkpoints/projet3_act_eval1_v1}"
STEPS="${STEPS:-006000 010000 016000 020000 026000 030000 036000 040000}"

export POLICY_DEVICE="${POLICY_DEVICE:-mps}"
export CAMERA_INDEX="${CAMERA_INDEX:-0}"
export CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
export CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
export CAMERA_FPS="${CAMERA_FPS:-30}"
export NUM_EPISODES=1
export EPISODE_TIME_S="${EPISODE_TIME_S:-20}"
export RESET_TIME_S="${RESET_TIME_S:-5}"
export PUSH_TO_HUB=false
export SINGLE_TASK="${SINGLE_TASK:-Pick block and place in bowl}"
export FOLLOWER_PORT

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/eval1_checkpoint_test/$RUN_TS"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.tsv"
printf 'step\twall_s\tslow_fps_warnings\teffective_fps\n' >"$SUMMARY"

echo "[test] follower=$FOLLOWER_PORT  device=$POLICY_DEVICE  cam=$CAMERA_INDEX  fps_target=$CAMERA_FPS"
echo "[test] log_dir=$LOG_DIR"
echo "[test] steps: $STEPS"
echo

i=0
total=$(echo "$STEPS" | wc -w | tr -d ' ')
for step in $STEPS; do
  i=$((i+1))
  policy_dir="$LOCAL_BASE/$step/pretrained_model"
  if [[ ! -d "$policy_dir" ]]; then
    echo "[test] SKIP step=$step (missing $policy_dir — run download_eval1_checkpoints.sh first)"
    continue
  fi

  log="$LOG_DIR/${step}.log"
  echo "================================================================"
  echo "[test] ($i/$total) step=$step  policy=$policy_dir"
  echo "[test] log -> $log"
  echo "[test] press right-arrow in the lerobot window to end this episode"
  echo "================================================================"

  export POLICY_PATH="$policy_dir"
  export EVAL_REPO_ID="local/eval_projet3_act_eval1_v1_${step}_${RUN_TS}"

  start=$(python3 -c 'import time;print(time.time())')
  set +e
  bash deploy/infer.sh 2>&1 | tee "$log"
  status="${PIPESTATUS[0]}"
  set -e
  end=$(python3 -c 'import time;print(time.time())')
  wall=$(python3 -c "print(f'{$end-$start:.2f}')")

  warnings=$(grep -c -i "running slower than target FPS" "$log" || true)
  # Effective FPS = target_fps * (episode_time_s / wall_time)  is a rough proxy;
  # better: read frames count from the saved dataset metadata if present.
  meta=$(find ~/.cache/huggingface/lerobot -type f -name 'episodes.parquet' \
            -path "*${EVAL_REPO_ID}*" 2>/dev/null | head -1 || true)
  eff_fps="n/a"
  if [[ -n "$meta" ]]; then
    eff_fps=$(python3 - <<PY 2>/dev/null || echo n/a
import pyarrow.parquet as pq
t = pq.read_table("$meta").to_pandas()
frames = int(t["length"].sum()) if "length" in t.columns else int(t.iloc[0]["length"])
print(f"{frames / float($wall):.1f}")
PY
)
  fi

  printf '%s\t%s\t%s\t%s\n' "$step" "$wall" "$warnings" "$eff_fps" >>"$SUMMARY"
  echo
  echo "[test] step=$step done. wall=${wall}s  slow-fps-warnings=$warnings  effective_fps=$eff_fps  exit=$status"
  echo

  if [[ $i -lt $total ]]; then
    read -r -p "[test] Press Enter for next checkpoint (Ctrl+C to abort)... " _
  fi
done

echo
echo "================================================================"
echo "[test] all done. Summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY"
