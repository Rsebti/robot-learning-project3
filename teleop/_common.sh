#!/usr/bin/env bash
# teleop/_common.sh — shared plumbing for SO-101 record/deploy scripts.
#
# Sourced by sibling scripts (record_eval2.sh / record_hilserl_demos.sh /
# future record scripts). Not executable on its own.
#
# Usage:
#   # at the top of a sibling script, after `set -euo pipefail`:
#   source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#   teleop_common_init                # sets DEFAULT_* vars + RUST_LOG
#   teleop_assert_hf_auth osammotg1   # bail if not logged in as that user
#   teleop_clear_stale_cache REPO_ID  # clear ~/.cache/huggingface/lerobot/<repo>
#
# Why this exists: lerobot-record and lerobot.rl.gym_manipulator share three
# Mac-specific gotchas (USB port enumeration, HF cache FileExistsError, wgpu
# log spam). Extracting them avoids drift between scripts and lets us fix in
# one place.

# Mac-default SO-101 ports + camera index. Override per session if `lerobot-find-port`
# reports different values. These match deploy/infer_smolvla.sh defaults.
: "${DEFAULT_FOLLOWER_PORT:=/dev/tty.usbmodem5B141129871}"
: "${DEFAULT_LEADER_PORT:=/dev/tty.usbmodem5B141128171}"
: "${DEFAULT_CAM_INDEX:=0}"
: "${DEFAULT_CAM_WIDTH:=640}"
: "${DEFAULT_CAM_HEIGHT:=480}"
: "${DEFAULT_CAM_FPS:=30}"

teleop_common_init() {
  # Mute the wgpu/Vulkan log spam that floods the terminal when display_data=true.
  # (Sanity-session learning, 2026-04-28 — see notes/sanity_results.md.)
  export RUST_LOG="${RUST_LOG:-error}"
}

teleop_assert_hf_auth() {
  local expected_user="$1"
  if ! command -v hf >/dev/null 2>&1; then
    printf '[teleop_common] ERROR: `hf` CLI not on PATH. Activate the lerobot env first.\n' >&2
    return 1
  fi
  # hf auth whoami emits ANSI color codes even when piped; substring-match the
  # username instead of parsing fields. Exit code is unreliable (returns 0 even
  # when not logged in on some hf-cli versions).
  local whoami_out
  whoami_out="$(hf auth whoami 2>&1)"
  if printf '%s' "$whoami_out" | grep -q "Not logged in\|not authenticated\|run.*login"; then
    printf '[teleop_common] ERROR: not logged in to HF. Run `hf auth login`.\n' >&2
    return 1
  fi
  if ! printf '%s' "$whoami_out" | grep -q "$expected_user"; then
    printf '[teleop_common] WARNING: hf user does not appear to be %s. Output was:\n%s\n' \
      "$expected_user" "$whoami_out" >&2
  fi
}

# Lerobot-record / gym_manipulator refuse to start if the local cache dir for
# this repo_id already exists (FileExistsError). This wipes the stale skeleton
# from an aborted prior run.
teleop_clear_stale_cache() {
  local repo_id="$1"
  local cache_dir="${HOME}/.cache/huggingface/lerobot/${repo_id}"
  if [[ -d "$cache_dir" ]]; then
    printf '[teleop_common] removing stale lerobot cache: %s\n' "$cache_dir"
    rm -rf "$cache_dir"
  fi
}

# Build the camera JSON arg for lerobot-record. The wrist key is the policy-side
# convention; SmolVLA uses camera1 (via --dataset.rename_map), ACT and HIL-SERL
# use wrist directly.
teleop_camera_json() {
  local key="${1:-wrist}"
  printf '{"%s": {"type": "opencv", "index_or_path": %s, "width": %s, "height": %s, "fps": %s, "warmup_s": 3}}' \
    "$key" "$DEFAULT_CAM_INDEX" "$DEFAULT_CAM_WIDTH" "$DEFAULT_CAM_HEIGHT" "$DEFAULT_CAM_FPS"
}
