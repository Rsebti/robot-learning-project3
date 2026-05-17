#!/usr/bin/env bash
# teleop/record_hilserl_demos.sh — record EE-space teleop demos for HIL-SERL.
#
# Wraps `python -m lerobot.rl.gym_manipulator --config_path ...` with our
# Mac-specific defaults (ports, cache cleanup, HF auth check). The actual
# recording behavior — robot/teleop type, EE bounds, IK pipeline, demo count,
# task string — is driven by the config JSON, not env vars.
#
# Usage:
#   bash teleop/record_hilserl_demos.sh                        # uses defaults
#   NUM_EPISODES=20 bash teleop/record_hilserl_demos.sh        # override demo count
#   CONFIG_PATH=sim/hilserl/configs/my_variant.json \
#     bash teleop/record_hilserl_demos.sh                      # different config
#   FOLLOWER_PORT=/dev/ttyACM0 LEADER_PORT=/dev/ttyACM1 \
#     bash teleop/record_hilserl_demos.sh                      # non-Mac ports
#
# Pre-reqs:
#   1. `bash sim/hilserl/configs/env_config_so101.json` exists and has
#      `processor.inverse_kinematics.end_effector_bounds` filled with real
#      values from `lerobot-find-joint-limits`.
#   2. The lerobot env is active (or `PY` is set to its python binary).
#   3. `hf auth login` done as osammotg1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=teleop/_common.sh
source "$SCRIPT_DIR/_common.sh"
teleop_common_init

PY="${PY:-/Users/admin/miniforge3/bin/python3.13}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/sim/hilserl/configs/env_config_so101.json}"
EXPECTED_HF_USER="${EXPECTED_HF_USER:-osammotg1}"

# Optional overrides that punch through into the config JSON via jq.
# Set to non-empty to override; leave unset/empty to use the config value.
NUM_EPISODES="${NUM_EPISODES:-}"
REPO_ID="${REPO_ID:-}"
FOLLOWER_PORT="${FOLLOWER_PORT:-$DEFAULT_FOLLOWER_PORT}"
LEADER_PORT="${LEADER_PORT:-$DEFAULT_LEADER_PORT}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  printf '[record_hilserl] ERROR: config not found at %s\n' "$CONFIG_PATH" >&2
  exit 1
fi

if [[ ! -x "$PY" ]]; then
  printf '[record_hilserl] ERROR: python binary not executable: %s\n' "$PY" >&2
  printf '[record_hilserl] override with PY=/path/to/python or activate the lerobot env first.\n' >&2
  exit 1
fi

teleop_assert_hf_auth "$EXPECTED_HF_USER"

# Pull the repo_id and num_episodes from the config (or the env-var overrides)
# to do a focused cache clean. jq is in the lerobot conda env already.
EFFECTIVE_REPO_ID="$(jq -r '.dataset.repo_id' "$CONFIG_PATH")"
if [[ -n "$REPO_ID" ]]; then
  EFFECTIVE_REPO_ID="$REPO_ID"
fi
teleop_clear_stale_cache "$EFFECTIVE_REPO_ID"

# Build the override args. gym_manipulator accepts dotted overrides via draccus.
OVERRIDES=(
  "--env.robot.port=$FOLLOWER_PORT"
  "--env.teleop.port=$LEADER_PORT"
)
if [[ -n "$NUM_EPISODES" ]]; then
  OVERRIDES+=("--dataset.num_episodes_to_record=$NUM_EPISODES")
fi
if [[ -n "$REPO_ID" ]]; then
  OVERRIDES+=("--dataset.repo_id=$REPO_ID")
fi

printf '[record_hilserl] config=%s\n'      "$CONFIG_PATH"
printf '[record_hilserl] python=%s\n'      "$PY"
printf '[record_hilserl] repo_id=%s\n'     "$EFFECTIVE_REPO_ID"
printf '[record_hilserl] follower=%s\n'    "$FOLLOWER_PORT"
printf '[record_hilserl] leader=%s\n'      "$LEADER_PORT"
printf '[record_hilserl] overrides: %s\n'  "${OVERRIDES[*]}"
printf '\n'

cd "$REPO_ROOT"
exec "$PY" -m lerobot.rl.gym_manipulator \
  --config_path "$CONFIG_PATH" \
  "${OVERRIDES[@]}"
