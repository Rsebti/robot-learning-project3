#!/usr/bin/env bash
# Usage: bash friend_handoff/examples/calibrate_six_cubes.sh path/to/snap.png
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMG="${1:?image with all 6 cube colors}"
cd "$ROOT"
python -m toolset.perception.calibrate_colors_from_images "$IMG" --viz
