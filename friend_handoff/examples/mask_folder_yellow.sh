#!/usr/bin/env bash
# Usage: bash friend_handoff/examples/mask_folder_yellow.sh deploy/_snaps/my_frames yellow
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE_DIR="${1:?image_dir}"
COLOR="${2:-yellow}"
cd "$ROOT"
python -m toolset.perception.cube_mask_batch --image_dir "$IMAGE_DIR" --color "$COLOR"
echo "Open: ${IMAGE_DIR}/cube_masks/gallery.html"
