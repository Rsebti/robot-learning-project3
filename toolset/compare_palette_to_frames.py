"""
compare_palette_to_frames.py — side-by-side: real frame vs extracted swatch.

For each entity in toolset/configs/color_palette.yaml, picks a representative
frame from the Eval-2 demos and renders:
   [ frame thumbnail ] [ extracted color swatch ] [ hex / RGB ]

Useful to eyeball whether the bowl really IS grayish, whether green really is
that dark, etc.

Output: toolset/figs/color_palette_vs_frames.png

Usage:
  python toolset/compare_palette_to_frames.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from detect_cube_cv import HSV_RANGES, read_frame_at  # noqa: E402
from extract_color_palette import (load_episode_videos, cube_pixels,        # noqa: E402
                                     bowl_pixels, table_pixels, COLORS)


def main():
    repo_id = "osammotg1/projet3-eval2-v1-tom-hugo"
    here = Path(__file__).parent
    palette_path = here / "configs" / "color_palette.yaml"
    if not palette_path.exists():
        raise SystemExit("Run extract_color_palette.py first")
    with open(palette_path) as f:
        palette = yaml.safe_load(f)

    # Reuse verification CSV to find a clean demo per color
    vdf = pd.read_csv(here / "configs" / "eval2_verification.csv")
    matched = vdf[vdf["match"] == True]

    print("Loading episode -> video mapping ...")
    ep_videos = load_episode_videos(repo_id)

    # Pick one representative episode per cube color (first matched one)
    rep_eps: dict[str, int] = {}
    for c in COLORS:
        eps = matched[matched["target"] == c]["episode"].tolist()
        if eps:
            rep_eps[c] = int(eps[0])

    # For bowl: same as the yellow demo (any frame with a bowl works)
    # For table: any pre-grasp frame works. Use blue's demo.
    rep_eps["bowl"]  = rep_eps.get("yellow", 0)
    rep_eps["table"] = rep_eps.get("blue", 5)

    video_cache: dict[tuple[int, int], str] = {}

    def get_frame(ep_idx, last=True, fps=30):
        row = ep_videos.loc[ep_idx]
        chunk = int(row["videos/observation.images.wrist/chunk_index"])
        fidx  = int(row["videos/observation.images.wrist/file_index"])
        ts_end = float(row["videos/observation.images.wrist/to_timestamp"]) - 1 / fps
        ts_pre = float(row["videos/observation.images.wrist/from_timestamp"]) + 1.0
        key = (chunk, fidx)
        if key not in video_cache:
            video_cache[key] = hf_hub_download(
                repo_id,
                f"videos/observation.images.wrist/chunk-{chunk:03d}/file-{fidx:03d}.mp4",
                repo_type="dataset",
            )
        ts = ts_end if last else ts_pre
        return read_frame_at(video_cache[key], ts)

    entities = COLORS + ["bowl", "table"]
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.flatten()

    for i, label in enumerate(entities):
        if label not in palette:
            continue
        ep = rep_eps.get(label, 0)
        bgr = get_frame(ep, last=(label != "table"))
        if bgr is None:
            continue

        rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Highlight the masked region with a contour
        if label in COLORS:
            # Cube mask
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in HSV_RANGES[label]:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                cv2.drawContours(rgb_img, [largest], -1, (255, 255, 255), 2)
        elif label == "bowl":
            # White blob outline
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            white = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 35, 255]))
            white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(rgb_img,
                                  [max(contours, key=cv2.contourArea)],
                                  -1, (0, 255, 255), 2)
        # for "table" we don't outline -- it's a scattered region

        ax_img = axes[i * 2]
        ax_swatch = axes[i * 2 + 1]

        ax_img.imshow(rgb_img)
        ax_img.set_title(f"ep {ep:03d} ({label})", fontsize=10)
        ax_img.axis("off")

        rgb = np.array(palette[label]["rgb_median"]) / 255.0
        ax_swatch.add_patch(mpatches.Rectangle((0, 0), 1, 1,
                                                facecolor=rgb, edgecolor="black"))
        ax_swatch.set_xlim(0, 1)
        ax_swatch.set_ylim(0, 1)
        ax_swatch.text(0.5, 0.5,
                        f"{label}\n{palette[label]['hex']}\nRGB {tuple(palette[label]['rgb_median'])}",
                        ha="center", va="center", fontsize=11, fontweight="bold",
                        color="white" if rgb.mean() < 0.5 else "black")
        ax_swatch.set_aspect("equal")
        ax_swatch.axis("off")

    plt.suptitle("Extracted palette vs. real frame (white outline = cube mask, "
                  "cyan = bowl mask, table from low-S pre-grasp pixels)",
                  fontsize=11, y=0.995)
    plt.tight_layout()
    out_png = here / "figs" / "color_palette_vs_frames.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
