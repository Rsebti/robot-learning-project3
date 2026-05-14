"""
extract_color_palette.py — measure the actual RGB/HSV of every scene entity
(6 cube colors, bowl, table) from the successful Eval-2 demo frames.

The TA only gives a color index for the table (~#B8ADA9). Everything else has
to be calibrated under our real lighting. This script does exactly that:

  - For each cube color, pick `--n_samples` matched episodes (from
    `configs/eval2_verification.csv`), open the last frame, erode the
    color mask to drop boundary pixels, and aggregate the interior pixels.
  - Bowl: largest white blob in the same last frames, with the cube pixels
    subtracted, eroded.
  - Table: pre-grasp frame of one demo per color, low-saturation pixels
    that aren't part of the bowl/cubes/gripper.

Outputs:
  toolset/configs/color_palette.yaml          mean/median RGB+HSV + count per entity
  toolset/figs/color_palette.png              swatch chart for visual validation

Usage:
  python toolset/extract_color_palette.py
  python toolset/extract_color_palette.py --n_samples 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from detect_cube_cv import HSV_RANGES, read_frame_at  # noqa: E402

# Make sure orange + red ranges match the eval-2 calibration
HSV_RANGES["red"]    = [((0, 120, 60), (4, 255, 255)),
                        ((172, 120, 60), (180, 255, 255))]
HSV_RANGES["orange"] = [((4, 120, 80), (22, 255, 255))]
COLORS = ["yellow", "orange", "red", "blue", "green", "violet"]


# --------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------

def cube_pixels(bgr: np.ndarray, color: str, erode_iters: int = 3) -> np.ndarray | None:
    """Return BGR pixels of the cube interior (eroded mask)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    if mask.sum() < 200 * 255:   # need at least ~200 pixels
        return None
    # Keep largest contour, erode to drop boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 400:
        return None
    blob = np.zeros_like(mask)
    cv2.drawContours(blob, [largest], -1, 255, -1)
    blob = cv2.erode(blob, np.ones((3, 3), np.uint8), iterations=erode_iters)
    if blob.sum() == 0:
        return None
    return bgr[blob > 0]   # (N, 3) BGR


def bowl_pixels(bgr: np.ndarray, cube_color: str | None = None,
                erode_iters: int = 4) -> np.ndarray | None:
    """Return BGR pixels inside the white bowl, excluding the cube.

    Tight white threshold (V>=190, S<=35) to avoid pulling in the table
    which sits around V~150, S~10.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 35, 255]))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 2000:
        return None
    bowl = np.zeros_like(white)
    cv2.drawContours(bowl, [largest], -1, 255, -1)
    bowl = cv2.erode(bowl, np.ones((3, 3), np.uint8), iterations=erode_iters)
    if cube_color is not None:
        # Subtract cube color from the bowl area
        cube_mask = np.zeros_like(white)
        for lo, hi in HSV_RANGES[cube_color]:
            cube_mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        cube_mask = cv2.dilate(cube_mask, np.ones((3, 3), np.uint8), iterations=2)
        bowl = cv2.bitwise_and(bowl, cv2.bitwise_not(cube_mask))
    if bowl.sum() == 0:
        return None
    return bgr[bowl > 0]


def table_pixels(bgr: np.ndarray) -> np.ndarray | None:
    """Return BGR pixels that look like the table (low-S medium gray).

    Table sits around V~150, S~10. We cap V at 185 to keep brighter bowl
    regions out, even when the bowl appears in pre-grasp frames.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    table = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 30, 185]))
    # Drop tiny scattered regions
    table = cv2.morphologyEx(table, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # Drop the bowl (largest connected white blob) - the table is the
    # SECOND biggest neutral blob, or the surrounding pixels around the bowl.
    contours, _ = cv2.findContours(table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Sort by area, take everything except the biggest (which is probably the bowl)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours_sorted) == 1:
        # only bowl detected, no separate table -> give up
        return None
    table_mask = np.zeros_like(table)
    for c in contours_sorted[1:]:
        if cv2.contourArea(c) > 200:
            cv2.drawContours(table_mask, [c], -1, 255, -1)
    if table_mask.sum() == 0:
        return None
    return bgr[table_mask > 0]


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def aggregate(pixels_bgr: np.ndarray) -> dict:
    """mean/median in BGR, RGB, HSV."""
    bgr = pixels_bgr.astype(float)
    rgb = bgr[:, ::-1]
    hsv = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    return {
        "n_pixels":  int(len(pixels_bgr)),
        "rgb_mean":   [round(float(v), 1) for v in rgb.mean(axis=0)],
        "rgb_median": [int(v) for v in np.median(rgb, axis=0)],
        "hsv_mean":   [round(float(v), 1) for v in hsv.mean(axis=0)],
        "hsv_median": [int(v) for v in np.median(hsv, axis=0)],
        "hex":        "#{:02X}{:02X}{:02X}".format(*[int(v) for v in np.median(rgb, axis=0)]),
    }


# --------------------------------------------------------------------------
# Episode -> video file lookup
# --------------------------------------------------------------------------

def load_episode_videos(repo_id: str) -> pd.DataFrame:
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    meta_files = sorted(f for f in files
                        if f.startswith("meta/episodes/") and f.endswith(".parquet"))
    cols = ["episode_index", "tasks", "length",
            "videos/observation.images.wrist/chunk_index",
            "videos/observation.images.wrist/file_index",
            "videos/observation.images.wrist/from_timestamp",
            "videos/observation.images.wrist/to_timestamp"]
    dfs = []
    for f in meta_files:
        p = hf_hub_download(repo_id, f, repo_type="dataset")
        dfs.append(pd.read_parquet(p, columns=cols))
    return pd.concat(dfs, ignore_index=True).set_index("episode_index")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id",   default="osammotg1/projet3-eval2-v1-tom-hugo")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of demos per cube color to sample from")
    parser.add_argument("--fps",       type=int, default=30)
    args = parser.parse_args()

    here = Path(__file__).parent

    # Source: matched episodes from yesterday's verification
    verify_csv = here / "configs" / "eval2_verification.csv"
    if not verify_csv.exists():
        raise SystemExit(f"Missing {verify_csv}. Run verify_eval2_demos.py first.")
    vdf = pd.read_csv(verify_csv)
    matched = vdf[vdf["match"] == True]
    print(f"[palette] {len(matched)}/{len(vdf)} matched episodes available")

    print("[palette] Loading episode -> video mapping ...")
    ep_videos = load_episode_videos(args.repo_id)

    # Buffer pixels per entity
    samples = {c: [] for c in COLORS}
    samples["bowl"]  = []
    samples["table"] = []

    video_cache: dict[tuple[int, int], str] = {}

    for color in COLORS:
        eps = matched[matched["target"] == color]["episode"].head(args.n_samples).tolist()
        print(f"\n[palette] {color}: sampling {len(eps)} demo(s) -> {eps}")
        for ep_idx in eps:
            if ep_idx not in ep_videos.index:
                continue
            row = ep_videos.loc[ep_idx]
            chunk = int(row["videos/observation.images.wrist/chunk_index"])
            fidx  = int(row["videos/observation.images.wrist/file_index"])
            ts_end = float(row["videos/observation.images.wrist/to_timestamp"]) - 1 / args.fps
            ts_pre = float(row["videos/observation.images.wrist/from_timestamp"]) + 1.0  # 1s in

            key = (chunk, fidx)
            if key not in video_cache:
                video_cache[key] = hf_hub_download(
                    args.repo_id,
                    f"videos/observation.images.wrist/chunk-{chunk:03d}/file-{fidx:03d}.mp4",
                    repo_type="dataset",
                )

            # Last frame -> cube + bowl
            bgr_end = read_frame_at(video_cache[key], ts_end)
            if bgr_end is not None:
                cp = cube_pixels(bgr_end, color)
                if cp is not None:
                    samples[color].append(cp)
                    print(f"   ep {ep_idx:03d}: cube {len(cp):5d} px")
                bp = bowl_pixels(bgr_end, cube_color=color)
                if bp is not None:
                    samples["bowl"].append(bp)
            # Early frame -> table
            bgr_pre = read_frame_at(video_cache[key], ts_pre)
            if bgr_pre is not None:
                tp = table_pixels(bgr_pre)
                if tp is not None:
                    samples["table"].append(tp)

    # --------- aggregate ---------
    palette = {}
    for label, sample_list in samples.items():
        if not sample_list:
            print(f"\n[palette] !! no samples for {label}")
            continue
        all_pixels = np.concatenate(sample_list, axis=0)
        # Subsample if huge
        if len(all_pixels) > 200_000:
            idx = np.random.choice(len(all_pixels), 200_000, replace=False)
            all_pixels = all_pixels[idx]
        palette[label] = aggregate(all_pixels)

    out_yaml = here / "configs" / "color_palette.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump(palette, f, sort_keys=False)
    print(f"\n[palette] Saved -> {out_yaml}")

    # --------- summary print ---------
    print(f"\n{'label':>8s} | n_pix |  RGB median  | HSV median | hex")
    print("-" * 70)
    for label, p in palette.items():
        rgb = p["rgb_median"]; hsv = p["hsv_median"]
        print(f"{label:>8s} | {p['n_pixels']:>5d} | "
              f"({rgb[0]:3d},{rgb[1]:3d},{rgb[2]:3d}) | "
              f"({hsv[0]:3d},{hsv[1]:3d},{hsv[2]:3d}) | {p['hex']}")

    # --------- visualization ---------
    fig, ax = plt.subplots(figsize=(11, 3))
    labels = list(palette.keys())
    for i, label in enumerate(labels):
        rgb = np.array(palette[label]["rgb_median"]) / 255.0
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=rgb, edgecolor="black"))
        ax.text(i + 0.5, -0.1, label, ha="center", va="top",
                fontsize=11, fontweight="bold")
        ax.text(i + 0.5, 1.1, palette[label]["hex"], ha="center", va="bottom",
                fontsize=9, family="monospace")
        ax.text(i + 0.5, 0.5,
                f"RGB {tuple(palette[label]['rgb_median'])}\nHSV {tuple(palette[label]['hsv_median'])}",
                ha="center", va="center", fontsize=8,
                color="white" if rgb.mean() < 0.5 else "black")
    ax.set_xlim(0, len(labels))
    ax.set_ylim(-0.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Scene color palette (calibrated under current lighting)")
    out_png = here / "figs" / "color_palette.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[palette] Plot  -> {out_png}")


if __name__ == "__main__":
    main()
