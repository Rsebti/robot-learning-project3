"""
verify_eval2_demos.py - check that each Eval-2 demo ends with the
*correct target color* in the bowl.

For each of the 101 episodes:
  1. Look up the target color from the task description (tasks.parquet)
  2. Read the LAST frame from the wrist camera video (using episode meta
     to find the right video chunk and timestamp)
  3. Run HSV detection for all 6 candidate colors
  4. The color with the largest blob is the cube in the bowl
  5. Flag any mismatch between detected color and target color

Outputs:
  configs/eval2_verification.csv      one row per episode
  figs/eval2_mismatches/ep_XXX.png    crop + label, only for mismatches

Usage:
  python teleop/verify_eval2_demos.py
  python teleop/verify_eval2_demos.py --repo_id osammotg1/projet3-eval2-v1-tom-hugo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from detect_cube_cv import HSV_RANGES, read_frame_at  # noqa: E402


COLORS = list(HSV_RANGES.keys())
# Tighten red to H=0-4 only; orange takes H=4-22 (the "orange" cube here
# reads H=5-7, which is reddish-orange in HSV).
HSV_RANGES["red"]    = [((0, 120, 60), (4, 255, 255)),
                        ((172, 120, 60), (180, 255, 255))]
HSV_RANGES["orange"] = [((4, 120, 80), (22, 255, 255))]
COLORS = ["yellow", "orange", "red", "blue", "green", "violet"]


def extract_target_color(task_desc: str) -> str:
    desc = task_desc.lower()
    for c in COLORS:
        if c in desc:
            return c
    return "unknown"


def find_bowl_region(bgr: np.ndarray) -> np.ndarray | None:
    """Return a filled mask of the white bowl interior (largest white blob)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # White-ish: low S, mid-to-high V
    white_mask = cv2.inRange(hsv, np.array([0, 0, 130]), np.array([180, 80, 255]))
    kernel = np.ones((7, 7), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 2000:
        return None
    filled = np.zeros_like(white_mask)
    cv2.drawContours(filled, [largest], -1, 255, -1)
    # Slightly dilate so the cube-bowl boundary pixels are included
    filled = cv2.dilate(filled, np.ones((5, 5), np.uint8), iterations=1)
    return filled


def best_color_blob(bgr: np.ndarray):
    """Return (best_color, best_area, mask, bbox).

    Strategy: find the white bowl region, then for each candidate color,
    count the area of that color WITHIN the bowl. Picks the dominant one.
    For orange vs red disambiguation we also look at the mean hue of the
    in-bowl colored pixels and remap to orange if it's clearly above ~8.
    """
    bowl = find_bowl_region(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    best = (None, 0, None, None)
    kernel = np.ones((3, 3), np.uint8)

    for color in COLORS:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in HSV_RANGES[color]:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        if bowl is not None:
            mask = cv2.bitwise_and(mask, bowl)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area > best[1]:
            bbox = cv2.boundingRect(largest)
            best = (color, area, mask, bbox)

    # If nothing detected within the bowl, fall back to full-image search
    if best[0] is None and bowl is not None:
        for color in COLORS:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in HSV_RANGES[color]:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(largest))
            if area > best[1]:
                bbox = cv2.boundingRect(largest)
                best = (color, area, mask, bbox)
    return best


def load_all_episodes_meta(repo_id: str) -> pd.DataFrame:
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    meta_files = sorted(f for f in files if f.startswith("meta/episodes/") and f.endswith(".parquet"))
    cols_needed = [
        "episode_index", "tasks", "length",
        "videos/observation.images.wrist/chunk_index",
        "videos/observation.images.wrist/file_index",
        "videos/observation.images.wrist/from_timestamp",
        "videos/observation.images.wrist/to_timestamp",
    ]
    dfs = []
    for f in meta_files:
        p = hf_hub_download(repo_id, f, repo_type="dataset")
        df = pd.read_parquet(p, columns=cols_needed)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True).sort_values("episode_index").reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="osammotg1/projet3-eval2-v1-tom-hugo")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    print(f"[verify] Loading episode metadata for {args.repo_id} ...")
    meta = load_all_episodes_meta(args.repo_id)
    print(f"[verify] {len(meta)} episodes\n")

    # Cache downloaded video files
    video_cache: dict[tuple[int, int], str] = {}

    rows = []
    mismatches = []
    for _, ep in meta.iterrows():
        ep_idx = int(ep["episode_index"])
        tasks_field = ep["tasks"]
        task_desc = tasks_field[0] if hasattr(tasks_field, "__iter__") else str(tasks_field)
        target = extract_target_color(task_desc)
        chunk_idx = int(ep["videos/observation.images.wrist/chunk_index"])
        file_idx  = int(ep["videos/observation.images.wrist/file_index"])
        ts_to     = float(ep["videos/observation.images.wrist/to_timestamp"])
        last_ts   = max(0.0, ts_to - 1.0 / args.fps)   # last frame timestamp

        # Download video (cached)
        key = (chunk_idx, file_idx)
        if key not in video_cache:
            video_cache[key] = hf_hub_download(
                args.repo_id,
                f"videos/observation.images.wrist/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4",
                repo_type="dataset",
            )

        bgr = read_frame_at(video_cache[key], last_ts)
        if bgr is None:
            print(f"  ep {ep_idx:03d}: failed to read last frame at t={last_ts:.3f}s")
            rows.append({"episode": ep_idx, "target": target, "detected": None, "area": 0, "match": False})
            continue

        color, area, _mask, bbox = best_color_blob(bgr)
        match = (color == target)
        rows.append({
            "episode":  ep_idx,
            "target":   target,
            "detected": color,
            "area":     int(area),
            "match":    bool(match),
        })

        flag = "OK" if match else "!! MISMATCH"
        print(f"  ep {ep_idx:03d}: target={target:7s}  detected={str(color):7s}  area={int(area):5d}  {flag}")

        # Save tagged last-frame for EVERY episode (semi-auto review)
        vis = bgr.copy()
        if bbox is not None:
            x, y, w, h = bbox
            box_color = (0, 255, 0) if match else (0, 0, 255)
            cv2.rectangle(vis, (x, y), (x + w, y + h), box_color, 2)
        # Banner at the top
        banner_color = (0, 200, 0) if match else (0, 0, 200)
        cv2.rectangle(vis, (0, 0), (640, 30), (40, 40, 40), -1)
        text = f"ep {ep_idx:03d}  target={target}  detected={color}  area={int(area)}"
        cv2.putText(vis, text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, banner_color, 2)
        out_dir = Path(__file__).parent / "figs" / "debug" / "eval2_color_check"
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"ep_{ep_idx:03d}.png"), vis)

        if not match:
            mismatches.append(ep_idx)

    # Save CSV
    out_csv = Path(__file__).parent / "configs" / "eval2_verification.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[verify] CSV saved: {out_csv}")

    # Summary
    n = len(rows)
    n_ok = sum(1 for r in rows if r["match"])
    n_bad = n - n_ok
    print(f"\n[verify] Summary: {n_ok}/{n} match, {n_bad} mismatch")
    if mismatches:
        print(f"[verify] Mismatches at episodes: {mismatches}")
        print(f"[verify] Visualizations: figs/eval2_mismatches/")


if __name__ == "__main__":
    main()
