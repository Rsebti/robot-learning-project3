"""
map_cube_positions_eval2.py - workspace coverage map for Eval-2 demos.

For each episode (101 total in osammotg1/projet3-eval2-v1-tom-hugo):
  1. Find grasp moment from gripper signal
  2. Target cube xy = fingertip xy at grasp (FK)
  3. Grasp yaw = jaw-axis yaw in base frame (FK rotation)
  4. Distractor color = largest non-target blob in a pre-grasp frame (CV)
  5. Distractor side = compare image-u of distractor vs target -> world side via
     pixel back-projection through configs/camera_calibration.yaml

Plot: bi-color circle per episode at target xy.
  - Split line oriented at grasp yaw (parallel to gripper jaw axis)
  - Half toward the distractor side = distractor color
  - Half away = target color
  - Episode # annotated inside

Outputs:
  toolset/configs/cube_positions_eval2.csv
  toolset/figs/positions/eval2_map.png

Usage:
  python toolset/map_cube_positions_eval2.py
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
from matplotlib.patches import Wedge

sys.path.insert(0, str(Path(__file__).parent))
from probe_bowl_position import fk_wrist, fk_tip            # noqa: E402
from detect_cube_cv import (HSV_RANGES, read_frame_at,        # noqa: E402
                             pixel_to_base as cv_pixel_to_base)
from map_cube_positions import (_COLOR_MAP, _extract_color,    # noqa: E402
                                  find_grasp_frame)


# --------------------------------------------------------------------------
# Make sure 'orange' is in HSV_RANGES with the tuned eval-2 range
# --------------------------------------------------------------------------
HSV_RANGES["red"]    = [((0, 120, 60), (4, 255, 255)),
                        ((172, 120, 60), (180, 255, 255))]
HSV_RANGES["orange"] = [((4, 120, 80), (22, 255, 255))]
ALL_COLORS = ["yellow", "orange", "red", "blue", "green", "violet"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def grasp_yaw_in_base(q_rad: list[float]) -> float:
    """Yaw angle (radians) of the gripper jaw axis in the robot base xy plane.

    Uses the local +Y axis of the wrist frame after FK -- after wrist_roll
    rotation this approximately points along the jaw closing direction.
    """
    T = fk_wrist(q_rad)
    jaw_in_base = T[:3, 1]   # second column = local +Y in base coords
    return float(np.arctan2(jaw_in_base[1], jaw_in_base[0]))


def detect_other_blob(bgr: np.ndarray, target_color: str):
    """Find the largest colored blob whose color != target_color.

    Returns (color, area, (u, v)) or (None, 0, None).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    best = (None, 0, None)
    for color in ALL_COLORS:
        if color == target_color:
            continue
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in HSV_RANGES[color]:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < 400:
            continue
        if area > best[1]:
            M = cv2.moments(largest)
            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])
            best = (color, area, (u, v))
    return best


def detect_target_blob(bgr: np.ndarray, target_color: str):
    """Largest blob of the target color. (u, v) or None."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES[target_color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 400:
        return None
    M = cv2.moments(largest)
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


# --------------------------------------------------------------------------
# Episode metadata
# --------------------------------------------------------------------------

def load_episodes_meta(repo_id: str) -> pd.DataFrame:
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    meta_files = sorted(f for f in files
                        if f.startswith("meta/episodes/") and f.endswith(".parquet"))
    cols = [
        "episode_index", "tasks", "length",
        "videos/observation.images.wrist/chunk_index",
        "videos/observation.images.wrist/file_index",
        "videos/observation.images.wrist/from_timestamp",
        "videos/observation.images.wrist/to_timestamp",
    ]
    dfs = []
    for f in meta_files:
        p = hf_hub_download(repo_id, f, repo_type="dataset")
        dfs.append(pd.read_parquet(p, columns=cols))
    return pd.concat(dfs, ignore_index=True).sort_values("episode_index").reset_index(drop=True)


def load_all_data(repo_id: str) -> pd.DataFrame:
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    data_files = sorted(f for f in files
                        if f.startswith("data/") and f.endswith(".parquet"))
    dfs = []
    for f in data_files:
        p = hf_hub_download(repo_id, f, repo_type="dataset")
        dfs.append(pd.read_parquet(p))
    return pd.concat(dfs, ignore_index=True)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_bicolor_marker(ax, xy, c_target, c_distractor, split_angle_deg,
                        target_on_positive_side: bool, radius=0.011):
    """
    Draw a circle split into two halves by a line at angle split_angle_deg.
      - the "positive side" of the split line is at angles
        (split_angle_deg, split_angle_deg + 180)
      - if target_on_positive_side -> target color is there, else distractor.
    """
    c1 = _COLOR_MAP.get(c_target,     "#888")
    c2 = _COLOR_MAP.get(c_distractor, "#888")
    if target_on_positive_side:
        col_pos, col_neg = c1, c2
    else:
        col_pos, col_neg = c2, c1
    w_pos = Wedge(xy, radius, split_angle_deg, split_angle_deg + 180,
                  facecolor=col_pos, edgecolor="black", linewidth=0.5, zorder=3)
    w_neg = Wedge(xy, radius, split_angle_deg + 180, split_angle_deg + 360,
                  facecolor=col_neg, edgecolor="black", linewidth=0.5, zorder=3)
    ax.add_patch(w_pos)
    ax.add_patch(w_neg)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="osammotg1/projet3-eval2-v1-tom-hugo")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pre_grasp_sec", type=float, default=1.0,
                        help="Seconds before grasp to sample the CV detection frame")
    args = parser.parse_args()

    here = Path(__file__).parent

    # Camera calibration (from eval-1 self-cal; assume same camera)
    calib_path = here / "configs" / "camera_calibration.yaml"
    if not calib_path.exists():
        raise SystemExit(f"Missing {calib_path}. Run calibrate_camera_from_demos.py first.")
    with open(calib_path) as f:
        calib = yaml.safe_load(f)
    K              = np.array(calib["K"])
    T_cam_in_wrist = np.array(calib["T_cam_in_wrist"])
    z_table        = float(calib.get("z_table", 0.01))
    print(f"[eval2-map] Loaded calibration from {calib_path}")

    print(f"[eval2-map] Loading metadata + data from {args.repo_id} ...")
    meta = load_episodes_meta(args.repo_id)
    df   = load_all_data(args.repo_id)
    print(f"[eval2-map] {len(meta)} episodes, {len(df)} frames\n")

    video_cache: dict[tuple[int, int], str] = {}
    results = []

    for _, ep_meta in meta.iterrows():
        ep_idx     = int(ep_meta["episode_index"])
        task_desc  = ep_meta["tasks"][0] if hasattr(ep_meta["tasks"], "__iter__") else str(ep_meta["tasks"])
        target_col = _extract_color(task_desc)
        chunk_idx  = int(ep_meta["videos/observation.images.wrist/chunk_index"])
        file_idx   = int(ep_meta["videos/observation.images.wrist/file_index"])

        ep_rows = df[df["episode_index"] == ep_idx].sort_values("frame_index").reset_index(drop=True)
        if len(ep_rows) == 0:
            print(f"  ep {ep_idx:03d}: no data, skip")
            continue
        states = np.array(ep_rows["observation.state"].tolist())
        gripper = states[:, 5]

        grasp_f = find_grasp_frame(gripper, fps=args.fps)
        if grasp_f < 0:
            print(f"  ep {ep_idx:03d} ({target_col:7s}): no grasp detected, skip")
            continue

        q_rad_grasp = np.deg2rad(states[grasp_f, :5]).tolist()
        T = fk_tip(q_rad_grasp)
        target_x, target_y, target_z = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
        yaw = grasp_yaw_in_base(q_rad_grasp)

        # Pre-grasp frame for distractor CV
        pre_f = max(0, grasp_f - int(args.pre_grasp_sec * args.fps))
        ts_pre = float(ep_rows.iloc[pre_f]["timestamp"])
        q_rad_pre = np.deg2rad(states[pre_f, :5]).tolist()

        key = (chunk_idx, file_idx)
        if key not in video_cache:
            video_cache[key] = hf_hub_download(
                args.repo_id,
                f"videos/observation.images.wrist/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4",
                repo_type="dataset",
            )
        bgr = read_frame_at(video_cache[key], ts_pre)
        distractor_col = None
        target_on_pos = True  # default arbitrary side
        if bgr is not None:
            distractor_col, dist_area, dist_pixel = detect_other_blob(bgr, target_col)
            target_pixel = detect_target_blob(bgr, target_col)
            if distractor_col and dist_pixel and target_pixel:
                # Back-project both to base frame, then decide which side of
                # the line through target_xy with normal (cos(yaw+90), sin(yaw+90))
                # the distractor is on.
                d_base = cv_pixel_to_base(*dist_pixel, K, T_cam_in_wrist, q_rad_pre, z_table=z_table)
                t_base = cv_pixel_to_base(*target_pixel, K, T_cam_in_wrist, q_rad_pre, z_table=z_table)
                if d_base is not None and t_base is not None:
                    # vector from target to distractor in xy
                    dx, dy = d_base[0] - t_base[0], d_base[1] - t_base[1]
                    # Side of the split line (split line direction = yaw):
                    # normal to the line points at angle yaw+90
                    nx, ny = -np.sin(yaw), np.cos(yaw)
                    side = dx * nx + dy * ny
                    # If distractor is on the "positive" side, target goes opposite.
                    target_on_pos = (side < 0)

        results.append({
            "episode":           ep_idx,
            "target_color":      target_col,
            "distractor_color":  distractor_col or "unknown",
            "x":                 target_x,
            "y":                 target_y,
            "z":                 target_z,
            "grasp_frame":       int(grasp_f),
            "yaw_deg":           float(np.degrees(yaw)),
            "target_on_pos":     bool(target_on_pos),
        })

        print(f"  ep {ep_idx:03d} ({target_col:7s} vs {str(distractor_col):7s}): "
              f"xy=({target_x:+.3f},{target_y:+.3f}) yaw={np.degrees(yaw):+6.1f}deg")

    if not results:
        print("[eval2-map] No usable episodes -- aborting.")
        return

    csv_path = here / "configs" / "cube_positions_eval2.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n[eval2-map] CSV -> {csv_path}")

    # ---------------- plot ----------------
    xs = np.array([r["x"] for r in results])
    ys = np.array([r["y"] for r in results])

    fig, ax = plt.subplots(figsize=(10, 8))
    for r in results:
        plot_bicolor_marker(
            ax, (r["x"], r["y"]),
            c_target=r["target_color"], c_distractor=r["distractor_color"],
            split_angle_deg=r["yaw_deg"],
            target_on_positive_side=r["target_on_pos"],
        )
    for r in results:
        ax.annotate(str(r["episode"]), (r["x"], r["y"]),
                    fontsize=6, ha="center", va="center", color="white")

    pad = 0.03
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_xlabel("x (m) - forward from robot base")
    ax.set_ylabel("y (m) - lateral from robot base")
    ax.set_title(f"Eval-2 cube positions across {len(results)} demos\n"
                 f"(bi-color = target / distractor, split oriented by grasp yaw)")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # Legend: one swatch per color seen
    used_colors = sorted(set([r["target_color"] for r in results] +
                              [r["distractor_color"] for r in results]))
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=_COLOR_MAP.get(c, "#888"),
                           markeredgecolor="black", markersize=10, label=c)
               for c in used_colors]
    ax.legend(handles=handles, loc="best", framealpha=0.9)

    png_path = here / "figs" / "positions" / "eval2_map.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"[eval2-map] Plot -> {png_path}")
    print(f"[eval2-map] x range: [{xs.min():+.3f}, {xs.max():+.3f}]  span {np.ptp(xs):.3f} m")
    print(f"[eval2-map] y range: [{ys.min():+.3f}, {ys.max():+.3f}]  span {np.ptp(ys):.3f} m")


if __name__ == "__main__":
    main()
