"""
detect_cube_cv.py - first-frame cube detection via classical CV.

Pipeline (Approach B):
  1. Load frame 0 of an episode from the wrist camera video
  2. HSV color segmentation for that episode's cube color (from task_index)
  3. Find centroid pixel of the largest colored blob
  4. Back-project: pixel -> camera ray -> wrist frame -> base frame
  5. Intersect ray with table plane (z = 0) -> cube xy
  6. Compare with FK-derived grasp-time position from configs/cube_positions.csv

This is a SCAFFOLD: intrinsics K and extrinsics T_cam_in_wrist below are
educated guesses. Once one frame works, iterate the constants until the
detected xy matches the FK-derived xy across multiple episodes.

Usage:
    python teleop/detect_cube_cv.py --episode 0           # ep 0, yellow
    python teleop/detect_cube_cv.py --episode 8           # ep 8, blue
    python teleop/detect_cube_cv.py --episode 0 --frame 50  # later frame
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
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from probe_bowl_position import fk_wrist  # noqa: E402

# ---------------------------------------------------------------------------
# CAMERA MODEL
# Default values are kept as a fallback. If configs/camera_calibration.yaml
# exists (produced by calibrate_camera_from_demos.py), it overrides these.
# ---------------------------------------------------------------------------

# Fallback intrinsics: 60 deg HFOV at 640x480 -> fx = fy = 320 / tan(30 deg) ~ 554
K = np.array([
    [554.0,   0.0, 320.0],
    [  0.0, 554.0, 240.0],
    [  0.0,   0.0,   1.0],
])

# Fallback extrinsics: T_cam_in_wrist with camera +Z = wrist +X (forward).
R_CAM_IN_WRIST = np.array([
    [0,  0, 1],
    [-1, 0, 0],
    [0, -1, 0],
])
T_CAM_IN_WRIST = np.eye(4)
T_CAM_IN_WRIST[:3, :3] = R_CAM_IN_WRIST
T_CAM_IN_WRIST[:3, 3]  = [0.02, 0.0, 0.03]

Z_TABLE = 0.01  # default table height in base frame

_CALIB_PATH = Path(__file__).parent / "configs" / "camera_calibration.yaml"
if _CALIB_PATH.exists():
    with open(_CALIB_PATH) as _f:
        _cfg = yaml.safe_load(_f)
    K               = np.array(_cfg["K"])
    T_CAM_IN_WRIST  = np.array(_cfg["T_cam_in_wrist"])
    Z_TABLE         = float(_cfg.get("z_table", Z_TABLE))
    print(f"[cv] Loaded calibration from {_CALIB_PATH}")

# ---------------------------------------------------------------------------
# HSV ranges (OpenCV: H in [0,180], S/V in [0,255])
# ---------------------------------------------------------------------------
HSV_RANGES = {
    "yellow": [((20, 100, 100), (35, 255, 255))],
    "blue":   [((100, 120,  60), (130, 255, 255))],
    "green":  [((40,  60,  60), ( 80, 255, 255))],
    "violet": [((130, 60,  60), (160, 255, 255))],
    "red":    [((  0, 120,  60), ( 10, 255, 255)),
               ((170, 120,  60), (180, 255, 255))],  # red wraps around H
}

# Episode -> color mapping (from tasks.parquet)
EPISODE_COLOR = (
    ["yellow"] * 8 +
    ["blue"]   * 8 +
    ["green"]  * 8 +
    ["violet"] * 7 +    # 7 because ep 30 (violet) was removed
    ["red"]    * 8
)
# Episode -> chunk video file mapping (8 eps per video file, but ep30 removed)
def episode_to_video_file(ep_idx: int) -> int:
    # Original 40 eps were split 8/file. After removing original ep30,
    # the trimmed eps 31-38 correspond to original eps 32-39 in file-004.
    if ep_idx <= 7:
        return 0
    if ep_idx <= 15:
        return 1
    if ep_idx <= 23:
        return 2
    if ep_idx <= 30:
        return 3  # trimmed eps 24-30 (was 24-29 + 31 originally)
    return 4      # trimmed eps 31-38


# ---------------------------------------------------------------------------
# Cube detection
# ---------------------------------------------------------------------------

def detect_cube_pixel(bgr: np.ndarray, color: str):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    # Clean noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Largest blob
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask, None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 50:
        return None, mask, area
    M = cv2.moments(largest)
    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    return (u, v), mask, area


# ---------------------------------------------------------------------------
# Geometric back-projection: pixel -> base-frame point on table
# ---------------------------------------------------------------------------

def pixel_to_base(u, v, K, T_cam_in_wrist, q_rad, z_table=0.0):
    """Return (x, y, z) in base frame where the ray through (u, v) hits z=z_table."""
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([u, v, 1.0])    # ray direction in camera frame

    T_wrist_in_base = fk_wrist(q_rad)
    T_cam_in_base   = T_wrist_in_base @ T_cam_in_wrist
    cam_origin = T_cam_in_base[:3, 3]
    ray_base   = T_cam_in_base[:3, :3] @ ray_cam

    if abs(ray_base[2]) < 1e-6:
        return None
    t = (z_table - cam_origin[2]) / ray_base[2]
    if t < 0:
        return None
    return cam_origin + t * ray_base


# ---------------------------------------------------------------------------
# Video reader (timestamp seek via PyAV, robust for AV1 codec)
# ---------------------------------------------------------------------------

def read_frame_at(video_path: str, timestamp_sec: float) -> np.ndarray | None:
    import av
    container = av.open(video_path)
    stream = container.streams.video[0]
    target_pts = int(timestamp_sec / float(stream.time_base))
    container.seek(target_pts, stream=stream)
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        if frame.pts >= target_pts:
            arr = frame.to_ndarray(format="bgr24")
            container.close()
            return arr
    container.close()
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="hudela390/projet3-eval1-bowl1-v1-trimmed")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame",   type=int, default=0,
                        help="Frame index within the episode (0 = first)")
    args = parser.parse_args()

    color = EPISODE_COLOR[args.episode]
    chunk_file_idx = episode_to_video_file(args.episode)
    print(f"[cv] Episode {args.episode}: color={color}, video file=file-{chunk_file_idx:03d}.mp4")

    # 1. Parquet
    parquet_path = hf_hub_download(
        repo_id=args.repo_id,
        filename="data/chunk-000/file-000.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(parquet_path)
    ep = df[df["episode_index"] == args.episode].sort_values("frame_index").reset_index(drop=True)
    if args.frame >= len(ep):
        print(f"[cv] frame {args.frame} out of range (episode has {len(ep)} frames)")
        return
    row = ep.iloc[args.frame]
    timestamp_sec = float(row["timestamp"])
    state_deg = np.asarray(row["observation.state"], dtype=float)
    q_rad = np.deg2rad(state_deg[:5]).tolist()
    print(f"[cv] Frame {args.frame}: timestamp={timestamp_sec:.3f}s")
    print(f"[cv] Joints (deg): {state_deg}")

    # 2. Video
    video_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=f"videos/observation.images.wrist/chunk-000/file-{chunk_file_idx:03d}.mp4",
        repo_type="dataset",
    )
    frame_bgr = read_frame_at(video_path, timestamp_sec)
    if frame_bgr is None:
        print("[cv] Failed to read frame from video.")
        return
    print(f"[cv] Read frame: shape={frame_bgr.shape}")

    # 3. Detect
    pixel, mask, area = detect_cube_pixel(frame_bgr, color)
    if pixel is None:
        print(f"[cv] No {color} blob found (area below threshold). Tune HSV_RANGES['{color}'].")
        cv2.imwrite("figs/_frame_raw.png", frame_bgr)
        cv2.imwrite("figs/_mask.png", mask)
        return
    u, v = pixel
    print(f"[cv] Detected {color} centroid: pixel=({u}, {v}), area={int(area)} px")

    # 4. Back-project
    point = pixel_to_base(u, v, K, T_CAM_IN_WRIST, q_rad, z_table=Z_TABLE)
    if point is None:
        print("[cv] Back-projection failed (ray parallel to table or behind camera).")
        return
    print(f"[cv] Back-projected cube xy = ({point[0]:+.3f}, {point[1]:+.3f}) m  "
          f"(z target = 0)")

    # 5. Compare with FK grasp-time
    csv_path = Path(__file__).parent / "configs" / "cube_positions.csv"
    if csv_path.exists():
        positions = pd.read_csv(csv_path)
        row = positions[positions["episode"] == args.episode]
        if len(row):
            r = row.iloc[0]
            print(f"[cv] FK grasp-time xy   = ({r['x']:+.3f}, {r['y']:+.3f}) m")
            dx = point[0] - r['x']
            dy = point[1] - r['y']
            err = np.hypot(dx, dy)
            print(f"[cv] Error |dxy|       = {err*100:.1f} cm  (dx={dx*100:+.1f}, dy={dy*100:+.1f})")

    # 6. Save visualization
    vis = frame_bgr.copy()
    cv2.circle(vis, (u, v), 12, (0, 0, 255), 2)
    cv2.drawMarker(vis, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 18, 1)
    cv2.putText(vis, f"{color} ({u},{v})", (u + 16, v - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(vis, f"base xy = ({point[0]:+.3f}, {point[1]:+.3f}) m",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    out_dir = Path(__file__).parent / "figs" / "debug" / "cv_detection"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cube_detect_ep{args.episode:03d}_f{args.frame:03d}.png"
    cv2.imwrite(str(out_path), vis)
    mask_path = out_dir / f"cube_mask_ep{args.episode:03d}_f{args.frame:03d}.png"
    cv2.imwrite(str(mask_path), mask)
    print(f"[cv] Saved: {out_path}")
    print(f"[cv] Saved mask: {mask_path}")


if __name__ == "__main__":
    main()
