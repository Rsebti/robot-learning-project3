"""
calibrate_camera_from_demos.py - solve for camera intrinsics + extrinsics
using the 39 demos as a self-supervised calibration set.

Idea: each episode gives us a (pixel, joint_angles, ground_truth_xy) triplet.
- pixel        : cube centroid detected in a pre-grasp frame
- joint_angles : robot joints at that same frame
- gt_xy        : grasp-moment FK cube position from cube_positions.csv

We optimize 10 params (4 intrinsics + 6-DOF T_cam_in_wrist) so that
back-projecting each pixel through the camera model + that episode's FK +
the table plane (z=z_table) lands on gt_xy.

Output: configs/camera_calibration.yaml + a residuals plot.

Usage:
    python teleop/calibrate_camera_from_demos.py
    python teleop/calibrate_camera_from_demos.py --pre_grasp_sec 1.0
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
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rscipy

sys.path.insert(0, str(Path(__file__).parent))
from probe_bowl_position import fk_wrist  # noqa: E402
from detect_cube_cv import (                                       # noqa: E402
    HSV_RANGES, EPISODE_COLOR, episode_to_video_file,
    detect_cube_pixel, read_frame_at,
)


# ---------------------------------------------------------------------------
# Camera model: 10 params [fx, fy, cx, cy, tx, ty, tz, rx, ry, rz]
# (tx, ty, tz, rx, ry, rz) describes T_cam_in_wrist using rotvec (axis-angle).
# ---------------------------------------------------------------------------

def params_to_K_T(p):
    fx, fy, cx, cy, tx, ty, tz, rx, ry, rz = p
    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]])
    R = Rscipy.from_rotvec([rx, ry, rz]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = [tx, ty, tz]
    return K, T


def back_project(u, v, K, T_cam_in_wrist, q_rad, z_table=0.0):
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([u, v, 1.0])
    T_wrist = fk_wrist(q_rad)
    T_cam_base = T_wrist @ T_cam_in_wrist
    origin = T_cam_base[:3, 3]
    ray = T_cam_base[:3, :3] @ ray_cam
    if abs(ray[2]) < 1e-6:
        return None
    t = (z_table - origin[2]) / ray[2]
    if t < 0:
        return None
    return origin + t * ray


# ---------------------------------------------------------------------------
# Build (pixel, joints, gt) dataset
# ---------------------------------------------------------------------------

def build_dataset(repo_id: str, pre_grasp_sec: float, fps: int):
    print(f"[calib] Loading {repo_id} ...")
    parquet_path = hf_hub_download(repo_id, "data/chunk-000/file-000.parquet", repo_type="dataset")
    df = pd.read_parquet(parquet_path)

    csv_path = Path(__file__).parent / "configs" / "cube_positions.csv"
    if not csv_path.exists():
        raise SystemExit(f"Need {csv_path} -- run map_cube_positions.py first")
    gt = pd.read_csv(csv_path)
    print(f"[calib] Loaded {len(gt)} grasp-time ground-truth positions")

    # Cache video paths per chunk
    video_cache = {}

    samples = []
    for _, gt_row in gt.iterrows():
        ep_idx = int(gt_row["episode"])
        color  = EPISODE_COLOR[ep_idx]
        grasp_f = int(gt_row["grasp_frame"])
        pre_f = max(0, grasp_f - int(pre_grasp_sec * fps))

        ep = df[df["episode_index"] == ep_idx].sort_values("frame_index").reset_index(drop=True)
        if pre_f >= len(ep):
            continue
        frame_row = ep.iloc[pre_f]
        ts = float(frame_row["timestamp"])
        q_rad = np.deg2rad(np.asarray(frame_row["observation.state"])[:5]).tolist()

        chunk = episode_to_video_file(ep_idx)
        if chunk not in video_cache:
            video_cache[chunk] = hf_hub_download(
                repo_id,
                f"videos/observation.images.wrist/chunk-000/file-{chunk:03d}.mp4",
                repo_type="dataset",
            )
        bgr = read_frame_at(video_cache[chunk], ts)
        if bgr is None:
            print(f"  ep {ep_idx:03d}: failed to read frame")
            continue
        pixel, _, area = detect_cube_pixel(bgr, color)
        if pixel is None:
            print(f"  ep {ep_idx:03d} ({color}): no detection at pre-grasp frame, skipping")
            continue
        u, v = pixel
        # Reject low-quality detections (tiny blobs or near image edges)
        if area is None or area < 500 or v < 50 or v > 430 or u < 40 or u > 600:
            print(f"  ep {ep_idx:03d} ({color:7s}): rejected (area={int(area or 0)} pixel=({u},{v}))")
            continue
        samples.append({
            "ep": ep_idx,
            "color": color,
            "u": u, "v": v,
            "q_rad": q_rad,
            "gt_x": float(gt_row["x"]),
            "gt_y": float(gt_row["y"]),
            "area": int(area),
        })
        print(f"  ep {ep_idx:03d} ({color:7s}): pixel=({u},{v}) area={int(area)}  gt=({gt_row['x']:+.3f},{gt_row['y']:+.3f})")

    print(f"\n[calib] Collected {len(samples)} valid samples")
    return samples


# ---------------------------------------------------------------------------
# Residual function for least_squares
# ---------------------------------------------------------------------------

def residuals(p, samples, z_table):
    K, T = params_to_K_T(p)
    res = []
    FAIL_PENALTY = 1.0   # 1 m residual on geometric failure (won't dominate)
    for s in samples:
        bp = back_project(s["u"], s["v"], K, T, s["q_rad"], z_table=z_table)
        if bp is None:
            res.extend([FAIL_PENALTY, FAIL_PENALTY])
            continue
        # Clip excessive residuals so far-off points don't dominate
        dx = np.clip(bp[0] - s["gt_x"], -1.0, 1.0)
        dy = np.clip(bp[1] - s["gt_y"], -1.0, 1.0)
        res.append(dx)
        res.append(dy)
    return np.array(res)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="hudela390/projet3-eval1-bowl1-v1-trimmed")
    parser.add_argument("--pre_grasp_sec", type=float, default=1.0,
                        help="Seconds before grasp to sample the calibration frame")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--z_table", type=float, default=0.01,
                        help="Table-top height in base frame (m)")
    args = parser.parse_args()

    samples = build_dataset(args.repo_id, args.pre_grasp_sec, args.fps)
    if len(samples) < 8:
        raise SystemExit(f"[calib] Only {len(samples)} samples; need >= 8")

    # Initial guess
    # rotvec corresponds to R = [[0,0,1],[-1,0,0],[0,-1,0]]:
    # camera_X = -wrist_Y, camera_Y = -wrist_Z, camera_Z = +wrist_X.
    p0 = np.array([
        554.0, 554.0, 320.0, 240.0,            # fx, fy, cx, cy
        0.02,  0.00,  0.03,                    # tx, ty, tz (camera in wrist frame)
        -1.2092, 1.2092, -1.2092,              # rotvec for the orientation above
    ])

    # Bounds: keep things reasonable so the optimizer doesn't wander
    lower = [200, 200, 200, 100,  -0.10, -0.10, -0.10,  -np.pi, -np.pi, -np.pi]
    upper = [1500, 1500, 500, 400,  0.20,  0.10,  0.20,   np.pi,  np.pi,  np.pi]

    print("\n[calib] Optimizing 10 parameters over {} samples ...".format(len(samples)))
    result = least_squares(residuals, p0, args=(samples, args.z_table),
                           bounds=(lower, upper), max_nfev=2000, verbose=2)

    p = result.x
    K, T = params_to_K_T(p)
    fx, fy, cx, cy, tx, ty, tz, rx, ry, rz = p

    # Per-sample residuals
    final_res = residuals(p, samples, args.z_table).reshape(-1, 2)
    errs = np.linalg.norm(final_res, axis=1)
    print("\n[calib] Optimization done.")
    print(f"[calib] Mean xy error: {errs.mean()*100:.2f} cm")
    print(f"[calib] Max  xy error: {errs.max()*100:.2f} cm")
    print(f"[calib] RMS  xy error: {np.sqrt((errs**2).mean())*100:.2f} cm")
    print(f"\n[calib] K = [[{fx:.1f}, 0, {cx:.1f}], [0, {fy:.1f}, {cy:.1f}], [0, 0, 1]]")
    print(f"[calib] T_cam_in_wrist translation = ({tx:+.4f}, {ty:+.4f}, {tz:+.4f}) m")
    print(f"[calib] T_cam_in_wrist rotvec      = ({rx:+.4f}, {ry:+.4f}, {rz:+.4f}) rad")
    print(f"[calib] FOV estimate: H={2*np.degrees(np.arctan(320/fx)):.1f}deg, V={2*np.degrees(np.arctan(240/fy)):.1f}deg")

    # Save
    out_path = Path(__file__).parent / "configs" / "camera_calibration.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "image_size":      [640, 480],
        "K":               K.tolist(),
        "T_cam_in_wrist":  T.tolist(),
        "z_table":         args.z_table,
        "calibrated_from": {
            "repo_id":        args.repo_id,
            "n_samples":      len(samples),
            "pre_grasp_sec":  args.pre_grasp_sec,
            "mean_err_cm":    float(errs.mean() * 100),
            "rms_err_cm":     float(np.sqrt((errs**2).mean()) * 100),
        },
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"\n[calib] Saved -> {out_path}")

    # Plot residuals
    fig, ax = plt.subplots(figsize=(8, 7))
    gt_xs = np.array([s["gt_x"] for s in samples])
    gt_ys = np.array([s["gt_y"] for s in samples])
    bp_xs = gt_xs + final_res[:, 0]
    bp_ys = gt_ys + final_res[:, 1]
    for i, s in enumerate(samples):
        ax.plot([gt_xs[i], bp_xs[i]], [gt_ys[i], bp_ys[i]], "k-", alpha=0.3, lw=0.7)
    ax.scatter(gt_xs, gt_ys, marker="o", c="green", s=60, label="FK (gt)",
               edgecolors="black", linewidths=0.5)
    ax.scatter(bp_xs, bp_ys, marker="x", c="red", s=60, label="back-projected",
               linewidths=1.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Self-calibration residuals (mean {errs.mean()*100:.2f} cm)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    plot_path = Path(__file__).parent / "figs" / "calibration_residuals.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[calib] Saved -> {plot_path}")


if __name__ == "__main__":
    main()
