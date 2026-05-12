"""
map_cube_positions.py — extract approximate cube positions from teleop demos.

Approach (no camera calibration needed):
  1. For each episode, find the grasp event (gripper closing)
  2. At grasp time, the wrist is directly above the cube
  3. Run FK at that frame -> wrist xy ≈ cube xy

Outputs:
  configs/cube_positions.csv — (episode, x, y, z, grasp_frame) in robot base frame
  figs/cube_position_map.png — 2D scatter plot of all initial positions

Usage:
  python teleop/map_cube_positions.py
  python teleop/map_cube_positions.py --repo_id hudela390/projet3-eval1-bowl1-v1-trimmed

Requirements:
  pip install pandas numpy matplotlib huggingface_hub
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from probe_bowl_position import fk_tip  # noqa: E402


# ---------------------------------------------------------------------------
# Grasp detection
# ---------------------------------------------------------------------------

# Color name -> matplotlib RGB used for plotting
_COLOR_MAP = {
    "yellow": "#F4C430",
    "orange": "#FF7F0E",
    "blue":   "#1F77FF",
    "green":  "#2CA02C",
    "violet": "#9467BD",
    "red":    "#D62728",
    "unknown": "#888888",
}


def _extract_color(task_desc: str) -> str:
    """Pull the color word out of a task description like 'Pick yellow block...'"""
    desc_low = task_desc.lower()
    for c in _COLOR_MAP:
        if c in desc_low:
            return c
    return "unknown"


def find_grasp_frame(gripper_traj: np.ndarray, fps: int = 30) -> int:
    """Return the frame of the FIRST gripper transition (grasp event).

    In a teleop demo the gripper goes: open -> closed (grasp) -> open (release).
    We want the FIRST transition. We skip 1 second at the start (to avoid
    trim-cut contamination), then return the first midpoint crossing.

    Returns -1 only on degenerate (gripper never moves) episodes.
    """
    skip = fps  # skip the first second
    if len(gripper_traj) < skip + 5:
        return -1

    g_max = float(gripper_traj.max())
    g_min = float(gripper_traj.min())
    span  = g_max - g_min

    if span < 1.0:  # gripper never moved -> no grasp
        return -1

    g_mid = 0.5 * (g_max + g_min)

    # Look from `skip` onwards. Direction inferred from the value at `skip`.
    start_val = gripper_traj[skip]
    if start_val > g_mid:
        for t in range(skip, len(gripper_traj)):
            if gripper_traj[t] < g_mid:
                return t
    else:
        for t in range(skip, len(gripper_traj)):
            if gripper_traj[t] > g_mid:
                return t
    return -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", default="hudela390/projet3-eval1-bowl1-v1-trimmed")
    parser.add_argument("--fps",     type=int, default=30)
    args = parser.parse_args()

    print(f"[map] Loading {args.repo_id} ...")
    from huggingface_hub import HfApi
    api = HfApi()
    all_files = api.list_repo_files(args.repo_id, repo_type="dataset")
    data_files = sorted(f for f in all_files if f.startswith("data/") and f.endswith(".parquet"))
    print(f"[map] Found {len(data_files)} parquet shard(s)")
    dfs = []
    for f in data_files:
        p = hf_hub_download(repo_id=args.repo_id, filename=f, repo_type="dataset")
        dfs.append(pd.read_parquet(p))
    df = pd.concat(dfs, ignore_index=True)

    # Load task labels (cube colors are embedded here)
    tasks_path = hf_hub_download(
        repo_id=args.repo_id,
        filename="meta/tasks.parquet",
        repo_type="dataset",
    )
    tasks = pd.read_parquet(tasks_path)
    # tasks: index = task description, column = task_index
    task_desc_by_idx = {int(v): k for k, v in tasks["task_index"].items()}
    color_by_task = {idx: _extract_color(desc) for idx, desc in task_desc_by_idx.items()}
    print("[map] Colors detected:")
    for idx, c in color_by_task.items():
        print(f"  task {idx}: {c}")

    n_episodes = int(df["episode_index"].max()) + 1
    print(f"\n[map] {n_episodes} episodes, {len(df)} frames total\n")

    results = []
    for ep_idx in range(n_episodes):
        ep = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        states = np.array(ep["observation.state"].tolist())   # (T, 6)
        gripper = states[:, 5]
        task_idx = int(ep["task_index"].iloc[0])
        color = color_by_task.get(task_idx, "unknown")

        grasp_frame = find_grasp_frame(gripper, fps=args.fps)
        if grasp_frame < 0:
            print(f"  ep {ep_idx:03d} ({color:7s}): no grasp detected, skipped")
            continue

        # Arm joints (degrees -> radians)
        q_rad = np.deg2rad(states[grasp_frame, :5]).tolist()
        T = fk_tip(q_rad)
        x, y, z = T[:3, 3]

        results.append({
            "episode":     ep_idx,
            "task_index":  task_idx,
            "color":       color,
            "x":           float(x),
            "y":           float(y),
            "z":           float(z),
            "grasp_frame": int(grasp_frame),
        })
        print(f"  ep {ep_idx:03d} ({color:7s}): grasp @ frame {grasp_frame:3d}  "
              f"xy=({x:+.3f}, {y:+.3f}) m,  z={z:+.3f} m")

    if not results:
        print("[map] No grasps detected — aborting.")
        return

    # Save CSV
    csv_path = Path(__file__).parent / "configs" / "cube_positions.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n[map] Saved CSV -> {csv_path}")

    # Plot
    xs = np.array([r["x"] for r in results])
    ys = np.array([r["y"] for r in results])
    eps = [r["episode"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Group by color so the legend lists each color once
    colors_present = sorted(set(r["color"] for r in results))
    for color in colors_present:
        pts = [r for r in results if r["color"] == color]
        cx = [r["x"] for r in pts]
        cy = [r["y"] for r in pts]
        ax.scatter(cx, cy, c=_COLOR_MAP.get(color, "#888"), s=160,
                   alpha=0.85, edgecolors="black", linewidths=0.7,
                   label=f"{color} (n={len(pts)})")
    for r in results:
        ax.annotate(str(r["episode"]), (r["x"], r["y"]),
                    fontsize=7, ha="center", va="center",
                    color="white" if r["color"] != "yellow" else "black")

    # Zoom in on the relevant region (with padding)
    pad = 0.03
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)

    ax.set_xlabel("x (m) - forward from robot base")
    ax.set_ylabel("y (m) - lateral from robot base")
    ax.set_title(f"Cube initial positions across {len(results)} demos\n"
                 f"(gripper fingertip xy at grasp, colored by cube color)")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    ax.legend(loc="best", framealpha=0.9)

    png_path = Path(__file__).parent / "figs" / "cube_position_map.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"[map] Saved plot  -> {png_path}")
    print(f"\n[map] x range: [{xs.min():+.3f}, {xs.max():+.3f}] m  (span {np.ptp(xs):.3f} m)")
    print(f"[map] y range: [{ys.min():+.3f}, {ys.max():+.3f}] m  (span {np.ptp(ys):.3f} m)")


if __name__ == "__main__":
    main()
