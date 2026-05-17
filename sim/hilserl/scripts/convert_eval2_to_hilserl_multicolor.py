#!/usr/bin/env python3
"""Convert ALL 6 task indices of osammotg1/projet3-eval2-v1-tom-hugo into
HIL-SERL EE-space frames (joint pos+vel observation, EE-delta+gripper action).

Generalizes convert_eval2_to_hilserl_yellow.py: no task_index filter, all
6 colors processed in one pass. Output is a single combined parquet plus
a sanity report; the v3 dataset wrapper is built by build_v3_dataset.py.

Run:
    uv run --no-sync python sim/hilserl/scripts/convert_eval2_to_hilserl_multicolor.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
URDF_PATH = REPO_ROOT / "sim/hilserl/assets/so101/urdf/so_arm101.urdf"

SOURCE_DATASET = "osammotg1/projet3-eval2-v1-tom-hugo"
SOURCE_SNAPSHOT_ROOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / f"datasets--{SOURCE_DATASET.replace('/', '--')}"
    / "snapshots"
)
SOURCE_FPS = 30

OUTPUT_DIR = REPO_ROOT / "outputs/datasets/projet3-hilserl-multicolor-v1-converted"
OUTPUT_PARQUET = OUTPUT_DIR / "all_ee_frames.parquet"
OUTPUT_REPORT = OUTPUT_DIR / "conversion_report.md"

EE_STEP_BOUND_M = 0.02  # env_config_so101.json clamp — diagnostic only


def find_latest_snapshot() -> Path:
    snaps = sorted(SOURCE_SNAPSHOT_ROOT.glob("*"), key=lambda p: p.stat().st_mtime)
    if not snaps:
        raise FileNotFoundError(
            f"No snapshot under {SOURCE_SNAPSHOT_ROOT}. "
            "Run huggingface_hub.snapshot_download first."
        )
    return snaps[-1]


def load_all_frames(snapshot: Path) -> pd.DataFrame:
    parquet_files = sorted(glob.glob(str(snapshot / "data/chunk-*/*.parquet")))
    frames = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def convert_episode(ep_df: pd.DataFrame, rk) -> pd.DataFrame:
    """One episode (joint-space) -> EE-space, last frame dropped.

    Output columns mirror the eventual HIL-SERL v3 schema:
      - observation.state         (12,)  6 joint pos (deg) + 6 joint vel (deg/s)
      - action                    (4,)   ee_dx, ee_dy, ee_dz (m), gripper_delta
      - observation.ee_pose       (3,)   audit only — xyz from FK
      - source_action             (6,)   original joint targets (audit)
      - episode_index, frame_index, timestamp, task_index, length-aware idx
    """
    if len(ep_df) < 2:
        return pd.DataFrame()

    # placo's C++ binding needs float64; lerobot.RobotKinematics.forward_kinematics
    # takes DEGREES and converts internally — don't pre-convert to radians.
    obs_state = np.stack(ep_df["observation.state"].values).astype(np.float64)  # (T,6) deg
    src_action = np.stack(ep_df["action"].values).astype(np.float64)             # (T,6) deg

    ee_xyz = np.array([rk.forward_kinematics(q)[:3, 3] for q in obs_state])      # (T,3) m

    ee_delta = ee_xyz[1:] - ee_xyz[:-1]                                          # (T-1,3) m
    joint_vel = (obs_state[1:] - obs_state[:-1]) * SOURCE_FPS                    # (T-1,6) deg/s
    gripper_delta = obs_state[1:, 5:6] - obs_state[:-1, 5:6]                     # (T-1,1)

    # HIL-SERL state: [pos_0..pos_5, vel_0..vel_5]
    obs12 = np.concatenate([obs_state[:-1], joint_vel], axis=1).astype(np.float32)  # (T-1,12)
    # HIL-SERL action: [dx, dy, dz, gripper_delta]
    act4 = np.concatenate([ee_delta, gripper_delta], axis=1).astype(np.float32)     # (T-1,4)

    return pd.DataFrame({
        "episode_index": ep_df["episode_index"].values[:-1],
        "frame_index": ep_df["frame_index"].values[:-1],
        "timestamp": ep_df["timestamp"].values[:-1].astype(np.float32),
        "task_index": ep_df["task_index"].values[:-1],
        "observation.state": list(obs12),
        "action": list(act4),
        # Audit columns (used by build_v3_dataset.py for stats + sanity):
        "observation.ee_pose": list(ee_xyz[:-1].astype(np.float32)),
        "source_action": list(src_action[:-1].astype(np.float32)),
        "source_obs_state": list(obs_state[:-1].astype(np.float32)),
    })


def main() -> int:
    snapshot = find_latest_snapshot()
    print(f"Source: {SOURCE_DATASET}\nSnapshot: {snapshot.name[:12]}")

    print("\n[1/4] Loading all frames...")
    src_df = load_all_frames(snapshot)
    eps = sorted(src_df["episode_index"].unique().tolist())
    task_counts = src_df.groupby("task_index").size().to_dict()
    print(f"   episodes: {len(eps)}  frames: {len(src_df)}")
    print(f"   task_index counts: {task_counts}")

    print("\n[2/4] Loading placo kinematics...")
    from lerobot.model.kinematics import RobotKinematics
    rk = RobotKinematics(urdf_path=str(URDF_PATH), target_frame_name="gripper_frame_link")
    print(f"   joints: {rk.joint_names}")

    print("\n[3/4] Per-episode FK + finite-diff conversion...")
    per_ep = []
    for ep in eps:
        ep_df = src_df[src_df["episode_index"] == ep].reset_index(drop=True)
        conv = convert_episode(ep_df, rk)
        if conv.empty:
            print(f"   ep {ep:3d}: SKIPPED (only {len(ep_df)} frame)")
            continue
        per_ep.append(conv)
        if ep < 5 or ep % 20 == 0:
            print(f"   ep {ep:3d} (task {ep_df.iloc[0]['task_index']}): "
                  f"{len(ep_df):4d} -> {len(conv):4d}")
    converted = pd.concat(per_ep, ignore_index=True)
    print(f"   total: {len(converted)} converted frames "
          f"(dropped {len(src_df) - len(converted)} last-of-episode frames)")

    print(f"\n[4/4] Writing {OUTPUT_PARQUET} ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    converted.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"   wrote {OUTPUT_PARQUET.stat().st_size // 1024} KB")

    # --- Sanity report ----------------------------------------------------
    print("\n=== Sanity report (all colors) ===")
    state_all = np.stack(converted["observation.state"].values)
    action_all = np.stack(converted["action"].values)
    ee_xyz_all = np.stack(converted["observation.ee_pose"].values)

    ee_delta = action_all[:, :3]
    grip_delta = action_all[:, 3]

    delta_abs = np.abs(ee_delta)
    over_bound_pct = 100.0 * (delta_abs.max(axis=1) > EE_STEP_BOUND_M).mean()

    lines = [
        "# Conversion report — multicolor (all 6 task_indices) → EE space\n\n",
        f"Source: `{SOURCE_DATASET}` (snapshot `{snapshot.name[:12]}`)\n",
        f"Frames converted: **{len(converted)}** across **{len(eps)}** episodes "
        f"(dropped {len(src_df) - len(converted)} last-of-episode frames for finite diff)\n\n",
        "## Per-task frame counts (converted)\n\n",
    ]
    for ti, n_src in sorted(task_counts.items()):
        n_conv = (converted["task_index"] == ti).sum()
        lines.append(f"- task {ti}: src {n_src} → converted {n_conv}\n")
    lines += [
        "\n## EE pose range (meters, robot frame)\n",
        f"- X: min={ee_xyz_all[:,0].min():+.3f}  max={ee_xyz_all[:,0].max():+.3f}\n",
        f"- Y: min={ee_xyz_all[:,1].min():+.3f}  max={ee_xyz_all[:,1].max():+.3f}\n",
        f"- Z: min={ee_xyz_all[:,2].min():+.3f}  max={ee_xyz_all[:,2].max():+.3f}\n",
        "\n## EE deltas (meters / 30-fps frame)\n",
        f"- |dx| mean={delta_abs[:,0].mean():.4f}  max={delta_abs[:,0].max():.4f}\n",
        f"- |dy| mean={delta_abs[:,1].mean():.4f}  max={delta_abs[:,1].max():.4f}\n",
        f"- |dz| mean={delta_abs[:,2].mean():.4f}  max={delta_abs[:,2].max():.4f}\n",
        f"- Frames where max-axis delta > {EE_STEP_BOUND_M} m: **{over_bound_pct:.2f}%**\n",
        "\n## Gripper delta (raw units / frame)\n",
        f"- min={grip_delta.min():+.3f}  mean={grip_delta.mean():+.4f}  max={grip_delta.max():+.3f}\n",
        "\n## Joint state (12-dim observation.state)\n",
        f"- joint pos range: min={state_all[:,:6].min():+.2f}  max={state_all[:,:6].max():+.2f} (deg)\n",
        f"- joint vel range: min={state_all[:,6:].min():+.2f}  max={state_all[:,6:].max():+.2f} (deg/s)\n",
    ]
    OUTPUT_REPORT.write_text("".join(lines))
    print("".join(lines))
    print(f"\nReport: {OUTPUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
