#!/usr/bin/env python3
"""Wrap the multicolor converter output as a loadable LeRobot v3 dataset.

Inputs:
  - outputs/datasets/projet3-hilserl-multicolor-v1-converted/all_ee_frames.parquet
    (one row per converted frame: episode_index, frame_index, timestamp,
     task_index, observation.state[12], action[4], plus audit columns)
  - source dataset snapshot under ~/.cache/huggingface/.../tom-hugo/...
    (used for: original episode→file packing, video files, tasks.parquet,
     per-episode meta scaffolding)

Output (at outputs/datasets/projet3-hilserl-multicolor-v1/):
  data/chunk-000/file-NNN.parquet      ← rebuilt with 12-dim state, 4-dim action
  videos/observation.images.wrist/chunk-000/file-NNN.mp4  ← symlinks to source
  meta/info.json                       ← new feature shapes
  meta/tasks.parquet                   ← copied from source (same 6 tasks)
  meta/episodes/chunk-000/file-NNN.parquet  ← updated lengths/timestamps/stats
  meta/stats.json                      ← recomputed aggregates

Run:
    uv run --no-sync python sim/hilserl/scripts/build_v3_dataset.py
"""

from __future__ import annotations

import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

SOURCE_DATASET = "osammotg1/projet3-eval2-v1-tom-hugo"
SOURCE_SNAPSHOT_ROOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / f"datasets--{SOURCE_DATASET.replace('/', '--')}"
    / "snapshots"
)

CONVERTED_PARQUET = (
    REPO_ROOT
    / "outputs/datasets/projet3-hilserl-multicolor-v1-converted/all_ee_frames.parquet"
)
OUT_ROOT = REPO_ROOT / "outputs/datasets/projet3-hilserl-multicolor-v1"

FPS = 30


def find_latest_snapshot() -> Path:
    snaps = sorted(SOURCE_SNAPSHOT_ROOT.glob("*"), key=lambda p: p.stat().st_mtime)
    if not snaps:
        raise FileNotFoundError(f"No snapshot under {SOURCE_SNAPSHOT_ROOT}")
    return snaps[-1]


def load_source_episodes(snapshot: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(snapshot / "meta/episodes/chunk-*/*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("episode_index").reset_index(drop=True)


def per_feature_stats(arr: np.ndarray, feature_dim: int) -> dict:
    """arr: (N, feature_dim). Returns dict with min/max/mean/std/count/q01/q10/q50/q90/q99."""
    arr = arr.astype(np.float64)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q10": np.quantile(arr, 0.10, axis=0).tolist(),
        "q50": np.quantile(arr, 0.50, axis=0).tolist(),
        "q90": np.quantile(arr, 0.90, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def scalar_stats(arr: np.ndarray) -> dict:
    """For 1-d scalar columns. Returns same shape as per_feature_stats but with [scalar] lists."""
    arr = arr.astype(np.float64)
    return {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [int(arr.shape[0])],
        "q01": [float(np.quantile(arr, 0.01))],
        "q10": [float(np.quantile(arr, 0.10))],
        "q50": [float(np.quantile(arr, 0.50))],
        "q90": [float(np.quantile(arr, 0.90))],
        "q99": [float(np.quantile(arr, 0.99))],
    }


def build_info_json(src_info: dict, new_total_frames: int) -> dict:
    """Mirror source info.json, mutate features for HIL-SERL schema."""
    info = json.loads(json.dumps(src_info))  # deep copy
    info["total_frames"] = new_total_frames
    # action: 6 joint targets in degrees → 4-dim EE delta + gripper delta
    info["features"]["action"] = {
        "dtype": "float32",
        "names": ["ee_dx", "ee_dy", "ee_dz", "gripper_delta"],
        "shape": [4],
    }
    # observation.state: 6 joint pos → 12 (6 pos + 6 vel)
    info["features"]["observation.state"] = {
        "dtype": "float32",
        "names": [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
            "shoulder_pan.vel", "shoulder_lift.vel", "elbow_flex.vel",
            "wrist_flex.vel", "wrist_roll.vel", "gripper.vel",
        ],
        "shape": [12],
    }
    # observation.images.wrist + scalar columns unchanged
    return info


def main() -> int:
    snapshot = find_latest_snapshot()
    print(f"Source snapshot: {snapshot.name[:12]}")

    print("\n[1/8] Loading converted frames...")
    conv = pd.read_parquet(CONVERTED_PARQUET)
    print(f"   {len(conv)} frames, {conv['episode_index'].nunique()} episodes")

    print("\n[2/8] Loading source episode meta (for packing)...")
    src_eps = load_source_episodes(snapshot)
    print(f"   {len(src_eps)} source episodes")

    # Per-episode summary in NEW dataset (each ep loses 1 frame).
    ep_summary = (
        conv.groupby("episode_index")
        .agg(
            new_length=("frame_index", "count"),
            task_index=("task_index", "first"),
        )
        .reset_index()
        .sort_values("episode_index")
        .reset_index(drop=True)
    )
    assert len(ep_summary) == len(src_eps), \
        f"episode count mismatch: converted {len(ep_summary)} vs source {len(src_eps)}"

    # Recompute dataset-global index ranges.
    ep_summary["new_dataset_from_index"] = (
        ep_summary["new_length"].cumsum() - ep_summary["new_length"]
    ).astype(int)
    ep_summary["new_dataset_to_index"] = ep_summary["new_dataset_from_index"] + ep_summary["new_length"]
    new_total_frames = int(ep_summary["new_length"].sum())
    src_total_frames = int(src_eps["length"].sum())
    print(f"   source total frames: {src_total_frames}  →  new total frames: {new_total_frames} "
          f"(diff: {src_total_frames - new_total_frames}, expected 101)")

    # Merge: per-episode lookup for packing info.
    ep_lookup = src_eps.merge(
        ep_summary[["episode_index", "new_length", "new_dataset_from_index", "new_dataset_to_index"]],
        on="episode_index",
        how="left",
    )

    # --- Layout output dir --------------------------------------------------
    if OUT_ROOT.exists():
        print(f"\n[!] Wiping existing {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)
    (OUT_ROOT / "data/chunk-000").mkdir(parents=True)
    (OUT_ROOT / "videos/observation.images.wrist/chunk-000").mkdir(parents=True)
    (OUT_ROOT / "meta/episodes/chunk-000").mkdir(parents=True)

    # --- Step 3: data parquets (mirror source packing) ----------------------
    print("\n[3/8] Writing data parquets (packed like source)...")
    # Source data column order:
    src_data0 = pd.read_parquet(sorted(glob.glob(str(snapshot / "data/chunk-000/file-*.parquet")))[0])
    # We want: action, observation.state, timestamp, frame_index, episode_index, index, task_index
    out_data_cols = ["action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"]

    # Per-episode global-index map (the `index` column in v3 is the dataset-global frame index).
    ep_to_dfrom = dict(zip(ep_lookup["episode_index"], ep_lookup["new_dataset_from_index"]))

    # Group episodes by their data/file_index (each data file holds several episodes).
    files_to_episodes: dict[int, list[int]] = {}
    for _, row in ep_lookup.iterrows():
        files_to_episodes.setdefault(int(row["data/file_index"]), []).append(int(row["episode_index"]))

    for file_idx in sorted(files_to_episodes):
        eps_in_file = sorted(files_to_episodes[file_idx])
        sub = conv[conv["episode_index"].isin(eps_in_file)].copy()
        sub = sub.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        # Build dataset-global index column.
        idx_global = np.concatenate([
            np.arange(ep_to_dfrom[ep], ep_to_dfrom[ep] + (sub["episode_index"] == ep).sum())
            for ep in eps_in_file
        ])
        sub["index"] = idx_global.astype(np.int64)
        # Cast types to match source.
        sub["frame_index"] = sub["frame_index"].astype(np.int64)
        sub["episode_index"] = sub["episode_index"].astype(np.int64)
        sub["task_index"] = sub["task_index"].astype(np.int64)
        sub["timestamp"] = sub["timestamp"].astype(np.float32)
        out_df = sub[out_data_cols]
        out_path = OUT_ROOT / f"data/chunk-000/file-{file_idx:03d}.parquet"
        out_df.to_parquet(out_path, index=False)
    n_data_files = len(files_to_episodes)
    print(f"   wrote {n_data_files} data parquet files")

    # --- Step 4: videos (symlink source) ------------------------------------
    print("\n[4/8] Symlinking videos from source...")
    src_videos = sorted(glob.glob(
        str(snapshot / "videos/observation.images.wrist/chunk-000/file-*.mp4")
    ))
    for src_vid in src_videos:
        name = Path(src_vid).name
        dst = OUT_ROOT / f"videos/observation.images.wrist/chunk-000/{name}"
        dst.symlink_to(Path(src_vid).resolve())
    print(f"   symlinked {len(src_videos)} video files (unchanged from source)")

    # --- Step 5: episodes meta parquets -------------------------------------
    print("\n[5/8] Writing meta/episodes parquets (per-episode stats + packing)...")
    # We rebuild stats columns per episode for action and observation.state,
    # mirroring source column layout.
    # Group episodes by their meta/episodes/file_index.
    meta_files_to_episodes: dict[int, list[int]] = {}
    for _, row in ep_lookup.iterrows():
        meta_files_to_episodes.setdefault(int(row["meta/episodes/file_index"]), []).append(int(row["episode_index"]))

    # Source meta/episodes columns (preserve order/dtype).
    src_meta_cols = list(pd.read_parquet(
        sorted(glob.glob(str(snapshot / "meta/episodes/chunk-000/file-*.parquet")))[0]
    ).columns)

    def ep_stats_block(ep_df: pd.DataFrame) -> dict:
        """Compute per-episode stat columns for the new dataset schema."""
        action = np.stack(ep_df["action"].values).astype(np.float64)
        state = np.stack(ep_df["observation.state"].values).astype(np.float64)
        ts = ep_df["timestamp"].values.astype(np.float64)
        fi = ep_df["frame_index"].values.astype(np.float64)
        ei = ep_df["episode_index"].values.astype(np.float64)
        idx = ep_df["index"].values.astype(np.float64)  # gets filled in below
        ti = ep_df["task_index"].values.astype(np.float64)

        out = {}
        # action / observation.state — vector stats
        for name, arr in [("action", action), ("observation.state", state)]:
            s = per_feature_stats(arr, arr.shape[1])
            for stat_key, vals in s.items():
                out[f"stats/{name}/{stat_key}"] = vals
        # scalar columns
        for name, arr in [
            ("timestamp", ts), ("frame_index", fi), ("episode_index", ei),
            ("index", idx), ("task_index", ti),
        ]:
            s = scalar_stats(arr)
            for stat_key, vals in s.items():
                out[f"stats/{name}/{stat_key}"] = (
                    vals if stat_key == "count" else [float(vals[0])]
                )
        # observation.images.wrist — keep source's image stats (we didn't change pixels).
        # We'll fill these from the source per-episode row in the caller.
        return out

    # Pre-load all converted frames per episode (already sorted in `conv`).
    conv_by_ep = {ep: g.reset_index(drop=True) for ep, g in conv.groupby("episode_index")}

    # For images stats, just lift from source meta row (videos are unchanged).
    src_image_stat_cols = [c for c in src_meta_cols if c.startswith("stats/observation.images.wrist/")]

    for meta_file_idx in sorted(meta_files_to_episodes):
        eps_in_meta = sorted(meta_files_to_episodes[meta_file_idx])
        rows = []
        for ep in eps_in_meta:
            src_row = src_eps[src_eps["episode_index"] == ep].iloc[0]
            ep_df = conv_by_ep[ep].copy()
            # Attach dataset-global index for THIS episode (matches what we wrote in data parquet).
            d_from = int(ep_lookup.loc[ep_lookup["episode_index"] == ep, "new_dataset_from_index"].iloc[0])
            ep_df["index"] = np.arange(d_from, d_from + len(ep_df), dtype=np.int64)

            stats = ep_stats_block(ep_df)

            new_length = int(len(ep_df))
            new_from_ts = float(src_row["videos/observation.images.wrist/from_timestamp"])
            new_to_ts = new_from_ts + new_length / FPS

            row = {
                "episode_index": int(ep),
                "tasks": src_row["tasks"],
                "length": new_length,
                "data/chunk_index": int(src_row["data/chunk_index"]),
                "data/file_index": int(src_row["data/file_index"]),
                "dataset_from_index": d_from,
                "dataset_to_index": d_from + new_length,
                "videos/observation.images.wrist/chunk_index": int(
                    src_row["videos/observation.images.wrist/chunk_index"]
                ),
                "videos/observation.images.wrist/file_index": int(
                    src_row["videos/observation.images.wrist/file_index"]
                ),
                "videos/observation.images.wrist/from_timestamp": new_from_ts,
                "videos/observation.images.wrist/to_timestamp": new_to_ts,
                "meta/episodes/chunk_index": int(src_row["meta/episodes/chunk_index"]),
                "meta/episodes/file_index": int(src_row["meta/episodes/file_index"]),
            }
            row.update(stats)
            # Copy image stats verbatim (videos unchanged).
            for c in src_image_stat_cols:
                row[c] = src_row[c]
            rows.append(row)

        out_df = pd.DataFrame(rows)
        # Reorder columns to match source.
        # Some columns will be present that aren't in src (we just renamed action/state stats —
        # they're identical paths). Keep source order; any extras go at the end.
        col_order = [c for c in src_meta_cols if c in out_df.columns] + \
                    [c for c in out_df.columns if c not in src_meta_cols]
        out_df = out_df[col_order]
        out_path = OUT_ROOT / f"meta/episodes/chunk-000/file-{meta_file_idx:03d}.parquet"
        out_df.to_parquet(out_path, index=False)
    print(f"   wrote {len(meta_files_to_episodes)} meta/episodes parquet files")

    # --- Step 6: tasks.parquet (unchanged) ----------------------------------
    print("\n[6/8] Copying tasks.parquet (unchanged)...")
    shutil.copy(snapshot / "meta/tasks.parquet", OUT_ROOT / "meta/tasks.parquet")

    # --- Step 7: info.json --------------------------------------------------
    print("\n[7/8] Writing meta/info.json...")
    src_info = json.loads((snapshot / "meta/info.json").read_text())
    info = build_info_json(src_info, new_total_frames)
    (OUT_ROOT / "meta/info.json").write_text(json.dumps(info, indent=4))
    print(f"   new total_frames: {info['total_frames']}, action.shape: {info['features']['action']['shape']}, "
          f"observation.state.shape: {info['features']['observation.state']['shape']}")

    # --- Step 8: stats.json (whole-dataset) ---------------------------------
    print("\n[8/8] Writing meta/stats.json...")
    all_action = np.stack(conv["action"].values)
    all_state = np.stack(conv["observation.state"].values)
    # Recompute dataset-global index for ALL frames.
    idx_all = np.concatenate([
        np.arange(ep_to_dfrom[ep], ep_to_dfrom[ep] + len(g))
        for ep, g in conv.groupby("episode_index", sort=True)
    ])
    stats = {
        "action": per_feature_stats(all_action, 4),
        "observation.state": per_feature_stats(all_state, 12),
        "timestamp": scalar_stats(conv["timestamp"].values),
        "frame_index": scalar_stats(conv["frame_index"].values),
        "episode_index": scalar_stats(conv["episode_index"].values),
        "index": scalar_stats(idx_all),
        "task_index": scalar_stats(conv["task_index"].values),
        # observation.images.wrist: copy source stats (3-channel mean/std/etc).
        "observation.images.wrist": src_info_load_image_stats(snapshot),
    }
    (OUT_ROOT / "meta/stats.json").write_text(json.dumps(stats, indent=4))
    print(f"   action mean (4-dim): {[f'{x:+.4f}' for x in stats['action']['mean']]}")
    print(f"   state mean (12-dim): {[f'{x:+.2f}' for x in stats['observation.state']['mean'][:6]]} (pos)")
    print(f"   state mean (12-dim): {[f'{x:+.2f}' for x in stats['observation.state']['mean'][6:]]} (vel)")

    print(f"\nDone. Dataset at: {OUT_ROOT}")
    return 0


def src_info_load_image_stats(snapshot: Path) -> dict:
    """Read source meta/stats.json and lift the image stats verbatim."""
    src_stats = json.loads((snapshot / "meta/stats.json").read_text())
    return src_stats["observation.images.wrist"]


if __name__ == "__main__":
    raise SystemExit(main())
