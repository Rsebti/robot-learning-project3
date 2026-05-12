"""
trim_still_frames.py — strip leading still frames from every episode.

Problem: each episode starts with the hand resetting the robot while joints
are not moving → hand is visible in the wrist camera but there is no useful
demonstration signal.

Fix: for each episode, find the first frame where joints start moving
(cumulative displacement from frame-0 exceeds --threshold degrees), then
drop all frames before that point. Video files are trimmed with ffmpeg.

Usage:
    # Preview what would be trimmed (no changes written):
    python teleop/trim_still_frames.py \\
        --repo_id osammotg1/projet3-eval1-bowl1-v1 \\
        --dry_run

    # Actually trim and push back to HF:
    python teleop/trim_still_frames.py \\
        --repo_id osammotg1/projet3-eval1-bowl1-v1

    # Tune sensitivity (default 3.0 degrees total across all joints):
    python teleop/trim_still_frames.py \\
        --repo_id osammotg1/projet3-eval1-bowl1-v1 \\
        --threshold 5.0

Requirements:
    pip install datasets huggingface_hub pandas numpy
    ffmpeg must be on PATH (https://ffmpeg.org/download.html)
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download


# ---------------------------------------------------------------------------
# Movement detection
# ---------------------------------------------------------------------------

def find_trim_frame(states: np.ndarray, threshold_deg: float = 3.0) -> int:
    """Return the index of the first frame where joints have moved.

    Args:
        states:        (T, 6) array of joint positions in degrees.
        threshold_deg: cumulative L1 displacement from frame-0 across all
                       joints that must be exceeded to count as "moving".

    Returns:
        Frame index to keep from (inclusive). Returns 0 if movement starts
        immediately (nothing to trim).
    """
    origin = states[0]
    for t in range(1, len(states)):
        disp = np.sum(np.abs(states[t] - origin))
        if disp > threshold_deg:
            # Back off a few frames to keep a little pre-motion context
            return max(0, t - 2)
    # No movement detected — keep everything (shouldn't happen on real demos)
    return 0


# ---------------------------------------------------------------------------
# Parquet trimming
# ---------------------------------------------------------------------------

def trim_episode_parquet(df: pd.DataFrame, ep_idx: int, trim_frame: int) -> pd.DataFrame:
    """Slice one episode's rows and re-index frame_index from 0."""
    ep_rows = df[df["episode_index"] == ep_idx].copy()
    ep_rows = ep_rows[ep_rows["frame_index"] >= trim_frame].copy()
    ep_rows["frame_index"] = ep_rows["frame_index"] - trim_frame
    return ep_rows


# ---------------------------------------------------------------------------
# Video trimming (ffmpeg)
# ---------------------------------------------------------------------------

def trim_video(src: Path, dst: Path, trim_frame: int, fps: int = 30) -> bool:
    """Trim video so it starts at trim_frame, re-encode for frame accuracy.

    Returns True on success.
    """
    if not shutil.which("ffmpeg"):
        print("  [WARN] ffmpeg not found — video NOT trimmed. Install ffmpeg and re-run.")
        shutil.copy2(src, dst)
        return False

    start_sec = trim_frame / fps
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",          # near-lossless
        "-an",                 # no audio
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] ffmpeg failed:\n{result.stderr[-500:]}")
        shutil.copy2(src, dst)
        return False
    return True


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_episodes_meta(meta_dir: Path) -> list[dict]:
    path = meta_dir / "episodes.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_episodes_meta(meta_dir: Path, episodes: list[dict]):
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")


def load_info(meta_dir: Path) -> dict:
    path = meta_dir / "info.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_info(meta_dir: Path, info: dict):
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id",   default="osammotg1/projet3-eval1-bowl1-v1",
                        help="HuggingFace dataset repo id")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="Joint displacement threshold in degrees (default 3.0)")
    parser.add_argument("--fps",       type=int,   default=30)
    parser.add_argument("--dry_run",   action="store_true",
                        help="Print trim points without modifying anything")
    parser.add_argument("--cache_dir", default=None,
                        help="Where to download the dataset (default: HF cache)")
    args = parser.parse_args()

    # 1. Download ----------------------------------------------------------------
    print(f"[trim] Downloading {args.repo_id} ...")
    local_dir = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            cache_dir=args.cache_dir,
            ignore_patterns=["*.git*"],
        )
    )
    print(f"[trim] Dataset at: {local_dir}")

    # Work in a copy so we never corrupt the HF cache
    work_dir = local_dir.parent.parent / (local_dir.name + "_trimmed")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(local_dir, work_dir)
    print(f"[trim] Working copy: {work_dir}")

    # 2. Load all parquet data ---------------------------------------------------
    data_dir = work_dir / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        print("[trim] ERROR: no parquet files found.")
        sys.exit(1)

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df_all = pd.concat(dfs, ignore_index=True).sort_values(
        ["episode_index", "frame_index"]
    ).reset_index(drop=True)

    n_episodes = int(df_all["episode_index"].max()) + 1
    print(f"[trim] {n_episodes} episodes, {len(df_all)} frames total")

    # 3. Detect trim points per episode -----------------------------------------
    trim_points: dict[int, int] = {}
    for ep_idx in range(n_episodes):
        ep_rows = df_all[df_all["episode_index"] == ep_idx].sort_values("frame_index")
        states = np.array(ep_rows["observation.state"].tolist())   # (T, 6)
        trim_frame = find_trim_frame(states, args.threshold)
        trim_points[ep_idx] = trim_frame
        orig_len = len(ep_rows)
        new_len  = orig_len - trim_frame
        status = f"  cut {trim_frame:3d} frames → keep {new_len:3d}/{orig_len:3d}"
        print(f"  ep {ep_idx:03d}: {status}")

    if args.dry_run:
        total_cut = sum(trim_points.values())
        print(f"\n[dry_run] Would cut {total_cut} frames total across {n_episodes} episodes.")
        print("[dry_run] Re-run without --dry_run to apply.")
        return

    # 4. Trim parquet ------------------------------------------------------------
    print("\n[trim] Trimming parquet data ...")
    trimmed_parts = []
    for ep_idx in range(n_episodes):
        trimmed_parts.append(
            trim_episode_parquet(df_all, ep_idx, trim_points[ep_idx])
        )

    df_trimmed = pd.concat(trimmed_parts, ignore_index=True)

    # Re-build global `index` column (must be 0..N-1 with no gaps)
    df_trimmed = df_trimmed.reset_index(drop=True)
    df_trimmed["index"] = df_trimmed.index

    # Write back — same chunk structure as original
    for pf in parquet_files:
        pf.unlink()  # remove old files from work_dir

    # Write one parquet per episode (simple, always works)
    chunk_dir = data_dir / "chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for ep_idx in range(n_episodes):
        ep_df = df_trimmed[df_trimmed["episode_index"] == ep_idx]
        out_path = chunk_dir / f"episode_{ep_idx:06d}.parquet"
        ep_df.to_parquet(out_path, index=False)

    print(f"[trim] Wrote {n_episodes} parquet files.")

    # 5. Trim videos -------------------------------------------------------------
    video_root = work_dir / "videos"
    if video_root.exists():
        print("\n[trim] Trimming videos ...")
        for ep_idx in range(n_episodes):
            trim_frame = trim_points[ep_idx]
            if trim_frame == 0:
                continue  # nothing to cut

            # Find the video file for this episode
            pattern = f"episode_{ep_idx:06d}.mp4"
            video_files = list(video_root.rglob(pattern))
            for vf in video_files:
                tmp = vf.with_suffix(".tmp.mp4")
                ok = trim_video(vf, tmp, trim_frame, args.fps)
                if ok:
                    vf.unlink()
                    tmp.rename(vf)
                else:
                    tmp.unlink(missing_ok=True)
                print(f"  ep {ep_idx:03d}: {vf.name} {'✓' if ok else '(copy, ffmpeg failed)'}")
    else:
        print("[trim] No videos/ directory found — skipping video trimming.")

    # 6. Update metadata ---------------------------------------------------------
    meta_dir = work_dir / "meta"
    if meta_dir.exists():
        print("\n[trim] Updating metadata ...")

        # episodes.jsonl
        episodes_meta = load_episodes_meta(meta_dir)
        if episodes_meta:
            for ep in episodes_meta:
                ep_idx = ep.get("episode_index", ep.get("index"))
                if ep_idx is not None and ep_idx in trim_points:
                    old_len = ep.get("length", ep.get("num_frames", 0))
                    new_len = old_len - trim_points[ep_idx]
                    ep["length"] = new_len
                    if "num_frames" in ep:
                        ep["num_frames"] = new_len
            save_episodes_meta(meta_dir, episodes_meta)
            print("  episodes.jsonl updated.")

        # info.json
        info = load_info(meta_dir)
        if info:
            info["total_frames"] = int(len(df_trimmed))
            save_info(meta_dir, info)
            print("  info.json updated.")

    # 7. Push to HF --------------------------------------------------------------
    print(f"\n[trim] Pushing trimmed dataset to {args.repo_id} ...")
    api = HfApi()
    api.upload_folder(
        folder_path=str(work_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=f"Trim leading still frames (threshold={args.threshold}°)",
    )
    print(f"[trim] Done. Dataset pushed to https://huggingface.co/datasets/{args.repo_id}")

    # Cleanup work dir
    shutil.rmtree(work_dir)
    print("[trim] Work directory cleaned up.")


if __name__ == "__main__":
    main()
