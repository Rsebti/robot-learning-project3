"""Clean a LeRobot dataset by trimming leading/trailing static frames per episode.

For each episode:
  - Detect leading frames where max-joint deviation from action[0]   < THRESHOLD
  - Detect trailing frames where max-joint deviation from action[-1] < THRESHOLD
  - If the kept window is too short (< MIN_KEEP), drop the episode entirely
  - If the trim is too aggressive (> MAX_TRIM_RATIO of the episode), drop entirely

Then rebuild a fresh LeRobotDataset (new repo) with only the kept frames.
The original dataset is never modified.

Usage:
  # Dry-run (just print the trim plan)
  python tools/clean_dataset.py \\
      --src-repo osammotg1/projet3-eval1-smoketest \\
      --dst-repo osammotg1/projet3-eval1-clean \\
      --src-root ~/.lerobot-data/osammotg1/projet3-eval1-smoketest \\
      --dst-root ~/.lerobot-data/osammotg1/projet3-eval1-clean \\
      --dry-run

  # For real (writes new dataset locally + pushes to HF)
  python tools/clean_dataset.py \\
      --src-repo osammotg1/projet3-eval1-smoketest \\
      --dst-repo osammotg1/projet3-eval1-clean \\
      --src-root ~/.lerobot-data/osammotg1/projet3-eval1-smoketest \\
      --dst-root ~/.lerobot-data/osammotg1/projet3-eval1-clean \\
      --push-to-hub
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Trim plan computation (does not require lerobot import — pure parquet read)
# ---------------------------------------------------------------------------

def load_action_table(src_root: Path) -> pd.DataFrame:
    data_dir = src_root / "data" / "chunk-000"
    files = sorted(p for p in data_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(p) for p in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    return df


def compute_trim_plan(
    df: pd.DataFrame,
    threshold_deg: float,
    min_keep: int,
    max_trim_ratio: float,
):
    plan = []
    for ep_idx, g in df.groupby("episode_index"):
        actions = np.stack(g["action"].values)  # T x 6
        T = len(actions)
        if T < 2:
            plan.append({"ep": int(ep_idx), "T": T, "lead": 0, "trail": 0,
                         "keep": T, "drop": True, "reason": "too short"})
            continue
        dev_start = np.max(np.abs(actions - actions[0]), axis=1)
        moving = np.where(dev_start > threshold_deg)[0]
        lead = int(moving[0]) if len(moving) else T

        dev_end = np.max(np.abs(actions - actions[-1]), axis=1)
        moving_end = np.where(dev_end > threshold_deg)[0]
        last_moving = int(moving_end[-1]) if len(moving_end) else -1
        trail = T - 1 - last_moving if last_moving >= 0 else T

        keep = T - lead - trail
        drop = False
        reason = ""
        if keep <= 0 or keep < min_keep:
            drop = True
            reason = f"keep={keep} < min_keep={min_keep}"
        elif (lead + trail) / T > max_trim_ratio:
            drop = True
            reason = f"trim_ratio={(lead+trail)/T:.2f} > max_trim_ratio={max_trim_ratio}"

        plan.append({
            "ep": int(ep_idx), "T": T, "lead": lead, "trail": trail,
            "keep": keep, "drop": drop, "reason": reason,
        })
    return pd.DataFrame(plan)


def print_plan(plan: pd.DataFrame, label: str):
    print(f"\n=== Trim plan for {label} ===")
    n_total = len(plan)
    n_drop = int(plan["drop"].sum())
    total_T = int(plan["T"].sum())
    total_kept = int(plan.loc[~plan["drop"], "keep"].sum())
    total_dropped_eps = int(plan.loc[plan["drop"], "T"].sum())
    total_trimmed = total_T - total_kept - total_dropped_eps
    print(f"  episodes: {n_total} → kept {n_total - n_drop}, dropped {n_drop}")
    print(f"  frames:   {total_T} → kept {total_kept} "
          f"(trimmed {total_trimmed}, in-dropped-eps {total_dropped_eps})")
    print(f"  reduction: {100 * (1 - total_kept / total_T):.1f}% of frames removed")
    if n_drop:
        print(f"\n  Dropped episodes:")
        print(plan[plan["drop"]].to_string(index=False))
    # Show worst trims (kept episodes)
    kept = plan[~plan["drop"]].copy()
    kept["trim_pct"] = 100 * (kept["lead"] + kept["trail"]) / kept["T"]
    print(f"\n  Top 10 episodes by trim ratio (still kept):")
    print(kept.nlargest(10, "trim_pct").to_string(index=False))


# ---------------------------------------------------------------------------
# Actual rebuild (requires lerobot)
# ---------------------------------------------------------------------------

def rebuild(
    src_repo: str,
    dst_repo: str,
    src_root: Path,
    dst_root: Path,
    plan: pd.DataFrame,
    push_to_hub: bool,
    private: bool,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if dst_root.exists():
        sys.exit(f"dst-root already exists: {dst_root}. Remove it first to avoid corrupting another dataset.")

    print(f"\n→ Loading source: {src_repo} (root={src_root})")
    # Force pyav backend — torchcodec is broken on this Mac
    src = LeRobotDataset(src_repo, root=str(src_root), video_backend="pyav")
    print(f"  fps={src.fps}, episodes={src.num_episodes}, frames={src.num_frames}")
    print(f"  features: {list(src.features.keys())}")

    plan_by_ep = plan.set_index("ep").to_dict("index")

    print(f"\n→ Creating destination: {dst_repo} (root={dst_root})")
    dst = LeRobotDataset.create(
        repo_id=dst_repo,
        fps=src.fps,
        features=src.features,
        root=str(dst_root),
        robot_type=src.meta.robot_type,
        use_videos=True,
        vcodec="libsvtav1",
        streaming_encoding=False,  # safer for batch processing
        image_writer_threads=0,    # 0 = no multiprocessing — avoid sem hangs on macOS
        image_writer_processes=0,
    )

    kept_eps = 0
    kept_frames_total = 0
    for ep_idx in range(src.num_episodes):
        info = plan_by_ep.get(ep_idx)
        if info is None:
            print(f"  ep {ep_idx}: no plan entry, skipping")
            continue
        if info["drop"]:
            print(f"  ep {ep_idx}: DROP ({info['reason']})")
            continue
        lead = int(info["lead"])
        trail = int(info["trail"])
        T = int(info["T"])

        ep_row = src.meta.episodes[ep_idx]
        ep_start = int(ep_row["dataset_from_index"])
        ep_end = int(ep_row["dataset_to_index"])
        assert ep_end - ep_start == T, f"ep {ep_idx}: T mismatch {T} vs {ep_end - ep_start}"
        tasks = ep_row.get("tasks", [])
        task_str = tasks[0] if len(tasks) else "unknown"

        # Auto-generated/derived keys lerobot adds itself; we must NOT pass them.
        AUTO_KEYS = {"frame_index", "task_index", "index", "episode_index", "timestamp"}
        # Keys we DO want to forward
        forward_keys = [k for k in src.features.keys() if k not in AUTO_KEYS]

        # Identify image keys for CHW→HWC conversion
        image_keys = set(src.meta.image_keys) | set(src.meta.video_keys)

        kept_in_ep = 0
        for f_idx in range(ep_start + lead, ep_end - trail):
            frame = src[f_idx]
            frame_data = {}
            for k in forward_keys:
                if k not in frame:
                    continue
                v = frame[k]
                if k in image_keys:
                    # src returns (3, H, W) tensor; add_frame wants (H, W, 3)
                    import torch
                    if isinstance(v, torch.Tensor):
                        v = v.permute(1, 2, 0).contiguous()
                    elif hasattr(v, "transpose"):  # numpy
                        v = v.transpose(1, 2, 0)
                frame_data[k] = v
            frame_data["task"] = task_str
            dst.add_frame(frame_data)
            kept_in_ep += 1

        dst.save_episode()
        kept_eps += 1
        kept_frames_total += kept_in_ep
        print(f"  ep {ep_idx}: kept {kept_in_ep}/{T} (trim {lead}+{trail}, task={task_str!r})")

    dst.finalize()
    print(f"\n✓ Rebuilt dataset: {kept_eps} episodes, {kept_frames_total} frames")

    if push_to_hub:
        print(f"\n→ Pushing to HF as {dst_repo}...")
        dst.push_to_hub(private=private)
        print(f"✓ Pushed: https://huggingface.co/datasets/{dst_repo}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-repo", required=True)
    ap.add_argument("--dst-repo", required=True)
    ap.add_argument("--src-root", required=True, type=Path)
    ap.add_argument("--dst-root", required=True, type=Path)
    ap.add_argument("--threshold-deg", type=float, default=1.0)
    ap.add_argument("--min-keep", type=int, default=30)
    ap.add_argument("--max-trim-ratio", type=float, default=0.6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    src_root = args.src_root.expanduser().resolve()
    dst_root = args.dst_root.expanduser().resolve()

    print(f"src: {args.src_repo}  ({src_root})")
    print(f"dst: {args.dst_repo}  ({dst_root})")
    print(f"threshold={args.threshold_deg}°  min_keep={args.min_keep}  max_trim_ratio={args.max_trim_ratio}")

    df = load_action_table(src_root)
    plan = compute_trim_plan(df, args.threshold_deg, args.min_keep, args.max_trim_ratio)
    print_plan(plan, args.src_repo)

    if args.dry_run:
        print("\n[dry-run] no changes written.")
        return

    rebuild(args.src_repo, args.dst_repo, src_root, dst_root, plan,
            push_to_hub=args.push_to_hub, private=args.private)


if __name__ == "__main__":
    main()
