"""
add_goal_conditioning.py - augment a LeRobot v3 dataset with goal info as
`observation.environment_state` so ACT can be trained goal-conditioned.

Per-frame the new field is the concatenation:
    [c_yellow, c_orange, c_red, c_blue, c_green, c_violet,  bowl_x_m, bowl_y_m]
                  (6D one-hot color)                        (2D bowl in meters)

Bowl xy convention (matches the TA task strings):
    x = right positive, left negative
    y = forward positive, back negative
Values come straight from the task description "bowl at (X,Y) cm" / 100.

What gets updated:
  - data/chunk-XXX/file-YYY.parquet  : new column observation.environment_state
  - meta/info.json                    : new feature declaration
  - meta/episodes/chunk-XXX/file-YYY.parquet : per-episode stats for the new feature
  - meta/stats.json (if present)      : global stats for the new feature

Output goes to a NEW HF dataset (default: hudela390/<original-name>-goal).

Usage:
    python toolset/add_goal_conditioning.py \\
        --src_repo osammotg1/projet3-eval2-v1-tom-hugo \\
        --dst_repo hudela390/projet3-eval2-v1-goal

    python toolset/add_goal_conditioning.py \\
        --src_repo osammotg1/projet3-eval1-bowl1-v1 \\
        --dst_repo hudela390/projet3-eval1-bowl1-v1-goal
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download


COLORS = ["yellow", "orange", "red", "blue", "green", "violet"]   # canonical order
# Schema is set in main() depending on whether cube_positions.csv is given:
#   no cube_xy: env_state = [color_6, bowl_x, bowl_y]              (8D, eval-2)
#   with cube_xy: env_state = [color_6, bowl_x, bowl_y, cube_x, cube_y]  (10D, eval-1)
ENV_DIM = len(COLORS) + 2
ENV_NAMES = [f"color_{c}" for c in COLORS] + ["bowl_x", "bowl_y"]

TASK_PAT = re.compile(
    r"Pick\s+(\w+)\s+block\s+and\s+place\s+in\s+bowl\s+at\s*\(\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*\)\s*cm",
    re.IGNORECASE,
)


def parse_task(task_desc: str) -> tuple[str, float, float] | None:
    m = TASK_PAT.search(task_desc)
    if not m:
        return None
    color = m.group(1).lower()
    bowl_x_m = float(m.group(2)) / 100.0
    bowl_y_m = float(m.group(3)) / 100.0
    return color, bowl_x_m, bowl_y_m


def env_state_for(color: str, bowl_x_m: float, bowl_y_m: float) -> np.ndarray:
    vec = np.zeros(ENV_DIM, dtype=np.float32)
    if color in COLORS:
        vec[COLORS.index(color)] = 1.0
    vec[-2] = bowl_x_m
    vec[-1] = bowl_y_m
    return vec


def build_task_map(tasks_parquet: Path) -> dict[int, np.ndarray]:
    """task_index -> env_state vector."""
    df = pd.read_parquet(tasks_parquet)
    # tasks.parquet: index = task description, col = task_index
    out: dict[int, np.ndarray] = {}
    for desc, row in df.iterrows():
        idx = int(row["task_index"])
        parsed = parse_task(str(desc))
        if parsed is None:
            print(f"  WARNING: could not parse task '{desc}', using zeros")
            out[idx] = np.zeros(ENV_DIM, dtype=np.float32)
        else:
            color, bx, by = parsed
            out[idx] = env_state_for(color, bx, by)
    return out


# --------------------------------------------------------------------------
# Data parquet augmentation
# --------------------------------------------------------------------------

def augment_data_parquets(data_dir: Path, task_map: dict[int, np.ndarray],
                            cube_map: dict[int, tuple[float, float]] | None = None):
    parquets = sorted(data_dir.rglob("*.parquet"))
    print(f"[goal] {len(parquets)} data shard(s) to augment")
    for pf in parquets:
        df = pd.read_parquet(pf)
        if cube_map is None:
            env_states = np.stack(
                [task_map[int(tid)] for tid in df["task_index"].values]
            )
        else:
            rows = []
            for tid, ep in zip(df["task_index"].values, df["episode_index"].values):
                base = task_map[int(tid)]                  # 8D
                cx, cy = cube_map.get(int(ep), (0.0, 0.0))  # default 0,0 if missing
                rows.append(np.concatenate([base, [cx, cy]]).astype(np.float32))
            env_states = np.stack(rows)
        df["observation.environment_state"] = list(env_states.astype(np.float32))
        df.to_parquet(pf, index=False)
        print(f"  {pf.relative_to(data_dir.parent)}: +{len(df)} rows")


# --------------------------------------------------------------------------
# Episode-meta augmentation (per-episode stats for the new feature)
# --------------------------------------------------------------------------

def _build_episode_stats(values: np.ndarray) -> dict:
    """values: (N_frames, 8). env_state is constant within an episode, so
    min == max == mean == q* and std == 0."""
    v0 = values[0]
    return {
        "stats/observation.environment_state/min":   [v0.tolist()],
        "stats/observation.environment_state/max":   [v0.tolist()],
        "stats/observation.environment_state/mean":  [v0.tolist()],
        "stats/observation.environment_state/std":   [np.zeros_like(v0).tolist()],
        "stats/observation.environment_state/count": [[int(len(values))]],
        "stats/observation.environment_state/q01":   [v0.tolist()],
        "stats/observation.environment_state/q10":   [v0.tolist()],
        "stats/observation.environment_state/q50":   [v0.tolist()],
        "stats/observation.environment_state/q90":   [v0.tolist()],
        "stats/observation.environment_state/q99":   [v0.tolist()],
    }


def augment_episodes_meta(meta_episodes_dir: Path, data_dir: Path,
                          task_map: dict[int, np.ndarray]):
    """Append stats columns for observation.environment_state to each episode-meta parquet."""
    if not meta_episodes_dir.exists():
        print("[goal] no meta/episodes/ - skipping per-episode stats")
        return
    meta_parquets = sorted(meta_episodes_dir.rglob("*.parquet"))
    print(f"[goal] {len(meta_parquets)} episode-meta shard(s) to augment")

    # Load ALL data parquets once to compute per-episode env_state
    data_dfs = [pd.read_parquet(p) for p in sorted(data_dir.rglob("*.parquet"))]
    big_df = pd.concat(data_dfs, ignore_index=True)

    for mp in meta_parquets:
        emp = pd.read_parquet(mp)
        rows = []
        for _, row in emp.iterrows():
            ep_idx = int(row["episode_index"])
            ep_rows = big_df[big_df["episode_index"] == ep_idx]
            if len(ep_rows) == 0:
                # No data for this episode in our shards (shouldn't happen)
                vec = np.zeros(ENV_DIM, dtype=np.float32)
                stats = _build_episode_stats(np.stack([vec]))
            else:
                vals = np.stack(list(ep_rows["observation.environment_state"].values))
                stats = _build_episode_stats(vals)
            rows.append(stats)
        for col in rows[0].keys():
            emp[col] = [r[col][0] for r in rows]
        emp.to_parquet(mp, index=False)
        print(f"  {mp.relative_to(meta_episodes_dir.parent.parent)} ({len(emp)} eps)")


# --------------------------------------------------------------------------
# info.json and stats.json updates
# --------------------------------------------------------------------------

def update_info_json(info_path: Path):
    with open(info_path) as f:
        info = json.load(f)
    info.setdefault("features", {})
    info["features"]["observation.environment_state"] = {
        "dtype":  "float32",
        "shape":  [ENV_DIM],
        "names":  ENV_NAMES,
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[goal] Updated {info_path.name}: + observation.environment_state ({ENV_DIM}D)")


def update_stats_json(stats_path: Path, data_dir: Path):
    if not stats_path.exists():
        return
    with open(stats_path) as f:
        stats = json.load(f)

    # Compute global stats from all data parquets
    data_dfs = [pd.read_parquet(p) for p in sorted(data_dir.rglob("*.parquet"))]
    big = np.stack(
        [np.asarray(v, dtype=np.float32)
         for df in data_dfs
         for v in df["observation.environment_state"].values]
    )   # (total_frames, 8)

    stats["observation.environment_state"] = {
        "min":  big.min(axis=0).tolist(),
        "max":  big.max(axis=0).tolist(),
        "mean": big.mean(axis=0).tolist(),
        "std":  big.std(axis=0).tolist(),
        "count": int(len(big)),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[goal] Updated {stats_path.name} with global env_state stats")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_repo", required=True)
    parser.add_argument("--dst_repo", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--cube_positions_csv", default=None,
                        help="Optional CSV with columns episode,x,y (meters). "
                             "When given, env_state becomes 10D = [color_6, bowl_xy, cube_xy].")
    parser.add_argument("--no_push", action="store_true",
                        help="Don't push to HF, just write local copy")
    parser.add_argument("--private", action="store_true",
                        help="Push as private repo (default: public)")
    args = parser.parse_args()

    # Schema: when cube_xy is provided, env_state becomes 10D. task_map stays
    # 8D (color + bowl); cube_xy gets concatenated per-frame in the augmenter.
    # ENV_DIM/ENV_NAMES are bumped AFTER task_map is built so info.json and
    # stats.json reflect the final 10D schema.
    cube_map: dict[int, tuple[float, float]] | None = None
    if args.cube_positions_csv:
        cdf = pd.read_csv(args.cube_positions_csv)
        cube_map = {int(r["episode"]): (float(r["x"]), float(r["y"])) for _, r in cdf.iterrows()}
        print(f"[goal] cube_xy from {args.cube_positions_csv}: {len(cube_map)} episodes -> "
              f"env_state will be 10D")
    else:
        print(f"[goal] no cube_positions_csv -> env_state stays 8D (color + bowl)")

    print(f"[goal] Downloading {args.src_repo} ...")
    local_dir = Path(snapshot_download(
        repo_id=args.src_repo, repo_type="dataset", cache_dir=args.cache_dir,
        ignore_patterns=["*.git*"],
    ))
    print(f"[goal] Source at {local_dir}")

    # Make a working copy so we never mutate the HF cache
    work_dir = local_dir.parent.parent / (local_dir.name + "_goal")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(local_dir, work_dir)
    print(f"[goal] Working copy: {work_dir}")

    tasks_path = work_dir / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        raise SystemExit(f"Missing {tasks_path}")
    task_map = build_task_map(tasks_path)
    print(f"\n[goal] task_index -> env_state mapping:")
    for idx in sorted(task_map.keys()):
        vec = task_map[idx]
        active = [ENV_NAMES[i] for i, v in enumerate(vec[:6]) if v > 0.5]
        print(f"  task {idx}: color={active[0] if active else '?':7s}  "
              f"bowl_xy=({vec[6]:+.3f}, {vec[7]:+.3f}) m")

    print()
    augment_data_parquets(work_dir / "data", task_map, cube_map=cube_map)

    # Now that env_state is its final width on disk, bump the global schema
    # so info.json + stats.json declare the correct dims (10D if cube_xy, else 8D).
    if cube_map is not None:
        global ENV_DIM, ENV_NAMES
        ENV_DIM = len(COLORS) + 4
        ENV_NAMES = [f"color_{c}" for c in COLORS] + ["bowl_x", "bowl_y", "cube_x", "cube_y"]

    augment_episodes_meta(work_dir / "meta" / "episodes", work_dir / "data", task_map)
    update_info_json(work_dir / "meta" / "info.json")
    update_stats_json(work_dir / "meta" / "stats.json", work_dir / "data")

    if args.no_push:
        print(f"\n[goal] Done. Local copy: {work_dir}")
        return

    print(f"\n[goal] Creating + uploading {args.dst_repo} ...")
    api = HfApi()
    api.create_repo(args.dst_repo, repo_type="dataset",
                    exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=str(work_dir),
        repo_id=args.dst_repo,
        repo_type="dataset",
        commit_message=f"Add goal-conditioning env_state (6D color one-hot + 2D bowl xy)",
    )
    print(f"[goal] Done. https://huggingface.co/datasets/{args.dst_repo}")


if __name__ == "__main__":
    main()
