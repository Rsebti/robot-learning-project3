"""Inspect the Eval 2 teleop dataset (Federico's HF dataset) structure.

Usage (from the conda `lerobot` env on this PC):

    conda activate lerobot
    python scripts/inspect_eval2_dataset.py

Prints:
- top-level metadata (n_episodes, n_frames, fps, feature keys)
- the 6 task strings (mapped by task_index)
- per-episode metadata for the first 5 episodes (target task + length)
- the structure of a single frame (key shapes/dtypes/values)

This tells us how to map (task_index) -> (target_color, bowl_xyz),
which we need before writing the BC training pipeline.
"""

from __future__ import annotations

import json

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "osammotg1/projet3-eval2-v1"


def main() -> None:
    print(f"loading dataset: {REPO_ID}")
    ds = LeRobotDataset(REPO_ID)

    print("\n" + "=" * 60)
    print("METADATA")
    print("=" * 60)
    print(f"total_episodes = {ds.meta.total_episodes}")
    print(f"total_frames   = {ds.meta.total_frames}")
    print(f"fps            = {ds.meta.fps}")
    print(f"feature keys   = {list(ds.meta.features.keys())}")

    print("\n" + "=" * 60)
    print("TASKS (the 6 tasks):")
    print("=" * 60)
    tasks = ds.meta.tasks
    if hasattr(tasks, "items"):
        for task_idx, task_text in tasks.items():
            print(f"  task_index={task_idx} -> {task_text!r}")
    else:
        # fallback: iterate as dict-like or list
        try:
            for task_idx, task_text in enumerate(tasks):
                print(f"  task_index={task_idx} -> {task_text!r}")
        except TypeError:
            print(f"  raw tasks object: {tasks!r}")

    print("\n" + "=" * 60)
    print("FIRST 5 EPISODES METADATA:")
    print("=" * 60)
    eps = ds.meta.episodes
    for ep_idx in range(min(5, ds.meta.total_episodes)):
        try:
            ep = eps[ep_idx]
            print(f"  ep{ep_idx}: {json.dumps(ep, default=str)[:300]}")
        except Exception as e:
            print(f"  ep{ep_idx}: <could not access: {type(e).__name__}: {e}>")
            break

    print("\n" + "=" * 60)
    print("SAMPLE FRAME (frame 0):")
    print("=" * 60)
    frame = ds[0]
    for k, v in frame.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v!r}")

    print("\n" + "=" * 60)
    print("FRAME 0 — observation.state values:")
    print("=" * 60)
    state = frame["observation.state"]
    action = frame["action"]
    print(f"  observation.state = {state.tolist()}")
    print(f"  action            = {action.tolist()}")

    print("\n" + "=" * 60)
    print("PER-EPISODE TASK_INDEX DISTRIBUTION:")
    print("=" * 60)
    # Count episodes per task by looking at first frame of each episode
    from collections import Counter
    ep_to_task = []
    for ep_idx in range(ds.meta.total_episodes):
        try:
            ep = eps[ep_idx]
            # episode metadata typically has 'tasks' or 'task_index' field
            ep_to_task.append((ep_idx, ep))
        except Exception:
            break
    if ep_to_task:
        # Show first 10 to inspect structure
        print("  (first 10 episodes, raw metadata):")
        for ep_idx, ep in ep_to_task[:10]:
            print(f"    ep{ep_idx}: {ep!r}")


if __name__ == "__main__":
    main()
