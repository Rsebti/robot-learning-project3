#!/usr/bin/env python3
"""Inspect first-frame joint state of a teleop dataset to diagnose
calibration mismatch between this machine and the dataset-recording machine.

Context: SO-101 calibration JSONs live at
~/.cache/huggingface/lerobot/calibration/robots/so_follower/ and are
machine-local. A policy trained on data recorded with calibration A,
deployed on a machine with calibration B, will silently produce a
physical pose offset (often 180 deg on shoulder_pan if the operator
zeroed the arm in the opposite reference orientation).

How to use the output:
  1. Run this script. It prints the per-joint angles (degrees) the
     demonstrator's follower was in at the start of each episode.
  2. Manually back-drive THIS machine's follower to those angles
     (e.g. via the leader + lerobot teleop).
  3. If the resulting physical pose matches what you remember as the
     teleop home pose -> calibration is consistent across machines.
     If it lands in a mirrored/rotated pose -> calibration mismatch
     confirmed, and the offending joint is whichever one is rotated.

Also prints the dataset-wide per-joint min/max. A large negative or
positive span that this machine's calibration JSON would map to
joint-limit violations is another diagnostic signal.
"""

import argparse
import sys

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# SO-101 follower joint order in observation.state / action.
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="osammotg1/projet3-eval2-v1-tom-hugo",
        help="HF dataset repo id to inspect.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="How many episode-starts to sample for the home-pose distribution.",
    )
    parser.add_argument(
        "--range-stride",
        type=int,
        default=10,
        help="Stride for dataset-wide range scan (1 = every frame).",
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.repo_id}")
    try:
        ds = LeRobotDataset(args.repo_id)
    except Exception as e:
        print(f"\nERROR loading dataset: {e}", file=sys.stderr)
        print(
            "Hints: confirm the dataset is cached at "
            "~/.cache/huggingface/hub/datasets--<owner>--<name>/, or "
            "that `huggingface-cli login` is configured.",
            file=sys.stderr,
        )
        return 1

    n_eps_total = ds.num_episodes
    n_eps = min(args.episodes, n_eps_total)
    print(
        f"Dataset has {n_eps_total} episodes, {len(ds)} frames. "
        f"Sampling first frame of {n_eps} episodes."
    )

    # ---- first frame of each sampled episode --------------------------------
    # lerobot 0.5.x: episode boundaries live in ds.meta.episodes (an arrow Dataset)
    # under the dataset_from_index / dataset_to_index columns.
    first_frames = []
    for ep in range(n_eps):
        start_idx = int(ds.meta.episodes[ep]["dataset_from_index"])
        state = ds[start_idx]["observation.state"].numpy()
        first_frames.append(state)
    first_frames = np.stack(first_frames)  # (n_eps, 6)

    print("\n=== First-frame state per episode (degrees) ===")
    header = f"{'joint':14s} " + "  ".join(f"ep{i:02d}".rjust(8) for i in range(n_eps))
    print(header)
    for j, name in enumerate(JOINT_NAMES):
        row = "  ".join(f"{v:+8.2f}" for v in first_frames[:, j])
        print(f"{name:14s} {row}")

    print("\n=== Home-pose summary across sampled episodes (degrees) ===")
    print(f"{'joint':14s} {'mean':>10s} {'std':>8s} {'min':>10s} {'max':>10s}")
    for j, name in enumerate(JOINT_NAMES):
        col = first_frames[:, j]
        print(
            f"{name:14s} {col.mean():+10.2f} {col.std():8.2f} "
            f"{col.min():+10.2f} {col.max():+10.2f}"
        )

    # ---- dataset-wide range (strided, to bound memory/time) -----------------
    total = len(ds)
    indices = list(range(0, total, args.range_stride))
    print(
        f"\nScanning {len(indices)} of {total} frames "
        f"(stride={args.range_stride}) for per-joint range..."
    )
    sampled = np.stack(
        [ds[i]["observation.state"].numpy() for i in indices]
    )  # (n_sampled, 6)

    print(
        f"\n=== Per-joint range across full dataset "
        f"(strided, {sampled.shape[0]} frames) ===\n"
    )
    print(f"{'joint':14s} {'min':>10s} {'max':>10s} {'span':>10s}")
    for j, name in enumerate(JOINT_NAMES):
        col = sampled[:, j]
        print(f"{name:14s} {col.min():+10.2f} {col.max():+10.2f} {col.max()-col.min():10.2f}")

    print(
        "\nNext step: back-drive this machine's follower to the 'mean' home pose "
        "above and visually compare to the teleop home pose. If the arm lands "
        "mirrored 180 deg, the joint with the inconsistent sign is the offender."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
