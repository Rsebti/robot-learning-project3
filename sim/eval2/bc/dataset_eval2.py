"""Wrap the lerobot teleop dataset into (obs_15d, action_6d) tensors for BC.

Source dataset: ``osammotg1/projet3-eval2-v1`` on HF.
Layout per frame:
    observation.state      : (6,)  joint_pos in DEGREES (raw teleop scale)
    action                 : (6,)  next-step joint targets (degrees)
    task_index             : ()    int64, 0..5 ; one of the 6 colors
    episode_index          : ()    int64
    observation.images.wrist : (3, 480, 640)  ← we DO NOT load this (too slow,
                              we said "BC sans caméra")

What we build per sample:
    obs (15,) = [joint_pos (6), target_color_one_hot (6), bowl_xyz (3)]
    target    = action (6,)

Bowl xyz is constant in the data (Federico's teleop bowl was fixed). We
inject a placeholder ``BOWL_XYZ_TELEOP`` here — the policy will see ONE
value during BC and learn to associate it with the demo trajectories.
RL fine-tuning will randomize it later.

Implementation note — speed:
    LeRobotDataset's __getitem__ decodes the wrist video on the fly,
    which is brutally slow (~10 ms/frame on this PC = ~4 min to walk all
    24k frames once). We bypass this by reading the underlying HF parquet
    dataset (``ds.hf_dataset``) which has the action / state / task_index
    columns WITHOUT touching the video. ~2 s to load everything in RAM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


# Mapping task_index → color name (from meta/tasks.parquet, see
# scripts/inspect_eval2_dataset.py output for confirmation):
TASK_INDEX_TO_COLOR: dict[int, str] = {
    0: "blue",
    1: "green",
    2: "violet",
    3: "yellow",
    4: "red",
    5: "orange",
}
NUM_COLORS = len(TASK_INDEX_TO_COLOR)

# Bowl position used during teleop (placeholder — actual value unknown,
# Federico had the bowl at one fixed pose). Picked to match the typical
# bowl position used in the Eval 2 sim env. The numerical value barely
# matters for BC: every demo sees the same vector, so the policy can
# treat it as a constant. What matters is that during RL warmstart the
# input slot exists (so PPO can vary it).
BOWL_XYZ_TELEOP: tuple[float, float, float] = (0.20, -0.15, 0.020)

OBS_DIM = 6 + NUM_COLORS + 3  # 15
ACTION_DIM = 6


@dataclass
class NormalizationStats:
    """Per-feature mean/std used inside the policy."""

    obs_mean: torch.Tensor   # (15,)
    obs_std: torch.Tensor    # (15,)
    action_mean: torch.Tensor  # (6,)
    action_std: torch.Tensor   # (6,)


class Eval2BCDataset(Dataset):
    """In-memory tensor dataset of (obs, action) pairs from the teleop demos.

    Args:
        repo_id:   HF dataset id.
        episode_indices: optional iterable of episode indices to keep
                          (used for train/val split). If None, keep all.
    """

    def __init__(
        self,
        repo_id: str = "osammotg1/projet3-eval2-v1",
        episode_indices: list[int] | None = None,
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id)
        # Direct access to the underlying HF parquet dataset — no video.
        hf = ds.hf_dataset

        # Vectorized columns (we have 24k rows, fits easily in RAM).
        # action / state are stored as Python lists per row; convert to
        # numpy then torch for batch operations.
        states = np.asarray(hf["observation.state"], dtype=np.float32)   # (N, 6)
        actions = np.asarray(hf["action"], dtype=np.float32)              # (N, 6)
        task_idx = np.asarray(hf["task_index"], dtype=np.int64)           # (N,)
        episode_idx = np.asarray(hf["episode_index"], dtype=np.int64)     # (N,)

        # Optional split: keep only frames whose episode is in episode_indices.
        if episode_indices is not None:
            keep = np.isin(episode_idx, np.asarray(episode_indices, dtype=np.int64))
            states = states[keep]
            actions = actions[keep]
            task_idx = task_idx[keep]
            episode_idx = episode_idx[keep]

        # Build the obs vector per sample.
        n = states.shape[0]
        color_oh = np.zeros((n, NUM_COLORS), dtype=np.float32)
        color_oh[np.arange(n), task_idx] = 1.0

        bowl_xyz = np.tile(
            np.asarray(BOWL_XYZ_TELEOP, dtype=np.float32), (n, 1)
        )

        obs = np.concatenate([states, color_oh, bowl_xyz], axis=1)         # (N, 15)
        assert obs.shape[1] == OBS_DIM

        self.states = torch.from_numpy(states)            # (N, 6)
        self.actions = torch.from_numpy(actions)          # (N, 6)
        self.obs = torch.from_numpy(obs)                  # (N, 15)
        self.task_idx = torch.from_numpy(task_idx)        # (N,)
        self.episode_idx = torch.from_numpy(episode_idx)  # (N,)

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.obs[i], self.actions[i]

    # ------------------------------------------------------------------
    def compute_normalization(self) -> NormalizationStats:
        """Compute per-feature mean/std for the policy's internal normalizer.

        - For joint_pos slots [0..5] : data-driven (degrees, large scale).
        - For color one-hot   [6..11]: identity (mean=0, std=1) — already
          well-scaled.
        - For bowl_xyz        [12..14]: identity. The value is constant
          here; we don't want std=0 (division blows up). The policy clamps
          std to 1e-6 anyway, but identity is cleaner.

        Action stats are data-driven (raw degrees).
        """
        obs_mean = torch.zeros(OBS_DIM)
        obs_std = torch.ones(OBS_DIM)
        # Joints (slots 0..5)
        obs_mean[:6] = self.states.mean(dim=0)
        obs_std[:6] = self.states.std(dim=0).clamp(min=1.0)  # >=1 deg
        # color one-hot and bowl xyz: identity (already set)

        action_mean = self.actions.mean(dim=0)
        action_std = self.actions.std(dim=0).clamp(min=1.0)

        return NormalizationStats(
            obs_mean=obs_mean,
            obs_std=obs_std,
            action_mean=action_mean,
            action_std=action_std,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def make_train_val_split(
        repo_id: str = "osammotg1/projet3-eval2-v1",
        val_fraction: float = 0.1,
        seed: int = 0,
    ) -> tuple["Eval2BCDataset", "Eval2BCDataset"]:
        """Episode-level split — important to avoid train/val leakage.

        Frames within an episode are temporally correlated, so a frame-level
        split would put adjacent frames in train and val and over-estimate
        the val accuracy.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id)
        n_eps = ds.meta.total_episodes
        rng = np.random.default_rng(seed)
        ep_perm = rng.permutation(n_eps)
        n_val = max(1, int(round(n_eps * val_fraction)))
        val_eps = sorted(ep_perm[:n_val].tolist())
        train_eps = sorted(ep_perm[n_val:].tolist())
        train = Eval2BCDataset(repo_id=repo_id, episode_indices=train_eps)
        val = Eval2BCDataset(repo_id=repo_id, episode_indices=val_eps)
        return train, val
