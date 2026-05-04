"""PyTorch Dataset that wraps the .pt file produced by capture_dataset.py."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class BlockPositionDataset(Dataset):
    """Loads (image, target) pairs from a .pt file.

    Args:
        data_path: Path to the .pt file. Must contain keys ``images`` (uint8,
            shape (N, H, W, 3)) and ``targets`` (float32, shape (N, 6)).
        augment: If True, applies brightness jitter only. We don't horizontal-
            flip because the targets are 3D world-frame positions — flipping
            the image without also transforming the targets in 3D would
            corrupt the labels.
    """

    def __init__(self, data_path: str | Path, augment: bool = False):
        data = torch.load(data_path, map_location="cpu")
        # (N, H, W, 3) uint8 -> keep on CPU; convert to float on __getitem__
        self.images = data["images"]
        self.targets = data["targets"]
        self.augment = augment
        assert self.images.shape[0] == self.targets.shape[0]
        assert self.targets.shape[1] == 6, (
            f"Expected 6-D targets (xyz_red + xyz_blue), got {self.targets.shape[1]}."
        )

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        # uint8 HWC -> float CHW in [0, 1]
        img = self.images[idx].permute(2, 0, 1).float() / 255.0  # (3, H, W)
        target = self.targets[idx].clone()                        # (6,)

        if self.augment:
            # Brightness jitter: scale by a factor in [0.8, 1.2], clamp to [0, 1].
            img = (img * (0.8 + 0.4 * torch.rand(1).item())).clamp(0, 1)
            # No horizontal flip for 3D targets.

        return img, target
