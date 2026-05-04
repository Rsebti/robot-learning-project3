"""Run the trained perception CNN on a single image and return the xyz of
each colored block in the robot frame.

This is what the deploy script will call at every step on the real robot:

    from sim.eval2.perception.inference import PerceptionPipeline

    perception = PerceptionPipeline(checkpoint="sim/eval2/perception/checkpoint.pt")
    block_red_xyz, block_blue_xyz = perception.predict(rgb_image)  # numpy or torch
    # both are (3,) tensors in meters, robot base frame.

The same pipeline is used to validate the model in sim before any
real-robot deployment.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .model import ColorBlockCNN


class PerceptionPipeline:
    """Wraps the trained CNN with image preprocessing.

    Args:
        checkpoint: Path to a .pt file produced by train.py.
        device: "cuda" or "cpu". Auto-detected if None.
        image_size: Side length of the network input. Must match training.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device | None = None,
        image_size: int = 84,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.image_size = image_size

        self.model = ColorBlockCNN().to(self.device).eval()
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

    @torch.no_grad()
    def predict(self, image) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            image: RGB image. Accepts:
                - torch.Tensor of shape (H, W, 3) or (3, H, W), uint8 or float
                - numpy array of shape (H, W, 3) uint8

        Returns:
            (block_red_xyz, block_blue_xyz), each a (3,) torch tensor on CPU,
            in meters, in the robot base frame.
        """
        x = self._preprocess(image)  # (1, 3, image_size, image_size) float on device
        out = self.model(x).squeeze(0).cpu()  # (6,)
        return out[:3], out[3:]

    def _preprocess(self, image) -> torch.Tensor:
        # Handle numpy array
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        # Detect layout: (H, W, 3) -> CHW. (3, H, W) -> already CHW.
        if image.ndim == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        # uint8 -> float in [0, 1]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            image = image.float()
            if image.max() > 1.5:
                image = image / 255.0
        # Resize to network input size.
        image = image.unsqueeze(0)  # (1, 3, H, W)
        if image.shape[-1] != self.image_size or image.shape[-2] != self.image_size:
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return image.to(self.device)


if __name__ == "__main__":
    # Smoke test on random noise — should run without crashing and return
    # 2x (3,) tensors. The values are nonsense, the test just checks plumbing.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path(__file__).parent / "checkpoint.pt"),
    )
    args = parser.parse_args()

    p = PerceptionPipeline(checkpoint=args.checkpoint)
    fake_img = torch.randint(0, 256, (84, 84, 3), dtype=torch.uint8)
    red_xyz, blue_xyz = p.predict(fake_img)
    print(f"red_xyz   shape={tuple(red_xyz.shape)} value={red_xyz.tolist()}")
    print(f"blue_xyz  shape={tuple(blue_xyz.shape)} value={blue_xyz.tolist()}")
