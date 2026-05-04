"""Tiny CNN that regresses 3D positions of red and blue blocks (in the robot
base frame) directly from a wrist-camera RGB image.

Architecture (~250 k params):
    input      (B, 3, 84, 84)
    conv1      3 -> 32, 3x3, padding=1, ReLU, MaxPool 2x2  -> (B, 32, 42, 42)
    conv2     32 -> 64, 3x3, padding=1, ReLU, MaxPool 2x2  -> (B, 64, 21, 21)
    conv3     64 -> 128, 3x3, padding=1, ReLU, AdaptiveAvgPool2d(4)
                                                           -> (B, 128, 4, 4) -> flatten
    fc1       2048 -> 128, ReLU
    fc2       128  -> 6  (linear output, no activation)
    output     (B, 6)  in the order
               [x_red, y_red, z_red, x_blue, y_blue, z_blue],
               in meters, in the robot base frame.

Why pool to 4x4 (not 1x1)
    Earlier the head pooled to 1x1 (= Global Average Pool), which
    collapses ALL spatial information: the FC layer only saw channel
    statistics ("how much red is in the image"), not WHERE the red is.
    The model converged to the mean of the position distribution
    (val_mae ~ (range / 4)) because it had no spatial cue to localize
    the blobs. Pooling to 4x4 keeps a 4x4 grid of feature vectors,
    which preserves enough spatial layout for the regression head to
    learn "red blob in the top-left of the grid -> x_red, y_red ~ ...".

Why output xyz directly (and not pixel coords)
    Approach (B) avoids any table-plane calibration: the CNN learns the
    geometry from the apparent size of the blocks in the image (their
    physical size is fixed at 2 cm, so the apparent size in pixels is a
    direct depth cue). At deploy time, the same image format goes in,
    we get xyz back, and feed it to the policy without any geometry
    recovery in user code.

Training loss
    Plain MSE between the 6 predicted floats and the 6 ground-truth
    floats (in meters). Mean absolute error on each coordinate is also
    reported so we can read it as "the model is wrong by ~1.2 cm on x".
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ColorBlockCNN(nn.Module):
    """Regresses (x_red, y_red, z_red, x_blue, y_blue, z_blue) in robot frame, meters.

    No BatchNorm: with our small dataset (~5k images) and the simple,
    consistent lighting from the sim, BN's running statistics in eval mode
    diverged from the per-batch statistics in train mode and produced
    garbage validation predictions. Removed entirely; the model still
    converges fine.
    """

    def __init__(self, in_channels: int = 3, num_outputs: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Pool to 4x4 instead of 1x1: keeps a 4x4 feature map so the
            # regression head can read WHERE the colored blobs are.
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_outputs),
            # Linear output: targets are in meters in the robot frame, no
            # squashing needed. The MSE loss + Adam will handle the scale.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, 3, H, W) RGB image, values in [0, 1] (float).

        Returns:
            (B, 6) tensor with [x_red, y_red, z_red, x_blue, y_blue, z_blue]
            in meters, in the robot base frame.
        """
        feats = self.features(x)
        return self.head(feats)


def count_parameters(model: nn.Module) -> int:
    """Quick helper for sanity checks."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ColorBlockCNN()
    print(f"ColorBlockCNN — {count_parameters(model):,} trainable parameters")
    dummy = torch.rand(4, 3, 84, 84)
    out = model(dummy)
    print(f"input shape : {tuple(dummy.shape)}")
    print(f"output shape: {tuple(out.shape)}  (expected (4, 6))")
