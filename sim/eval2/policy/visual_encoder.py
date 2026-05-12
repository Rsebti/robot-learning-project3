"""Frozen ResNet-18 (ImageNet) visual encoder, used as the wrist-cam
feature extractor for our Phase B policy.

Architecture choice: standard ResNet-18 from torchvision, with the final
fully-connected classification head dropped, leaving a 512-dimensional
feature vector after the global average pooling layer. The encoder is
loaded once, frozen (`requires_grad=False`), and run in `eval()` mode so
its BatchNorm statistics stay at their ImageNet values — important for
on-policy RL where mini-batches are correlated and would otherwise corrupt
running BatchNorm stats.

Why frozen rather than fine-tuned: the wrist camera images in our LeIsaac
table-with-cube scene are visually similar enough to ImageNet (real
textures, lit objects, sharp depth) that the off-the-shelf features carry
useful information for "where is the cube relative to the gripper". Fine
tuning the encoder during PPO would (a) blow up the parameter count being
optimized, (b) risk catastrophic forgetting of the ImageNet prior, and
(c) require BatchNorm→GroupNorm conversion to be PPO-stable.

Why a module-level singleton instead of a per-env instance: there is one
ResNet shared by all 4096 envs at every step. Instantiating it in the
ManagerBasedRLEnv config object would build it on each gym.make() call
which fights the PEP 526 attribute typing of `@configclass`. The
singleton is built lazily on the first `encode_image()` call (so config
parsing stays cheap) and gets moved to whatever device the input image
tensor lives on.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Lazy-loaded singleton instance (filled in by `_lazy_init`).
_ENCODER: nn.Module | None = None
_ENCODER_DEVICE: torch.device | None = None
_MEAN: torch.Tensor | None = None
_STD: torch.Tensor | None = None


def _lazy_init(device: torch.device) -> None:
    """Build (or move) the ResNet-18 to the requested device."""
    global _ENCODER, _ENCODER_DEVICE, _MEAN, _STD

    if _ENCODER is not None and _ENCODER_DEVICE == device:
        return

    if _ENCODER is None:
        # Import here so that env_cfg parsing (which imports this module
        # transitively) doesn't pull torchvision and download weights.
        import torchvision.models as models

        resnet = models.resnet18(weights="IMAGENET1K_V1")
        # Drop the final fully-connected classification head; keep up to
        # the global average pooling layer (output shape: B x 512 x 1 x 1).
        encoder = nn.Sequential(*list(resnet.children())[:-1])
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        _ENCODER = encoder

    _ENCODER = _ENCODER.to(device)
    _ENCODER_DEVICE = device

    # ImageNet normalization stats (broadcast against (B, 3, H, W)).
    _MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


@torch.no_grad()
def encode_image(image: torch.Tensor) -> torch.Tensor:
    """Encode a batch of RGB images through the frozen ResNet-18.

    Accepts both Isaac Lab's NHWC convention (output of `TiledCamera.data
    .output["rgb"]`) and the more standard NCHW. Auto-detects the channel
    axis. Auto-detects whether values are uint8/0–255 or already in [0, 1].
    Resizes to 224x224 if the input is a different resolution.

    Args:
        image: ``(B, H, W, 3)`` or ``(B, 3, H, W)`` tensor, dtype uint8 or
            float in either [0, 1] or [0, 255].

    Returns:
        ``(B, 512)`` feature tensor.
    """
    if image.numel() == 0:
        # Empty batch — return shape-compatible empty features.
        return torch.empty(0, 512, device=image.device, dtype=torch.float32)

    _lazy_init(image.device)

    x = image
    # uint8 → float in [0, 1]
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    elif x.dtype != torch.float32:
        x = x.float()

    # Heuristic: if the float range looks like 0–255 (typical raw GPU
    # output from Isaac Lab cameras), normalize to 0–1.
    if x.dtype == torch.float32 and x.max() > 2.0:
        x = x / 255.0

    # NHWC → NCHW
    if x.shape[-1] == 3 and x.shape[1] != 3:
        x = x.permute(0, 3, 1, 2).contiguous()

    # Resize to 224x224 if needed (ResNet's pretrained input size).
    if x.shape[-1] != 224 or x.shape[-2] != 224:
        x = torch.nn.functional.interpolate(
            x, size=(224, 224), mode="bilinear", align_corners=False
        )

    # ImageNet normalize and forward through the encoder.
    x = (x - _MEAN) / _STD
    feats = _ENCODER(x)              # (B, 512, 1, 1)
    return feats.squeeze(-1).squeeze(-1)  # (B, 512)


def get_frozen_resnet18(device: torch.device | str = "cuda") -> nn.Module:
    """Public accessor (mostly for tests / sanity checks)."""
    if isinstance(device, str):
        device = torch.device(device)
    _lazy_init(device)
    return _ENCODER  # type: ignore[return-value]
