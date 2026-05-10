"""Custom observation terms for our LeIsaac-based RL tasks.

We expose `wrist_image_features`: a 512-D feature vector produced by the
frozen ResNet-18 encoder applied to the wrist camera RGB output. This is
the key piece of Phase B: the policy gets visual context without paying
the cost of carrying a 224×224×3 image through the rollout buffer.

Why pre-encode in the observation pipeline rather than inside the policy:
the rsl_rl rollout buffer holds num_envs × num_steps × obs_dim floats. A
raw image obs (4096 × 50 × 150528 × 4 bytes ≈ 123 GB) does not fit; a
512-D encoded feature (4096 × 50 × 512 × 4 bytes ≈ 0.4 GB) does. We pay
one ResNet forward per env per step (~4 ms on RTX 5070 for 4096 images),
which is acceptable for training and keeps the policy MLP small.
"""
from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ..policy.visual_encoder import encode_image


def wrist_image_features(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist"),
) -> torch.Tensor:
    """Read RGB from the named camera, encode with frozen ResNet-18.

    Args:
        env: The Isaac Lab env.
        sensor_cfg: SceneEntityCfg pointing at the wrist `TiledCamera`.

    Returns:
        ``(num_envs, 512)`` tensor of visual features. Returns zeros at
        the very first step if the camera buffer hasn't populated yet
        (Isaac Lab fills it on the first sim render after env construction).
    """
    sensor = env.scene[sensor_cfg.name]
    image = sensor.data.output["rgb"]  # (num_envs, H, W, 3) uint8 typically

    # Defensive: if the camera buffer is unexpectedly empty (very first
    # step, before the first render), return zeros instead of crashing.
    if image is None or image.numel() == 0:
        return torch.zeros(env.num_envs, 512, device=env.device, dtype=torch.float32)

    return encode_image(image)


# ---------------------------------------------------------------------------
# Phase C compatibility placeholders — used in V2.10 to pre-allocate obs
# dimensions for `target_color_one_hot` and `bowl_xyz` so a V2.10 checkpoint
# can be warm-started in Phase C without obs-dim mismatch.
# ---------------------------------------------------------------------------


def target_color_zero(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Placeholder for the Phase C target color one-hot (6 colors).

    Returns ``(num_envs, 6)`` zeros. In Phase C this term will be replaced
    with the actual one-hot encoding of the requested color.
    """
    return torch.zeros(env.num_envs, 6, device=env.device, dtype=torch.float32)


def bowl_xyz_zero(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Placeholder for the Phase C bowl xyz position (in robot root frame).

    Returns ``(num_envs, 3)`` zeros. In Phase C this term will be replaced
    with the bowl's actual xyz coordinates from the bowl asset.
    """
    return torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
