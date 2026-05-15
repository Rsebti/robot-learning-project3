"""Observation terms — match Squint's PlaceCube obs EXACTLY.

After ``FlattenRGBDObservationWrapper`` strips everything but the canonical
``state`` and ``rgb`` keys, Squint's policy sees:

    obs["state"] : float32, shape (12,)
        = [noisy_qpos[0..5], controller_target_qpos[0..5]]
    obs["rgb"]   : uint8,   shape (16, 16, 3)
        downsampled wrist camera (raw 128x128 -> 16x16, mode='area')

This file exposes those two terms ONLY, in the same byte layout. Anything
else (item_pose, bin_pose, ...) is INTERNAL to the env's reward / debug;
the policy never reads it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# State observations (qpos | target_qpos)
# ---------------------------------------------------------------------------


def joint_pos_with_noise(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    noise_std: float = 0.0,
) -> torch.Tensor:
    """Robot qpos.  Squint adds Gaussian noise (std = 5deg) when DR is on; we
    leave it at 0 for the deploy-mode env so the obs matches the canonical
    audit dump byte-for-byte. Pass ``noise_std`` > 0 if you reintroduce DR.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    qpos = asset.data.joint_pos
    if noise_std > 0:
        qpos = qpos + torch.randn_like(qpos) * noise_std
    return qpos


NUM_COLORS = 6
COLOR_PALETTE = (
    (1.0, 0.0, 0.0),  # 0 red
    (0.0, 0.0, 1.0),  # 1 blue
    (0.0, 1.0, 0.0),  # 2 green
    (1.0, 1.0, 0.0),  # 3 yellow
    (0.6, 0.0, 0.8),  # 4 purple
    (1.0, 0.5, 0.0),  # 5 orange
)


def goal_color_one_hot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """6-d goal-color one-hot vector — reads ``env._goal_color_idx``
    (set per-episode by ``reset_goal_and_distractor_colors`` event)."""
    one_hot = torch.zeros(env.num_envs, NUM_COLORS, device=env.device, dtype=torch.float32)
    if hasattr(env, "_goal_color_idx") and env._goal_color_idx is not None:
        gi = env._goal_color_idx
        one_hot.scatter_(1, gi.view(-1, 1), 1.0)
    else:
        one_hot[:, 0] = 1.0  # fallback: red
    return one_hot


def controller_target_qpos(
    env: "ManagerBasedRLEnv",
    action_term_name: str = "arm_and_gripper",
) -> torch.Tensor:
    """Integrated joint-position target maintained by our delta-target action.

    Mirrors Squint's ``agent.controller.get_state()`` for the
    ``pd_joint_target_delta_pos`` controller, which returns the controller's
    internal ``_target_qpos`` tensor.

    Important: this MUST come from the same source as Squint, otherwise
    state[6:12] will be off and the policy will see a different obs.
    """
    action_term = env.action_manager.get_term(action_term_name)
    return action_term.target_qpos


def bowl_xyz_world(
    env: "ManagerBasedRLEnv",
    asset_name: str = "bowl",
) -> torch.Tensor:
    """Bowl root position in env-local frame (3 floats per env).

    Goal-conditioning extra: the bowl is only marginally visible from the
    wrist cam at home pose (FOV covers the cube spawn zone, not necessarily
    the bowl spawn zone), so we expose its xyz to the policy directly via
    the state vector. This lets the policy plan the place phase without
    needing to "see" the bowl on cam first.

    Important — frame choice: Isaac Lab's ``root_pos_w`` is the simulator's
    GLOBAL world frame. With N parallel envs laid out on a grid (spacing
    ``env.scene.env_origins``), every env would see a different absolute
    bowl xyz even when the bowl is in the same place relative to its robot.
    We subtract ``env_origins`` so the value is in the env-local frame —
    which equals the robot-base frame here because the SO-101 base is
    anchored at each env's origin. The policy then sees a consistent
    "bowl xyz wrt my robot" signal across the batch.

    For training warmstart from Squint ckpt 8 (which has 18-d state and
    does NOT include bowl_xyz), pad the state_proj weight with zeros on
    these 3 new input columns — the policy ignores bowl_xyz initially and
    learns to use it through gradient updates.
    """
    bowl = env.scene[asset_name]
    return bowl.data.root_pos_w[:, :3] - env.scene.env_origins


# ---------------------------------------------------------------------------
# Image observation (16x16 RGB, downsampled with area-mode interpolation)
# ---------------------------------------------------------------------------


def _apply_color_jitter(
    rgb: torch.Tensor,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.3,
    hue: float = 0.05,
) -> torch.Tensor:
    """Vectorised color jitter on a (N, H, W, 3) uint8 batch — replicates
    Squint's ``utils.ColorJitterWrapper`` (torchvision ColorJitter) but in
    pure PyTorch so it runs on GPU without a torchvision dependency.

    Sampling matches torchvision conventions:
    - brightness ∈ uniform[max(0, 1-b), 1+b]
    - contrast   ∈ uniform[max(0, 1-c), 1+c]
    - saturation ∈ uniform[max(0, 1-s), 1+s]
    - hue        ∈ uniform[-h, +h]  (applied as a shift in HSV space)

    Per-env (independent) for sim2real diversity across the batch.
    """
    if not torch.is_grad_enabled():
        # Cheap path; we don't need gradients through this anyway.
        pass
    n = rgb.shape[0]
    device = rgb.device
    img = rgb.to(torch.float32) / 255.0  # (N, H, W, 3) in [0, 1]

    # Brightness × contrast on luminance.
    b = torch.empty(n, 1, 1, 1, device=device).uniform_(max(0.0, 1.0 - brightness), 1.0 + brightness)
    img = (img * b).clamp(0.0, 1.0)

    c = torch.empty(n, 1, 1, 1, device=device).uniform_(max(0.0, 1.0 - contrast), 1.0 + contrast)
    mean = img.mean(dim=(1, 2, 3), keepdim=True)
    img = ((img - mean) * c + mean).clamp(0.0, 1.0)

    # Saturation: scale chroma against per-pixel grey.
    s = torch.empty(n, 1, 1, 1, device=device).uniform_(max(0.0, 1.0 - saturation), 1.0 + saturation)
    grey = img.mean(dim=-1, keepdim=True)
    img = (grey + (img - grey) * s).clamp(0.0, 1.0)

    # Hue shift via simple RGB rotation (cheap proxy for HSV rotation).
    # For hue=0.05 this is a small ±0.05 * 2π rotation around the [1,1,1] axis.
    if hue > 0:
        theta = torch.empty(n, device=device).uniform_(-hue, +hue) * 2 * 3.141593
        cosT = theta.cos().view(n, 1, 1, 1)
        sinT = theta.sin().view(n, 1, 1, 1)
        # Mean-axis rotation: R_axis(theta) applied to (img - mean) + mean.
        # Simplified to per-channel matrix mul — approximate but visually similar.
        m = img.mean(dim=-1, keepdim=True)
        centred = img - m
        rolled = torch.stack([centred[..., 1], centred[..., 2], centred[..., 0]], dim=-1)
        img = (m + cosT * centred + sinT * rolled).clamp(0.0, 1.0)

    return (img * 255.0).clamp(0.0, 255.0).to(torch.uint8)


def wrist_rgb_16(
    env: "ManagerBasedRLEnv",
    camera_name: str = "wrist",
    greenscreen_bg_rgb: tuple[int, int, int] | None = None,
    greenscreen_keep_keywords: tuple[str, ...] = ("Cube", "Bin", "Robot"),
    apply_jitter: bool = False,
) -> torch.Tensor:
    """16x16 RGB from the wrist camera, matching Squint's deploy pipeline.

    Squint downsamples the raw 128x128 wrist RGB to 16x16 with
    ``F.interpolate(..., mode='area')`` (the canonical audit confirmed this).

    If ``greenscreen_bg_rgb`` is set (e.g. ``(184, 173, 169)`` for Squint's
    #B8ADA9 taupe), this also composites the RGB over a solid background:
    pixels whose segmentation ID does NOT correspond to a "kept" object
    (matched by ``greenscreen_keep_keywords`` against the seg-id labels)
    are replaced with the background colour BEFORE the area downsample.

    Returns
    -------
    torch.Tensor
        Shape ``(N, 16, 16, 3)``, dtype ``uint8``.
    """
    cam = env.scene.sensors[camera_name]
    rgb = cam.data.output["rgb"]  # (N, H, W, 3 or 4) uint8
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    if greenscreen_bg_rgb is not None:
        seg_key = None
        for cand in ("instance_segmentation_fast", "semantic_segmentation",
                     "instance_segmentation"):
            if cand in cam.data.output:
                seg_key = cand
                break
        if seg_key is not None:
            seg = cam.data.output[seg_key]  # (N, H, W) or (N, H, W, 1) int
            if seg.dim() == 4 and seg.shape[-1] == 1:
                seg = seg.squeeze(-1)
            # Build the set of seg IDs to KEEP based on the per-camera info
            # mapping ``id -> {'class': prim_path}`` populated by the renderer.
            keep_ids: set[int] = set()
            try:
                info_for_env0 = cam.data.info[0].get(seg_key, None)
                if isinstance(info_for_env0, dict):
                    id_to_label = info_for_env0.get("idToLabels", info_for_env0)
                    for sid, lbl in id_to_label.items():
                        try:
                            sid_int = int(sid)
                        except (ValueError, TypeError):
                            continue
                        lbl_str = str(lbl)
                        if any(kw.lower() in lbl_str.lower() for kw in greenscreen_keep_keywords):
                            keep_ids.add(sid_int)
            except Exception:
                pass

            if keep_ids:
                keep_tensor = torch.tensor(sorted(keep_ids), device=seg.device, dtype=seg.dtype)
                mask = torch.isin(seg, keep_tensor)  # (N, H, W) bool
                bg = torch.tensor(greenscreen_bg_rgb, device=rgb.device, dtype=rgb.dtype).view(1, 1, 1, 3)
                rgb = torch.where(mask.unsqueeze(-1), rgb, bg.expand_as(rgb))

    rgb_chw = rgb.permute(0, 3, 1, 2).float()
    rgb_16 = F.interpolate(rgb_chw, size=(16, 16), mode="area")
    rgb_16 = rgb_16.permute(0, 2, 3, 1).to(torch.uint8)
    if apply_jitter:
        rgb_16 = _apply_color_jitter(rgb_16)
    return rgb_16
