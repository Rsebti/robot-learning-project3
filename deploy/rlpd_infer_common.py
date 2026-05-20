"""Load Squint RLPD checkpoints and build deploy observations (project3)."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_DEPLOY = Path(__file__).resolve().parent
_PROJECT = _DEPLOY.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# Squint RLPD image / action contract (train_rlpd.py defaults)
IMAGE_HW_CANDIDATES = [(80, 144), (36, 64), (32, 42), (64, 64), (16, 16)]
CONTROL_HZ = 30
DELTA_CAP = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.20], dtype=np.float32)
N_COLORS = 6
GRASP_THRESHOLD_RAD = 0.5

_STATE_SLICES_58 = {
    "noisy_qpos": (0, 6),
    "controller_target": (6, 12),
    "goal_color": (12, 18),
    "bowl_xyz_robot_frame": (18, 21),
    "qvel": (21, 27),
    "is_item_grasped": (27, 28),
    "tcp_pose": (42, 49),
}


@dataclass
class RlpdCheckpointMeta:
    path: str
    global_step: int | str
    n_state: int
    n_act: int
    image_h: int
    image_w: int
    encoder_repr_dim: int
    use_bowl_xyz: bool
    colour_conditioned: bool
    is_full_state_58: bool
    ckpt_keys: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _manifest_for(ckpt: Path) -> dict[str, Any]:
    mp = ckpt.with_name(ckpt.stem + ".manifest.json")
    if mp.is_file():
        return json.loads(mp.read_text(encoding="utf-8"))
    return {}


def is_rlpd_checkpoint(ckpt: dict) -> bool:
    if not isinstance(ckpt, dict):
        return False
    if "encoder" not in ckpt or "actor" not in ckpt:
        return False
    if "critic" in ckpt:
        return True
    enc = ckpt["encoder"]
    return "conv.4.weight" in enc and int(enc["conv.0.weight"].shape[-1]) >= 7


def resolve_rlpd_checkpoint(
    ckpt: str | Path | None = None,
    *,
    name: str | None = None,
) -> Path:
    if ckpt is not None:
        p = Path(ckpt).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    if name:
        for base in (Path.home() / "squint-rlpd" / "runs", _PROJECT.parent / "squint-rlpd" / "runs"):
            if not base.is_dir():
                continue
            hits = sorted(base.glob(f"**/{name}/ckpt*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
            if hits:
                return hits[0].resolve()
            hits = sorted(base.glob(f"**/*{name}*/ckpt*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
            if hits:
                return hits[0].resolve()
        raise FileNotFoundError(f"No run matching name={name!r} under squint-rlpd/runs")
    for base in (Path.home() / "squint-rlpd" / "runs", _PROJECT.parent / "squint-rlpd" / "runs"):
        if not base.is_dir():
            continue
        for pattern in ("**/ckpt_best.pt", "**/ckpt.pt"):
            hits = sorted(base.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
            if hits:
                return hits[0].resolve()
    raise FileNotFoundError(
        "No RLPD checkpoint found. Pass --ckpt path/to/ckpt.pt or --name <run_folder> "
        "(under ~/squint-rlpd/runs)."
    )


def probe_rlpd_checkpoint(path: Path) -> RlpdCheckpointMeta:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: expected dict, got {type(ckpt)}")
    if not is_rlpd_checkpoint(ckpt):
        raise ValueError(f"{path}: not an RLPD checkpoint (missing critic / RLPD encoder layout)")

    n_state = int(ckpt["actor"]["proj.state_proj.0.weight"].shape[1])
    repr_dim = int(ckpt["actor"]["proj.rgb_proj.0.weight"].shape[1])
    n_act = int(ckpt["actor"]["fc_mean.weight"].shape[0])
    image_h, image_w, enc_repr = _match_encoder_hw(ckpt["encoder"], repr_dim)

    manifest = _manifest_for(path)
    meta = RlpdCheckpointMeta(
        path=str(path.resolve()),
        global_step=ckpt.get("global_step", "?"),
        n_state=n_state,
        n_act=n_act,
        image_h=image_h,
        image_w=image_w,
        encoder_repr_dim=enc_repr,
        use_bowl_xyz=n_state >= 21,
        colour_conditioned=n_state >= 18,
        is_full_state_58=n_state >= 58,
        ckpt_keys=sorted(ckpt.keys()),
        manifest=manifest,
    )
    if n_state not in (18, 21, 58):
        meta.warnings.append(f"Unusual n_state={n_state}; deploy supports 18, 21, or 58.")
    if image_h != 80 or image_w != 144:
        meta.warnings.append(
            f"Policy image {image_h}x{image_w} differs from teleop default 80x144; "
            "resize camera accordingly."
        )
    return meta


class CNNEncoder(nn.Module):
    """Matches train_rlpd.CNNEncoder (Atari strides for H>=56)."""

    def __init__(self, image_h: int, image_w: int):
        super().__init__()
        H, W = int(image_h), int(image_w)
        if H >= 56:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                nn.Flatten(),
            )
        elif H in (32, 36):
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                nn.Flatten(),
            )
        elif H == 16:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=1), nn.ReLU(),
                nn.Flatten(),
            )
        else:
            raise ValueError(f"Unsupported CNN height {H}")
        with torch.no_grad():
            dummy = torch.zeros(1, 3, H, W)
            self.repr_dim = int(self.conv(dummy).shape[-1])

    def forward(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        x = rgb_uint8.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        return self.conv(x)


class Projection(nn.Module):
    def __init__(self, n_obs: int, n_state: int):
        super().__init__()
        self.repr_dim = 50 + 256
        self.rgb_proj = nn.Sequential(
            nn.Linear(n_obs, 50), nn.LayerNorm(50), nn.Tanh(),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(n_state, 256), nn.LayerNorm(256), nn.ReLU(),
        )

    def forward(self, rgb_feat: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.rgb_proj(rgb_feat), self.state_proj(state)], dim=-1)


class Actor(nn.Module):
    def __init__(self, n_obs: int, n_state: int, n_act: int = 6):
        super().__init__()
        hidden = 256
        self.proj = Projection(n_obs, n_state)
        self.fc = nn.Sequential(
            nn.Linear(self.proj.repr_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden, n_act)
        self.register_buffer("action_scale", torch.ones(n_act))
        self.register_buffer("action_bias", torch.zeros(n_act))

    def eval_action(self, rgb_feat: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = self.fc(self.proj(rgb_feat, state))
        return torch.tanh(self.fc_mean(x)) * self.action_scale + self.action_bias


def _match_encoder_hw(encoder_sd: dict, repr_dim: int) -> tuple[int, int, int]:
    last_err = None
    for h, w in IMAGE_HW_CANDIDATES:
        enc = CNNEncoder(h, w)
        try:
            enc.load_state_dict(encoder_sd, strict=True)
            if enc.repr_dim == repr_dim:
                return h, w, enc.repr_dim
        except Exception as exc:
            last_err = exc
    raise RuntimeError(
        f"Could not match encoder weights to known (H,W); repr_dim={repr_dim}. Last error: {last_err}"
    )


def load_rlpd_policy(
    ckpt_path: Path,
    *,
    device: str = "cpu",
) -> tuple[CNNEncoder, Actor, RlpdCheckpointMeta]:
    meta = probe_rlpd_checkpoint(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder = CNNEncoder(meta.image_h, meta.image_w).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    actor = Actor(meta.encoder_repr_dim, meta.n_state, meta.n_act).to(device).eval()
    actor.load_state_dict(ckpt["actor"])
    return encoder, actor, meta


def preprocess_rgb(rgb: np.ndarray | torch.Tensor, *, h: int, w: int) -> torch.Tensor:
    import cv2

    if torch.is_tensor(rgb):
        arr = rgb.cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
    else:
        arr = np.asarray(rgb)
        if arr.ndim == 4:
            arr = arr[0]
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    resized = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(resized).unsqueeze(0).to(torch.uint8)


def _tcp_pose_from_qpos_rad(qpos_rad: np.ndarray) -> np.ndarray:
    from toolset.kinematics.config import KinematicsConfig
    from toolset.kinematics.urdf_fk import SO101FK

    kcfg = KinematicsConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)
    q5 = np.asarray(qpos_rad, dtype=float).reshape(6)[:5]
    tip = fk.fk(q5, target="gripper_tip")["position"]
    pose = np.zeros(7, dtype=np.float32)
    pose[:3] = tip.astype(np.float32)
    pose[6] = 1.0
    return pose


def build_state_vector(
    qpos_rad: np.ndarray,
    target_qpos_rad: np.ndarray,
    *,
    goal_color: int,
    bowl_xyz: np.ndarray | None,
    prev_qpos_rad: np.ndarray | None,
    n_state: int,
) -> torch.Tensor:
    q = np.asarray(qpos_rad, dtype=np.float32).reshape(6)
    t = np.asarray(target_qpos_rad, dtype=np.float32).reshape(6)

    if n_state <= 21:
        onehot = np.zeros(N_COLORS, dtype=np.float32)
        onehot[int(goal_color) % N_COLORS] = 1.0
        parts = [q, t, onehot]
        if n_state == 21:
            if bowl_xyz is None:
                bowl_xyz = np.zeros(3, dtype=np.float32)
            parts.append(np.asarray(bowl_xyz, dtype=np.float32).reshape(3))
        return torch.from_numpy(np.concatenate(parts)).unsqueeze(0).float()

    state = np.zeros(58, dtype=np.float32)
    state[_STATE_SLICES_58["noisy_qpos"][0]:_STATE_SLICES_58["noisy_qpos"][1]] = q
    state[_STATE_SLICES_58["controller_target"][0]:_STATE_SLICES_58["controller_target"][1]] = t
    state[_STATE_SLICES_58["goal_color"][0] + int(goal_color % N_COLORS)] = 1.0
    if bowl_xyz is not None:
        state[_STATE_SLICES_58["bowl_xyz_robot_frame"][0]:_STATE_SLICES_58["bowl_xyz_robot_frame"][1]] = (
            np.asarray(bowl_xyz, dtype=np.float32).reshape(3)
        )
    if prev_qpos_rad is not None:
        dt = 1.0 / CONTROL_HZ
        qvel = (q - np.asarray(prev_qpos_rad, dtype=np.float32).reshape(6)) / dt
        state[_STATE_SLICES_58["qvel"][0]:_STATE_SLICES_58["qvel"][1]] = qvel
    state[_STATE_SLICES_58["is_item_grasped"][0]] = float(q[5] > GRASP_THRESHOLD_RAD)
    state[_STATE_SLICES_58["tcp_pose"][0]:_STATE_SLICES_58["tcp_pose"][1]] = _tcp_pose_from_qpos_rad(q)
    return torch.from_numpy(state).unsqueeze(0).float()
