"""Squint SAC+C51 modules — full architecture port.

Faithful port of every NN class from
https://github.com/aalmuzairee/squint/blob/main/train_squint.py:

- ``CNNEncoder``: 16x16 wrist-cam → 1024-D feature vector
  (Squint variants for 16/32/64; we keep all three so changing
  ``image_size`` only requires changing the CNNEncoder argument)
- ``Projection``: rgb_features (1024 → 50, Tanh) ⊕ state (n_state → 256, ReLU) → 306-D
- ``SquintActor``: Projection → 3-layer MLP (256 hidden, LayerNorm+ReLU) → mean / log_std heads
- ``SquintCritic``: distributional C51 ensemble (101 atoms, [-20, 20])
  with **vmap'd Q networks** via ``tensordict.from_modules`` —
  matches Squint's exact computation graph so ``torch.compile`` can
  fuse it the same way.

Adaptations vs Squint that DO NOT change behavior:
- We use the same ``orthogonal_`` weight init (Squint uses
  ``calculate_gain('relu')`` for conv layers — we replicate that).
- Action scale/bias buffers exactly as Squint.
- ``LOG_STD_MIN=-5, LOG_STD_MAX=2`` — Squint defaults.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules


# ---------------------------------------------------------------------------
# Weight init — matches Squint's `weight_init` function exactly
# ---------------------------------------------------------------------------


def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


# ---------------------------------------------------------------------------
# CNN encoder — Squint defaults to 16x16 input → 1024-D output
# ---------------------------------------------------------------------------


class CNNEncoder(nn.Module):
    """Convolutional encoder for tiny RGB images.

    Squint deliberately uses very small input (16x16x3) to keep buffer
    memory low and gradient compute fast. The "encoder feature dim" is
    1024 = 64 channels × 4 × 4 spatial after the conv stack.

    Three image_size variants supported (mirroring Squint):
    - 16: 2 conv layers (3→32 stride 2, 32→64 stride 1) → flatten 1024
    - 32: 3 conv layers (3→32 stride 2, 32→64 stride 2, 64→64 stride 1) → flatten ~1024
    - 64: 3 conv layers (3→32 stride 4, 32→64 stride 2, 64→64 stride 1) → flatten ~1024
    """

    def __init__(self, n_obs: tuple[int, int, int], device: torch.device | None = None):
        super().__init__()
        assert len(n_obs) == 3 and n_obs[0] == n_obs[1], f"image must be square HWC: got {n_obs}"
        self.num_channels = n_obs[2]
        self.image_size = n_obs[0]
        self.repr_dim = 1024  # all variants flatten to ~1024

        if self.image_size == 64:
            self.conv = nn.Sequential(
                nn.Conv2d(self.num_channels, 32, 8, stride=4, device=device), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, device=device), nn.ReLU(),
                nn.Flatten(),
            )
        elif self.image_size == 32:
            self.conv = nn.Sequential(
                nn.Conv2d(self.num_channels, 32, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, device=device), nn.ReLU(),
                nn.Flatten(),
            )
        elif self.image_size == 16:
            self.conv = nn.Sequential(
                nn.Conv2d(self.num_channels, 32, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=1, device=device), nn.ReLU(),
                nn.Flatten(),
            )
        else:
            raise ValueError(f"Unsupported image_size: {self.image_size}")

        self.apply(weight_init)
        self.conv = self.conv.to(memory_format=torch.channels_last)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward: HWC uint8 ``(B, H, W, C)`` → flat features ``(B, 1024)``."""
        obs = obs.permute(0, 3, 1, 2)
        obs = obs.contiguous(memory_format=torch.channels_last)
        obs = obs / 255.0 - 0.5
        return self.conv(obs)


# ---------------------------------------------------------------------------
# Projection — combines rgb_features and state into a single MLP input
# ---------------------------------------------------------------------------


class Projection(nn.Module):
    """rgb_proj(rgb_features) ⊕ state_proj(state) — Squint exact.

    rgb_proj: Linear(1024 → 50) + LayerNorm + Tanh.
    state_proj: Linear(n_state → 256) + LayerNorm + ReLU.
    Output: concat dim = 306.
    """

    def __init__(self, n_rgb_repr: int, n_state: int, device: torch.device | None = None):
        super().__init__()
        self.repr_dim = 50 + 256
        self.rgb_proj = nn.Sequential(
            nn.Linear(n_rgb_repr, 50, device=device),
            nn.LayerNorm(50, device=device),
            nn.Tanh(),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(n_state, 256, device=device),
            nn.LayerNorm(256, device=device),
            nn.ReLU(),
        )

    def forward(self, rgb_features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.rgb_proj(rgb_features), self.state_proj(state)], dim=-1)


# ---------------------------------------------------------------------------
# Actor — Squint's tanh-Gaussian SAC actor
# ---------------------------------------------------------------------------


class SquintActor(nn.Module):
    """SAC actor with Projection input (rgb_features, state) + 3-layer MLP."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        n_rgb_repr: int,
        n_state: int,
        n_act: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        hidden_dim: int = 256,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.proj = Projection(n_rgb_repr, n_state, device=device)
        self.fc = nn.Sequential(
            nn.Linear(self.proj.repr_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, n_act, device=device)
        self.fc_logstd = nn.Linear(hidden_dim, n_act, device=device)

        action_scale = ((action_high - action_low) / 2.0).to(device, dtype=torch.float32)
        action_bias = ((action_high + action_low) / 2.0).to(device, dtype=torch.float32)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

        self.apply(weight_init)

    def forward(
        self,
        rgb_features: torch.Tensor,
        state: torch.Tensor,
        get_log_std: bool = False,
    ):
        x = self.proj(rgb_features, state)
        x = self.fc(x)
        mean = self.fc_mean(x)
        if get_log_std:
            log_std = self.fc_logstd(x)
            log_std = torch.tanh(log_std)
            log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1.0)
            return mean, log_std
        return mean

    def get_eval_action(self, rgb_features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        mean = self.forward(rgb_features, state)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def get_action(
        self,
        rgb_features: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(rgb_features, state, get_log_std=True)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


# ---------------------------------------------------------------------------
# Critic — distributional C51 ensemble with vmap'd Q nets
# ---------------------------------------------------------------------------


class SquintCritic(nn.Module):
    """Distributional C51 critic with vmap'd Q-network ensemble.

    Architecture (per Squint):
    - Projection (rgb_proj + state_proj) → 306-D combined obs
    - Concat with action → q_input (306 + n_act)
    - For each Q-net: 3-layer MLP (512 hidden, LayerNorm+ReLU) →
      num_atoms logits
    - Q ensemble registered via ``tensordict.from_modules`` so a single
      vmap dispatches across num_q networks
    """

    def __init__(
        self,
        n_rgb_repr: int,
        n_state: int,
        n_act: int,
        num_atoms: int = 101,
        v_min: float = -20.0,
        v_max: float = 20.0,
        num_q: int = 2,
        hidden_dim: int = 512,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_q = num_q
        self.v_min = v_min
        self.v_max = v_max
        self.register_buffer(
            "q_support",
            torch.linspace(v_min, v_max, num_atoms, device=device),
        )
        self.proj = Projection(n_rgb_repr, n_state, device=device)
        self.proj.apply(weight_init)

        q_input_dim = self.proj.repr_dim + n_act

        # Build num_q Q-networks, init weights, then stack into q_params.
        q_nets = [
            self._build_q_network(q_input_dim, num_atoms, hidden_dim=hidden_dim, device=device)
            for _ in range(num_q)
        ]
        for qn in q_nets:
            qn.apply(weight_init)

        # Squint's tensordict trick: one stacked param container, one
        # meta-device template for vmap dispatch. Hidden from
        # parameters() / state_dict() via object.__setattr__.
        self.q_params = from_modules(*q_nets, as_module=True)
        object.__setattr__(
            self,
            "_q_meta",
            self._build_q_network(q_input_dim, num_atoms, hidden_dim=hidden_dim, device="meta"),
        )

    @staticmethod
    def _build_q_network(input_dim: int, num_atoms: int, hidden_dim: int = 512, device=None) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, num_atoms, device=device),
        )

    def _vmap_q(self, params, x):
        """Single Q forward dispatched via vmap. ``params`` is a TensorDict."""
        with params.to_module(self._q_meta):
            return self._q_meta(x)

    def forward(
        self,
        rgb_features: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Returns logits ``(num_q, batch, num_atoms)``."""
        proj = self.proj(rgb_features, state)
        x = torch.cat([proj, actions], dim=-1)
        return torch.vmap(self._vmap_q, (0, None))(self.q_params, x)

    def get_q_values(
        self,
        rgb_features: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        detach_critic: bool = False,
    ) -> torch.Tensor:
        """Expected Q-values ``(num_q, batch)``.

        If ``detach_critic=True``, projection runs under ``no_grad`` and
        Q-net params are passed via ``.data`` so gradients flow through
        ``actions`` only — exactly Squint's actor-update path.
        """
        if detach_critic:
            with torch.no_grad():
                proj = self.proj(rgb_features, state)
            x = torch.cat([proj, actions], dim=-1)
            logits = torch.vmap(self._vmap_q, (0, None))(self.q_params.data, x)
        else:
            logits = self.forward(rgb_features, state, actions)
        probs = F.softmax(logits, dim=-1)
        return torch.sum(probs * self.q_support, dim=-1)

    def categorical(
        self,
        rgb_features: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
    ) -> torch.Tensor:
        """C51 categorical projection — verbatim from Squint."""
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]
        device = rewards.device

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount * self.q_support
        target_z = target_z.clamp(self.v_min, self.v_max)

        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_integer = upper == lower
        lower = torch.where(torch.logical_and(lower > 0, is_integer), lower - 1, lower)
        upper = torch.where(torch.logical_and(lower == 0, is_integer), upper + 1, upper)

        logits = self.forward(rgb_features, state, actions)
        next_dists = F.softmax(logits, dim=-1)

        total_batch = self.num_q * batch_size
        next_dists_flat = next_dists.reshape(-1, self.num_atoms)
        offset = torch.arange(total_batch, device=device).unsqueeze(1) * self.num_atoms

        lower_exp = lower.unsqueeze(0).expand(self.num_q, -1, -1).reshape(total_batch, self.num_atoms)
        upper_exp = upper.unsqueeze(0).expand(self.num_q, -1, -1).reshape(total_batch, self.num_atoms)
        b_exp = b.unsqueeze(0).expand(self.num_q, -1, -1).reshape(total_batch, self.num_atoms)

        max_index = total_batch * self.num_atoms - 1
        lower_indices = torch.clamp((lower_exp + offset).view(-1), 0, max_index)
        upper_indices = torch.clamp((upper_exp + offset).view(-1), 0, max_index)

        proj_dist_flat = torch.zeros_like(next_dists_flat)
        proj_dist_flat.view(-1).index_add_(
            0, lower_indices, (next_dists_flat * (upper_exp.float() - b_exp)).view(-1)
        )
        proj_dist_flat.view(-1).index_add_(
            0, upper_indices, (next_dists_flat * (b_exp - lower_exp.float())).view(-1)
        )

        return proj_dist_flat.reshape(self.num_q, batch_size, self.num_atoms)
