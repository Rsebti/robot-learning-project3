"""MLP-Gaussian policy for BC pretrain — compatible with rsl_rl warmstart.

Architecture mirrors rsl_rl's ``ActorCritic.actor`` exactly so the BC
checkpoint can be loaded straight into PPO without weight surgery:

    - hidden dims [256, 128, 64]
    - ELU activations
    - final linear layer to action_dim (no activation)
    - log_std as a separate learnable parameter (state-independent)

Used in two places:
    - ``train_mlp_bc.py`` — for BC training (we ignore the std and
      regress mean against demo action via L1).
    - later in PPO warmstart — we copy these weights into
      ``ActorCritic.actor.*`` and the ``std`` parameter.

Input layout (15D):
    [0:6]   joint_pos (degrees, raw teleop scale, normalized inside the model)
    [6:12]  target_color one-hot (6D)
    [12:15] bowl_xyz (m, robot frame; constant per episode in BC,
            randomized later in RL)

Output layout (6D):
    next-step joint position targets (degrees, same convention as the data
    ``action`` field).
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _make_mlp(in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
    """Build a Sequential MLP with ELU activations between hidden layers."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ELU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MLPGaussianPolicy(nn.Module):
    """State-only Gaussian policy with state-independent log_std.

    Args:
        obs_dim:        observation feature dim (default 15 = 6 joint + 6
                        color one-hot + 3 bowl xyz).
        action_dim:     action dim (default 6 = SO-101 joints).
        hidden_dims:    actor MLP hidden sizes.
        init_noise_std: std at init (matches PPO's ``init_noise_std=0.5``).
        obs_mean/std:   per-feature normalization stats. Stored as buffers
                        so they survive ``state_dict`` round-trips. We
                        normalize the obs INSIDE the forward pass — that way
                        the policy is self-contained and we don't have to
                        re-normalize when loading into PPO (rsl_rl has its
                        own empirical normalization, so we'll set those to
                        1.0 / 0.0 when transferring).
        action_mean/std: target normalization for the BC regression.
                        We predict normalized actions and de-normalize on
                        the way out, matching the demos' scale.
    """

    def __init__(
        self,
        obs_dim: int = 15,
        action_dim: int = 6,
        hidden_dims: list[int] | None = None,
        init_noise_std: float = 0.5,
        obs_mean: torch.Tensor | None = None,
        obs_std: torch.Tensor | None = None,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = list(hidden_dims)

        self.actor = _make_mlp(obs_dim, self.hidden_dims, action_dim)
        # log_std as a learnable parameter (NOT state-dependent), matching
        # rsl_rl. Initialized so exp(log_std) == init_noise_std.
        self.log_std = nn.Parameter(
            torch.full((action_dim,), math.log(init_noise_std))
        )

        # Normalization stats. Stored as buffers so they're saved/loaded
        # but not trained.
        if obs_mean is None:
            obs_mean = torch.zeros(obs_dim)
        if obs_std is None:
            obs_std = torch.ones(obs_dim)
        if action_mean is None:
            action_mean = torch.zeros(action_dim)
        if action_std is None:
            action_std = torch.ones(action_dim)
        self.register_buffer("obs_mean", obs_mean.float())
        self.register_buffer("obs_std", obs_std.float())
        self.register_buffer("action_mean", action_mean.float())
        self.register_buffer("action_std", action_std.float())

    # ------------------------------------------------------------------
    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # Avoid division by zero on constant features (e.g. bowl_xyz fixed).
        std = self.obs_std.clamp(min=1e-6)
        return (obs - self.obs_mean) / std

    def _denormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        std = self.action_std.clamp(min=1e-6)
        return normalized_action * std + self.action_mean

    # ------------------------------------------------------------------
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic forward pass — used for BC training and inference.

        Returns the predicted action in DATA UNITS (degrees, raw teleop scale).
        """
        x = self._normalize_obs(obs)
        normalized_action = self.actor(x)
        return self._denormalize_action(normalized_action)

    def predict_normalized(self, obs: torch.Tensor) -> torch.Tensor:
        """Same as forward but returns the action BEFORE de-normalization.

        Used by the BC loss so we regress in normalized space (better
        gradient scaling across joints with very different ranges).
        """
        x = self._normalize_obs(obs)
        return self.actor(x)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Stochastic forward pass — used at deployment if we want
        exploration noise. Returns (action_in_data_units, log_prob).
        """
        normalized_mean = self.predict_normalized(obs)
        std = self.log_std.exp().expand_as(normalized_mean)
        dist = torch.distributions.Normal(normalized_mean, std)
        normalized_sample = dist.rsample()
        log_prob = dist.log_prob(normalized_sample).sum(dim=-1)
        action = self._denormalize_action(normalized_sample)
        return action, log_prob
