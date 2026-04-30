"""Event functions for Eval 2 (custom, on top of Isaac Lab's standard events).

Isaac Lab calls these via the EventManager at the configured ``mode`` (typically
``"reset"`` or ``"interval"``). They mutate state on the env in-place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_target_color(env: ManagerBasedRLEnv, env_ids: torch.Tensor, num_classes: int = 2) -> None:
    """Sample a new target color index (uniform integer in [0, num_classes)) for the resetting envs.

    The result is stored on ``env.target_color`` (a long tensor of shape (num_envs,)).
    Observations, rewards and terminations read from this buffer to dispatch to
    the right block.
    """
    if not hasattr(env, "target_color"):
        env.target_color = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env.target_color[env_ids] = torch.randint(
        low=0, high=num_classes, size=(len(env_ids),), device=env.device
    )
