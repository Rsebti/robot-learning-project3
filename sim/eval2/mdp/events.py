"""Event functions for Eval 2 (custom, on top of Isaac Lab's standard events).

Isaac Lab calls these via the EventManager at the configured ``mode`` (typically
``"reset"`` or ``"interval"``). They mutate state on the env in-place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_stage_buffer(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Reset env.episode_max_stage to 0 for the resetting envs (v1.7).

    Pairs with rewards.stage_progress_reward, which tracks the highest stage
    each env has reached during the current episode. On reset, the buffer
    must drop back to 0 so the new episode starts fresh.
    """
    if not hasattr(env, "episode_max_stage"):
        env.episode_max_stage = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env.episode_max_stage[env_ids] = 0


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


def reset_cluster_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    asset_names: Sequence[str] = ("block_red", "block_blue"),
) -> None:
    """Randomize a *cluster* of objects together by applying the same xy shift
    to all of them. Preserves their relative positions (= adjacency).

    Per the TA spec for Eval 2:
        "Two blocks of different colors are placed adjacent to each other
         (flat cluster). Object positions will be randomized across rollouts."

    So the cluster as a whole moves, but the two blocks stay in contact.

    ``position_range`` is a dict with keys ``"x"`` and ``"y"`` mapping to
    ``(low, high)`` tuples in meters.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    device = env.device

    # One (dx, dy) per env, shared by all assets in the cluster.
    dx = torch.empty(n, device=device).uniform_(*position_range.get("x", (0.0, 0.0)))
    dy = torch.empty(n, device=device).uniform_(*position_range.get("y", (0.0, 0.0)))

    for name in asset_names:
        asset = env.scene[name]
        default_state = asset.data.default_root_state[env_ids].clone()  # (n, 13)
        # Translate xy in the world frame (Isaac Lab default state is already env-local;
        # the env_origins offset is added internally by write_root_pose_to_sim).
        default_state[:, 0] = default_state[:, 0] + dx + env.scene.env_origins[env_ids, 0]
        default_state[:, 1] = default_state[:, 1] + dy + env.scene.env_origins[env_ids, 1]
        default_state[:, 2] = default_state[:, 2] + env.scene.env_origins[env_ids, 2]
        asset.write_root_pose_to_sim(default_state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(default_state[:, 7:13], env_ids=env_ids)
