"""Termination terms — replicate Squint's success criterion.

Squint's success (``envs/place.py:Place.evaluate``):
    success = is_item_above_bin & (~robot_touching_item)
              & is_robot_static & (~robot_touching_bin)

We approximate it cheaply on the Isaac side using world poses and qvel:
- ``is_item_above_bowl`` : cube xy within bowl half-extents AND cube above
                           bowl floor (z=0).
- ``is_cube_static``     : ``|cube_lin_vel| <= 0.05 m/s``.

Cheap proxies err on the GENEROUS side (we'd rather mark a marginal place
as a success than miss it). Tighten later if needed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Bowl AABB half-extents — matches ``mesh_bowl_from_ply.py`` output.
_BOWL_HALF_X = 0.0740
_BOWL_HALF_Y = 0.0745
_CUBE_HALF_SIZE = 0.01


def success(
    env: "ManagerBasedRLEnv",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    cube_static_vel_thresh: float = 0.05,
) -> torch.Tensor:
    """Boolean (N,) — True when the cube is placed inside the bowl AND not
    being held / moved.

    Mirrors Squint's rectangular-AABB check (NOT a circular check, even
    though the bowl is round — matches the source implementation).
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]

    cube_pos = cube.data.root_pos_w
    bowl_pos = bowl.data.root_pos_w
    env_origins = env.scene.env_origins

    # Cube xy inside bowl half-extents (rectangular AABB).
    dx = cube_pos[:, 0] - bowl_pos[:, 0]
    dy = cube_pos[:, 1] - bowl_pos[:, 1]
    inside_x = torch.abs(dx) < _BOWL_HALF_X
    inside_y = torch.abs(dy) < _BOWL_HALF_Y
    # Cube above bowl floor (z = bowl root z = table top).
    above_bowl_z = cube_pos[:, 2] > (env_origins[:, 2] - 1e-3)
    is_item_above_bowl = inside_x & inside_y & above_bowl_z

    cube_vel = torch.linalg.norm(cube.data.root_lin_vel_w, dim=-1)
    is_cube_static = cube_vel <= cube_static_vel_thresh

    return is_item_above_bowl & is_cube_static
