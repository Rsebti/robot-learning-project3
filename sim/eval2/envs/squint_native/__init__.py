"""Squint-native Isaac Lab env — port of Squint's SO101PlaceCube-v1.

Built from scratch to mirror Squint's ManiSkill env 1:1, WITHOUT inheriting
from LeIsaac. Goal: deploy the same checkpoint trained on ManiSkill and
verify whether the residual gap is purely PhysX vs SAPIEN dynamics, or
something we still mis-aligned.

Modules
-------
- ``squint_robot``: ArticulationCfg using the converted Squint URDF→USD
- ``squint_actions``: custom delta-target joint position action term
- ``squint_observations``: state vector matching Squint's PlaceCube obs
- ``squint_events``: reset to start keyframe + spawn item/bin
- ``squint_scene``: InteractiveSceneCfg (robot, cube, bin, cam, table, lights)
- ``squint_env_cfg``: ManagerBasedRLEnvCfg pulling everything together

Registered tasks
----------------
- ``Isaac-SquintNative-Place-v0`` (training shape)
- ``Isaac-SquintNative-Place-Play-v0`` (single-env eval shape)
"""
from .squint_env_cfg import (  # noqa: F401
    SquintNativePlaceEnvCfg,
    SquintNativePlaceEnvCfg_PLAY,
    SquintNativePlaceEnvCfg_REPLAY,
)
