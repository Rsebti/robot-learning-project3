"""
config.py - load kinematics & control configuration from YAML.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent / "configs" / "kinematics" / "config.yaml"
)
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class KinematicsConfig:
    urdf_path: Path
    gripper_tip_offset_m: tuple[float, float, float]
    joint_limits_rad: dict[str, tuple[float, float]]
    workspace: dict
    table_z_m: float
    cube_side_m: float
    cube_half_height_m: float
    ik: dict
    path: dict

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "KinematicsConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        urdf = Path(data["urdf_path"])
        if not urdf.is_absolute():
            urdf = PROJECT_ROOT / urdf
        tip = data.get("gripper_tip_offset_m", {})
        return cls(
            urdf_path=urdf,
            gripper_tip_offset_m=(
                float(tip.get("x", 0.0)),
                float(tip.get("y", 0.0)),
                float(tip.get("z", 0.0)),
            ),
            joint_limits_rad={
                k: (float(v[0]), float(v[1]))
                for k, v in data["joint_limits_rad"].items()
            },
            workspace=data["workspace"],
            table_z_m=float(data["table"]["z_m"]),
            cube_side_m=float(data["cube"]["side_m"]),
            cube_half_height_m=float(data["cube"]["half_height_m"]),
            ik=data["ik"],
            path=data["path"],
        )

    def in_workspace(self, xyz: np.ndarray) -> bool:
        ws = self.workspace
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        if not (ws["x_range_m"][0] <= x <= ws["x_range_m"][1]):
            return False
        if not (ws["y_range_m"][0] <= y <= ws["y_range_m"][1]):
            return False
        if not (ws["z_range_m"][0] <= z <= ws["z_range_m"][1]):
            return False
        if (x * x + y * y) ** 0.5 > ws["max_reach_xy_m"]:
            return False
        return True

    def clip_to_joint_limits(self, q_rad: np.ndarray) -> np.ndarray:
        """Clip 5-vector of URDF radians into joint-limit polytope."""
        from .urdf_fk import CHAIN_JOINTS
        out = np.asarray(q_rad, dtype=float).copy()
        for i, name in enumerate(CHAIN_JOINTS):
            lo, hi = self.joint_limits_rad[name]
            out[i] = float(np.clip(out[i], lo, hi))
        return out
