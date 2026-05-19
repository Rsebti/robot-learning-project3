"""
motor_to_urdf.py - convert lerobot motor angles to URDF joint angles.

lerobot's `get_observation()` returns angles in degrees where 0 == the
midpoint of each motor's calibrated range, NOT the URDF zero pose. So
URDF_angle = sign * (motor_angle - motor_offset_deg)         [in degrees]

The offsets and signs are robot-specific (depend on how the operator ran
`lerobot-calibrate`). The defaults in DEFAULT_OFFSETS / DEFAULT_SIGNS are
populated by `calibration_scripts/calibrate_motor_offsets.py` once per
physical robot; they're loaded from `configs/kinematics/motor_offsets.yaml`
at runtime.

If no offset file is found, the conversion is the identity (motor == URDF)
which is wrong but at least obvious - FK output will be nonsense, and the
caller can run the calibration script to fix it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .urdf_fk import CHAIN_JOINTS

DEFAULT_OFFSETS_PATH = (
    Path(__file__).parent.parent / "configs" / "kinematics" / "motor_offsets.yaml"
)


@dataclass
class MotorToUrdfConfig:
    """Per-joint linear transform: urdf_rad = sign * (motor_deg - offset_deg) * pi/180.

    offsets_deg: dict joint_name -> motor degrees that correspond to URDF zero.
    signs: dict joint_name -> +1 or -1 to flip the rotation direction.
    """
    offsets_deg: dict[str, float] = field(default_factory=lambda: {n: 0.0 for n in CHAIN_JOINTS})
    signs: dict[str, int] = field(default_factory=lambda: {n: +1 for n in CHAIN_JOINTS})

    @classmethod
    def load(cls, path: Path | str = DEFAULT_OFFSETS_PATH) -> "MotorToUrdfConfig":
        path = Path(path)
        if not path.exists():
            print(f"[motor_to_urdf] no offsets file at {path}; using identity (FK will be wrong).")
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        cfg = cls()
        cfg.offsets_deg.update({k: float(v) for k, v in (data.get("offsets_deg") or {}).items()})
        cfg.signs.update({k: int(v) for k, v in (data.get("signs") or {}).items()})
        return cfg

    def save(self, path: Path | str = DEFAULT_OFFSETS_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(
                {"offsets_deg": self.offsets_deg, "signs": self.signs},
                f,
                default_flow_style=False,
                sort_keys=True,
            )

    def motor_to_urdf_rad(self, motor_deg: dict[str, float] | np.ndarray | list[float]) -> np.ndarray:
        """Returns 5-vector of URDF radians for the manipulator joints."""
        if isinstance(motor_deg, dict):
            md = np.array([motor_deg[n] for n in CHAIN_JOINTS], dtype=float)
        else:
            md = np.asarray(motor_deg, dtype=float).reshape(-1)
            if md.shape[0] not in (5, 6):
                raise ValueError(f"motor_deg must have 5 or 6 elements, got {md.shape[0]}")
            md = md[:5]
        deg = np.array(
            [self.signs[n] * (md[i] - self.offsets_deg[n]) for i, n in enumerate(CHAIN_JOINTS)],
            dtype=float,
        )
        return np.deg2rad(deg)

    def urdf_rad_to_motor_deg(self, urdf_rad: np.ndarray | list[float]) -> np.ndarray:
        """Inverse: returns 5-vector of motor degrees."""
        ur = np.asarray(urdf_rad, dtype=float).reshape(-1)
        if ur.shape[0] != 5:
            raise ValueError(f"urdf_rad must have 5 elements, got {ur.shape[0]}")
        deg = np.rad2deg(ur)
        out = np.array(
            [self.signs[n] * deg[i] + self.offsets_deg[n] for i, n in enumerate(CHAIN_JOINTS)],
            dtype=float,
        )
        return out


def read_motor_deg_from_obs(obs: dict) -> np.ndarray:
    """Pull 6 motor degrees from a lerobot observation dict (manipulator + gripper)."""
    full = ["shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper"]
    return np.array([float(obs[f"{n}.pos"]) for n in full], dtype=float)
