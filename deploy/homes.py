"""
Single source of truth for inference home / rest poses.

Edit ONLY this file when tuning where the arm parks before a rollout.
All deploy infer scripts load poses by name via --home_pose.

Joint order (canonical):
    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

ACT deploy (infer_eval*_act*.py, infer_eval2.py):
    Values are **servo degrees** (same convention as LeRobot .pos on the follower).

SAC legacy (infer_*_sac_legacy.py):
    Uses ``sac_radians`` when set (sim-space radians for that checkpoint).
    Otherwise falls back to ``numpy.deg2rad`` of the degree table (approximate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

DEFAULT_HOME_POSE = "eval1_rest"


@dataclass(frozen=True)
class HomePose:
    """Named rest pose. ``degrees`` is always defined; ``sac_radians`` optional."""

    degrees: Mapping[str, float]
    sac_radians: tuple[float, ...] | None = None
    description: str = ""

    def as_deg_array(self) -> np.ndarray:
        return np.array([float(self.degrees[j]) for j in JOINT_ORDER], dtype=np.float32)

    def as_sac_rad_array(self) -> np.ndarray:
        if self.sac_radians is not None:
            if len(self.sac_radians) != len(JOINT_ORDER):
                raise ValueError(
                    f"sac_radians length {len(self.sac_radians)} != {len(JOINT_ORDER)}"
                )
            return np.array(self.sac_radians, dtype=np.float32)
        return np.deg2rad(self.as_deg_array())


# ---------------------------------------------------------------------------
# Presets — add / edit poses here
# ---------------------------------------------------------------------------

# Universal home pose given by the user (2026-05-18, post motor-3 fix +
# full recalibration). Applied to every preset below so ACT and SAC both
# ramp to the same physical pose.
_UNIVERSAL_HOME_DEG = {
    "shoulder_pan":  -2.242,
    "shoulder_lift": -80.791,
    "elbow_flex":    36.747,
    "wrist_flex":    86.901,
    "wrist_roll":   -82.154,
    "gripper":      -14.686,
}
_UNIVERSAL_HOME_RAD = (
    -0.039131,
    -1.410071,
     0.641358,
     1.516708,
    -1.433855,
    -0.256321,
)

HOMES: dict[str, HomePose] = {
    # ---- ACT (universal) ------------------------------------------------
    "eval1_rest": HomePose(
        description="Universal home (2026-05-18 post motor-3 fix + full recalibration). "
                    "Used by all ACT deploy scripts.",
        degrees=dict(_UNIVERSAL_HOME_DEG),
    ),
    "eval2_rest": HomePose(
        description="Universal home (same as eval1_rest).",
        degrees=dict(_UNIVERSAL_HOME_DEG),
    ),
    # ---- SAC legacy (training-time sim rest) ----------------------------
    # The squint SAC checkpoint was trained with rest_qpos =
    #   [0, 0, 0, +pi/2, -pi/2, deg2rad(60)] sim-radians.
    # Deploy home MUST match this (in sim convention) or the policy
    # observes OOD state and outputs saturating actions. Keep separate
    # from the universal ACT home above.
    "eval1_sac_legacy": HomePose(
        description="Original Eval-1 SAC training rest (sim radians). Do NOT replace "
                    "with the universal home or the policy will jump on first step.",
        degrees={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 90.0,
            "wrist_roll": -90.0,
            "gripper": 60.0,
        },
        sac_radians=(
            0.0,
            0.0,
            0.0,
            float(np.pi / 2),
            float(-np.pi / 2),
            float(np.deg2rad(60.0)),
        ),
    ),
    # Optional A/B preset: SAC starting at the universal home (will jump).
    "eval1_sac_universal": HomePose(
        description="SAC policies trained starting from the universal physical home "
                    "(2026+ friend checkpoints, e1*lat / 1792-dim CNN head). "
                    "Uses sim-space sac_radians matching that pose. "
                    "Legacy sim-only trainings: use --home_pose eval1_sac_legacy.",
        degrees=dict(_UNIVERSAL_HOME_DEG),
        sac_radians=_UNIVERSAL_HOME_RAD,
    ),
}


def list_home_poses() -> list[str]:
    return sorted(HOMES.keys())


def get_home(name: str | None = None) -> HomePose:
    key = name or DEFAULT_HOME_POSE
    if key not in HOMES:
        raise KeyError(f"Unknown home pose {key!r}. Choose from {list_home_poses()}")
    return HOMES[key]


def get_home_deg(name: str | None = None) -> np.ndarray:
    return get_home(name).as_deg_array()


def get_home_rad(name: str | None = None) -> np.ndarray:
    return get_home(name).as_sac_rad_array()
