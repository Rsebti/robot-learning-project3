"""ArticulationCfg for the Squint-native SO-101 robot.

Uses the URDF→USD conversion produced by ``sim/eval2/scripts/convert_squint_urdf.py``.
Matches Squint's controller defaults exactly:
- pd_joint_target_delta_pos with stiffness 1000, damping 100, force_limit 100
- start keyframe: [0, 0, 0, π/2, -π/2, 60°]
- robot rooted at world origin, no rotation (so101 case from Squint's place.py)
"""
from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


# ---------------------------------------------------------------------------
# Asset path
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
SQUINT_USD = _PROJECT_ROOT / "squint" / "converted_usd" / "so101_squint_renamed.usd"
if not SQUINT_USD.exists():
    raise FileNotFoundError(
        f"Squint USD not found at {SQUINT_USD}. Run "
        f"`python -m sim.eval2.scripts.rename_squint_urdf` then "
        f"`python -m sim.eval2.scripts.convert_squint_urdf` first."
    )


# ---------------------------------------------------------------------------
# Squint home qpos (from squint/envs/robot/so101.py keyframe['start'])
# ---------------------------------------------------------------------------

SQUINT_HOME_QPOS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": math.pi / 2,        # +90°
    "wrist_roll": -math.pi / 2,       # -90°
    "gripper": 60.0 * math.pi / 180,  # +60° (slightly open)
}


# ---------------------------------------------------------------------------
# ArticulationCfg
# ---------------------------------------------------------------------------

SQUINT_SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SQUINT_USD),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Full gravity — matches the flattable_woodcube_run1 training.
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # Squint/SAPIEN uses solver_pos_iter=15, vel_iter=1. Matching
            # these here reduces our settle drift (~7cm in 5 steps -> ~mm)
            # and aligns the PD tracking precision.
            solver_position_iteration_count=15,
            solver_velocity_iteration_count=1,
        ),
        # Apply PhysxContactReportAPI to every rigid body under the robot so
        # ``ContactSensorCfg`` can read net forces on the jaw bodies.
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),       # robot root at world origin (Squint so101 case)
        rot=(1.0, 0.0, 0.0, 0.0),  # identity quat → arm forward = world +X
        joint_pos=dict(SQUINT_HOME_QPOS),
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Single actuator group for all 6 joints — matches Squint's single
        # PDJointPosControllerConfig with shared stiffness/damping.
        "all": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ],
            stiffness=1000.0,
            damping=100.0,
            effort_limit=100.0,
            velocity_limit=100.0,
        ),
    },
)
