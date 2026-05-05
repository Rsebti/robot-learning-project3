"""Scripted pick-and-place controller for Eval 2 v1 (BC warmstart data source).

Drives the SO-101 in the Eval 2 v1 env via Isaac Lab's DifferentialIKController
+ a per-env phase state machine, deterministically solving "pick the
{target_color} block, place it in the bowl".

Action contract (must match v1 env's ActionsCfg)
------------------------------------------------
The env expects a 6-D action per step:
  - action[:5] = JointPositionAction over [shoulder_pan, shoulder_lift,
    elbow_flex, wrist_flex, wrist_roll], with scale=0.5 and use_default_offset.
    So:    target_joint_pos = default_joint_pos + 0.5 * action[:5]
  - action[5]  = BinaryJointPositionAction over [gripper], positive -> open
    (target=0.5), negative -> close (target=0.0).

We compute target_joint_pos via IK toward a per-phase target end-effector
position, then invert the affine map: action[:5] = 2 * (target - default).

Per-env state
-------------
phase (long, 0..7), phase_step (long, steps spent in phase). Phase advances when
EITHER the EE is within POS_TOL of target, OR a per-phase step budget elapses.
Phases CLOSE and OPEN always run their full budget so the gripper has time to
physically move.

Phase sequence
--------------
  0 APPROACH    EE 8 cm above target block, gripper open
  1 DESCEND     EE at block height, gripper open
  2 CLOSE       hold pose, gripper closing (fixed steps)
  3 LIFT        EE 12 cm above original block xy, gripper closed
  4 ABOVE_BOWL  EE 8 cm above bowl floor, gripper closed
  5 OPEN        hold pose, gripper opening (fixed steps)
  6 RETREAT     EE 15 cm above bowl, gripper open
  7 DONE        idle, holds RETREAT pose until episode ends

Approximations / caveats
------------------------
- The env's FrameTransformer "tip" sits at gripper_link + offset (0.01, 0,
  -0.09). We control gripper_link directly and approximate "place tip at
  (x,y,z)" as "place gripper_link at (x, y, z + 0.09)". That holds while the
  gripper points roughly down (wrist_flex near 1.57), which is the case under
  the small per-step IK deltas we issue. If the smoke test shows the wrist
  drifting, switch to command_type="pose" with a hardcoded down quaternion.
- IK uses command_type="position" (3-DOF goal) on a 5-DOF arm. Orientation is
  whatever falls out of dls — empirically smooth around the home pose.
- We clip the per-step Cartesian target to MAX_STEP_XYZ from the current EE
  position. Without this, IK at the start of APPROACH would issue a huge delta
  joint command (8 cm to cover) that saturates the actuators and makes the
  motion jerky.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import subtract_frame_transforms

from sim.eval2.bc.analytical_ik import analytical_ik_so101

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ScriptedPickController:
    """Deterministic IK-based controller solving Eval 2 v1."""

    # ------------------------------------------------------------------ phases
    # v2 (post-doc): added DESCEND_TO_RELEASE between ABOVE_BOWL (transport)
    # and OPEN. Without it, OPEN was firing while the block was still 8 cm
    # above the bowl rim — the block then bounced off the wall instead of
    # falling clean. The intermediate phase brings it down to ~4 cm above
    # the bowl floor before releasing.
    APPROACH = 0
    DESCEND = 1
    CLOSE = 2
    LIFT = 3
    ABOVE_BOWL = 4
    DESCEND_TO_RELEASE = 5
    OPEN = 6
    RETREAT = 7
    DONE = 8
    NUM_PHASES = 9

    PHASE_NAMES = (
        "APPROACH", "DESCEND", "CLOSE", "LIFT",
        "ABOVE_BOWL", "DESCEND_TO_RELEASE", "OPEN", "RETREAT", "DONE",
    )

    # ------------------------------------------------------------- geometry
    # All meters. Heights are RELATIVE to the surface they reference (block top
    # or bowl floor). Block_pos_w[:, 2] returns the block's center, which sits
    # at half-height = 0.01 m for our 2 cm cubes.
    # v4: tip 2 cm ABOVE the cube's top face. Top face = block_center +
    # block_half_size = 0.010 + 0.010 = 0.020. So tip target z = 0.040,
    # which is APPROACH_HEIGHT = 0.030 above block center.
    APPROACH_HEIGHT = 0.03
    # Tip 5 mm BELOW block center -> tip at z=0.005 (5 mm above table top).
    # Jaws straddle the cube's lower half before closing.
    DESCEND_HEIGHT = -0.005
    LIFT_HEIGHT = 0.12            # clears the 2.5 cm bowl walls comfortably
    ABOVE_BOWL_HEIGHT = 0.08      # transport hover, well above the rim
    # v2: NEW — descend over the bowl to ~4 cm above the bowl floor before
    # releasing. With bowl_h = 2.5 cm and block_h = 2 cm, dropping from 4 cm
    # leaves only 1.5 cm of free fall — block lands cleanly without bouncing.
    RELEASE_HEIGHT = 0.04
    RETREAT_HEIGHT = 0.15

    # gripper_link sits ~9 cm ABOVE the FrameTransformer tip when pointing down
    TIP_Z_OFFSET = 0.09

    # v2: 2 cm/step (was 4 cm). Doc advice — 4 cm saturated the SO-101
    # actuators (effort_limit=1.9 N.m) at almost every step, which made
    # the actual EE motion sluggish and inconsistent. 2 cm leaves headroom
    # so the actuators track the IK target precisely.
    MAX_STEP_XYZ = 0.02

    # v3: 5 mm (was 1 cm). v2 diagnostic showed phase advancing at
    # ~1 cm error, which was 1 cm OFF in z — fingers ended up above the
    # cube. With 5 mm, the descent must converge within half-cube width
    # before CLOSE fires.
    POS_TOL = 0.005

    # ------------------------------------------------------------- step budgets
    # v2: added DESCEND_TO_RELEASE budget. Sum of non-DONE = 50+100+25+40+100
    # +60+25+20 = 420 steps. With env episode_length=10s = 500 steps (set
    # in run_scripted launcher), leaves 80 for DONE.
    PHASE_MAX_STEPS = (
        50,   # APPROACH
        150,  # DESCEND   v3: 100 -> 150, give time to converge to POS_TOL=5mm
        25,   # CLOSE
        40,   # LIFT
        100,  # ABOVE_BOWL          longest xy travel
        60,   # DESCEND_TO_RELEASE  vertical 4 cm over bowl
        25,   # OPEN
        20,   # RETREAT
        500,  # DONE — until episode timeout
    )

    FIXED_DURATION_PHASES = (CLOSE, OPEN)

    # +1 = open, -1 = close. One per phase index.
    GRIPPER_PER_PHASE = (
        +1.0,  # APPROACH           open
        +1.0,  # DESCEND             open
        -1.0,  # CLOSE               closing
        -1.0,  # LIFT                stay closed
        -1.0,  # ABOVE_BOWL          stay closed
        -1.0,  # DESCEND_TO_RELEASE  stay closed (dropping in)
        +1.0,  # OPEN                opening
        +1.0,  # RETREAT             open
        +1.0,  # DONE                open
    )

    # ---------------------------------------------------------- gripper phi
    # Per-phase gripper orientation in the arm plane (radians, absolute
    # angle of the last link from horizontal). Calibrated so that the
    # IK solution for typical workspace targets keeps wrist_flex_urdf
    # within +/-1.55 (URDF soft limit is +/-1.658, leaving 0.1 rad of
    # safety margin). We tilt the gripper "back" (more negative phi)
    # for low-z phases and keep it closer to vertical for high-z phases.
    #
    # phi calibration done analytically by hand for the central target
    # (cube at r=0.18 m). For perturbations within the eval workspace
    # (cubes/bowls in +/-10 cm box), wrist_flex shifts by < 0.15 rad,
    # still well within the +/-1.658 hard limit.
    PHI_PER_PHASE = (
        -1.85,  # APPROACH            tip 2 cm above cube top (z~0.04)
        -1.90,  # DESCEND             tip 5 mm above table (z~0.005)
        -1.90,  # CLOSE               hold at grasp height
        -1.55,  # LIFT                tip 12 cm above grasp z (z~0.13)
        -1.70,  # ABOVE_BOWL          tip 8 cm above bowl floor (z~0.08)
        -1.85,  # DESCEND_TO_RELEASE  tip 4 cm above bowl floor (z~0.04)
        -1.85,  # OPEN                hold at release height
        -1.55,  # RETREAT             tip 15 cm above bowl (z~0.15)
        -1.55,  # DONE                idle, same as RETREAT
    )

    # ------------------------------------------------------------------ init
    def __init__(self, env: "ManagerBasedRLEnv"):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

        self.robot = env.scene["robot"]

        # Arm joints (5) + gripper (1).
        self.arm_joint_ids, self.arm_joint_names = self.robot.find_joints(
            ["shoulder_.*", "elbow_flex", "wrist_.*"]
        )
        self.gripper_joint_id = self.robot.find_joints(["gripper"])[0][0]

        # End-effector frame — used to read the actual tip position for
        # phase-transition checks. The "ee_frame" SceneEntity is the
        # FrameTransformer that puts the tip at gripper_link + offset.
        self.ee_frame = env.scene["ee_frame"]
        # gripper_link body index — kept for debug logs (run_scripted
        # prints body_pose_w[ee_body_idx] to inspect the gripper world
        # position). Not used in the IK pipeline.
        body_ids, _ = self.robot.find_bodies("gripper_link")
        self.ee_body_idx = body_ids[0]

        # Default joint pos (URDF convention) — to invert the
        # JointPositionAction affine map (action = 2*(joint - default)).
        self.default_arm_pos = (
            self.robot.data.default_joint_pos[:, self.arm_joint_ids].clone()
        )

        # Load SO-101 kinematic constants measured by
        # ``measure_link_lengths.py``. These are repo-local (gitignored
        # via the json suffix in the repo's .gitignore? — if not, just
        # ship them in).
        link_cfg_path = Path(__file__).parent / "so101_link_lengths.json"
        if not link_cfg_path.exists():
            raise FileNotFoundError(
                f"{link_cfg_path} not found. Run "
                f"`uv run python -m sim.eval2.bc.measure_link_lengths` first."
            )
        with open(link_cfg_path) as f:
            link_cfg = json.load(f)
        self.L1 = float(link_cfg["L1"])
        self.L2 = float(link_cfg["L2"])
        self.L3 = float(link_cfg["L3"])
        self.base_offset_z = float(link_cfg["base_offset_z"])
        self.base_offset_r = float(link_cfg.get("base_offset_r", 0.0))
        self.offset_theta2 = float(link_cfg["offset_theta2"])
        self.offset_theta3 = float(link_cfg["offset_theta3"])
        self.offset_theta4 = float(link_cfg["offset_theta4"])
        # Per-phase gripper phi tensor (see PHI_PER_PHASE constant).
        # SO-101 is 5-DoF, so the gripper orientation is a used-up DoF
        # — fixing phi = -pi/2 strict for all phases makes the IK
        # demand wrist_flex_urdf beyond +/-1.658 for low-z targets.
        # We instead pick a phase-specific phi calibrated to keep all
        # joint limits comfortably satisfied, with no runtime search.
        self._phi_per_phase_t = torch.tensor(
            self.PHI_PER_PHASE, device=self.device, dtype=torch.float32
        )

        # Snapshot of the LIFT-phase target. Captured at the CLOSE -> LIFT
        # transition (= the block position at the moment of grasp + 12 cm).
        # Without this snapshot, target_xyz would chase the block as it
        # rises with the gripper, making the distance criterion unreachable
        # (the target moves at the same rate as the tip).
        # Bowl-relative phases don't need this because the bowl is
        # kinematic — it never moves.
        self.lift_target_xyz = torch.zeros(self.num_envs, 3, device=self.device)

        # Per-env state buffers.
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Pre-build constant tensors used in compute_action / _advance_phases.
        self._gripper_per_phase_t = torch.tensor(
            self.GRIPPER_PER_PHASE, device=self.device, dtype=torch.float32
        )
        self._max_steps_t = torch.tensor(
            self.PHASE_MAX_STEPS, device=self.device, dtype=torch.long
        )
        self._fixed_duration_mask_t = torch.zeros(
            self.NUM_PHASES, dtype=torch.bool, device=self.device
        )
        for p in self.FIXED_DURATION_PHASES:
            self._fixed_duration_mask_t[p] = True

        # Per-phase z heights split into "block-relative" vs "bowl-relative"
        # tables for vectorized lookup. Unused entries are zeroed.
        # 9 entries — one per phase index in PHASE_NAMES.
        self._z_block_phase = torch.tensor([
            self.APPROACH_HEIGHT,  # 0 APPROACH           block + 8 cm
            self.DESCEND_HEIGHT,   # 1 DESCEND            block + 5 mm
            self.DESCEND_HEIGHT,   # 2 CLOSE              hold at grasp height
            self.LIFT_HEIGHT,      # 3 LIFT               block + 12 cm
            0.0,                   # 4 ABOVE_BOWL         (bowl-relative, unused here)
            0.0,                   # 5 DESCEND_TO_RELEASE (bowl-relative)
            0.0,                   # 6 OPEN               (bowl-relative)
            0.0,                   # 7 RETREAT            (bowl-relative)
            0.0,                   # 8 DONE               (bowl-relative)
        ], device=self.device)
        self._z_bowl_phase = torch.tensor([
            0.0,                       # 0 APPROACH
            0.0,                       # 1 DESCEND
            0.0,                       # 2 CLOSE
            0.0,                       # 3 LIFT
            self.ABOVE_BOWL_HEIGHT,    # 4 ABOVE_BOWL          bowl + 8 cm
            self.RELEASE_HEIGHT,       # 5 DESCEND_TO_RELEASE  bowl + 4 cm
            self.RELEASE_HEIGHT,       # 6 OPEN                hold at release height
            self.RETREAT_HEIGHT,       # 7 RETREAT             bowl + 15 cm
            self.RETREAT_HEIGHT,       # 8 DONE                bowl + 15 cm
        ], device=self.device)

    # ------------------------------------------------------------------- API
    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset per-env state for the given env ids (or all if None)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.phase[env_ids] = 0
        self.phase_step[env_ids] = 0

    @torch.no_grad()
    def compute_action(self) -> torch.Tensor:
        """One env-step worth of action for all envs. Returns (num_envs, 6).

        Pipeline (analytical IK, no Jacobian, no iteration):
          1. State machine -> target tip xyz in world frame.
          2. World -> base frame transform.
          3. analytical_ik_so101 -> URDF joint angles (5,).
          4. Invert JointPositionAction map -> raw action (5,).
          5. Append gripper command from state machine -> action (6,).
          6. Advance the state machine.
        """
        # 1. Per-phase target xyz of the TIP in world frame.
        target_pos_w = self._compute_target_xyz_world()                      # (N, 3)

        # 2. Express the target in the robot base frame.
        root_pose_w = self.robot.data.root_pose_w                            # (N, 7)
        target_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], target_pos_w
        )                                                                    # (N, 3)

        # 3. Closed-form IK -> URDF joint angles for the 5 arm joints.
        # phi is picked per-phase from PHI_PER_PHASE (see comment
        # there). elbow_up=False because the URDF home pose is in the
        # elbow-down branch (theta_3_ik = -1.29 at home).
        phi = self._phi_per_phase_t[self.phase]                              # (N,)
        ik_result = analytical_ik_so101(
            target_pos_b,
            L1=self.L1, L2=self.L2, L3=self.L3,
            phi=phi,
            elbow_up=False,
            base_offset_z=self.base_offset_z,
            base_offset_r=self.base_offset_r,
            offset_theta2=self.offset_theta2,
            offset_theta3=self.offset_theta3,
            offset_theta4=self.offset_theta4,
        )
        joint_pos_des = ik_result.joint_pos                                  # (N, 5)

        # 4. Invert the JointPositionAction affine map. The env applies
        #    processed = action * 0.5 + default, so:
        #        action = (target - default) / 0.5 = 2 * (target - default)
        # NOTE: we deliberately DO NOT clip arm_action to [-1, 1] here.
        # With env scale=0.5, clipping would limit each joint to default
        # +/- 0.5 rad, which is not enough to reach a cube ~25 cm away
        # from the base (typical IK targets need elbow_flex >= 2 rad,
        # i.e. arm_action[2] >= 4). The env's PD controller follows the
        # raw target without clipping. When we do BC/DAPG later, we
        # either normalize the action range or increase the env's
        # action scale so the policy can replicate these magnitudes.
        arm_action = 2.0 * (joint_pos_des - self.default_arm_pos)            # (N, 5)

        # 5. Gripper open/close from the phase machine.
        gripper_action = self._gripper_per_phase_t[self.phase].unsqueeze(-1) # (N, 1)

        action = torch.cat([arm_action, gripper_action], dim=-1)             # (N, 6)

        # 6. Advance state machine using the ACTUAL tip position.
        actual_tip_w = self.ee_frame.data.target_pos_w[:, 0, :]              # (N, 3)
        self._advance_phases(target_pos_w, actual_tip_w)

        return action

    # ------------------------------------------------------------ internals
    def _target_block_pos_w(self) -> torch.Tensor:
        """Per-env target block position in world frame (N, 3)."""
        red = self.env.scene["block_red"].data.root_pos_w                      # (N, 3)
        blue = self.env.scene["block_blue"].data.root_pos_w                    # (N, 3)
        is_red = (self.env.target_color == 0).unsqueeze(-1)                    # (N, 1)
        return torch.where(is_red, red, blue)

    def _compute_target_xyz_world(self) -> torch.Tensor:
        """Per-env target TIP xyz in world frame for the current phase."""
        block_pos_w = self._target_block_pos_w()                               # (N, 3)
        bowl_pos_w = self.env.scene["bowl_floor"].data.root_pos_w              # (N, 3)

        # Phases 0..3 reference the block; 4..8 reference the bowl.
        is_block_phase = self.phase < self.ABOVE_BOWL                          # (N,)
        # Phase 3 (LIFT) uses a SNAPSHOT instead of the live block position.
        is_lift_phase = (self.phase == self.LIFT)                              # (N,)

        # xy choice
        xy = torch.where(
            is_block_phase.unsqueeze(-1),
            block_pos_w[:, :2],
            bowl_pos_w[:, :2],
        )                                                                       # (N, 2)
        # Override xy for LIFT envs with the snapshot xy.
        xy = torch.where(
            is_lift_phase.unsqueeze(-1),
            self.lift_target_xyz[:, :2],
            xy,
        )

        # z choice: surface_z + per-phase height offset.
        z_block = block_pos_w[:, 2] + self._z_block_phase[self.phase]          # (N,)
        z_bowl = bowl_pos_w[:, 2] + self._z_bowl_phase[self.phase]             # (N,)
        z = torch.where(is_block_phase, z_block, z_bowl)                       # (N,)
        # Override z for LIFT envs with the snapshot z.
        z = torch.where(is_lift_phase, self.lift_target_xyz[:, 2], z)

        return torch.cat([xy, z.unsqueeze(-1)], dim=-1)                        # (N, 3)

    def _advance_phases(self, target_pos_w: torch.Tensor, ee_pos_w: torch.Tensor):
        """Update self.phase and self.phase_step in place.

        Transition rules:
          - Cartesian-target phases (APPROACH, DESCEND, LIFT, ABOVE_BOWL,
            DESCEND_TO_RELEASE, RETREAT): advance ONLY when the actual tip
            reaches within POS_TOL of the target. No timeout — the env's
            episode truncation handles the worst case.
          - Fixed-duration phases (CLOSE, OPEN): advance ONLY on step
            count, not distance. The gripper joint needs ~17 steps to
            traverse 0.5 rad mechanically; the tip position is irrelevant
            during the gripper actuation.
        """
        # Distance from the actual tip world pos to the commanded target.
        dist = torch.norm(ee_pos_w - target_pos_w, dim=-1)                     # (N,)

        max_steps_now = self._max_steps_t[self.phase]                          # (N,)
        timed_out = (self.phase_step + 1) >= max_steps_now                     # (N,)
        reached = dist < self.POS_TOL                                          # (N,)
        is_fixed = self._fixed_duration_mask_t[self.phase]                     # (N,)

        # Distance-based advance for Cartesian phases; step-count for fixed.
        advance_non_fixed = reached & ~is_fixed
        advance_fixed = timed_out & is_fixed
        advance_now = (advance_non_fixed | advance_fixed) & (self.phase < self.DONE)

        # Snapshot the LIFT target xyz at the CLOSE -> LIFT transition.
        # We need this BEFORE updating self.phase so we can detect the
        # transition. envs that just satisfy advance_now AND were in CLOSE
        # are about to enter LIFT.
        becomes_lift = advance_now & (self.phase == self.CLOSE)
        if becomes_lift.any():
            block_pos_w = self._target_block_pos_w()                           # (N, 3)
            new_xy = block_pos_w[:, :2]                                        # (N, 2)
            new_z = block_pos_w[:, 2] + self.LIFT_HEIGHT                       # (N,)
            new_target = torch.cat([new_xy, new_z.unsqueeze(-1)], dim=-1)      # (N, 3)
            self.lift_target_xyz = torch.where(
                becomes_lift.unsqueeze(-1), new_target, self.lift_target_xyz
            )

        self.phase = torch.where(advance_now, self.phase + 1, self.phase)
        self.phase_step = torch.where(
            advance_now, torch.zeros_like(self.phase_step), self.phase_step + 1
        )
