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

from typing import TYPE_CHECKING

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms

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
    APPROACH_HEIGHT = 0.08
    # v3: -5 mm BELOW block center. Visual diagnostic on v2 showed the tip
    # landing at z=0.024 (1.4 cm above block center), which put one finger
    # on the cube TOP and the other to the side — gripper closed on empty
    # space. We need the tip at block CENTER (z=0.010) for the jaws to
    # straddle the cube faces at mid-height. With POS_TOL=5mm now, aiming
    # at block_z - 0.005 means the actual tip lands in [0, 0.010] — fingers
    # squarely around the cube.
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

        # v3-revised: IK operates on 3 ACTIVE joints (pan, lift, elbow) only.
        # The 2 wrist joints (flex, roll) are hard-locked to their defaults
        # so the IK must NOT plan motion for them — otherwise it computes
        # delta_q assuming wrist motion that we then prevent in the action,
        # making the actual EE motion not match the IK's prediction.
        # Joint order in arm_joint_ids: [pan, lift, elbow, wrist_flex, wrist_roll].
        self.active_arm_joint_ids = self.arm_joint_ids[:3]
        self.active_arm_joint_names = self.arm_joint_names[:3]

        # End-effector body (gripper_link) + jacobian index.
        body_ids, _ = self.robot.find_bodies("gripper_link")
        self.ee_body_idx = body_ids[0]
        # Fixed-base articulation: jacobian index = body index - 1.
        self.ee_jacobi_idx = self.ee_body_idx - 1

        # Default joint pos for the arm — used to invert the JointPositionAction map.
        self.default_arm_pos = self.robot.data.default_joint_pos[:, self.arm_joint_ids].clone()
        # Default for the 3 active joints (subset of default_arm_pos).
        self.default_active_pos = self.robot.data.default_joint_pos[:, self.active_arm_joint_ids].clone()

        # v3-revised: switch BACK to position-only IK + hard-lock both wrist
        # joints in the action.
        #
        # The pose-mode IK kept wrist_roll under control (with the previous
        # hard-lock) but let wrist_flex drift from 1.57 (down) to 1.07 (61°
        # from vertical) — visible in v3 logs. The gripper tilted forward,
        # so the lower finger hit the cube TOP instead of going down its
        # front face. Hard-locking BOTH wrist joints in the action handles
        # this directly; we no longer need the IK to track orientation.
        #
        # With wrist_flex and wrist_roll forced to defaults, the IK has
        # effectively 3 DOF (shoulder_pan, shoulder_lift, elbow_flex) to
        # reach a 3-D goal — well-determined, no slack variables to drift.
        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        )
        self.ik = DifferentialIKController(
            ik_cfg, num_envs=self.num_envs, device=self.device
        )

        # Capture the home pose's gripper_link orientation in WORLD frame.
        # This is the "gripper points down" quat. We re-use it as the target
        # orientation for every phase. Cached lazily on the first compute_action()
        # call (the env must be stepped once before robot.data is populated).
        self._target_quat_w: torch.Tensor | None = None

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
        self.ik.reset(env_ids)

    @torch.no_grad()
    def compute_action(self) -> torch.Tensor:
        """One env-step worth of action for all envs. Returns (num_envs, 6)."""
        # 1. Per-phase target xyz of gripper_link in world frame.
        target_pos_w = self._compute_target_xyz_world()                      # (N, 3)

        # 2. Read current robot state.
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_body_idx]          # (N, 7)
        root_pose_w = self.robot.data.root_pose_w                              # (N, 7)
        # IK runs on the 3 ACTIVE joints (wrist joints are hard-locked, so we
        # must NOT include them in the IK plan — otherwise IK assumes wrist
        # motion that the action override prevents).
        joint_pos_active = self.robot.data.joint_pos[:, self.active_arm_joint_ids]  # (N, 3)

        # 3. Express target & current EE in robot base frame for IK.
        target_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], target_pos_w
        )
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7],
            ee_pose_w[:, :3], ee_pose_w[:, 3:7],
        )

        # 4. Cap the per-step Cartesian target to MAX_STEP_XYZ.
        delta = target_pos_b - ee_pos_b                                        # (N, 3)
        delta_norm = torch.norm(delta, dim=-1, keepdim=True).clamp(min=1e-9)
        scale = torch.clamp(self.MAX_STEP_XYZ / delta_norm, max=1.0)
        clipped_target_b = ee_pos_b + delta * scale                            # (N, 3)

        # 5. Jacobian for the 3 ACTIVE joints only, rotated into base frame.
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_idx, :, self.active_arm_joint_ids
        ].clone()                                                              # (N, 6, 3)
        base_rot_matrix = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))     # (N, 3, 3)
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        # 6. IK -> active-joint targets (3-DOF position-only).
        self.ik.set_command(clipped_target_b, ee_quat=ee_quat_b)
        joint_pos_des_active = self.ik.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos_active
        )                                                                      # (N, 3)

        # 7. Build the 5-D arm action: active joints from IK, wrist joints
        #    locked.
        arm_action = torch.zeros(self.num_envs, 5, device=self.device)
        arm_action[:, 0:3] = 2.0 * (joint_pos_des_active - self.default_active_pos)
        # arm_action[:, 3] (wrist_flex) stays at 0 (raw=0 + default 1.57
        #   -> processed = 1.57 = gripper points down strict).
        # arm_action[:, 4] (wrist_roll): DYNAMIC compensation of
        #   shoulder_pan. When shoulder_pan rotates by theta to face the
        #   cube, the whole arm (including the gripper) rotates by theta
        #   in world frame — so the jaws rotate too. To keep the jaws
        #   oriented in a FIXED world direction (perpendicular to the
        #   arm's home pointing axis), we set wrist_roll = -shoulder_pan.
        #   In action space: arm_action[:, 0] = 2*shoulder_pan_des, so
        #   wrist_roll_des = -shoulder_pan_des  =>
        #   arm_action[:, 4] = 2*(-shoulder_pan_des) = -arm_action[:, 0].
        arm_action[:, 4] = -arm_action[:, 0]

        # 10. Gripper sign for the current phase.
        gripper_action = self._gripper_per_phase_t[self.phase].unsqueeze(-1)  # (N, 1)

        # 11. Final action.
        action = torch.cat([arm_action, gripper_action], dim=-1)              # (N, 6)

        # 12. Advance state machine for the next call.
        self._advance_phases(target_pos_w, ee_pose_w[:, :3])

        return action

    # ------------------------------------------------------------ internals
    def _target_block_pos_w(self) -> torch.Tensor:
        """Per-env target block position in world frame (N, 3)."""
        red = self.env.scene["block_red"].data.root_pos_w                      # (N, 3)
        blue = self.env.scene["block_blue"].data.root_pos_w                    # (N, 3)
        is_red = (self.env.target_color == 0).unsqueeze(-1)                    # (N, 1)
        return torch.where(is_red, red, blue)

    def _compute_target_xyz_world(self) -> torch.Tensor:
        """Compute per-env target gripper_link xyz in world frame for current phase.

        The output is the position the gripper_link body should occupy. To put
        the FrameTransformer "tip" at a point (x, y, z), we put gripper_link at
        (x, y, z + TIP_Z_OFFSET) — assumes the gripper points roughly down.
        """
        block_pos_w = self._target_block_pos_w()                               # (N, 3)
        bowl_pos_w = self.env.scene["bowl_floor"].data.root_pos_w              # (N, 3)

        # Phases 0..3 (APPROACH..LIFT) reference the block; 4..8 reference the bowl.
        is_block_phase = self.phase < self.ABOVE_BOWL                          # (N,)

        # xy choice
        xy = torch.where(
            is_block_phase.unsqueeze(-1),
            block_pos_w[:, :2],
            bowl_pos_w[:, :2],
        )                                                                       # (N, 2)

        # z choice: surface_z + per-phase offset (+ tip compensation)
        z_block = block_pos_w[:, 2] + self._z_block_phase[self.phase]          # (N,)
        z_bowl = bowl_pos_w[:, 2] + self._z_bowl_phase[self.phase]             # (N,)
        z = torch.where(is_block_phase, z_block, z_bowl) + self.TIP_Z_OFFSET   # (N,)

        return torch.cat([xy, z.unsqueeze(-1)], dim=-1)                        # (N, 3)

    def _advance_phases(self, target_pos_w: torch.Tensor, ee_pos_w: torch.Tensor):
        """Update self.phase and self.phase_step in place."""
        # Distance from the actual gripper_link world pos to the commanded one.
        dist = torch.norm(ee_pos_w - target_pos_w, dim=-1)                     # (N,)

        max_steps_now = self._max_steps_t[self.phase]                          # (N,)
        timed_out = (self.phase_step + 1) >= max_steps_now                     # (N,)
        reached = dist < self.POS_TOL                                          # (N,)
        is_fixed = self._fixed_duration_mask_t[self.phase]                     # (N,)

        # Advance: timeout always advances; reached advances only on non-fixed phases.
        # Don't advance past DONE.
        advance_now = (timed_out | (reached & ~is_fixed)) & (self.phase < self.DONE)

        self.phase = torch.where(advance_now, self.phase + 1, self.phase)
        self.phase_step = torch.where(
            advance_now, torch.zeros_like(self.phase_step), self.phase_step + 1
        )
