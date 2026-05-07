"""Scripted pick-and-place controller for Eval 2 v1 (BC warmstart data source).

Drives the SO-101 in the Eval 2 v1 env via **pytorch_kinematics DLS IK**
+ a per-env phase state machine, deterministically solving "pick the
{target_color} block, place it in the bowl".

Uses pytorch_kinematics' batched DLS IK with **position-only** targeting
(orientation_weight=0). Why: SO-101 is 5-DoF, so a 6-DoF pose goal is
overdetermined — we let DLS pick whatever wrist orientation the geometry
favors. Empirically this gives top-down-ish gripper for normal pick-and-
place targets, which is what we want.

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

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.sim.schemas import modify_collision_properties, modify_rigid_body_properties
from isaaclab.sim.schemas.schemas_cfg import (
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from sim.eval2.bc.ik_solver import SO101IKSolver

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Path to the SO-101 URDF used by pytorch_kinematics. We use the URDF
# that ships with isaac_so_arm101 (same one Isaac Sim loads), so the
# FK chain matches the simulated robot exactly.
_URDF_PATH = (
    Path("C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101")
    / "src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf"
)


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
    SLIDE_FORWARD = 2     # NEW: with fixed-jaw 4cm BEHIND cube after DESCEND,
                          # slide forward in +x until fixed jaw makes contact
    CLOSE = 3
    LIFT = 4
    ABOVE_BOWL = 5
    DESCEND_TO_RELEASE = 6
    OPEN = 7
    RETREAT = 8
    DONE = 9
    NUM_PHASES = 10

    PHASE_NAMES = (
        "APPROACH", "DESCEND", "SLIDE_FORWARD", "CLOSE", "LIFT",
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
    # v6.4: tip target AT block center -> z=+0.010 (cube center, 1 cm
    # above table). The vertical gripper has its jaws extending a few
    # mm below gripper_frame_link; with the previous target z=0.005
    # the jaw tips were touching the table. At z=0.010 (cube center)
    # the jaws straddle the cube nicely without hitting the floor.
    DESCEND_HEIGHT = 0.000
    # 8 cm above grasp height. Needed to clear the bowl walls during the
    # ABOVE_BOWL lateral transport: bowl walls are 2.5 cm tall (top at
    # z=0.030), and we want the gripper + cube to pass clearly above
    # them. With LIFT_HEIGHT=0.08, cube bottom ends at z=0.080 (after
    # cube center at 0.090, half=0.010) — way above the wall top. The
    # arm body is even higher so it also clears the walls.
    LIFT_HEIGHT = 0.08
    # Steps to RAMP the LIFT z target. Cube is kinematic during attach
    # so the grasp is decoupled from physics — no risk dropping the
    # cube on a fast ramp. Was 12 steps (0.24 s); 6 steps (0.12 s) cuts
    # the lift time in half. Per-step Δz = 0.08/6 = 13.3 mm, still under
    # MAX_STEP_XYZ=20 mm so the IK delta never saturates.
    LIFT_RAMP_STEPS = 6
    ABOVE_BOWL_HEIGHT = 0.08      # transport hover, well above the rim
    # v2: NEW — descend over the bowl to ~4 cm above the bowl floor before
    # releasing. With bowl_h = 2.5 cm and block_h = 2 cm, dropping from 4 cm
    # leaves only 1.5 cm of free fall — block lands cleanly without bouncing.
    RELEASE_HEIGHT = 0.04
    # RETREAT_HEIGHT same as RELEASE_HEIGHT: RETREAT is now skipped
    # (max_steps=1, no motion). The gripper just stays where it released
    # the cube, then the episode terminates.
    RETREAT_HEIGHT = 0.04

    # gripper_link sits ~9 cm ABOVE the FrameTransformer tip when pointing down
    TIP_Z_OFFSET = 0.09

    # v2: 2 cm/step (was 4 cm). Doc advice — 4 cm saturated the SO-101
    # actuators (effort_limit=1.9 N.m) at almost every step, which made
    # the actual EE motion sluggish and inconsistent. 2 cm leaves headroom
    # so the actuators track the IK target precisely.
    MAX_STEP_XYZ = 0.02

    # v5 (pytorch_kinematics IK): 15 mm. PD tracking error empirically
    # ~10-12 mm even after IK has fully converged (FK == Isaac to 0 mm
    # but PD overshoots/lags by ~0.05 rad on each joint, cumulative
    # gripper error ~1 cm). 5 mm was unreachable; 15 mm is still well
    # within half-cube width (10 mm) so the jaws still grip on CLOSE.
    POS_TOL = 0.015

    # GRASP_X_OFFSET = -0.040 (40 mm backward in robot's +x). With the
    # SLIDE_FORWARD phase added, the new strategy is:
    #   APPROACH/DESCEND : fixed jaw 4 cm BEHIND cube (gripper open)
    #   SLIDE_FORWARD    : slide horizontally toward cube until fixed jaw
    #                      makes contact (no offset; target = cube xyz)
    #   CLOSE            : moving jaw closes; magic-attach when it touches
    # The big offset prevents the gripper from descending INTO the cube
    # area where the moving jaw would hit during closing. Sliding in
    # horizontally is much gentler — the fixed jaw approaches the cube's
    # back face along a flat trajectory, with no vertical impact.
    GRASP_X_OFFSET = -0.040

    # ------------------------------------------------------------- step budgets
    # v2: added DESCEND_TO_RELEASE budget. Sum of non-DONE = 50+100+25+40+100
    # +60+25+20 = 420 steps. With env episode_length=10s = 500 steps (set
    # in run_scripted launcher), leaves 80 for DONE.
    PHASE_MAX_STEPS = (
        50,   # APPROACH
        150,  # DESCEND
        80,   # SLIDE_FORWARD       advances early on fixed-jaw contact
        120,  # CLOSE
        12,   # LIFT      LIFT_RAMP_STEPS=6 + 6 buffer; advances on dist
        60,   # ABOVE_BOWL          longer xy travel, advances on dist
        40,   # DESCEND_TO_RELEASE  short vertical drop into bowl
        50,   # OPEN      symmetric with CLOSE
        1,    # RETREAT   skipped — episode ends right after OPEN
        500,  # DONE — until episode timeout
    )

    FIXED_DURATION_PHASES = (CLOSE, OPEN)

    # +1 = open, -1 = close. One per phase index.
    GRIPPER_PER_PHASE = (
        +1.0,  # APPROACH            open
        +1.0,  # DESCEND             open
        +1.0,  # SLIDE_FORWARD       still open while sliding toward cube
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
    # NOTE — PHI_PER_PHASE used to live here (per-phase gripper tilt for
    # the analytical IK). We dropped it: pytorch_kinematics DLS picks
    # the wrist orientation automatically given a position-only target,
    # warm-started by current_joints (orientation stays continuous).

    # ------------------------------------------------------------------ init
    # Magic-attach trigger thresholds.
    # GRASP_FORCE_THRESHOLD_N: kept for log compatibility / fallback. The
    # PRIMARY trigger is now PROXIMITY (below) — physical contact in PhysX
    # is too unreliable on the SO-101 jaws (halo, bouncing, etc.). The
    # contact sensor is still read for diagnostic logging.
    GRASP_FORCE_THRESHOLD_N = 0.05
    # PROXIMITY trigger: during CLOSE, if the target cube is within
    # ATTACH_PROXIMITY_M of gripper_frame_link, magic-attach. 3 cm is
    # generous: with GRASP_X_OFFSET=-0.04 + SLIDE_FORWARD aiming at cube_x,
    # the gripper-to-cube xy distance during CLOSE should be a few mm at
    # most. 3 cm is also tolerant to PD lag and small position errors.
    ATTACH_PROXIMITY_M = 0.03
    # Earliest CLOSE step at which we'll consider attaching. Avoids
    # attaching during the very first frame of CLOSE (gripper just arrived,
    # might be transient overshoot). After this many steps the gripper
    # has had time to settle.
    ATTACH_AFTER_STEPS_IN_CLOSE = 5
    # MAX_RETRIES: how many times we'll go back to APPROACH if the grasp
    # check fails. After this we give up and let the trajectory continue
    # (the LIFT will fail visibly and the episode end-phase histogram
    # records it).
    MAX_RETRIES = 2

    def __init__(self, env: "ManagerBasedRLEnv"):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

        self.robot = env.scene["robot"]
        # Contact sensors (added in PickInClutterSceneCfg). Each reports
        # net force per filtered body (red + blue cubes).
        self.contact_gripper_link = env.scene["contact_gripper_link"]
        self.contact_moving_jaw = env.scene["contact_moving_jaw"]
        # Per-env retry counter: incremented each time the grasp check
        # fails at end of CLOSE. Reset on episode reset.
        self.grasp_retries = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # ---------------------------- Magic attach state -----------------
        # The PhysX rigid-body simulation of the SO-101 jaws on a 2 cm cube
        # is intrinsically unstable (jaw rotation produces explosive normal
        # force on contact, gripper bounces above the cube). To work around
        # this and still validate the grasp via real contact: we read the
        # contact sensors every step, and AS SOON AS both jaws have force >
        # threshold on the target cube, we "magic attach" — capture the
        # cube-in-gripper relative pose, then every subsequent step write
        # the cube's world pose to gripper_pose * relative_pose. The cube
        # stays glued to the gripper through LIFT/ABOVE_BOWL/DESCEND_TO_
        # RELEASE. At OPEN entry we detach (cube falls under gravity).
        self.cube_attached = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.cube_attach_pos_local = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.cube_attach_quat_local = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.cube_attach_quat_local[:, 0] = 1.0  # init to identity

        # Arm joints (5) + gripper (1).
        self.arm_joint_ids, self.arm_joint_names = self.robot.find_joints(
            ["shoulder_.*", "elbow_flex", "wrist_.*"]
        )
        self.gripper_joint_id = self.robot.find_joints(["gripper"])[0][0]

        # End-effector frame — used to read the actual tip position for
        # phase-transition checks. We read ``gripper_frame_link`` body
        # state directly (NOT the env's "ee_frame" FrameTransformer,
        # which has an offset at gripper_link). Why: the IK chain
        # targets ``gripper_frame_link`` (pytorch_kinematics' chain end),
        # and the distance comparison MUST use the same frame, otherwise
        # the IK can converge perfectly to its target while the distance
        # stays >POS_TOL forever (mismatch ~1 cm between gripper_link +
        # offset(0.01, 0, -0.09) and gripper_frame_link).
        self.ee_frame = env.scene["ee_frame"]  # kept for backwards-compat
        gframe_ids, _ = self.robot.find_bodies("gripper_frame_link")
        self.gripper_frame_idx = gframe_ids[0]
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

        # IK solver: manual DLS (position-only) on the URDF chain, warm-
        # started with current_joints each step. Iterates to convergence
        # (n_iter=100) — pk.PseudoInverseIK was returning partial-step
        # solutions due to its early-stopping heuristic, which made the
        # commanded joint targets crawl toward the IK solution over many
        # env steps instead of jumping there in one IK call.
        self.ik = SO101IKSolver(
            urdf_path=_URDF_PATH,
            num_envs=self.num_envs,
            device=self.device,
            n_iter=100,
            lr=0.3,
            damping=0.01,
            early_stop_mm=0.5,
            converged_threshold_mm=5.0,
            debug_log_every=200,  # log convergence every 200 IK calls
        )

        # Running tally for IK convergence diagnostics.
        self._ik_calls = 0
        self._ik_converged = 0

        # Snapshot of the LIFT-phase target. Captured at the CLOSE -> LIFT
        # transition (= the block position at the moment of grasp + 12 cm).
        # Without this snapshot, target_xyz would chase the block as it
        # rises with the gripper, making the distance criterion unreachable
        # (the target moves at the same rate as the tip).
        # Bowl-relative phases don't need this because the bowl is
        # kinematic — it never moves.
        self.lift_target_xyz = torch.zeros(self.num_envs, 3, device=self.device)
        # Z at which LIFT started (= ee_pos_w[:, 2] at CLOSE -> LIFT).
        # Used to RAMP the z target linearly from start_z up to start_z +
        # LIFT_HEIGHT over LIFT_RAMP_STEPS, instead of jumping the full
        # 12 cm at once.
        self.lift_start_z = torch.zeros(self.num_envs, device=self.device)
        # Cube-anchored LIFT: offset between gripper and cube xy at LIFT
        # entry. During LIFT we use ``cube_xy_live + offset`` as the target
        # xy instead of a fixed gripper snapshot. This way if the cube
        # shifts laterally, the gripper tracks it (preserving the relative
        # gripper-cube alignment that grants the grasp). If the grasp is
        # solid the cube doesn't move relative to the gripper, so this is
        # equivalent to the snapshot — but if the grip is marginal the
        # tracking can recover instead of leaving the cube behind.
        self.lift_gripper_to_cube_offset_xy = torch.zeros(
            self.num_envs, 2, device=self.device
        )

        # Snapshot of the JOINT POSITIONS at the DESCEND -> CLOSE
        # transition. During CLOSE, we COMMAND THESE JOINTS instead
        # of running IK. Why: while the gripper closes around the cube,
        # the cube can move slightly (jaws frotter contre le cube).
        # The IK target is computed from block_pos_w → if the block
        # moves, the target moves, the IK gives new joint angles, and
        # the ARM MOVES MID-CLOSE. This violent motion (vel up to
        # 3.95 rad/s on wrist_flex observed in logs) ejects the cube.
        # Freezing joints during CLOSE breaks this feedback loop —
        # arm holds position, only gripper joint actuates.
        self.close_arm_snapshot = torch.zeros(self.num_envs, 5, device=self.device)

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
        # 10 entries — one per phase index in PHASE_NAMES.
        self._z_block_phase = torch.tensor([
            self.APPROACH_HEIGHT,  # 0 APPROACH           block + 3 cm
            self.DESCEND_HEIGHT,   # 1 DESCEND            block center
            self.DESCEND_HEIGHT,   # 2 SLIDE_FORWARD      block center
            self.DESCEND_HEIGHT,   # 3 CLOSE              hold at grasp height
            self.LIFT_HEIGHT,      # 4 LIFT               block + 3 cm
            0.0,                   # 5 ABOVE_BOWL         (bowl-relative)
            0.0,                   # 6 DESCEND_TO_RELEASE (bowl-relative)
            0.0,                   # 7 OPEN               (bowl-relative)
            0.0,                   # 8 RETREAT            (bowl-relative)
            0.0,                   # 9 DONE               (bowl-relative)
        ], device=self.device)
        self._z_bowl_phase = torch.tensor([
            0.0,                       # 0 APPROACH
            0.0,                       # 1 DESCEND
            0.0,                       # 2 SLIDE_FORWARD
            0.0,                       # 3 CLOSE
            0.0,                       # 4 LIFT
            self.ABOVE_BOWL_HEIGHT,    # 5 ABOVE_BOWL          bowl + 8 cm
            self.RELEASE_HEIGHT,       # 6 DESCEND_TO_RELEASE  bowl + 4 cm
            self.RELEASE_HEIGHT,       # 7 OPEN                hold at release height
            self.RETREAT_HEIGHT,       # 8 RETREAT             bowl + 15 cm
            self.RETREAT_HEIGHT,       # 9 DONE                bowl + 15 cm
        ], device=self.device)

    # ------------------------------------------------------------------- API
    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset per-env state for the given env ids (or all if None)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # If any env being reset has cube_attached (= cube still kinematic),
        # restore it to dynamic. Otherwise the env reset would try to
        # respawn the cube while it's kinematic — randomization would
        # not take effect.
        reset_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        reset_mask[env_ids] = True
        needs_dynamic_restore = self.cube_attached & reset_mask
        if needs_dynamic_restore.any():
            self._set_cube_kinematic(needs_dynamic_restore, kinematic=False)
        self.phase[env_ids] = 0
        self.phase_step[env_ids] = 0
        # Reset the grasp-retry counter at episode start.
        self.grasp_retries[env_ids] = 0
        # Detach any magic-attached cubes for these envs.
        self.cube_attached[env_ids] = False

    @torch.no_grad()
    def compute_action(self) -> torch.Tensor:
        """One env-step worth of action for all envs. Returns (num_envs, 6).

        Pipeline:
          1. State machine -> target tip xyz in world frame.
          2. World -> base frame transform.
          3. SO101IKSolver (pytorch_kinematics DLS, position-only,
             warm-started with current joints) -> URDF joint angles.
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

        # 3. pytorch_kinematics DLS IK -> URDF joint angles for the 5
        # arm joints. Warm-started with the CURRENT joint pose so the
        # solution stays continuous frame-to-frame (no elbow flips, no
        # "ostrich jump" pattern reported in lerobot issue #2531).
        # NOTE: GRASP_X_OFFSET is already baked into target_pos_w (and
        # therefore into target_pos_b) by _compute_target_xyz_world for
        # grasp-side phases. So both the IK and the phase-advance check
        # below use the SAME shifted target — without this consistency,
        # the phase never advances because the IK satisfies a target
        # 25 mm offset from where advance_phases is measuring distance.
        current_arm = self.robot.data.joint_pos[:, self.arm_joint_ids]       # (N, 5)
        # Wrist constraint ON for all phases (including LIFT). With the
        # right GRASP_X_OFFSET=-0.010 the cube is properly gripped by
        # both jaws, so during LIFT the gripper just needs to go
        # straight up while staying vertical. The constraint formula
        # w_f = pi/2 - s_l - e_f keeps the gripper vertical, which
        # preserves jaw alignment on the cube during the lift. With
        # LIFT_HEIGHT=0.03 (small enough to stay in elbow-up branch),
        # the constraint doesn't force the IK into elbow flips.
        joint_pos_des, ik_converged = self.ik.solve(
            target_pos_b=target_pos_b,
            current_joints=current_arm,
        )
        # If IK failed to converge (target unreachable / hit joint
        # limits), hold the current pose for that env. Better to freeze
        # than send a half-baked target the PD will track wildly.
        joint_pos_des = torch.where(
            ik_converged.unsqueeze(-1), joint_pos_des, current_arm
        )
        # ENFORCE wrist locks at the action layer too, regardless of IK.
        # Why: in fallback (IK didn't converge) we'd otherwise command
        # the joint to its DRIFTED current value, perpetuating the drift.
        # We always want the commanded target for wrist_flex to be the
        # vertical-gripper constraint, and wrist_roll to be 0.
        import math as _math
        joint_pos_des[:, 3] = (
            (_math.pi / 2.0) - joint_pos_des[:, 1] - joint_pos_des[:, 2]
        )
        joint_pos_des[:, 4] = 0.0

        # FREEZE the arm during CLOSE. Snapshot joint_pos_des at the
        # DESCEND -> CLOSE transition (phase_step == 0 of CLOSE), then
        # use that snapshot for the entire CLOSE phase. This stops the
        # IK from chasing the cube as it gets pushed by the gripping
        # jaws — arm holds position, only the gripper joint moves.
        in_close = (self.phase == self.CLOSE)
        just_entered_close = in_close & (self.phase_step == 0)
        if just_entered_close.any():
            self.close_arm_snapshot = torch.where(
                just_entered_close.unsqueeze(-1),
                joint_pos_des,
                self.close_arm_snapshot,
            )
        joint_pos_des = torch.where(
            in_close.unsqueeze(-1), self.close_arm_snapshot, joint_pos_des
        )

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
        # IMPORTANT: use gripper_frame_link directly (matches what the IK
        # solves for). The env's ee_frame is gripper_link + offset, ~1 cm
        # away from gripper_frame_link — using it would mean the IK can
        # converge perfectly while the distance check stays > POS_TOL.
        actual_tip_w = self.robot.data.body_state_w[
            :, self.gripper_frame_idx, :3
        ]
        self._advance_phases(target_pos_w, actual_tip_w)

        # 7. Magic attach: detect contact, attach if needed, maintain attach,
        # detach on release. Done LAST so the cube pose is written to sim
        # right before the next physics step uses it.
        self._update_magic_attach()

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
        # Override xy for LIFT envs with cube-anchored target:
        #   target_xy = live_cube_xy + frozen_offset
        # The offset is captured at LIFT entry. If the cube shifts laterally
        # during LIFT (because the grip is marginal and the cube wobbles),
        # the gripper tracks it instead of going off-axis. If the cube is
        # solidly held, the relative xy is invariant and this is identical
        # to a fixed-gripper snapshot.
        lift_xy = block_pos_w[:, :2] + self.lift_gripper_to_cube_offset_xy
        xy = torch.where(
            is_lift_phase.unsqueeze(-1),
            lift_xy,
            xy,
        )

        # z choice: surface_z + per-phase height offset.
        z_block = block_pos_w[:, 2] + self._z_block_phase[self.phase]          # (N,)
        z_bowl = bowl_pos_w[:, 2] + self._z_bowl_phase[self.phase]             # (N,)
        z = torch.where(is_block_phase, z_block, z_bowl)                       # (N,)
        # Override z for LIFT envs with a RAMPED snapshot: z target rises
        # linearly from lift_start_z to lift_start_z + LIFT_HEIGHT over
        # LIFT_RAMP_STEPS. This avoids a 12 cm step at LIFT entry, which
        # would saturate the IK delta and shake the cube loose.
        ramp_progress = (
            self.phase_step.float() / float(self.LIFT_RAMP_STEPS)
        ).clamp(max=1.0)                                                       # (N,)
        z_lift = self.lift_start_z + ramp_progress * self.LIFT_HEIGHT           # (N,)
        z = torch.where(is_lift_phase, z_lift, z)

        target_xyz = torch.cat([xy, z.unsqueeze(-1)], dim=-1)                  # (N, 3)

        # Apply GRASP_X_OFFSET only to APPROACH and DESCEND. With the new
        # SLIDE_FORWARD phase, those two phases position the fixed jaw 4 cm
        # BEHIND the cube. SLIDE_FORWARD then drives the gripper toward
        # cube_x (no offset) until the fixed jaw makes contact. CLOSE
        # holds at the same position (cube_x). LIFT/ABOVE_BOWL/etc don't
        # use this offset.
        is_offset_phase = (self.phase == self.APPROACH) | (self.phase == self.DESCEND)
        target_xyz[:, 0] = torch.where(
            is_offset_phase, target_xyz[:, 0] + self.GRASP_X_OFFSET, target_xyz[:, 0]
        )
        return target_xyz

    def _advance_phases(self, target_pos_w: torch.Tensor, ee_pos_w: torch.Tensor):
        """Update self.phase and self.phase_step in place.

        Transition rules:
          - Cartesian-target phases (APPROACH, DESCEND, LIFT, ABOVE_BOWL,
            DESCEND_TO_RELEASE, RETREAT): advance when the actual tip
            reaches within POS_TOL of the target OR the per-phase step
            budget elapses. The timeout fallback handles cases where
            the IK target is geometrically unreachable (e.g. DESCEND
            target below the table is blocked by physical contact —
            the gripper sits at z=0 but the target is z=-0.005, so
            distance > POS_TOL forever without the timeout).
          - Fixed-duration phases (CLOSE, OPEN): advance ONLY on step
            count, not distance. The gripper joint needs time to close
            mechanically; tip position is irrelevant during gripper
            actuation.
        """
        # Distance from the actual tip world pos to the commanded target.
        # Special case for LIFT: compare against the FINAL lift target
        # (= lift_start_z + LIFT_HEIGHT), not the ramped intermediate.
        # Otherwise the ramped target sits exactly at current tip pos
        # at phase_step=0, dist=0, the phase advances after 1 step,
        # and the gripper never actually rises.
        in_lift = (self.phase == self.LIFT)
        dist_target_w = torch.where(
            in_lift.unsqueeze(-1), self.lift_target_xyz, target_pos_w
        )
        dist = torch.norm(ee_pos_w - dist_target_w, dim=-1)                    # (N,)

        max_steps_now = self._max_steps_t[self.phase]                          # (N,)
        timed_out = (self.phase_step + 1) >= max_steps_now                     # (N,)
        reached = dist < self.POS_TOL                                          # (N,)
        is_fixed = self._fixed_duration_mask_t[self.phase]                     # (N,)

        # SLIDE_FORWARD has an extra advance criterion: fixed jaw contact.
        # As soon as the fixed jaw makes any contact with the target cube
        # (force > 0.05 N), we know we've slid all the way to it and can
        # transition to CLOSE. This is more reliable than waiting for the
        # Cartesian distance criterion (which would require pushing the
        # cube — bad).
        in_slide = (self.phase == self.SLIDE_FORWARD)
        if in_slide.any():
            ff_data = self.contact_gripper_link.data.force_matrix_w
            if ff_data is not None:
                target_idx = self.env.target_color.long()
                env_arange = torch.arange(self.num_envs, device=self.device)
                fixed_force = ff_data.squeeze(1)[env_arange, target_idx].norm(dim=-1)
                fixed_contact = fixed_force > 0.05
                # Advance on contact even before timeout/reached.
                reached = reached | (in_slide & fixed_contact)

        # Distance OR timeout for Cartesian phases; step-count only for fixed.
        advance_non_fixed = (reached | timed_out) & ~is_fixed
        advance_fixed = timed_out & is_fixed
        advance_now = (advance_non_fixed | advance_fixed) & (self.phase < self.DONE)

        # Snapshot the LIFT target xyz at the CLOSE -> LIFT transition.
        # We need this BEFORE updating self.phase so we can detect the
        # transition. envs that just satisfy advance_now AND were in CLOSE
        # are about to enter LIFT.
        # IMPORTANT: snapshot the GRIPPER position, NOT the cube position.
        # If we snapshot the cube, the LIFT target xy = cube_xy (no
        # GRASP_X_OFFSET applied since LIFT is not a grasp phase). But
        # the gripper is at cube_xy + GRASP_X_OFFSET (-25 mm in x). So
        # LIFT would yank the gripper +25 mm forward, ejecting the
        # freshly-grasped cube. Snapshotting the gripper position means
        # LIFT goes STRAIGHT UP from the grasp point — cube stays in.
        becomes_lift = advance_now & (self.phase == self.CLOSE)
        # GRASP CHECK: at the CLOSE -> LIFT transition, verify both jaws
        # have ≥ GRASP_FORCE_THRESHOLD_N contact force on the *target*
        # cube. If yes, proceed to LIFT. If no AND retries remaining,
        # bounce the env back to APPROACH (force_to_lift = False masks
        # this env out of becomes_lift, so the phase update below stays
        # at CLOSE; we then explicitly rewind to APPROACH and increment
        # the retry counter).
        retry_now = torch.zeros_like(becomes_lift)
        if becomes_lift.any():
            # With magic attach: a "successful grasp" means cube_attached is
            # True (set during CLOSE when contact was first detected). We
            # don't re-check live contact forces here because once attached
            # we teleport the cube each step → contact sensors read 0.
            grasp_ok = self.cube_attached.clone()
            retry_allowed = self.grasp_retries < self.MAX_RETRIES               # (N,)
            for env_idx in becomes_lift.nonzero(as_tuple=False).flatten().tolist():
                ok_str = "OK (attached)" if grasp_ok[env_idx].item() else "FAIL (no contact during CLOSE)"
                print(
                    f"[scripted] env{env_idx} CLOSE -> LIFT decision: {ok_str}",
                    flush=True,
                )
            # Envs that MUST retry: at CLOSE end, no attach, retries left.
            retry_now = becomes_lift & (~grasp_ok) & retry_allowed
            becomes_lift = becomes_lift & ~retry_now
            advance_now = advance_now & ~retry_now

        if becomes_lift.any():
            new_xy = ee_pos_w[:, :2]                                           # (N, 2)
            start_z = ee_pos_w[:, 2]                                           # (N,)
            new_z = start_z + self.LIFT_HEIGHT                                 # (N,)
            new_target = torch.cat([new_xy, new_z.unsqueeze(-1)], dim=-1)      # (N, 3)
            self.lift_target_xyz = torch.where(
                becomes_lift.unsqueeze(-1), new_target, self.lift_target_xyz
            )
            # Capture starting z for the ramp in _compute_target_xyz_world.
            self.lift_start_z = torch.where(
                becomes_lift, start_z, self.lift_start_z
            )
            # Capture the gripper-to-cube xy offset at LIFT entry. During
            # LIFT, target_xy = live_cube_xy + this_offset, so the gripper
            # tracks the cube laterally with the same relative alignment
            # that produced the grasp.
            cube_xy_now = self._target_block_pos_w()[:, :2]
            new_offset = ee_pos_w[:, :2] - cube_xy_now
            self.lift_gripper_to_cube_offset_xy = torch.where(
                becomes_lift.unsqueeze(-1),
                new_offset,
                self.lift_gripper_to_cube_offset_xy,
            )

        self.phase = torch.where(advance_now, self.phase + 1, self.phase)
        self.phase_step = torch.where(
            advance_now, torch.zeros_like(self.phase_step), self.phase_step + 1
        )

        # Retry envs: rewind to APPROACH, reset phase_step, increment counter.
        if retry_now.any():
            self.phase = torch.where(
                retry_now, torch.full_like(self.phase, self.APPROACH), self.phase
            )
            self.phase_step = torch.where(
                retry_now, torch.zeros_like(self.phase_step), self.phase_step
            )
            self.grasp_retries = self.grasp_retries + retry_now.long()
            n = int(retry_now.sum().item())
            print(
                f"[scripted] grasp check FAILED on {n} env(s) at end of CLOSE -> "
                f"retrying from APPROACH (counter now {self.grasp_retries[retry_now][0].item()}/{self.MAX_RETRIES})",
                flush=True,
            )

    @torch.no_grad()
    def _grasp_check_per_env(self):
        """Return (grasp_ok, force_fixed, force_moving), all shaped (N,).

        ``grasp_ok`` is True iff BOTH jaws have ≥ threshold contact force on
        the *target color* cube. The two force tensors are returned for
        diagnostic logging in the caller.

        Reads ContactSensorData.force_matrix_w which has shape
        (N, B=1, M=2, 3) — N envs, 1 sensor body, 2 filtered cubes, xyz force.
        """
        # force_matrix_w shape: (N, 1, 2, 3)
        f_fixed = self.contact_gripper_link.data.force_matrix_w               # (N, 1, 2, 3)
        f_moving = self.contact_moving_jaw.data.force_matrix_w                # (N, 1, 2, 3)
        if f_fixed is None or f_moving is None:
            # Sensors not initialized yet (shouldn't happen mid-episode);
            # accept the grasp by default to not block the trajectory, but
            # mark the forces as -1 in the return so the print clearly
            # signals "no sensor data" rather than "0 N contact".
            sentinel = torch.full(
                (self.num_envs,), -1.0, device=self.device
            )
            return (
                torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
                sentinel,
                sentinel,
            )
        # Squeeze the body dim (always 1 here).
        f_fixed_2 = f_fixed.squeeze(1)                                         # (N, 2, 3)
        f_moving_2 = f_moving.squeeze(1)                                       # (N, 2, 3)
        # Pick the target-color cube column (target_color: 0 = red, 1 = blue).
        # filter_prim_paths_expr order: [BlockRed, BlockBlue] -> idx 0 = red, 1 = blue.
        target_idx = self.env.target_color.long()                              # (N,)
        env_arange = torch.arange(self.num_envs, device=self.device)
        f_fixed_target = f_fixed_2[env_arange, target_idx]                     # (N, 3)
        f_moving_target = f_moving_2[env_arange, target_idx]                   # (N, 3)
        force_fixed = f_fixed_target.norm(dim=-1)                              # (N,)
        force_moving = f_moving_target.norm(dim=-1)                            # (N,)
        thr = self.GRASP_FORCE_THRESHOLD_N
        # OR: attach as soon as EITHER jaw makes contact. Catches the first
        # touch instantly and freezes the cube to the gripper before PhysX
        # can build up explosive contact forces.
        grasp_ok = (force_fixed > thr) | (force_moving > thr)
        return grasp_ok, force_fixed, force_moving

    @torch.no_grad()
    def _update_magic_attach(self):
        """Manage the cube↔gripper magic attach (every step, called from
        ``compute_action``).

        Trigger: as soon as both jaws report contact force > threshold on
        the target cube, snapshot the cube's pose relative to gripper_frame_
        link and mark the env as attached.

        Maintain: each step where the env is attached, overwrite the cube's
        world pose to ``gripper_pose * snapshot_relative_pose``. Cube stays
        glued to the gripper through LIFT/ABOVE_BOWL/DESCEND_TO_RELEASE
        regardless of physics-engine contact dynamics.

        Detach: on entry to OPEN phase, clear the attached flag. The cube
        is then free to fall under gravity (no more pose overrides).
        """
        # 1. Trigger: as soon as the FIXED JAW makes contact with the
        # target cube during SLIDE_FORWARD or early CLOSE. This is the
        # moment when the gripper has horizontally arrived at the cube,
        # the cube hasn't been pushed around yet, and the cube↔fixed_jaw
        # contact surface is well-defined. We capture the relative pose
        # AS-IS — cube stays exactly where it is, just gets locked to
        # gripper. Subsequent moving-jaw closing motion (even if it
        # self-blocks at +0.36 due to convex_decomposition) doesn't
        # matter; the cube is already firmly attached.
        _, force_fixed, force_moving = self._grasp_check_per_env()
        in_grip_phase = (self.phase == self.SLIDE_FORWARD) | (self.phase == self.CLOSE)
        fixed_jaw_contact = force_fixed > self.GRASP_FORCE_THRESHOLD_N
        can_attach = in_grip_phase & fixed_jaw_contact & ~self.cube_attached
        if can_attach.any():
            cube_pose_w = self._target_block_pose_w()                          # (N, 7)
            gripper_pose_w = self.robot.data.body_state_w[
                :, self.gripper_frame_idx, :7
            ]                                                                  # (N, 7)
            # Clamp the cube's z BEFORE computing the relative pose, so
            # the cube can never end up below table level. The cube's
            # half-height is 1cm, table top at z=0 → cube center must
            # be ≥ 0.010m. Without this clamp, when the gripper has PD
            # overshoot during DESCEND/CLOSE and is below cube level, the
            # captured relative offset means "cube AT gripper z" — and
            # later as gripper rises during LIFT, cube tracks but the
            # initial relative was already wrong.
            cube_pos_clamped = cube_pose_w[:, :3].clone()
            cube_pos_clamped[:, 2] = cube_pos_clamped[:, 2].clamp(min=0.010)
            rel_pos, rel_quat = subtract_frame_transforms(
                gripper_pose_w[:, :3], gripper_pose_w[:, 3:7],
                cube_pos_clamped, cube_pose_w[:, 3:7],
            )
            self.cube_attach_pos_local = torch.where(
                can_attach.unsqueeze(-1), rel_pos, self.cube_attach_pos_local
            )
            self.cube_attach_quat_local = torch.where(
                can_attach.unsqueeze(-1), rel_quat, self.cube_attach_quat_local
            )
            self.cube_attached = self.cube_attached | can_attach
            # Set the just-attached cubes to KINEMATIC so they don't react
            # to contact forces from the moving jaw / gripper. A kinematic
            # body is rigidly moved by our pose writes (no physics
            # interference). Reverted at OPEN/detach.
            self._set_cube_kinematic(can_attach, kinematic=True)
            for env_idx in can_attach.nonzero(as_tuple=False).flatten().tolist():
                ff = force_fixed[env_idx].item()
                fm = force_moving[env_idx].item()
                phase_name = self.PHASE_NAMES[self.phase[env_idx].item()]
                print(
                    f"[scripted] env{env_idx} MAGIC ATTACH on FIXED JAW contact in {phase_name} "
                    f"(forces fixed={ff:.2f}N moving={fm:.2f}N) — cube locked KINEMATIC",
                    flush=True,
                )

        # 2. Detach: any env in OPEN phase or later.
        in_release_phase = self.phase >= self.OPEN
        must_detach = self.cube_attached & in_release_phase
        if must_detach.any():
            self.cube_attached = self.cube_attached & ~must_detach
            # Restore cubes to DYNAMIC so they fall under gravity into bowl.
            self._set_cube_kinematic(must_detach, kinematic=False)
            for env_idx in must_detach.nonzero(as_tuple=False).flatten().tolist():
                phase_name = self.PHASE_NAMES[self.phase[env_idx].item()]
                print(
                    f"[scripted] env{env_idx} MAGIC DETACH (entering {phase_name}) — cube DYNAMIC",
                    flush=True,
                )

        # 3. Maintain attach: write cube pose AND velocity every step for
        # attached envs. The velocity must match the gripper's velocity
        # so the cube "rides along" during the physics step. Otherwise
        # the cube starts each step with v=0 while the gripper has
        # nonzero v → they separate during the step, and contact forces
        # build up at the next teleport. Matching velocity = no separation
        # = no contact force = stable rigid attach.
        if not self.cube_attached.any():
            return
        body_state = self.robot.data.body_state_w[
            :, self.gripper_frame_idx, :
        ]                                                                       # (N, 13)
        gripper_pose_w = body_state[:, :7]                                     # (N, 7)
        gripper_lin_vel_w = body_state[:, 7:10]                                # (N, 3)
        gripper_ang_vel_w = body_state[:, 10:13]                               # (N, 3)
        new_pos, new_quat = combine_frame_transforms(
            gripper_pose_w[:, :3], gripper_pose_w[:, 3:7],
            self.cube_attach_pos_local, self.cube_attach_quat_local,
        )
        # Hard floor: cube center can never go below z=0.010 (cube
        # half-height; cube bottom at table top z=0). Without this clamp,
        # if the gripper has PD overshoot during CLOSE and dips below
        # cube level, the rigidly-attached cube follows it INTO the table.
        new_pos[:, 2] = new_pos[:, 2].clamp(min=0.010)
        # Cube linear velocity at its center: v_cube = v_gripper +
        # ω_gripper × (cube_pos_world - gripper_pos_world). Since the cube
        # is right at the gripper, the lever arm is small and gripper's
        # linear velocity dominates.
        rel_pos_world = new_pos - gripper_pose_w[:, :3]                        # (N, 3)
        cube_lin_vel_w = gripper_lin_vel_w + torch.linalg.cross(
            gripper_ang_vel_w, rel_pos_world, dim=-1
        )                                                                       # (N, 3)
        cube_ang_vel_w = gripper_ang_vel_w                                     # (N, 3)
        cube_vel = torch.cat([cube_lin_vel_w, cube_ang_vel_w], dim=-1)         # (N, 6)
        # Apply per-env: each env's target_color decides which cube prim
        # gets its pose overwritten (red or blue).
        target_color = self.env.target_color.long()                            # (N,)
        for color_idx, block_name in enumerate(("block_red", "block_blue")):
            color_mask = self.cube_attached & (target_color == color_idx)
            if not color_mask.any():
                continue
            env_ids = color_mask.nonzero(as_tuple=False).flatten()
            new_pose = torch.cat(
                [new_pos[env_ids], new_quat[env_ids]], dim=-1
            )                                                                  # (M, 7)
            block = self.env.scene[block_name]
            block.write_root_pose_to_sim(new_pose, env_ids=env_ids)
            block.write_root_velocity_to_sim(cube_vel[env_ids], env_ids=env_ids)

    def _target_block_pose_w(self) -> torch.Tensor:
        """Per-env target block pose (pos+quat) in world frame (N, 7)."""
        red_pose = self.env.scene["block_red"].data.root_pose_w
        blue_pose = self.env.scene["block_blue"].data.root_pose_w
        is_red = (self.env.target_color == 0).unsqueeze(-1)
        return torch.where(is_red, red_pose, blue_pose)

    def _set_cube_kinematic(self, env_mask: torch.Tensor, kinematic: bool):
        """Toggle KINEMATIC + COLLISION on the target cube of each env in mask.

        Magic-attach state (kinematic=True):
            - kinematic_enabled = True   → no force response (rigidly moved
              only by our pose writes; no wobble/sliding from contact forces)
            - collision_enabled = False  → cube is a "ghost", passes
              through gripper jaws. Without this, the moving jaw closes
              into the kinematic cube and gets PUSHED BACK (joint opens
              instead of closing).

        Detached state (kinematic=False):
            - both restored to their dynamic-cube defaults so the cube
              falls under gravity and lands in the bowl.
        """
        if not env_mask.any():
            return
        rb_cfg = RigidBodyPropertiesCfg(kinematic_enabled=kinematic)
        col_cfg = CollisionPropertiesCfg(collision_enabled=not kinematic)
        target_color = self.env.target_color.long()
        for env_idx in env_mask.nonzero(as_tuple=False).flatten().tolist():
            color_idx = int(target_color[env_idx].item())
            block_name = "BlockRed" if color_idx == 0 else "BlockBlue"
            prim_path = f"/World/envs/env_{env_idx}/{block_name}"
            modify_rigid_body_properties(prim_path, rb_cfg)
            modify_collision_properties(prim_path, col_cfg)

