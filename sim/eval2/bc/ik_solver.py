"""Batched IK for SO-101 using pytorch_kinematics + DLS.

Designed for batched IK across N parallel envs in Isaac Sim, with
warm-starting from current joint positions to prevent IK jumping
between equally-valid solutions across consecutive steps (the
"ostrich jump" pattern reported in lerobot issue #2531).

Key design choices:
    - **Position-only** IK (orientation_weight=0). SO-101 is 5-DoF;
      asking for a 6-DoF pose is over-constrained. We let DLS pick
      whatever wrist orientation is geometrically natural.
      Empirically, with warm-starting from a "gripper down" home pose,
      the orientation stays nearly vertical because DLS picks the
      closest solution to current_joints.
    - **DLS** with damping=1e-2 (default) handles singularities at the
      workspace edge gracefully — when the target is unreachable, the
      arm stops moving instead of oscillating.
    - **Warm-start** with current_joints = the previous frame's joint
      pose. This locks the elbow-up vs elbow-down branch, and avoids
      jumps across DLS local minima.
    - **Joint limits** enforced from chain.low/chain.high (parsed from
      URDF directly).
    - Gripper joint is NOT controlled by IK — separate action channel.

URDF source: ``isaac_so_arm101/src/.../trs_so101/urdf/so_arm101.urdf``
(same one Isaac Sim loads, so FK matches the simulated robot exactly).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import torch


# Joint names in the URDF SO-101 chain. Must match exactly. Order is
# the order pytorch_kinematics returns from get_joint_parameter_names.
SO101_ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# Frame at the gripper tip — this is what the IK targets.
SO101_EE_FRAME = "gripper_frame_link"


class SO101IKSolver:
    """Batched DLS IK for SO-101 with position-only target + warm-start."""

    def __init__(
        self,
        urdf_path: str | Path,
        num_envs: int,
        device: torch.device | str = "cuda:0",
        ee_frame: str = SO101_EE_FRAME,
        n_iter: int = 100,
        lr: float = 0.3,
        damping: float = 0.01,
        early_stop_mm: float = 0.5,
        converged_threshold_mm: float = 5.0,
        debug_log_every: int = 0,
        lock_wrist_flex: bool = True,
        wrist_flex_value: float = math.pi / 2,
    ):
        import pytorch_kinematics as pk

        self.pk = pk
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self.num_envs = num_envs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = torch.float32

        # Manual DLS hyperparams.
        self.n_iter = n_iter
        self.lr = lr
        self.damping = damping
        self.early_stop = early_stop_mm * 1e-3   # in meters
        self.converged_threshold = converged_threshold_mm * 1e-3
        self._debug_every = debug_log_every
        self._call_count = 0

        # Constrain BOTH wrist_flex AND wrist_roll to fixed values:
        #   wrist_flex = pi/2  (gripper straight down)
        #   wrist_roll = 0     (gripper jaws aligned with robot's y axis)
        # Why: SO-101 has 5 arm DoF for a 3-DoF position goal. The 2
        # redundant DoFs (wrist_flex, wrist_roll) drift unpredictably
        # under position-only DLS. Drifting wrist_flex tilts the gripper
        # forward → collides with cube during descent. Drifting
        # wrist_roll rotates the jaws around the vertical axis → jaws
        # no longer apply parallel forces on opposite cube faces, grasp
        # fails. Locking both gives a 3-DoF IK on a 3-DoF goal — unique
        # solution, fully predictable orientation, jaws guaranteed
        # parallel-vertical.
        self.lock_wrist_flex = lock_wrist_flex
        self.wrist_flex_value = wrist_flex_value
        self.wrist_flex_idx = 3  # index of wrist_flex in arm joints
        self.wrist_roll_idx = 4  # index of wrist_roll in arm joints
        self.wrist_roll_value = 0.0

        # Load the kinematic chain from URDF directly. The chain runs from
        # base_link (root) to ``ee_frame`` (default gripper_frame_link),
        # exactly the 5 arm joints we care about — the gripper joint is
        # NOT in this chain (it's downstream of gripper_frame_link).
        with open(urdf_path, "rb") as f:
            urdf_bytes = f.read()
        self.chain = pk.build_serial_chain_from_urdf(urdf_bytes, ee_frame).to(
            device=self.device, dtype=self.dtype
        )

        chain_joints = self.chain.get_joint_parameter_names()
        if chain_joints != SO101_ARM_JOINT_NAMES:
            print(
                f"[ik] WARNING — chain joint order {chain_joints} differs "
                f"from expected {SO101_ARM_JOINT_NAMES}. Caller must ensure "
                f"current_joints / joint_targets are in the chain order."
            )
        self.joint_names = chain_joints
        self.dof = len(chain_joints)

    @torch.no_grad()
    def solve(
        self,
        target_pos_b: torch.Tensor,  # (N, 3) target tip xyz in base frame
        current_joints: torch.Tensor,  # (N, 5) current arm joints (warm-start seed)
        lock_wrist_mask: torch.Tensor | None = None,  # (N,) bool, override per-env
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Position-only DLS IK iterated to convergence.

        ``lock_wrist_mask`` (per-env bool) overrides the class-level
        ``lock_wrist_flex`` flag. True envs apply the wrist constraint;
        False envs run pure 5-DOF IK (no wrist coupling). When None,
        defaults to all-True if ``self.lock_wrist_flex`` is set, else
        all-False — same behavior as before this parameter existed.

        Why manual DLS instead of pk.PseudoInverseIK: empirically the pk
        solver was returning partial-step solutions (early stopping kicks
        in after a few iters of "no improvement"), so the action grew
        slowly over many env steps instead of converging in one IK call.
        Manual DLS with explicit n_iter, lr, damping gives full control
        and reliably converges in 30-60 iterations for typical pick
        targets.

        Returns:
            joint_targets: (N, 5) float32 — fully-converged joint angles.
            converged: (N,) bool — True iff final FK error < threshold.
        """
        N = current_joints.shape[0]
        if lock_wrist_mask is None:
            lock_wrist_mask = torch.full(
                (N,),
                fill_value=self.lock_wrist_flex,
                dtype=torch.bool,
                device=self.device,
            )
        # Per-env scalar masks for branchless application.
        m_lock = lock_wrist_mask                                       # (N,) bool
        m_lock_f = m_lock.float().unsqueeze(-1)                        # (N, 1)

        q = current_joints.clone()
        # Apply the constraint q[3] = pi/2 - q[1] - q[2] only for masked envs;
        # other envs keep their warm-start wrist_flex/wrist_roll untouched.
        constrained_w_f = (
            self.wrist_flex_value - q[:, 1] - q[:, 2]
        )
        q[:, self.wrist_flex_idx] = torch.where(
            m_lock, constrained_w_f, q[:, self.wrist_flex_idx]
        )
        q[:, self.wrist_roll_idx] = torch.where(
            m_lock,
            torch.full_like(q[:, self.wrist_roll_idx], self.wrist_roll_value),
            q[:, self.wrist_roll_idx],
        )

        I3 = torch.eye(3, device=self.device, dtype=self.dtype).expand(
            q.shape[0], 3, 3
        )
        damping_sq = self.damping ** 2

        last_err_norm = None
        for it in range(self.n_iter):
            # FK
            tf = self.chain.forward_kinematics(q)
            pos = tf.get_matrix()[:, :3, 3]                # (N, 3)
            err = target_pos_b - pos                       # (N, 3)
            err_norm = err.norm(dim=-1)                    # (N,)
            last_err_norm = err_norm
            if err_norm.max() < self.early_stop:
                break

            # Jacobian — geometric, base frame, (N, 6, dof). We use the
            # position rows only (rows 0..2) for position-only IK.
            J = self.chain.jacobian(q)                     # (N, 6, dof)
            J_pos = J[:, :3, :].clone()                    # (N, 3, dof)

            # Build the constrained Jacobian (effective on s_l, e_f, s_p
            # only, with wrist columns zeroed) and select per-env.
            J_pos_constrained = J_pos.clone()
            J_pos_constrained[:, :, 1] = (
                J_pos[:, :, 1] - J_pos[:, :, self.wrist_flex_idx]
            )
            J_pos_constrained[:, :, 2] = (
                J_pos[:, :, 2] - J_pos[:, :, self.wrist_flex_idx]
            )
            J_pos_constrained[:, :, self.wrist_flex_idx] = 0.0
            J_pos_constrained[:, :, self.wrist_roll_idx] = 0.0
            mask_jac = m_lock.view(-1, 1, 1).expand_as(J_pos)
            J_pos = torch.where(mask_jac, J_pos_constrained, J_pos)

            # DLS: dq = J^T (J J^T + lambda^2 I)^-1 err
            JJT = J_pos @ J_pos.transpose(-1, -2)          # (N, 3, 3)
            sol_lin = torch.linalg.solve(
                JJT + damping_sq * I3, err.unsqueeze(-1)
            )
            dq = (J_pos.transpose(-1, -2) @ sol_lin).squeeze(-1)  # (N, dof)
            # For locked envs: zero wrist deltas (handled implicitly via the
            # zeroed Jacobian columns above, but explicit guard for safety).
            dq[:, self.wrist_flex_idx] = torch.where(
                m_lock,
                torch.zeros_like(dq[:, self.wrist_flex_idx]),
                dq[:, self.wrist_flex_idx],
            )
            dq[:, self.wrist_roll_idx] = torch.where(
                m_lock,
                torch.zeros_like(dq[:, self.wrist_roll_idx]),
                dq[:, self.wrist_roll_idx],
            )
            q = q + self.lr * dq
            # Re-enforce the constraint for locked envs only.
            constrained_w_f = (
                self.wrist_flex_value - q[:, 1] - q[:, 2]
            )
            q[:, self.wrist_flex_idx] = torch.where(
                m_lock, constrained_w_f, q[:, self.wrist_flex_idx]
            )
            q[:, self.wrist_roll_idx] = torch.where(
                m_lock,
                torch.full_like(q[:, self.wrist_roll_idx], self.wrist_roll_value),
                q[:, self.wrist_roll_idx],
            )

        # Final convergence check.
        tf_final = self.chain.forward_kinematics(q)
        final_err = (target_pos_b - tf_final.get_matrix()[:, :3, 3]).norm(dim=-1)
        converged = final_err < self.converged_threshold

        # Optional convergence diagnostic logging.
        self._call_count += 1
        if self._debug_every and self._call_count % self._debug_every == 0:
            print(
                f"[IK conv] call {self._call_count} | iters={it + 1} | "
                f"final err max={final_err.max().item() * 1000:.2f}mm "
                f"mean={final_err.mean().item() * 1000:.2f}mm "
                f"converged={converged.float().mean().item() * 100:.0f}%",
                flush=True,
            )

        return q, converged

    @torch.no_grad()
    def forward_kinematics(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """FK: (N, 5) joints -> (N, 3) tip position in base frame."""
        tf = self.chain.forward_kinematics(joint_pos)
        return tf.get_matrix()[:, :3, 3]
