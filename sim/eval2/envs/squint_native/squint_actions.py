"""Custom delta-target joint position action — replicates Squint's
``pd_joint_target_delta_pos`` controller.

Semantics (Hamilton, ManiSkill convention):
    target_qpos[t+1] = target_qpos[t] + clamp(action × scale, -bound, +bound)
    target_qpos      = clamp(target_qpos, joint_limits)
    PD command       = target_qpos

where ``action ∈ [-1, +1]`` (normalized) and ``scale = bound`` per joint.

Squint's so101 bounds:
    arm joints:    ±0.1 rad / step
    gripper joint: ±0.2 rad / step

Difference vs. ``RelativeJointPositionAction`` (Isaac Lab built-in):
    Isaac's relative action adds the delta to the CURRENT qpos every step.
    Squint's target accumulates: drift in the controller (PD lag, gravity)
    does NOT cause the target to drift. This is critical for trained
    policies that learned to issue small deltas around an integrated target.
"""
from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ---------------------------------------------------------------------------
# Action term
# ---------------------------------------------------------------------------


class DeltaTargetJointPositionAction(ActionTerm):
    """Action term that integrates a normalized delta into a stored joint target.

    Each call:
        1. ``self._target += clamp(action × scale_per_joint, ±bound)``
        2. Clamp ``self._target`` to soft joint limits
        3. Set the articulation's joint position target to ``self._target``

    On reset, the stored target is reinitialized to the current joint pos so
    the policy's first delta applies relative to the home pose.
    """

    cfg: "DeltaTargetJointPositionActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "DeltaTargetJointPositionActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        # Resolve joint indices / names in the order requested.
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=True
        )
        self._num_joints = len(self._joint_ids)

        if self._num_joints == 0:
            raise ValueError(
                f"DeltaTargetJointPositionAction: no joints matched "
                f"{self.cfg.joint_names!r}"
            )

        # Per-joint delta bound (== scale for normalize_action=True).
        bounds = self.cfg.bounds
        if isinstance(bounds, (int, float)):
            scale = torch.full(
                (self.num_envs, self._num_joints), float(bounds), device=self.device
            )
        else:
            if len(bounds) != self._num_joints:
                raise ValueError(
                    f"bounds length {len(bounds)} != num_joints {self._num_joints}"
                )
            scale = torch.tensor(
                list(bounds), dtype=torch.float32, device=self.device
            ).unsqueeze(0).expand(self.num_envs, -1).clone()
        self._scale = scale  # (N, J)

        # Action storage
        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        # Integrated target — initialised lazily on first reset.
        self._target = torch.zeros_like(self._raw_actions)
        self._target_initialized = False

        # Soft joint limits to clamp the integrated target.
        self._joint_lo = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0]
        self._joint_hi = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1]

    # -- Properties -------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def target_qpos(self) -> torch.Tensor:
        """The integrated joint-position target the controller is tracking.

        Used by the observation manager to expose Squint's
        ``controller.get_state()`` (12-d state vector = qpos | target_qpos).
        """
        return self._target

    # -- Operations -------------------------------------------------------
    def _refresh_target_from_qpos(self, env_ids: Sequence[int] | None = None) -> None:
        """Snap the stored target to the robot's current qpos.

        We read ``joint_pos`` instead of ``default_joint_pos`` so that any
        reset-time noise the env applied to qpos is captured by the target
        too — preserves Squint's invariant that ``target == qpos`` right
        after reset (within numerical noise). The PD controller then keeps
        qpos at that target across the settle period.
        """
        # Force a fresh read of joint state from sim — needed because the
        # reset_robot_to_home event called write_joint_state_to_sim moments
        # before us, and Articulation buffers may still hold pre-reset data.
        self._asset.update(dt=0.0)
        qpos = self._asset.data.joint_pos[:, self._joint_ids]
        if env_ids is None:
            self._target[:] = qpos
        else:
            self._target[env_ids] = qpos[env_ids]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Re-anchor the target to the (just-reset) qpos.
        self._refresh_target_from_qpos(env_ids)
        # Zero the raw actions for the reset envs.
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
        self._target_initialized = True

    def process_actions(self, actions: torch.Tensor) -> None:
        # Lazy first-time init in case reset() never fires before the first step.
        if not self._target_initialized:
            self._refresh_target_from_qpos(None)
            self._target_initialized = True

        # Clamp normalized action to [-1, +1] then scale (per-joint bound).
        clipped = torch.clamp(actions, -1.0, 1.0)
        delta = clipped * self._scale  # (N, J), in radians

        self._raw_actions[:] = actions
        self._processed_actions[:] = delta

        # Integrate into the stored target, clamp to joint limits.
        self._target = torch.clamp(self._target + delta, self._joint_lo, self._joint_hi)

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._target, joint_ids=self._joint_ids)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@configclass
class DeltaTargetJointPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`DeltaTargetJointPositionAction`."""

    class_type: type = DeltaTargetJointPositionAction
    """The action term class."""

    joint_names: list[str] = MISSING
    """Names of the joints driven by this action term, in order."""

    bounds: float | list[float] = MISSING
    """Per-joint absolute delta bound in radians (== scale for normalized action).

    For Squint SO-101: ``[0.1, 0.1, 0.1, 0.1, 0.1, 0.2]`` (arm + gripper).
    """
