"""ManagerBasedRLEnv subclass that explicitly tracks the wrist cam to the
gripper body each step — replicates Squint's
``self.wrist_camera_mount.set_pose(gripper_pose * local_offset)``.

We don't trust Isaac's CameraCfg.OffsetCfg inheritance because in our
converted Squint USD, the resulting cam world pose does not match
``gripper_pos + R_gripper · local_offset`` (verified with a side-by-side
audit vs ManiSkill — Isaac places the cam ~12 cm too far in x/z).

So we set the offset to zero in CameraCfg and override ``step()`` to write
the correct world pose explicitly using ``set_world_poses(convention='world')``.
"""
from __future__ import annotations

import math

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .squint_scene import WRIST_CAM_POS, WRIST_CAM_QUAT


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two batched quaternions (..., 4) wxyz."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector ``v`` (..., 3) by quaternion ``q`` (..., 4) wxyz."""
    qw, qx, qy, qz = q.unbind(-1)
    vx, vy, vz = v.unbind(-1)
    rx = (1 - 2 * (qy * qy + qz * qz)) * vx + 2 * (qx * qy - qw * qz) * vy + 2 * (qx * qz + qw * qy) * vz
    ry = 2 * (qx * qy + qw * qz) * vx + (1 - 2 * (qx * qx + qz * qz)) * vy + 2 * (qy * qz - qw * qx) * vz
    rz = 2 * (qx * qz - qw * qy) * vx + 2 * (qy * qz + qw * qx) * vy + (1 - 2 * (qx * qx + qy * qy)) * vz
    return torch.stack([rx, ry, rz], dim=-1)


class SquintNativePlaceEnv(ManagerBasedRLEnv):
    """Same as ManagerBasedRLEnv but updates the wrist cam pose to follow
    the gripper body explicitly at each control step (after the physics
    sub-steps and before the cam is rendered)."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        # Cache cam offset (constant) on the device we're using.
        device = self.device
        self._cam_local_pos = torch.tensor(WRIST_CAM_POS, device=device, dtype=torch.float32)
        self._cam_local_quat = torch.tensor(WRIST_CAM_QUAT, device=device, dtype=torch.float32)
        # Track gripper body index lazily (resolved on first call).
        self._gripper_body_idx: int | None = None
        self._cam_sensor_name: str = "wrist"

    # -- helpers ---------------------------------------------------------
    def _resolve_gripper_idx(self) -> int:
        if self._gripper_body_idx is None:
            robot = self.scene["robot"]
            if "gripper" not in robot.body_names:
                raise RuntimeError(
                    f"Could not find 'gripper' body in robot.body_names = {robot.body_names}"
                )
            self._gripper_body_idx = robot.body_names.index("gripper")
        return self._gripper_body_idx

    def _update_wrist_cam_pose(self, force_render: bool = False) -> None:
        """Set the wrist camera world pose = gripper world pose ⊗ local offset.

        If ``force_render=True``, also force the cam to re-render so cam.data
        (pos_w, quat_w_world, output['rgb']) reflects the new pose immediately.
        Otherwise the buffers are stale until the next physics-loop scene.update.
        """
        if self._cam_sensor_name not in self.scene.sensors:
            return
        cam = self.scene.sensors[self._cam_sensor_name]

        robot = self.scene["robot"]
        gi = self._resolve_gripper_idx()
        g_pos = robot.data.body_pos_w[:, gi]    # (N, 3)
        g_quat = robot.data.body_quat_w[:, gi]  # (N, 4)

        local_pos = self._cam_local_pos.unsqueeze(0).expand(g_pos.shape[0], -1)
        cam_pos_w = g_pos + _quat_rotate(g_quat, local_pos)
        local_quat = self._cam_local_quat.unsqueeze(0).expand(g_quat.shape[0], -1)
        cam_quat_w = _quat_mul(g_quat, local_quat)

        # set_world_poses with convention="world" matches Squint's SAPIEN cam
        cam.set_world_poses(positions=cam_pos_w, orientations=cam_quat_w, convention="world")

        if force_render:
            # Mark all cam env-ids as outdated so the next cam.data read
            # triggers _update_buffers_impl (which calls _update_poses + render).
            try:
                cam._is_outdated.fill_(True)
                cam._update_outdated_buffers()
            except Exception:
                pass

    # -- overrides -------------------------------------------------------
    def step(self, action):
        # Update cam pose BEFORE the physics loop so the cam render that
        # happens INSIDE super().step() captures a view from the up-to-date
        # gripper pose. The cam pose comes from the gripper's body_pos_w,
        # which was last refreshed at the previous step's end of physics —
        # so this is "last-step gripper" accurate (1 control-step lag),
        # which is invisible at 10 Hz with small inter-step motions.
        self._update_wrist_cam_pose(force_render=False)
        return super().step(action)

    def reset(self, *args, **kwargs):
        out = super().reset(*args, **kwargs)
        # First-frame cam alignment so the very next env.step renders from
        # the correct pose. The obs returned here is the post-reset obs;
        # downstream code typically discards it and runs warmup steps.
        self._update_wrist_cam_pose(force_render=False)
        return out
