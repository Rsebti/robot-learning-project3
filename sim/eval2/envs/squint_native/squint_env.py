"""ManagerBasedRLEnv subclass that tracks the wrist cam to the gripper body
each substep — replicates Squint's
``self.wrist_camera_mount.set_pose(gripper_pose * local_offset)``.

Isaac's CameraCfg.OffsetCfg inheritance gives the wrong world pose under
our converted Squint USD, so the cam offset stays zero in CameraCfg and
``step()`` writes the correct world pose each substep via
``cam.set_world_poses(convention='world')``.
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
        """Set wrist cam world pose = gripper world pose ⊗ local offset."""
        if self._cam_sensor_name not in self.scene.sensors:
            return
        cam = self.scene.sensors[self._cam_sensor_name]

        robot = self.scene["robot"]
        gi = self._resolve_gripper_idx()
        g_pos = robot.data.body_pos_w[:, gi]    # (N, 3)
        g_quat = robot.data.body_quat_w[:, gi]  # (N, 4)

        # Guard against zero-norm / NaN quats — scipy's Rotation.from_quat
        # (used downstream by set_world_poses) raises on any quaternion that
        # has zero norm or is non-finite.
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=g_quat.device, dtype=g_quat.dtype)
        # Always renormalise the gripper quat — body_quat_w isn't guaranteed
        # to be exactly unit norm on the first frame after a multi-env reset.
        g_norm = g_quat.norm(dim=-1, keepdim=True)
        g_pos_finite = torch.isfinite(g_pos).all(dim=-1, keepdim=True)
        bad = (~torch.isfinite(g_norm)) | (g_norm < 1e-4) | (~g_pos_finite)
        g_quat = torch.where(bad, identity_quat.expand_as(g_quat), g_quat / g_norm.clamp_min(1e-9))

        local_pos = self._cam_local_pos.unsqueeze(0).expand(g_pos.shape[0], -1)
        cam_pos_w = g_pos + _quat_rotate(g_quat, local_pos)
        local_quat = self._cam_local_quat.unsqueeze(0).expand(g_quat.shape[0], -1)
        cam_quat_w = _quat_mul(g_quat, local_quat)
        cam_quat_w = cam_quat_w / cam_quat_w.norm(dim=-1, keepdim=True).clamp_min(1e-9)

        # Sanity-clamp positions to a sane workspace box — Isaac's transform
        # math can return NaN if positions go to inf during a degenerate
        # multi-env reset frame.
        cam_pos_w = torch.where(
            torch.isfinite(cam_pos_w), cam_pos_w, torch.zeros_like(cam_pos_w)
        )

        try:
            cam.set_world_poses(positions=cam_pos_w, orientations=cam_quat_w, convention="world")
        except Exception:
            # Drop a frame rather than crash the training loop — the cam
            # stays at its previous pose; cam.data.pos_w refreshes next step.
            pass

        if force_render:
            try:
                cam._is_outdated.fill_(True)
                cam._update_outdated_buffers()
            except Exception:
                pass

    # -- overrides -------------------------------------------------------
    def step(self, action):
        """ManagerBasedRLEnv.step() with the wrist cam re-synced to the
        gripper AFTER every sim substep (otherwise the finger appears to
        jiggle in the rendered view because the cam stays at the start-of-
        decimation pose while the gripper drifts over the 10 substeps).
        """
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            self._update_wrist_cam_pose(force_render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # Capture the PRE-RESET obs for envs that ended so the SAC replay
            # buffer can bootstrap from V(true_final_state) instead of
            # V(next_episode_initial_state). Mirrors ManiSkill's
            # ``infos["final_observation"]`` convention.
            pre_reset_obs = self.observation_manager.compute()
            final_obs: dict = {}
            if isinstance(pre_reset_obs, dict):
                for k, v in pre_reset_obs.items():
                    if isinstance(v, torch.Tensor):
                        final_obs[k] = v.detach().clone()
                    elif isinstance(v, dict):
                        final_obs[k] = {
                            kk: vv.detach().clone() for kk, vv in v.items()
                            if isinstance(vv, torch.Tensor)
                        }
            self.extras["final_observation"] = final_obs
            self.extras["final_observation_env_ids"] = reset_env_ids.detach().clone()

            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)
        else:
            # No reset this step — clear stale final_observation so the
            # training loop doesn't reuse it.
            self.extras.pop("final_observation", None)
            self.extras.pop("final_observation_env_ids", None)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # Final cam pose update before the obs render (covers post-reset too).
        self._update_wrist_cam_pose(force_render=False)

        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def reset(self, *args, **kwargs):
        out = super().reset(*args, **kwargs)
        self._update_wrist_cam_pose(force_render=False)
        return out
