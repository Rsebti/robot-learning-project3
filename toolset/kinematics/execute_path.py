"""
execute_path.py - drive the SO-101 through a list of (x, y, z) waypoints
using IK + joint-space interpolation.

Speed presets (from config.yaml):
    slow   - 10 Hz, 1.0 deg max step
    normal - 30 Hz, 2.5 deg max step
    fast   - 30 Hz, 6.0 deg max step (use only in free space)

Each waypoint is either:
    {"xyz": [x, y, z], "frame": "user" | "urdf",
     "open_gripper": bool, "close_gripper": bool,
     "settle_s": float, "hold_s": float}

Frame mappings (deploy-facing user frame -> URDF base frame):
    x_user = -y_urdf   ->  y_urdf = -x_user
    y_user = +x_urdf   ->  x_urdf = +y_user
    z_user =  z_urdf
i.e. user (right+, forward+, up+) <-> URDF (forward+x, left+y, up+z).
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from toolset.kinematics.config import KinematicsConfig
from toolset.kinematics.ik import IKResult, IKSolver
from toolset.kinematics.motor_to_urdf import (
    MotorToUrdfConfig, read_motor_deg_from_obs,
)
from toolset.kinematics.urdf_fk import CHAIN_JOINTS, SO101FK


@dataclass
class Waypoint:
    xyz: np.ndarray          # 3-vec in `frame`
    frame: str = "user"      # "user" or "urdf"
    open_gripper: bool = False
    close_gripper: bool = False
    settle_s: float = 0.0
    hold_s: float = 0.0
    mode: str | None = None  # per-waypoint override; falls back to executor.mode


def user_to_urdf_xyz(xyz_user: np.ndarray) -> np.ndarray:
    return np.array([+xyz_user[1], -xyz_user[0], xyz_user[2]], dtype=float)


class PathExecutor:
    def __init__(
        self,
        kcfg: KinematicsConfig | None = None,
        mcfg: MotorToUrdfConfig | None = None,
        gripper_open_deg: float = 100.0,
        gripper_close_deg: float = 0.0,
    ):
        self.kcfg = kcfg or KinematicsConfig.load()
        self.mcfg = mcfg or MotorToUrdfConfig.load()
        self.fk = SO101FK(
            self.kcfg.urdf_path, gripper_tip_offset=self.kcfg.gripper_tip_offset_m,
        )
        self.ik = IKSolver(self.kcfg, self.fk)
        self.gripper_open_deg = float(gripper_open_deg)
        self.gripper_close_deg = float(gripper_close_deg)

    def _resolve_waypoint(self, wp: Waypoint) -> np.ndarray:
        if wp.frame == "user":
            return user_to_urdf_xyz(np.asarray(wp.xyz, dtype=float))
        if wp.frame == "urdf":
            return np.asarray(wp.xyz, dtype=float)
        raise ValueError(f"unknown frame: {wp.frame!r}")

    def _send_joint_command(self, robot, q_urdf_rad: np.ndarray, gripper_deg: float):
        motor_deg = self.mcfg.urdf_rad_to_motor_deg(q_urdf_rad)
        action = {f"{n}.pos": float(motor_deg[i]) for i, n in enumerate(CHAIN_JOINTS)}
        action["gripper.pos"] = float(gripper_deg)
        robot.send_action(action)

    def _interpolate_joint(
        self,
        q_from_rad: np.ndarray,
        q_to_rad: np.ndarray,
        max_step_rad: float,
    ) -> np.ndarray:
        delta = q_to_rad - q_from_rad
        max_abs = float(np.max(np.abs(delta)))
        n = max(1, int(np.ceil(max_abs / max_step_rad)))
        return np.linspace(q_from_rad, q_to_rad, n + 1)[1:]

    def execute(
        self,
        robot,
        waypoints: list[Waypoint],
        speed: str = "normal",
        target: str = "gripper_tip",
        mode: str = "position_vertical",
        dry_run: bool = False,
    ) -> bool:
        """Run the path. Returns True if completed without IK failure."""
        speed_cfg = self.kcfg.path["speeds"][speed]
        fps = int(speed_cfg["fps"])
        max_step_rad = float(np.deg2rad(speed_cfg["max_delta_deg_per_step"]))
        period = 1.0 / fps

        obs = robot.get_observation() if not dry_run else None
        if obs is not None:
            motor_now = read_motor_deg_from_obs(obs)
            q_urdf = self.mcfg.motor_to_urdf_rad(motor_now)
            gripper_now = float(motor_now[5])
        else:
            q_urdf = np.zeros(5)
            gripper_now = self.gripper_open_deg

        for i, wp in enumerate(waypoints):
            tgt_urdf = self._resolve_waypoint(wp)
            if not self.kcfg.in_workspace(tgt_urdf):
                print(f"[path] waypoint {i} {tgt_urdf.round(3).tolist()} OUT OF WORKSPACE; abort.")
                return False
            wp_mode = wp.mode or mode
            ik_res: IKResult = self.ik.solve(
                tgt_urdf, q_init=q_urdf, target=target, mode=wp_mode,
            )
            if not ik_res.converged:
                print(f"[path] waypoint {i}: IK reason={ik_res.reason}  "
                      f"pos_err={ik_res.pos_err_m*1000:.2f}mm; abort.")
                return False

            traj = self._interpolate_joint(q_urdf, ik_res.joints_rad, max_step_rad)
            grip_target = (
                self.gripper_close_deg if wp.close_gripper else
                self.gripper_open_deg if wp.open_gripper else gripper_now
            )
            print(f"[path] wp {i}: tgt_urdf={tgt_urdf.round(3).tolist()}  "
                  f"steps={len(traj)}  gripper->{grip_target:.0f}")
            if dry_run:
                q_urdf = ik_res.joints_rad
                gripper_now = grip_target
                continue
            for q_step in traj:
                t0 = time.time()
                self._send_joint_command(robot, q_step, grip_target)
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
            q_urdf = ik_res.joints_rad
            gripper_now = grip_target

            if wp.settle_s > 0:
                time.sleep(wp.settle_s)
            if wp.hold_s > 0:
                # keep sending the same command to hold against drift
                t_end = time.time() + wp.hold_s
                while time.time() < t_end:
                    self._send_joint_command(robot, q_urdf, gripper_now)
                    time.sleep(period)
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_waypoints_yaml(path: Path) -> list[Waypoint]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    wps = []
    for entry in data.get("waypoints", []):
        wps.append(Waypoint(
            xyz=np.asarray(entry["xyz"], dtype=float),
            frame=entry.get("frame", "user"),
            open_gripper=bool(entry.get("open_gripper", False)),
            close_gripper=bool(entry.get("close_gripper", False)),
            settle_s=float(entry.get("settle_s", 0.0)),
            hold_s=float(entry.get("hold_s", 0.0)),
            mode=entry.get("mode"),
        ))
    return wps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--camera_index", type=int, default=1)
    parser.add_argument("--waypoints_yaml", type=Path, required=True,
                        help="YAML with a 'waypoints' list (xyz, frame, gripper, ...).")
    parser.add_argument("--speed", choices=["slow", "normal", "fast"], default="normal")
    parser.add_argument("--target", default="gripper_tip",
                        choices=["wrist", "wrist_roll", "gripper_frame", "gripper_tip"])
    parser.add_argument("--mode", default="position_vertical",
                        choices=["position", "position_vertical"])
    parser.add_argument("--dry_run", action="store_true",
                        help="Don't drive the robot; just print the IK trajectory.")
    args = parser.parse_args()

    wps = _load_waypoints_yaml(args.waypoints_yaml)
    print(f"[path] loaded {len(wps)} waypoint(s) from {args.waypoints_yaml}")

    executor = PathExecutor()

    if args.dry_run:
        print("[path] DRY RUN (no robot).")
        executor.execute(None, wps, speed=args.speed, target=args.target, mode=args.mode, dry_run=True)
        return

    from lerobot.robots.utils import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    rconf = SO101FollowerConfig(
        port=args.port, id="so101_follower",
        cameras={"wrist": OpenCVCameraConfig(
            index_or_path=args.camera_index, fps=30, width=640, height=480,
        )},
    )
    robot = make_robot_from_config(rconf)
    print(f"[path] Connecting to {args.port} ...")
    robot.connect()
    try:
        ok = executor.execute(robot, wps, speed=args.speed, target=args.target, mode=args.mode)
        print(f"[path] {'COMPLETED' if ok else 'ABORTED'}.")
    finally:
        robot.disconnect()
        print("[path] Disconnected.")


if __name__ == "__main__":
    main()
