"""
scout_cube.py - wrist-camera cube position at the scout/home pose.

Used by probe sessions and deploy/run_probe_episodes.py before each policy rollout.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from toolset.kinematics.config import KinematicsConfig
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig
from toolset.kinematics.urdf_fk import SO101FK
from toolset.perception.cube_localization import CubeLocalizer
from toolset.perception.estimate_cube_xy import load_hsv_ranges, loosen_hsv, urdf_xyz_to_user
from toolset.perception.probe_camera_vs_fk import extract_wrist_bgr

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def scout_bowl_xyz_user(
    robot,
    localizer: CubeLocalizer,
    *,
    color: str,
    n_frames: int = 10,
    obs_hz: float = 5.0,
    camera_key: str | None = None,
) -> tuple[np.ndarray | None, list[dict]]:
    """Median cube xyz in user frame (right+, forward+, up+) from home pose."""
    period = 1.0 / max(0.1, obs_hz)
    xyzs_urdf: list[np.ndarray] = []
    log: list[dict] = []

    for fi in range(n_frames):
        t0 = time.time()
        obs = robot.get_observation()
        motor_deg = np.array([float(obs[f"{n}.pos"]) for n in MOTOR_NAMES], dtype=float)
        bgr, key_used = extract_wrist_bgr(
            obs, camera_key=camera_key, camera_index=None,
        )
        rec: dict = {"frame": fi, "image_key": key_used, "motor_deg": motor_deg.tolist()}

        if bgr is None:
            rec["status"] = "no_image"
        else:
            det = localizer.locate(bgr, motor_deg, color)
            if det is None:
                px, _, area = localizer.detect_pixel(bgr, color)
                rec["status"] = "no_pixel" if px is None else "no_backproj"
                rec["area_px"] = float(area)
            else:
                rec["status"] = "ok"
                rec["pixel"] = list(det.pixel)
                rec["base_xyz_urdf"] = det.base_xyz_m.tolist()
                xyzs_urdf.append(det.base_xyz_m.copy())

        log.append(rec)
        if fi == 0:
            print(f"  [scout 0] key={key_used!r} status={rec.get('status')}", flush=True)
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)

    if not xyzs_urdf:
        return None, log
    med = np.median(np.stack(xyzs_urdf), axis=0)
    return urdf_xyz_to_user(med), log


def write_scout_result(
    path: Path,
    *,
    bowl_xyz_user: np.ndarray | None,
    color: str,
    frame_log: list[dict],
    home_pose: str,
    episode: int | None = None,
) -> None:
    payload = {
        "color": color,
        "home_pose": home_pose,
        "episode": episode,
        "bowl_xyz_user_m": None if bowl_xyz_user is None else bowl_xyz_user.tolist(),
        "bowl_xyz_policy": None,
        "frame_log": frame_log,
        "timestamp": time.time(),
    }
    if bowl_xyz_user is not None:
        # Policy bowl_xyz is user-frame (x right, y forward, z up) — same as deploy defaults.
        payload["bowl_xyz_policy"] = [float(v) for v in bowl_xyz_user]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def make_localizer(loosen_color: str | None = None) -> CubeLocalizer:
    kcfg = KinematicsConfig.load()
    mcfg = MotorToUrdfConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)
    hsv = load_hsv_ranges(Path(__file__).parent / "hsv_config.yaml")
    if loosen_color:
        hsv = loosen_hsv(hsv, loosen_color)
    return CubeLocalizer(kcfg=kcfg, mcfg=mcfg, fk=fk, hsv_ranges=hsv)
