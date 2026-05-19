"""
calibrate_extrinsics_from_probe.py — fit T_cam_in_wrist from probe_camera_vs_fk sessions.

No chessboard: each trial is one calibration sample —
  - at home: red pixel (u,v) + joint angles (arm at eval1_rest)
  - at grasp: FK tip position = cube location (you centered grasp on cube)

Assumes the cube did NOT move between home view and grasp.

Opt-in only — overwrites toolset/configs/kinematics/camera_in_wrist.npy.
For dense mapping without refitting extrinsics, use probe_camera_vs_fk --map_only
and probe_space_map.py instead.

Usage:
  1) Collect data (10+ trials, different cube spots):
       python -m toolset.perception.probe_camera_vs_fk `
           --port COM3 --camera_index 0 --color red --n_trials 12 --save_frames --map_only

  2) Fit extrinsics (keeps mate's K from camera_intrinsics.npz):
       python -m toolset.calibration_scripts.calibrate_extrinsics_from_probe `
           --session_dir deploy/_snaps/probe_XXXXX

  3) Check error:
       python -m toolset.perception.probe_session_estimate `
           --session_dir deploy/_snaps/probe_XXXXX --loosen_hsv

Writes:
  toolset/configs/kinematics/camera_in_wrist.npy
  toolset/configs/kinematics/camera_in_wrist.yaml
  <session_dir>/extrinsics_fit_report.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rscipy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolset.kinematics.config import KinematicsConfig
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig
from toolset.kinematics.urdf_fk import SO101FK
from toolset.perception.cube_localization import (
    KIN_CONFIG_DIR,
    _load_intrinsics,
    _scale_K_to_stream,
    load_intrinsics_image_size,
    WRIST_CAM_HEIGHT,
    WRIST_CAM_WIDTH,
)
def user_xyz_to_urdf(xyz_user: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(xyz_user, dtype=float).reshape(3)
    return np.array([y, -x, z], dtype=float)


def T_from_params(tx: float, ty: float, tz: float, rx: float, ry: float, rz: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rscipy.from_rotvec([rx, ry, rz]).as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T


def back_project_urdf(
    u: float,
    v: float,
    K: np.ndarray,
    T_cam_in_wrist: np.ndarray,
    q_rad: np.ndarray,
    fk: SO101FK,
    z_table: float,
) -> np.ndarray | None:
    import cv2

    pt = np.array([[[u, v]]], dtype=np.float32)
    dist = np.zeros(5)
    pt_u = cv2.undistortPoints(pt, K, dist, P=K)
    px_u, py_u = float(pt_u[0, 0, 0]), float(pt_u[0, 0, 1])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ray_cam = np.array([(px_u - cx) / fx, (py_u - cy) / fy, 1.0])
    ray_cam /= np.linalg.norm(ray_cam) + 1e-12
    T_wrist = fk.fk(q_rad, target="wrist")["T"]
    T_cam = T_wrist @ T_cam_in_wrist
    origin = T_cam[:3, 3]
    ray = T_cam[:3, :3] @ ray_cam
    if abs(ray[2]) < 1e-8:
        return None
    t = (z_table - origin[2]) / ray[2]
    if t < 0:
        return None
    return origin + t * ray


def collect_samples(session_dir: Path, mcfg: MotorToUrdfConfig, fk: SO101FK, kcfg: KinematicsConfig):
    meta = json.loads((session_dir / "session.json").read_text())
    z_table = kcfg.table_z_m + kcfg.cube_half_height_m
    samples = []

    for t in meta.get("trials", []):
        tri = int(t["trial"])
        trial_dir = session_dir / f"trial_{tri:03d}"
        frames_path = trial_dir / "home" / "frames_home.json"
        if not frames_path.is_file():
            frames_path = trial_dir / "home" / "frames.json"
        if not frames_path.is_file():
            continue

        frames = json.loads(frames_path.read_text())
        pixels = [fr["pixel"] for fr in frames if fr.get("pixel") is not None]
        home_deg_list = [fr["motor_deg"] for fr in frames if fr.get("motor_deg") is not None]
        if not pixels or not home_deg_list:
            continue

        fk_user = t.get("fk_xyz_user_m")
        if not fk_user or not np.all(np.isfinite(fk_user)):
            continue

        med_px = np.median(np.asarray(pixels, dtype=float), axis=0)
        home_deg = np.median(np.asarray(home_deg_list, dtype=float), axis=0)
        q_rad = mcfg.motor_to_urdf_rad(home_deg)
        gt_urdf = user_xyz_to_urdf(np.asarray(fk_user, dtype=float))

        samples.append({
            "trial": tri,
            "u": float(med_px[0]),
            "v": float(med_px[1]),
            "q_rad": q_rad,
            "gt_urdf": gt_urdf,
            "gt_user": np.asarray(fk_user, dtype=float),
            "z_table": z_table,
        })
        print(
            f"  trial {tri:03d}: pixel=({med_px[0]:.0f},{med_px[1]:.0f})  "
            f"fk_user=({fk_user[0]:+.3f},{fk_user[1]:+.3f},{fk_user[2]:+.3f})",
        )

    return samples, meta


def residuals_T(p: np.ndarray, samples: list[dict], K: np.ndarray, fk: SO101FK) -> np.ndarray:
    T = T_from_params(*p)
    res = []
    fail = 0.5
    for s in samples:
        bp = back_project_urdf(s["u"], s["v"], K, T, s["q_rad"], fk, s["z_table"])
        if bp is None:
            res.extend([fail, fail])
            continue
        res.append(bp[0] - s["gt_urdf"][0])
        res.append(bp[1] - s["gt_urdf"][1])
    return np.array(res)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session_dir", type=Path, required=True)
    p.add_argument("--min_samples", type=int, default=5)
    args = p.parse_args()

    session_dir = args.session_dir.resolve()
    kcfg = KinematicsConfig.load()
    mcfg = MotorToUrdfConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)

    intr = _load_intrinsics()
    if intr is None:
        raise SystemExit("Need camera_intrinsics.npz (mate's K) first.")
    K, _ = intr
    native = load_intrinsics_image_size()
    if native:
        K = _scale_K_to_stream(K, native)

    print(f"[fit] session {session_dir}")
    print(f"[fit] K from npz (stream {WRIST_CAM_WIDTH}x{WRIST_CAM_HEIGHT})")
    samples, meta = collect_samples(session_dir, mcfg, fk, kcfg)
    if len(samples) < args.min_samples:
        raise SystemExit(f"Need >= {args.min_samples} trials with home pixel + grasp FK; got {len(samples)}")

    # Seed from legacy yaml if present
    legacy = PROJECT_ROOT / "toolset" / "configs" / "camera_calibration.yaml"
    if legacy.is_file():
        d = yaml.safe_load(legacy.read_text()) or {}
        T0 = np.array(d["T_cam_in_wrist"], dtype=float)
        rv = Rscipy.from_matrix(T0[:3, :3]).as_rotvec()
        p0 = np.concatenate([T0[:3, 3], rv])
    else:
        p0 = np.array([0.02, 0.0, 0.03, -1.2, 1.2, -1.2])

    print(f"[fit] optimizing T_cam_in_wrist over {len(samples)} samples ...")
    result = least_squares(
        residuals_T, p0, args=(samples, K, fk),
        max_nfev=2000, verbose=2,
    )
    T = T_from_params(*result.x)
    errs = residuals_T(result.x, samples, K, fk).reshape(-1, 2)
    valid = errs[np.isfinite(errs).all(axis=1)]
    xy_mm = np.linalg.norm(valid, axis=1) * 1000 if len(valid) else np.array([np.nan])

    out_npy = KIN_CONFIG_DIR / "camera_in_wrist.npy"
    out_yaml = KIN_CONFIG_DIR / "camera_in_wrist.yaml"
    np.save(out_npy, T)
    with open(out_yaml, "w") as f:
        yaml.safe_dump({
            "T_cam_in_wrist": T.tolist(),
            "calibrated_from": {
                "session_dir": str(session_dir),
                "n_samples": len(samples),
                "mean_xy_err_mm": float(np.mean(xy_mm)) if len(xy_mm) else None,
                "color": meta.get("color"),
                "camera_index": meta.get("camera_index"),
            },
        }, f, sort_keys=False)

    report_lines = [
        "extrinsics fit from probe session",
        f"session: {session_dir}",
        f"samples: {len(samples)}",
        f"mean xy error mm: {np.mean(xy_mm):.2f}" if len(xy_mm) else "n/a",
        f"wrote: {out_npy}",
        "",
        "Optional: probe_session_estimate --session_dir ... (do not run unless refitting extrinsics)",
    ]
    report = session_dir / "extrinsics_fit_report.txt"
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"\n[fit] mean xy error {np.mean(xy_mm):.2f} mm")
    print(f"[fit] saved {out_npy}")
    print(f"[fit] report {report}")


if __name__ == "__main__":
    main()
