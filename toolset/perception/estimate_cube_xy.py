"""
estimate_cube_xy.py - cube position in robot base frame from one wrist image + joints.

Uses CubeLocalizer (HSV + K + T_cam_in_wrist + motor_to_urdf + FK).

Usage:
    # single saved frame + joint angles (servo degrees, 6-vector)
    python -m toolset.perception.estimate_cube_xy `
        --image deploy/_snaps/probe_XXX/trial_000/home/frame_00.png `
        --motor_deg -2.2 -80.8 36.7 86.9 -82.2 -14.7 `
        --color red

    # median over a probe trial folder (home frames + frames.json)
    python -m toolset.perception.estimate_cube_xy `
        --trial_dir deploy/_snaps/probe_XXX/trial_000 --phase home --color red
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolset.kinematics.config import KinematicsConfig
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig
from toolset.kinematics.urdf_fk import SO101FK
from toolset.perception.cube_localization import CubeLocalizer, DEFAULT_HSV


def urdf_xyz_to_user(xyz_urdf: np.ndarray) -> np.ndarray:
    return np.array([-xyz_urdf[1], xyz_urdf[0], xyz_urdf[2]], dtype=float)


def load_hsv_ranges(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {
        color: [(tuple(lo), tuple(hi)) for lo, hi in ranges]
        for color, ranges in data.items()
    }


def loosen_hsv(hsv: dict, color: str, s_delta: int = 40, v_delta: int = 40) -> dict:
    """Widen S/V lower bounds for one color (replay / tuning only)."""
    out = {k: list(v) for k, v in hsv.items()}
    if color not in out:
        return out
    widened = []
    for lo, hi in out[color]:
        lo = list(lo)
        lo[1] = max(0, lo[1] - s_delta)
        lo[2] = max(0, lo[2] - v_delta)
        widened.append((tuple(lo), tuple(hi)))
    out[color] = widened
    return out


def make_localizer(
    *,
    hsv_path: Path | None = None,
    motor_offsets_path: Path | None = None,
    loosen_color: str | None = None,
) -> CubeLocalizer:
    kcfg = KinematicsConfig.load()
    if motor_offsets_path and motor_offsets_path.is_file():
        mcfg = MotorToUrdfConfig.load(motor_offsets_path)
    else:
        mcfg = MotorToUrdfConfig.load()

    hsv = load_hsv_ranges(hsv_path) or DEFAULT_HSV
    if loosen_color:
        hsv = loosen_hsv(hsv, loosen_color)
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)
    return CubeLocalizer(kcfg=kcfg, mcfg=mcfg, fk=fk, hsv_ranges=hsv)


def estimate_trial_home(
    trial_dir: Path,
    localizer: CubeLocalizer,
    color: str,
) -> tuple[np.ndarray | None, list[dict]]:
    home = trial_dir / "home"
    meta_path = home / "frames.json"
    if not home.is_dir():
        return None, []
    if meta_path.is_file():
        frame_log = json.loads(meta_path.read_text())
    else:
        pngs = [
            p for p in sorted(home.glob("frame_*.png"))
            if "_mask" not in p.name and "_overlay" not in p.name
        ]
        frame_log = [{"frame": i, "motor_deg": None} for i in range(len(pngs))]

    xyzs: list[np.ndarray] = []
    details: list[dict] = []
    for rec in frame_log:
        fi = rec["frame"]
        img_path = home / f"frame_{fi:02d}.png"
        if not img_path.is_file():
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        motor = rec.get("motor_deg")
        if motor is None:
            details.append({"frame": fi, "status": "no_motor_deg"})
            continue
        motor_deg = np.asarray(motor, dtype=float)
        det = localizer.locate(bgr, motor_deg, color)
        if det is None:
            px, mask, area = localizer.detect_pixel(bgr, color)
            st = "no_pixel" if px is None else "no_backproj"
            details.append({"frame": fi, "status": st, "area": float(area)})
            continue
        xyzs.append(det.base_xyz_m)
        details.append({
            "frame": fi,
            "status": "ok",
            "pixel": list(det.pixel),
            "area": float(det.contour_area_px),
            "confidence": float(det.confidence),
            "xyz_user_m": urdf_xyz_to_user(det.base_xyz_m).tolist(),
        })

    if not xyzs:
        return None, details
    med = np.median(np.stack(xyzs), axis=0)
    return med, details


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, default=None)
    p.add_argument("--motor_deg", type=float, nargs=6, metavar="DEG")
    p.add_argument("--trial_dir", type=Path, default=None)
    p.add_argument("--phase", choices=["home", "grasp"], default="home")
    p.add_argument("--color", required=True)
    p.add_argument("--hsv_config", type=Path, default=None)
    p.add_argument("--motor_offsets", type=Path, default=None)
    p.add_argument("--loosen_hsv", action="store_true",
                   help="Widen S/V mins for --color (tuning).")
    args = p.parse_args()

    loc = make_localizer(
        hsv_path=args.hsv_config,
        motor_offsets_path=args.motor_offsets,
        loosen_color=args.color if args.loosen_hsv else None,
    )

    if args.trial_dir is not None:
        if args.phase == "home":
            med, details = estimate_trial_home(args.trial_dir, loc, args.color)
            if med is None:
                print(f"[estimate] no detection in {args.trial_dir / 'home'}")
                for d in details[:5]:
                    print(f"  {d}")
                return 1
            user = urdf_xyz_to_user(med)
            print(f"[estimate] trial {args.trial_dir.name} home median (user m): "
                  f"x={user[0]:+.4f} y={user[1]:+.4f} z={user[2]:+.4f}  "
                  f"(n_ok={sum(1 for d in details if d.get('status')=='ok')}/{len(details)})")
            print(f"  ik_relative: --bowl_xy_m {user[0]:.3f} {user[1]:.3f}")
            return 0

        grasp_img = args.trial_dir / "grasp" / "frame.png"
        meta_path = args.trial_dir / "grasp" / "frame_meta.json"
        if not grasp_img.is_file():
            print(f"[estimate] missing {grasp_img}")
            return 1
        bgr = cv2.imread(str(grasp_img))
        if meta_path.is_file():
            motor_deg = np.array(json.loads(meta_path.read_text())["motor_deg"], dtype=float)
        elif args.motor_deg:
            motor_deg = np.array(args.motor_deg, dtype=float)
        else:
            print("[estimate] need grasp/frame_meta.json or --motor_deg")
            return 1
        det = loc.locate(bgr, motor_deg, args.color)
        if det is None:
            print("[estimate] grasp frame: no detection (cube often occluded at grasp)")
            return 1
        user = urdf_xyz_to_user(det.base_xyz_m)
        print(f"[estimate] grasp (user m): x={user[0]:+.4f} y={user[1]:+.4f} z={user[2]:+.4f}")
        return 0

    if args.image is None or args.motor_deg is None:
        p.error("Provide --image and --motor_deg, or --trial_dir")
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        print(f"[estimate] cannot read {args.image}")
        return 1
    det = loc.locate(bgr, np.array(args.motor_deg, dtype=float), args.color)
    if det is None:
        print("[estimate] no detection")
        return 1
    user = urdf_xyz_to_user(det.base_xyz_m)
    print(f"[estimate] pixel={det.pixel} area={det.contour_area_px:.0f} conf={det.confidence:.2f}")
    print(f"[estimate] xyz user (m): x={user[0]:+.4f} y={user[1]:+.4f} z={user[2]:+.4f}")
    print(f"  --bowl_xy_m {user[0]:.3f} {user[1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
