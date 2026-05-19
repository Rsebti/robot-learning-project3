"""
probe_session_estimate.py - offline cube position from a probe_camera_vs_fk session.

Always exports FK grasp positions (joints from session.json).
If trial_NNN/home/ frames exist, re-runs CubeLocalizer and compares cam vs FK.

Outputs in <session_dir>/:
  estimates.csv           per-trial cam (home) + fk (grasp) + error
  cube_positions.yaml     deploy-ready cube xy/xyz (user frame) per trial
  estimate_report.txt     summary + ik_relative hints

Grasp reference (see --fk_target / --grasp_reference):
  tip_centered   FK at gripper_tip — you centered the symmetric jaw on the cube (default).
  gripper_frame  FK at gripper_frame_link — better when the tool is offset / one jaw opens
                 to the side and the contact point is not the symmetric tip center.

Usage:
    python -m toolset.perception.probe_session_estimate `
        --session_dir deploy/_snaps/probe_1779186663

    # replay saved images with relaxed HSV (red often needs this)
    python -m toolset.perception.probe_session_estimate `
        --session_dir deploy/_snaps/probe_XXX --loosen_hsv

    # use FK at gripper_frame for side-ready grasps (future probes)
    python -m toolset.perception.probe_session_estimate `
        --session_dir deploy/_snaps/probe_XXX --fk_target gripper_frame
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolset.kinematics.config import KinematicsConfig
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig
from toolset.kinematics.urdf_fk import SO101FK
from toolset.perception.estimate_cube_xy import (
    estimate_trial_home,
    load_hsv_ranges,
    loosen_hsv,
    make_localizer,
    urdf_xyz_to_user,
)
from toolset.perception.cube_localization import CubeLocalizer, DEFAULT_HSV


def fk_grasp_user(
    motor_deg: list[float],
    mcfg: MotorToUrdfConfig,
    fk: SO101FK,
    target: str,
) -> np.ndarray:
    urdf = mcfg.motor_to_urdf_rad(np.asarray(motor_deg, dtype=float))
    pos = fk.fk(urdf, target=target)["position"]
    return urdf_xyz_to_user(pos)


def session_paths(session_dir: Path) -> tuple[Path | None, Path | None]:
    hsv = session_dir / "hsv_config.yaml"
    motor = session_dir / "motor_offsets.yaml"
    return (
        hsv if hsv.is_file() else None,
        motor if motor.is_file() else None,
    )


def build_localizer_for_session(
    session_dir: Path,
    *,
    color: str,
    loosen_hsv_flag: bool,
) -> CubeLocalizer:
    hsv_p, motor_p = session_paths(session_dir)
    hsv = load_hsv_ranges(hsv_p)
    if hsv is None:
        hsv = DEFAULT_HSV
    if loosen_hsv_flag:
        hsv = loosen_hsv(hsv, color)
    kcfg = KinematicsConfig.load()
    mcfg = MotorToUrdfConfig.load(motor_p) if motor_p else MotorToUrdfConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)
    return CubeLocalizer(kcfg=kcfg, mcfg=mcfg, fk=fk, hsv_ranges=hsv)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session_dir", type=Path, required=True,
                   help="Folder probe_<timestamp> under deploy/_snaps/")
    p.add_argument("--color", default=None,
                   help="Override color (default: from session.json).")
    p.add_argument("--fk_target", default=None,
                   choices=["gripper_tip", "gripper_frame", "wrist", "wrist_roll"],
                   help="FK frame at grasp (default: session meta or gripper_tip).")
    p.add_argument("--grasp_reference", default=None,
                   choices=["tip_centered", "gripper_frame", "wrist"],
                   help="Alias: tip_centered->gripper_tip, gripper_frame->gripper_frame, wrist->wrist.")
    p.add_argument("--loosen_hsv", action="store_true",
                   help="Widen HSV for replay (helps red under dim light).")
    args = p.parse_args()

    session_dir = args.session_dir.resolve()
    meta_path = session_dir / "session.json"
    if not meta_path.is_file():
        raise SystemExit(f"Missing {meta_path}")

    meta = json.loads(meta_path.read_text())
    color = args.color or meta.get("color", "yellow")
    fk_target = args.fk_target
    if args.grasp_reference:
        fk_target = {
            "tip_centered": "gripper_tip",
            "gripper_frame": "gripper_frame",
            "wrist": "wrist",
        }[args.grasp_reference]
    if fk_target is None:
        fk_target = meta.get("fk_target", "gripper_tip")

    kcfg = KinematicsConfig.load()
    _, motor_p = session_paths(session_dir)
    mcfg = MotorToUrdfConfig.load(motor_p) if motor_p else MotorToUrdfConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)

    has_home_frames = any(session_dir.glob("trial_*/home/frame_*.png"))
    localizer = None
    if has_home_frames:
        localizer = build_localizer_for_session(
            session_dir, color=color, loosen_hsv_flag=args.loosen_hsv,
        )
        print(f"[estimate] replaying home frames with color={color!r} loosen_hsv={args.loosen_hsv}")
    else:
        print("[estimate] no trial_*/home/frame_*.png — FK-only export from session.json")

    trials = meta.get("trials", [])
    rows: list[dict] = []
    yaml_trials: dict = {}

    for t in trials:
        tri = int(t["trial"])
        trial_dir = session_dir / f"trial_{tri:03d}"

        motor = t.get("motor_deg")
        if motor is None:
            continue
        fk_user = fk_grasp_user(motor, mcfg, fk, fk_target)
        # prefer stored fk if same target
        if t.get("fk_xyz_user_m") and fk_target == meta.get("fk_target", "gripper_tip"):
            fk_user = np.asarray(t["fk_xyz_user_m"], dtype=float)

        cam_user = np.array([np.nan, np.nan, np.nan])
        n_cam_ok = 0
        cam_status = "no_frames"
        if localizer is not None and (trial_dir / "home").is_dir():
            med, details = estimate_trial_home(trial_dir, localizer, color)
            n_cam_ok = sum(1 for d in details if d.get("status") == "ok")
            if med is not None:
                cam_user = urdf_xyz_to_user(med)
                cam_status = f"ok_{n_cam_ok}"
            else:
                cam_status = details[0]["status"] if details else "no_detect"

        err = cam_user - fk_user
        err_norm = float(np.linalg.norm(err) * 1000) if np.all(np.isfinite(cam_user)) else float("nan")

        row = {
            "trial": tri,
            "color": color,
            "cam_status": cam_status,
            "cam_n_ok": n_cam_ok,
            "cam_x_user": cam_user[0],
            "cam_y_user": cam_user[1],
            "cam_z_user": cam_user[2],
            "fk_x_user": fk_user[0],
            "fk_y_user": fk_user[1],
            "fk_z_user": fk_user[2],
            "fk_target": fk_target,
            "err_x_mm": err[0] * 1000 if np.isfinite(err[0]) else "",
            "err_y_mm": err[1] * 1000 if np.isfinite(err[1]) else "",
            "err_z_mm": err[2] * 1000 if np.isfinite(err[2]) else "",
            "err_norm_mm": err_norm if np.isfinite(err_norm) else "",
            "motor_deg_grasp": motor,
            "urdf_rad_grasp": t.get("urdf_rad"),
            "home_motor_deg": t.get("home_motor_deg"),
        }
        rows.append(row)

        label = f"trial_{tri:03d}"
        yaml_trials[label] = {
            "cube_xy_user_m": [round(float(fk_user[0]), 4), round(float(fk_user[1]), 4)],
            "cube_xyz_user_m": [round(float(v), 4) for v in fk_user],
            "source": "fk_grasp",
            "fk_target": fk_target,
            "grasp_note": (
                "tip_centered symmetric grasp on cube center"
                if fk_target == "gripper_tip"
                else "offset / frame reference — not symmetric tip center"
            ),
            "motor_deg_grasp": [round(float(x), 3) for x in motor],
            "urdf_rad_grasp": t.get("urdf_rad"),
        }
        if np.all(np.isfinite(cam_user)):
            yaml_trials[label]["cam_xyz_user_m"] = [round(float(v), 4) for v in cam_user]
            yaml_trials[label]["cam_xy_user_m"] = [
                round(float(cam_user[0]), 4), round(float(cam_user[1]), 4)
            ]

    csv_path = session_dir / "estimates.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[estimate] -> {csv_path}")

    cube_yaml = {
        "session_dir": str(session_dir),
        "color": color,
        "fk_target": fk_target,
        "n_trials": len(rows),
        "frame_convention_user": "x_right_plus_y_forward_plus_z_up",
        "grasp_protocol": (
            "Operator places gripper in good orientation; default FK uses gripper_tip "
            "(centered, one jaw opens). For side-ready offset grasps use fk_target=gripper_frame."
        ),
        "trials": yaml_trials,
    }
    yaml_path = session_dir / "cube_positions.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cube_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"[estimate] -> {yaml_path}")

    lines = [
        "probe_session_estimate report",
        f"session: {session_dir}",
        f"color: {color}  fk_target: {fk_target}",
        f"trials: {len(rows)}  home_frames: {has_home_frames}",
        "",
    ]
    fk_xy = np.array([[r["fk_x_user"], r["fk_y_user"]] for r in rows])
    if len(fk_xy):
        med_fk = np.median(fk_xy, axis=0)
        lines.append(f"FK grasp median xy (user m): x={med_fk[0]:+.3f} y={med_fk[1]:+.3f}")
        lines.append(f"  ik_relative example: --bowl_xy_m {med_fk[0]:.3f} {med_fk[1]:.3f}")
    cam_ok = [r for r in rows if r["cam_n_ok"] > 0]
    lines.append(f"Trials with camera detection on replay: {len(cam_ok)}/{len(rows)}")
    if cam_ok:
        errs = [float(r["err_norm_mm"]) for r in cam_ok if r["err_norm_mm"] != ""]
        if errs:
            lines.append(f"||cam-fk|| mm: mean={np.mean(errs):.1f} median={np.median(errs):.1f}")
    elif has_home_frames:
        lines.append("Camera replay: 0 detections — try --loosen_hsv or tune hsv_config.yaml")
    elif not has_home_frames:
        lines.append("No saved home images — re-run probe with --save_frames for camera replay.")
    lines.append("")
    for r in rows[:25]:
        lines.append(
            f"  {r['trial']:03d} cam={r['cam_status']:12s} "
            f"fk=({r['fk_x_user']:+.3f},{r['fk_y_user']:+.3f}) "
            f"err_mm={r['err_norm_mm']}"
        )
    report_path = session_dir / "estimate_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[estimate] -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
