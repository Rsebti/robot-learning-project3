"""
probe_live_estimate_grasp.py — live map estimate, compare to nearest training probe, then manual FK grasp.

Workflow:
  1. Ramp home, capture N wrist frames, refined-mask detect
  2. kNN map -> estimated cube xyz; find closest training trial (du,dv)
  3. Show comparison + save overlay (open in viewer on Windows)
  4. Press Enter -> torque off -> you align grasp on cube -> Enter -> record FK
  5. Print est vs nearest-train vs measured; append validation log

Does not modify ik_relative.

Usage:
    python deploy/probe_live_estimate_grasp.py `
        --map_session_dir deploy/_snaps/probe_1779205245

    python deploy/probe_live_estimate_grasp.py `
        --map_session_dir deploy/_snaps/probe_1779205245 `
        --n_frames 10 --label test_spot_A

    # Already captured — grasp FK only, compare to saved estimate:
    python deploy/probe_live_estimate_grasp.py `
        --map_session_dir deploy/_snaps/probe_1779205245 `
        --probe_only --from_dir deploy/_snaps/test_probe_1779225334
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_DSHOW", "1000")

_DEPLOY = Path(__file__).resolve().parent
_PROJECT = _DEPLOY.parent
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from homes import get_home_deg  # noqa: E402
from robot_calibration import LOCAL_CALIBRATION_DIR, make_so101_follower_config  # noqa: E402
from wrist_camera_config import make_wrist_opencv_camera_config  # noqa: E402
from lerobot.robots.utils import make_robot_from_config  # noqa: E402
from toolset.kinematics.config import KinematicsConfig  # noqa: E402
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig  # noqa: E402
from toolset.kinematics.urdf_fk import SO101FK  # noqa: E402
from toolset.perception.mask_square_fit import draw_final_verification_overlay  # noqa: E402
from toolset.perception.probe_camera_vs_fk import (  # noqa: E402
    MOTOR_NAMES,
    ramp_to_target_deg,
    save_frame_bundle,
)
from toolset.perception.probe_map_data import load_mapping_rows  # noqa: E402
from toolset.perception.probe_space_map import (  # noqa: E402
    WRIST_CAM_HEIGHT,
    WRIST_CAM_WIDTH,
    build_index,
    load_samples,
    query,
)
from toolset.perception.refined_cube_detect import detect_refined_pixel, pixel_to_du_dv  # noqa: E402

OUT_BASE = _DEPLOY / "_snaps"


def _bgr_from_obs(obs) -> np.ndarray | None:
    bgr = obs.get("observation.images.base_camera")
    if bgr is None:
        for k, v in obs.items():
            if "base_camera" in str(k) and hasattr(v, "shape"):
                bgr = v
                break
    if bgr is None:
        return None
    if hasattr(bgr, "cpu"):
        bgr = bgr.cpu().numpy()
    if bgr.ndim == 3 and bgr.shape[0] == 3:
        bgr = np.transpose(bgr, (1, 2, 0))
    if bgr.dtype != np.uint8:
        bgr = (np.clip(bgr, 0, 1) * 255).astype(np.uint8)
    if bgr.shape[2] == 3:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
    return bgr


def capture_home_median(
    robot,
    *,
    map_session_dir: Path,
    color: str,
    n_frames: int,
    period_s: float,
) -> tuple[np.ndarray | None, np.ndarray | None, list[np.ndarray], list[dict]]:
    """Returns (home_pixel uv, motor_deg median, list bgr frames, per-frame detect meta)."""
    pixels: list[np.ndarray] = []
    motors: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    metas: list[dict] = []

    for _ in range(n_frames):
        obs = robot.get_observation()
        bgr = _bgr_from_obs(obs)
        if bgr is None:
            time.sleep(period_s)
            continue
        frames.append(bgr.copy())
        motor = np.array([float(obs[f"{n}.pos"]) for n in MOTOR_NAMES], dtype=float)
        motors.append(motor)
        det = detect_refined_pixel(bgr, color, session_dir=map_session_dir)
        if det.pixel is not None:
            pixels.append(np.array(det.pixel, dtype=float))
        metas.append({
            "pixel": det.pixel,
            "method": det.method,
            "area": det.area_px,
        })
        time.sleep(period_s)

    if not pixels:
        return None, None, frames, metas
    med_px = np.median(np.stack(pixels), axis=0)
    med_motor = np.median(np.stack(motors), axis=0) if motors else None
    return med_px, med_motor, frames, metas


def nearest_training_row(rows: list[dict], du: float, dv: float) -> tuple[dict, float]:
    best = rows[0]
    best_d = float("inf")
    for r in rows:
        d = float(np.hypot(r["du"] - du, r["dv"] - dv))
        if d < best_d:
            best_d = d
            best = r
    return best, best_d


def fk_user_from_motor_deg(motor_deg: np.ndarray, fk: SO101FK, mcfg: MotorToUrdfConfig) -> np.ndarray:
    urdf = mcfg.motor_to_urdf_rad(motor_deg)
    p = fk.fk(urdf, target="gripper_tip")["position"]
    return np.array([-p[1], p[0], p[2]], dtype=float)


def draw_comparison_overlay(
    bgr: np.ndarray,
    *,
    pixel: tuple[float, float],
    est: np.ndarray,
    near: dict,
    dist_px: float,
) -> np.ndarray:
    fk = near["fk"]
    vis = bgr.copy()
    if pixel:
        c = (int(pixel[0]), int(pixel[1]))
        cv2.circle(vis, c, 10, (0, 255, 255), 2)
    lines = [
        f"MAP est:  ({est[0]:+.3f}, {est[1]:+.3f}, {est[2]:+.3f}) m",
        f"NEAR trial {near['trial']:03d} ({dist_px:.0f}px): ({fk[0]:+.3f}, {fk[1]:+.3f}, {fk[2]:+.3f})",
        f"delta est-near xy: {np.linalg.norm(est[:2]-fk[:2])*1000:.0f} mm",
    ]
    y = 22
    for ln in lines:
        cv2.putText(vis, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return vis


def show_overlay(path: Path) -> None:
    try:
        os.startfile(str(path))  # noqa: S606 — Windows only
    except Exception:
        print(f"[view] open manually: {path}")


def load_saved_estimate(from_dir: Path) -> dict:
    """From live_estimate.json or analyze-test results.csv (median over ok frames)."""
    from_dir = from_dir.resolve()
    for rel in ("live_estimate.json", "home/live_estimate.json"):
        p = from_dir / rel
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))

    for csv_rel in ("analysis/results.csv", "results.csv"):
        csv_path = from_dir / csv_rel
        if not csv_path.is_file():
            continue
        ok = [
            r for r in csv.DictReader(csv_path.open(encoding="utf-8"))
            if r.get("status") == "ok" and r.get("est_x_m") not in ("", None)
        ]
        if not ok:
            raise ValueError(f"No ok rows in {csv_path}")
        r0 = ok[0]
        return {
            "label": r0.get("label", from_dir.name),
            "map_session": "",
            "pixel_u": float(np.median([float(r["pixel_u"]) for r in ok])),
            "pixel_v": float(np.median([float(r["pixel_v"]) for r in ok])),
            "du_px": float(np.median([float(r["du_px"]) for r in ok])),
            "dv_px": float(np.median([float(r["dv_px"]) for r in ok])),
            "est_xyz_user_m": [
                float(np.median([float(r["est_x_m"]) for r in ok])),
                float(np.median([float(r["est_y_m"]) for r in ok])),
                float(np.median([float(r["est_z_m"]) for r in ok])),
            ],
            "neighbor_trials": json.loads(r0["neighbor_trials"]) if r0.get("neighbor_trials") else [],
            "nearest_train_trial": int(r0["nearest_train_trial"]),
            "nearest_train_dist_px": float(r0["nearest_train_dist_px"]),
            "nearest_train_xyz_m": [
                float(r0["nearest_train_x_m"]),
                float(r0["nearest_train_y_m"]),
                float(r0.get("nearest_train_z_m") or r0["est_z_m"]),
            ],
        }

    raise FileNotFoundError(
        f"No live_estimate.json or analysis/results.csv under {from_dir}",
    )


def resolve_near_row(summary: dict, train_rows: list[dict]) -> tuple[dict, float]:
    du = float(summary["du_px"])
    dv = float(summary["dv_px"])
    tri = summary.get("nearest_train_trial")
    if tri is not None:
        for r in train_rows:
            if int(r["trial"]) == int(tri):
                d = float(summary.get("nearest_train_dist_px", np.hypot(r["du"] - du, r["dv"] - dv)))
                return r, d
    return nearest_training_row(train_rows, du, dv)


def _append_validation_log(
    *,
    ts: int,
    label: str,
    out_dir: Path,
    est: np.ndarray,
    measured: np.ndarray,
    near: dict,
    near_px: float,
) -> None:
    log_path = OUT_BASE / "live_probe_validation.csv"
    new_file = not log_path.is_file()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp", "label", "out_dir", "est_x", "est_y", "est_z",
            "meas_x", "meas_y", "meas_z", "near_trial", "near_px",
            "err_est_meas_xy_mm", "err_est_meas_z_mm",
        ])
        if new_file:
            w.writeheader()
        w.writerow({
            "timestamp": ts,
            "label": label,
            "out_dir": str(out_dir),
            "est_x": f"{est[0]:+.4f}",
            "est_y": f"{est[1]:+.4f}",
            "est_z": f"{est[2]:+.4f}",
            "meas_x": f"{measured[0]:+.4f}",
            "meas_y": f"{measured[1]:+.4f}",
            "meas_z": f"{measured[2]:+.4f}",
            "near_trial": near["trial"],
            "near_px": f"{near_px:.1f}",
            "err_est_meas_xy_mm": f"{np.linalg.norm(est[:2] - measured[:2]) * 1000:.1f}",
            "err_est_meas_z_mm": f"{(est[2] - measured[2]) * 1000:+.1f}",
        })
    print(f"[live] appended {log_path}")


def run_probe_only(
    *,
    map_dir: Path,
    from_dir: Path,
    train_rows: list[dict],
    port: str,
    camera_index: int,
    color: str,
    fk: SO101FK,
    mcfg: MotorToUrdfConfig,
) -> None:
    summary = load_saved_estimate(from_dir)
    est = np.asarray(summary["est_xyz_user_m"], dtype=float)
    near, near_px = resolve_near_row(summary, train_rows)
    nf = near["fk"]
    label = summary.get("label", from_dir.name)

    out_dir = from_dir.resolve()
    grasp_dir = out_dir / "grasp"
    grasp_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("SAVED MAP ESTIMATE (from capture — not re-running camera)")
    print("=" * 60)
    print(f"  source:          {from_dir}")
    print(f"  label:           {label}")
    print(f"  pixel (u,v):     ({summary['pixel_u']:.0f}, {summary['pixel_v']:.0f})")
    print(f"  offset (du,dv):  ({summary['du_px']:+.0f}, {summary['dv_px']:+.0f}) px")
    if summary.get("neighbor_trials"):
        print(f"  kNN neighbors:   trials {summary['neighbor_trials']}")
    print(f"  MAP estimate:    ({est[0]:+.3f}, {est[1]:+.3f}, {est[2]:+.3f}) m")
    print(f"  Closest train:   trial {near['trial']:03d}  ({near_px:.0f} px in image)")
    print(f"                   ({nf[0]:+.3f}, {nf[1]:+.3f}, {nf[2]:+.3f}) m")
    print("=" * 60)
    print("\nCube must still be at the SAME spot. Move arm to grasp (not home).")
    print("Press Enter to start PROBE (torque off -> align grasp -> Enter = FK). Ctrl+C abort.")
    input()

    cfg = make_so101_follower_config(
        port,
        cameras={"base_camera": make_wrist_opencv_camera_config(camera_index)},
        calibration_dir=LOCAL_CALIBRATION_DIR,
        use_degrees=True,
    )
    robot = make_robot_from_config(cfg)
    robot.connect()
    try:
        measured, grasp_motor = manual_grasp_record(robot, fk, mcfg)

        print("\n" + "=" * 60)
        print("VALIDATION: estimate vs measured grasp FK")
        print("=" * 60)
        print(f"  MAP estimate:  ({est[0]:+.3f}, {est[1]:+.3f}, {est[2]:+.3f})")
        print(f"  Measured FK:   ({measured[0]:+.3f}, {measured[1]:+.3f}, {measured[2]:+.3f})")
        err_xy = float(np.linalg.norm(est[:2] - measured[:2]) * 1000)
        err_z = float((est[2] - measured[2]) * 1000)
        print(f"  err est-meas:  xy {err_xy:.1f} mm   z {err_z:+.1f} mm")
        print(f"  err near-meas: xy {np.linalg.norm(nf[:2]-measured[:2])*1000:.1f} mm")
        if err_xy > 40:
            print("  >> XY error large — map estimate unreliable for IK at this spot.")
        else:
            print("  >> XY within ~40 mm — map may be OK for approach test.")
        print("=" * 60)

        obs = robot.get_observation()
        bgr = _bgr_from_obs(obs)
        if bgr is not None:
            det = detect_refined_pixel(bgr, color, session_dir=map_dir)
            save_frame_bundle(
                grasp_dir / "frame",
                bgr,
                det.final_mask if det.final_mask is not None else det.bright_mask,
                det.pixel,
                det.area_px,
            )
        (grasp_dir / "grasp_record.json").write_text(
            json.dumps({
                "grasp_motor_deg": grasp_motor.tolist(),
                "fk_xyz_user_m": measured.tolist(),
                "est_xyz_user_m": est.tolist(),
                "nearest_train_trial": int(near["trial"]),
                "source_dir": str(from_dir),
            }, indent=2),
            encoding="utf-8",
        )
        if not (out_dir / "live_estimate.json").is_file():
            summary["est_xyz_user_m"] = est.tolist()
            (out_dir / "live_estimate.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8",
            )

        _append_validation_log(
            ts=int(time.time()),
            label=label,
            out_dir=out_dir,
            est=est,
            measured=measured,
            near=near,
            near_px=near_px,
        )
    finally:
        robot.disconnect()
        print("[live] disconnected")


def manual_grasp_record(
    robot, fk: SO101FK, mcfg: MotorToUrdfConfig,
) -> tuple[np.ndarray, np.ndarray]:
    print("\n>>> PROBE GRASP")
    print(">>> Enter = disable torque (support arm)")
    input(">>> ")
    robot.bus.disable_torque()
    print(">>> Align perfect side grasp on cube center, then Enter to record FK")
    input(">>> ")
    obs = robot.get_observation()
    motor_deg = np.array([float(obs[f"{n}.pos"]) for n in MOTOR_NAMES], dtype=float)
    xyz = fk_user_from_motor_deg(motor_deg, fk, mcfg)
    print(f"[fk] measured ({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}) m")
    return xyz, motor_deg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map_session_dir", type=Path, required=True)
    p.add_argument("--port", default="COM3")
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--home_pose", default="eval1_rest")
    p.add_argument("--color", default=None)
    p.add_argument("--n_frames", type=int, default=10)
    p.add_argument("--obs_hz", type=float, default=5.0)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--label", default="live_test")
    p.add_argument("--no_probe", action="store_true",
                   help="Estimate only; skip Enter-to-grasp.")
    p.add_argument("--probe_only", action="store_true",
                   help="Skip capture; use --from_dir estimate and run grasp FK only.")
    p.add_argument("--from_dir", type=Path, default=None,
                   help="Folder with live_estimate.json or analysis/results.csv (required for --probe_only).")
    p.add_argument("--fk_target", default="gripper_tip")
    args = p.parse_args()

    if args.probe_only and args.from_dir is None:
        p.error("--probe_only requires --from_dir (your capture folder)")

    map_dir = args.map_session_dir.resolve()
    meta_path = map_dir / "session.json"
    color = args.color
    if not color and meta_path.is_file():
        color = json.loads(meta_path.read_text(encoding="utf-8")).get("color", "yellow")
    color = color or "yellow"

    train_rows = load_mapping_rows(map_dir, use_refined=True)
    if not train_rows:
        raise SystemExit(f"No training rows in {map_dir}")

    kcfg = KinematicsConfig.load()
    mcfg = MotorToUrdfConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)

    if args.probe_only:
        run_probe_only(
            map_dir=map_dir,
            from_dir=args.from_dir.resolve(),
            train_rows=train_rows,
            port=args.port,
            camera_index=args.camera_index,
            color=color or "yellow",
            fk=fk,
            mcfg=mcfg,
        )
        return

    samples = load_samples(map_dir, use_refined=True)
    index_path = map_dir / "space_map_index_refined.yaml"
    if not index_path.is_file():
        index = build_index(samples)
        with open(index_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(index, f, sort_keys=False)
    else:
        index = yaml.safe_load(index_path.read_text(encoding="utf-8"))

    ts = int(time.time())
    out_dir = OUT_BASE / f"live_probe_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = make_so101_follower_config(
        args.port,
        cameras={"base_camera": make_wrist_opencv_camera_config(args.camera_index)},
        calibration_dir=LOCAL_CALIBRATION_DIR,
        use_degrees=True,
    )
    robot = make_robot_from_config(cfg)
    print(f"[live] out -> {out_dir}")
    print(f"[live] training map: {map_dir} ({len(train_rows)} samples)")
    robot.connect()

    try:
        home_deg = get_home_deg(args.home_pose)
        print("[live] ramp home ...")
        ramp_to_target_deg(robot, home_deg)
        time.sleep(0.4)

        print(f"[live] capture {args.n_frames} frames @ {args.obs_hz} Hz ...")
        med_px, med_motor, frames, _ = capture_home_median(
            robot,
            map_session_dir=map_dir,
            color=color,
            n_frames=args.n_frames,
            period_s=1.0 / args.obs_hz,
        )
        if med_px is None:
            raise SystemExit("No cube detected in live frames — check color / HSV / lighting.")

        du, dv = pixel_to_du_dv(float(med_px[0]), float(med_px[1]))
        q = query(index, du, dv, k=args.k)
        est = np.asarray(q["fk_xyz_user_m"], dtype=float)
        near, near_px = nearest_training_row(train_rows, du, dv)

        print("\n" + "=" * 60)
        print("LIVE MAP ESTIMATE vs CLOSEST TRAINING PROBE")
        print("=" * 60)
        print(f"  pixel (u,v):     ({med_px[0]:.0f}, {med_px[1]:.0f})")
        print(f"  offset (du,dv):  ({du:+.0f}, {dv:+.0f}) px")
        print(f"  kNN neighbors:   trials {q['neighbor_trials']}  dist_px {q['neighbor_dist_px']}")
        print(f"  MAP estimate:    ({est[0]:+.3f}, {est[1]:+.3f}, {est[2]:+.3f}) m  (user frame)")
        nf = near["fk"]
        print(f"  Closest train:   trial {near['trial']:03d}  ({near_px:.0f} px away in image)")
        print(f"                   ({nf[0]:+.3f}, {nf[1]:+.3f}, {nf[2]:+.3f}) m")
        print(f"  est - nearest:   xy {np.linalg.norm(est[:2]-nf[:2])*1000:.1f} mm  "
              f"z {(est[2]-nf[2])*1000:+.1f} mm")
        print("=" * 60)

        home_dir = out_dir / "home"
        home_dir.mkdir(exist_ok=True)
        if frames:
            last = frames[-1]
            det = detect_refined_pixel(last, color, session_dir=map_dir)
            if det.final_mask is not None:
                vis0 = draw_final_verification_overlay(last, det.final_mask, centroid=tuple(med_px))
            else:
                vis0 = last.copy()
            overlay = draw_comparison_overlay(
                vis0, pixel=tuple(med_px), est=est, near=near, dist_px=near_px,
            )
            ov_path = home_dir / "live_estimate_overlay.png"
            cv2.imwrite(str(ov_path), overlay)
            cv2.imwrite(str(home_dir / "frame_last.png"), last)
            print(f"\n[live] overlay -> {ov_path}")
            show_overlay(ov_path)

        summary = {
            "label": args.label,
            "map_session": str(map_dir),
            "pixel_u": float(med_px[0]),
            "pixel_v": float(med_px[1]),
            "du_px": float(du),
            "dv_px": float(dv),
            "est_xyz_user_m": est.tolist(),
            "neighbor_trials": q["neighbor_trials"],
            "nearest_train_trial": int(near["trial"]),
            "nearest_train_dist_px": float(near_px),
            "nearest_train_xyz_m": nf.tolist(),
            "est_minus_nearest_xy_mm": float(np.linalg.norm(est[:2] - nf[:2]) * 1000),
        }
        (out_dir / "live_estimate.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8",
        )

        if args.no_probe:
            print("[live] --no_probe: done.")
            return

        print("\n[live] Press Enter to PROBE (manual grasp + FK record). Ctrl+C to abort.")
        input()

        measured, grasp_motor = manual_grasp_record(robot, fk, mcfg)

        print("\n" + "=" * 60)
        print("VALIDATION: estimate vs measured grasp FK")
        print("=" * 60)
        print(f"  MAP estimate:  ({est[0]:+.3f}, {est[1]:+.3f}, {est[2]:+.3f})")
        print(f"  Measured FK:   ({measured[0]:+.3f}, {measured[1]:+.3f}, {measured[2]:+.3f})")
        print(f"  err est-meas xy: {np.linalg.norm(est[:2]-measured[:2])*1000:.1f} mm  "
              f"z {(est[2]-measured[2])*1000:+.1f} mm")
        print(f"  err near-meas xy: {np.linalg.norm(nf[:2]-measured[:2])*1000:.1f} mm")
        print("=" * 60)

        grasp_dir = out_dir / "grasp"
        grasp_dir.mkdir(exist_ok=True)
        obs = robot.get_observation()
        bgr = _bgr_from_obs(obs)
        if bgr is not None:
            det = detect_refined_pixel(bgr, color, session_dir=map_dir)
            save_frame_bundle(
                grasp_dir / "frame",
                bgr,
                det.final_mask if det.final_mask is not None else det.bright_mask,
                det.pixel,
                det.area_px,
            )
        (grasp_dir / "grasp_record.json").write_text(
            json.dumps({
                "grasp_motor_deg": grasp_motor.tolist(),
                "fk_xyz_user_m": measured.tolist(),
                "est_xyz_user_m": est.tolist(),
                "nearest_train_trial": int(near["trial"]),
            }, indent=2),
            encoding="utf-8",
        )

        _append_validation_log(
            ts=ts, label=args.label, out_dir=out_dir,
            est=est, measured=measured, near=near, near_px=near_px,
        )

        print("[live] ramp home ...")
        ramp_to_target_deg(robot, home_deg)

    except KeyboardInterrupt:
        print("\n[live] aborted")
    finally:
        robot.disconnect()
        print("[live] disconnected")


if __name__ == "__main__":
    main()
