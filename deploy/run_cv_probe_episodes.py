"""
run_cv_probe_episodes.py — CV cube probing (no policy).

Dense mapping (no FK grasp): --map_only saves home pixel + joints only.
Dense map + FK ground truth: use probe_camera_vs_fk --map_only instead.
Hover default is 2D pixel_servo (--hover_mode pixel); use cv3d only if extrinsics trusted.

Usage:
    python deploy/run_cv_probe_episodes.py `
        --port COM3 --color red --n_episodes 30 --map_only --loosen_hsv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import yaml

_DEPLOY = Path(__file__).resolve().parent
_PROJECT = _DEPLOY.parent
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

import cv2  # noqa: E402

_OriginalVideoCapture = cv2.VideoCapture


class _DShowVideoCapture(_OriginalVideoCapture):
    def __init__(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], int) and args[1] in (
            cv2.CAP_ANY, cv2.CAP_MSMF
        ):
            args = (args[0], cv2.CAP_DSHOW)
        elif len(args) == 1 and isinstance(args[0], int):
            args = (args[0], cv2.CAP_DSHOW)
        super().__init__(*args, **kwargs)


cv2.VideoCapture = _DShowVideoCapture

from homes import get_home_deg  # noqa: E402
from robot_calibration import make_so101_follower_config  # noqa: E402
from lerobot.robots.utils import make_robot_from_config  # noqa: E402

from direct_wrist_camera import DirectWristCamera  # noqa: E402
from wrist_camera_config import make_wrist_opencv_camera_config  # noqa: E402

from toolset.kinematics.config import KinematicsConfig  # noqa: E402
from toolset.kinematics.ik_relative import (  # noqa: E402
    CHAIN_JOINTS,
    urdf_xyz_to_user,
    user_offset_to_urdf,
)
from toolset.perception.probe_camera_vs_fk import log_robot_cameras  # noqa: E402
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig  # noqa: E402
from toolset.kinematics.urdf_fk import SO101FK  # noqa: E402
from toolset.perception.estimate_cube_xy import urdf_xyz_to_user as urdf_xyz_to_user_est  # noqa: E402
from toolset.perception.probe_camera_vs_fk import (  # noqa: E402
    MOTOR_NAMES,
    _np_to_json,
    build_calibration_bundle,
    capture_at_home,
    extract_wrist_bgr,
    ramp_to_target_deg,
    write_calibration_bundle,
)
from toolset.perception.cube_localization import (  # noqa: E402
    WRIST_CAM_HEIGHT,
    WRIST_CAM_WIDTH,
    log_intrinsics_camera_size_match,
)
from toolset.perception.scout_cube import make_localizer  # noqa: E402


def discard_camera_frames(robot, n: int) -> None:
    for _ in range(max(0, n)):
        robot.get_observation()


def run_camera_check(
    *,
    robot,
    out_dir: Path,
    camera_key: str,
    camera_index: int,
    show_preview: bool,
    extra_warmup: int,
) -> bool:
    """Grab one frame after warmup; save PNG and report brightness. Returns True if usable."""
    discard_camera_frames(robot, extra_warmup)
    obs = robot.get_observation()
    bgr, key_used = extract_wrist_bgr(
        obs, camera_key=camera_key, camera_index=camera_index,
    )
    check_dir = out_dir / "camera_check"
    check_dir.mkdir(parents=True, exist_ok=True)
    if bgr is None:
        print(f"[cv-probe] camera_check FAILED: no image (key={key_used})", flush=True)
        return False
    mean_bgr = float(bgr.mean())
    path = check_dir / "live_frame.png"
    cv2.imwrite(str(path), bgr)
    print(
        f"[cv-probe] camera_check index={camera_index} key={key_used} "
        f"shape={bgr.shape} mean={mean_bgr:.1f} -> {path}",
        flush=True,
    )
    if show_preview:
        cv2.imshow("cv-probe camera_check", bgr)
        cv2.waitKey(1500)
    ok = mean_bgr >= 12.0
    if not ok:
        print(
            "[cv-probe] Frame still nearly black — try another USB port, "
            "close Teams/Camera app, or --camera_fourcc YUYV",
            flush=True,
        )
    return ok


def countdown_pause(seconds: float, message: str) -> None:
    if seconds <= 0:
        return
    print(message, flush=True)
    n = max(1, int(round(seconds)))
    step = seconds / n
    for k in range(n, 0, -1):
        print(f"  {k} ...", flush=True)
        time.sleep(step)


def tip_user_ik_relative_convention(robot, fk: SO101FK, target: str = "gripper_tip") -> np.ndarray:
    """Same start-tip FK as ik_relative.py (motor deg -> rad, first 5 joints)."""
    obs = robot.get_observation()
    q_motor = np.array(
        [float(obs[f"{n}.pos"]) for n in CHAIN_JOINTS] + [float(obs["gripper.pos"])],
        dtype=float,
    )
    q_motor = np.deg2rad(q_motor)
    q_urdf = q_motor[:5].copy()
    xyz_urdf = fk.fk(q_urdf, target=target)["position"]
    return urdf_xyz_to_user(xyz_urdf)


def median_pixel_from_log(log_home: list[dict]) -> tuple[float, float] | None:
    pixels = [fr["pixel"] for fr in log_home if fr.get("pixel") is not None]
    if not pixels:
        return None
    med = np.median(np.asarray(pixels, dtype=float), axis=0)
    return float(med[0]), float(med[1])


def hover_offset_from_camera(
    robot,
    fk: SO101FK,
    cube_xyz_user: np.ndarray,
    hover_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """delta_user so ik_relative --offsets moves tip to cube + hover_m (full 3D backproj)."""
    tip_user = tip_user_ik_relative_convention(robot, fk)
    cube_user = np.asarray(cube_xyz_user, dtype=float).reshape(3)
    hover_user = cube_user + np.array([0.0, 0.0, float(hover_m)], dtype=float)
    delta_user = hover_user - tip_user
    return delta_user, tip_user, hover_user


def hover_offset_from_pixel(
    robot,
    fk: SO101FK,
    pixel: tuple[float, float],
    K: np.ndarray,
    hover_m: float,
    *,
    z_est_m: float = 0.22,
    max_horiz_m: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hover from 2D red pixel only (pinhole @ ~table depth). Use when hand-eye 3D is wrong."""
    u, v = pixel
    cx, cy = float(K[0, 2]), float(K[1, 2])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    # Cube vs image center -> move tip in user frame (right+, forward+, up+).
    right_m = (u - cx) / fx * z_est_m
    forward_m = (v - cy) / fy * z_est_m
    horiz = np.array([right_m, forward_m], dtype=float)
    hn = float(np.linalg.norm(horiz))
    if hn > max_horiz_m and hn > 1e-9:
        horiz *= max_horiz_m / hn
    tip_user = tip_user_ik_relative_convention(robot, fk)
    delta_user = np.array([horiz[0], horiz[1], -float(hover_m)], dtype=float)
    hover_user = tip_user + delta_user
    return delta_user, tip_user, hover_user


def run_ik_relative_offsets(
    *,
    port: str,
    camera_index: int,
    delta_user: np.ndarray,
    speed: str = "fluid",
    target: str = "gripper_tip",
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    """Invoke the real ik_relative.py module (must own the serial port)."""
    dx, dy, dz = [float(v) for v in np.asarray(delta_user).reshape(3)]
    cmd = [
        sys.executable,
        "-m",
        "toolset.kinematics.ik_relative",
        "--port",
        port,
        "--camera_index",
        str(camera_index),
        "--speed",
        speed,
        "--target",
        target,
        "--offsets",
        f"{dx:.5f}",
        f"{dy:.5f}",
        f"{dz:.5f}",
        "--no-return_to_start",
        "--no-go_home",
    ]
    if dry_run:
        cmd.append("--dry_run")
    print(f"[cv-probe] ik_relative: {' '.join(cmd[2:])}", flush=True)
    rc = subprocess.run(
        cmd, cwd=str(_PROJECT), check=False,
        capture_output=True, text=True,
    )
    if rc.stdout:
        print(rc.stdout, end="" if rc.stdout.endswith("\n") else "\n", flush=True)
    if rc.stderr:
        print(rc.stderr, end="" if rc.stderr.endswith("\n") else "\n", flush=True)
    return rc


def run_ik_relative_hover_staged(
    *,
    port: str,
    camera_index: int,
    delta_user: np.ndarray,
    speed: str,
    target: str = "gripper_tip",
) -> tuple[bool, str]:
    """XY then Z as two ik_relative runs (lift_place-style), if one-shot fails."""
    dx, dy, dz = np.asarray(delta_user, dtype=float).reshape(3)
    if np.linalg.norm([dx, dy, dz]) < 1e-5:
        return True, "delta ~ 0"

    rc = run_ik_relative_offsets(
        port=port, camera_index=camera_index,
        delta_user=delta_user, speed=speed, target=target,
    )
    out = (rc.stdout or "") + (rc.stderr or "")
    if rc.returncode == 0 and "FAILED:" not in out:
        return True, "ik_relative one-shot OK"
    if rc.returncode == 0 and "FAILED:" in out:
        print("[cv-probe] ik_relative printed FAILED but exit 0", flush=True)

    print("[cv-probe] ik_relative one-shot failed; trying XY then Z ...", flush=True)
    if np.linalg.norm([dx, dy]) > 1e-4:
        rc1 = run_ik_relative_offsets(
            port=port, camera_index=camera_index,
            delta_user=np.array([dx, dy, 0.0]), speed=speed, target=target,
        )
        if rc1.returncode != 0:
            return False, f"ik_relative XY failed (exit {rc1.returncode})"
    if abs(dz) > 1e-4:
        rc2 = run_ik_relative_offsets(
            port=port, camera_index=camera_index,
            delta_user=np.array([0.0, 0.0, dz]), speed=speed, target=target,
        )
        if rc2.returncode != 0:
            return False, f"ik_relative Z failed (exit {rc2.returncode})"
    return True, "ik_relative staged XY+Z OK"


def write_cv_report(out_dir: Path, meta: dict) -> None:
    lines = [
        "cv_probe_episodes report (hover via ik_relative.py)",
        f"session: {out_dir}",
        f"color: {meta.get('color')}",
        f"hover_m: {meta.get('hover_m')}",
        f"episodes: {len(meta.get('trials', []))}",
        "",
    ]
    ok_home = ok_hover = 0
    for t in meta.get("trials", []):
        if t.get("home_frames_detected", 0) > 0:
            ok_home += 1
        if t.get("hover_frames_detected", 0) > 0:
            ok_hover += 1
        lines.append(
            f"  ep {t['episode']:03d}  ik={t.get('ik_hover_ok')}  "
            f"delta_user={t.get('ik_offset_user_m')}  "
            f"home_xyz={t.get('cam_xyz_user_m')}  hover_xyz={t.get('hover_cam_xyz_user_m')}"
        )
    lines.append(f"\nHome detections: {ok_home}/{len(meta.get('trials', []))}")
    lines.append(f"Hover detections: {ok_hover}/{len(meta.get('trials', []))}")
    (out_dir / "cv_probe_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="COM3")
    p.add_argument("--camera_index", type=int, default=0,
                   help="OpenCV device index (default 0 = wrist USB on this PC).")
    p.add_argument("--camera_key", default="base_camera",
                   help="Observation dict key for RGB (default base_camera = wrist USB cam).")
    p.add_argument("--camera_fourcc", default="MJPG",
                   help="USB format for 1080p (default MJPG). Try YUYV if black frames.")
    p.add_argument("--camera_warmup_s", type=int, default=3,
                   help="LeRobot camera warmup at connect (seconds).")
    p.add_argument("--extra_warmup_frames", type=int, default=15,
                   help="Extra get_observation() discards after connect (exposure settle).")
    p.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True,
                   help="Live HSV overlay window every frame (no keypress; default on).")
    p.add_argument("--direct_camera", action=argparse.BooleanOptionalAction, default=True,
                   help="640x480 DSHOW direct OpenCV on --camera_index (default on).")
    p.add_argument("--camera_check", action="store_true",
                   help="Only save camera_check/live_frame.png and exit.")
    p.add_argument("--home_pose", default="eval1_rest")
    p.add_argument("--color", default="red")
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--n_frames", type=int, default=10)
    p.add_argument("--n_frames_coarse", type=int, default=5)
    p.add_argument("--obs_hz", type=float, default=5.0)
    p.add_argument("--setup_pause_s", type=float, default=5.0)
    p.add_argument("--hover_m", type=float, default=0.01)
    p.add_argument("--map_only", action="store_true",
                   help="Collect home pixels only; no IK / 3D hover (use probe_camera_vs_fk for dense map).")
    p.add_argument("--hover_mode", default="pixel", choices=["auto", "cv3d", "pixel"],
                   help="pixel: 2D servo (default). auto/cv3d use backproj when trusted.")
    p.add_argument("--hover_z_est_m", type=float, default=0.22,
                   help="Table distance for pixel hover_mode (pinhole scale).")
    p.add_argument("--hover_speed", default="fluid",
                   choices=["slow", "fluid", "smooth", "normal", "fast", "max"])
    p.add_argument("--ik_staged", action=argparse.BooleanOptionalAction, default=True,
                   help="If one-shot ik_relative fails, retry XY then Z (default: on).")
    p.add_argument("--ik_dry_run", action="store_true",
                   help="Only print ik_relative plan, no motion.")
    p.add_argument("--loosen_hsv", action="store_true")
    p.add_argument("--no_hover", action="store_true")
    args = p.parse_args()

    use_hover = (args.hover_m > 0) and not args.no_hover and not args.map_only

    out_dir = _DEPLOY / "_snaps" / f"cv_probe_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "mode": (
            "spatial_map_collect" if args.map_only
            else ("cv_ik_relative_hover" if use_hover else "cv_home_only")
        ),
        "map_only": args.map_only,
        "out_dir": str(out_dir),
        "color": args.color,
        "home_pose": args.home_pose,
        "hover_m": args.hover_m if use_hover else 0.0,
        "n_episodes": args.n_episodes,
        "setup_pause_s": args.setup_pause_s,
        "hover_speed": args.hover_speed,
        "started_at": time.time(),
        "trials": [],
    }

    cx_img = WRIST_CAM_WIDTH / 2.0
    cy_img = WRIST_CAM_HEIGHT / 2.0

    csv_path = out_dir / "trials.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    if args.map_only:
        csv_w.writerow([
            "episode", "color", "n_pixel_frames", "pixel_u", "pixel_v", "du_px", "dv_px",
            *MOTOR_NAMES,
        ])
    else:
        csv_w.writerow([
            "episode", "color", "home_detected", "hover_detected",
            "home_x", "home_y", "home_z",
            "hover_x", "hover_y", "hover_z",
            "ik_dx", "ik_dy", "ik_dz", "ik_hover_ok",
            *MOTOR_NAMES,
        ])

    home_deg = get_home_deg(args.home_pose)
    kcfg = KinematicsConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)
    mcfg = MotorToUrdfConfig.load()
    localizer = make_localizer(loosen_color=args.color if args.loosen_hsv else None)
    period = 1.0 / max(0.1, args.obs_hz)

    print(f"[cv-probe] intrinsics: {getattr(localizer, 'intrinsics_source', '?')}")
    log_intrinsics_camera_size_match("cv-probe")
    print(f"[cv-probe] extrinsics: legacy T_cam_in_wrist (eval1) unless camera_in_wrist.npy exists")
    print(
        f"[cv-probe] K fx={localizer.K[0,0]:.1f} fy={localizer.K[1,1]:.1f} "
        f"cx={localizer.K[0,2]:.1f} cy={localizer.K[1,2]:.1f}",
        flush=True,
    )

    print(f"[cv-probe] session -> {out_dir}")
    print(f"[cv-probe] home {args.home_pose!r}")
    if args.map_only:
        print("[cv-probe] map_only: home pixels + joints only (no IK / no 3D labels)")
    if use_hover:
        print(f"[cv-probe] hover: ik_relative --offsets from home tip -> CV cube + {args.hover_m} m")

    cal_args = types.SimpleNamespace(
        port=args.port, camera_index=args.camera_index, camera_key=args.camera_key,
        home_pose=args.home_pose, color=args.color,
        n_obs_frames=args.n_frames, obs_hz=args.obs_hz,
    )
    write_calibration_bundle(
        out_dir,
        build_calibration_bundle(
            args=cal_args, kcfg=kcfg, mcfg=mcfg, localizer=localizer, home_deg=home_deg,
        ),
    )

    direct_cam: DirectWristCamera | None = None
    if args.direct_camera:
        print(
            f"[cv-probe] direct wrist camera index={args.camera_index} "
            f"{WRIST_CAM_WIDTH}x{WRIST_CAM_HEIGHT} (DSHOW, same as manual color scripts)",
            flush=True,
        )
        direct_cam = DirectWristCamera(
            args.camera_index, WRIST_CAM_WIDTH, WRIST_CAM_HEIGHT,
            warmup_reads=args.extra_warmup_frames,
        )
        robot_cameras = None
    else:
        fourcc = args.camera_fourcc if args.camera_fourcc.upper() != "NONE" else None
        robot_cameras = {
            "base_camera": make_wrist_opencv_camera_config(
                args.camera_index, fourcc=fourcc, warmup_s=args.camera_warmup_s,
            ),
        }

    cfg = make_so101_follower_config(args.port, cameras=robot_cameras, use_degrees=True)
    if robot_cameras:
        log_robot_cameras(cfg, "cv-probe")
        print(f"[cv-probe] LeRobot key {args.camera_key!r}", flush=True)

    read_bgr = (lambda: direct_cam.read_bgr()) if direct_cam is not None else None
    yaml_trials: dict = {}
    try:
        if args.camera_check:
            check_dir = out_dir / "camera_check"
            check_dir.mkdir(parents=True, exist_ok=True)
            bgr = read_bgr() if read_bgr else None
            if bgr is not None:
                path = check_dir / "live_frame.png"
                cv2.imwrite(str(path), bgr)
                print(f"[cv-probe] camera_check -> {path} mean={float(bgr.mean()):.1f}", flush=True)
                if args.preview:
                    cv2.imshow("cv-probe camera_check", bgr)
                    cv2.waitKey(1500)
            return

        for ep in range(args.n_episodes):
            countdown_pause(
                args.setup_pause_s,
                f"\n[cv-probe] Episode {ep + 1}/{args.n_episodes}: place/move {args.color} cube:",
            )

            print(f"\n========== Episode {ep + 1}/{args.n_episodes} ==========", flush=True)

            robot = make_robot_from_config(cfg)
            robot.connect()
            print("[cv-probe] ramp to home ...", flush=True)
            ramp_to_target_deg(robot, home_deg)
            time.sleep(0.3)
            if direct_cam is not None:
                for _ in range(args.extra_warmup_frames):
                    direct_cam.read_bgr()

            trial_dir = out_dir / f"trial_{ep:03d}"
            n_coarse = args.n_frames_coarse if use_hover else args.n_frames

            cam_home_urdf, log_home = capture_at_home(
                robot, localizer, mcfg, fk,
                color=args.color, n_frames=n_coarse, period=period,
                trial_dir=trial_dir, save_frames=True,
                camera_key=args.camera_key,
                camera_index=args.camera_index,
                subdir="home",
                show_preview=args.preview,
                read_bgr=read_bgr,
                map_only=args.map_only,
            )

            cam_user = np.array([np.nan, np.nan, np.nan])
            hover_user = np.array([np.nan, np.nan, np.nan])
            delta_user = None
            ik_ok = ""
            cam_hover_urdf: list = []
            log_hover: list = []
            home_motor_deg = None

            if cam_home_urdf and not args.map_only:
                cam_user = urdf_xyz_to_user_est(np.median(np.stack(cam_home_urdf), axis=0))
                print(f"[cv-probe] home CV 3D (untrusted): {cam_user.round(3).tolist()}", flush=True)

            home_motors = [
                np.asarray(fr["motor_deg"], dtype=float)
                for fr in log_home if fr.get("motor_deg") is not None
            ]
            if home_motors:
                home_motor_deg = np.median(np.stack(home_motors), axis=0).tolist()

            n_ok = sum(1 for fr in log_home if fr.get("status") == "ok")
            n_pix = sum(1 for fr in log_home if fr.get("pixel") is not None)
            if args.map_only and med_px is not None:
                cx, cy = WRIST_CAM_WIDTH / 2.0, WRIST_CAM_HEIGHT / 2.0
                print(
                    f"[cv-probe] map pixel ({med_px[0]:.0f},{med_px[1]:.0f}) "
                    f"offset ({med_px[0]-cx:+.0f},{med_px[1]-cy:+.0f}) px",
                    flush=True,
                )
            if not args.map_only and n_ok == 0 and n_pix > 0:
                print(
                    f"[cv-probe] {n_pix} frame(s) with red pixel but 0 with 3D xyz "
                    f"(statuses: { {s: sum(1 for fr in log_home if fr.get('status')==s) for s in set(fr.get('status') for fr in log_home)} })",
                    flush=True,
                )

            med_px = median_pixel_from_log(log_home)
            hover_mode_used = ""
            run_ik = False

            if use_hover and med_px is not None:
                tip_user = tip_user_ik_relative_convention(robot, fk)
                cv_pack = None
                in_ws = True
                if len(cam_home_urdf) > 0 and np.all(np.isfinite(cam_user)):
                    cv_pack = hover_offset_from_camera(robot, fk, cam_user, args.hover_m)
                    hover_urdf = user_offset_to_urdf(cv_pack[2])
                    in_ws = kcfg.in_workspace(hover_urdf)

                use_pixel = args.hover_mode == "pixel" or (
                    args.hover_mode == "auto"
                    and (
                        cv_pack is None
                        or not in_ws
                        or float(np.linalg.norm(cv_pack[0])) > 0.35
                    )
                )
                if use_pixel:
                    delta_user, tip_user, hover_user = hover_offset_from_pixel(
                        robot, fk, med_px, localizer.K, args.hover_m,
                        z_est_m=args.hover_z_est_m,
                    )
                    hover_mode_used = "pixel_servo"
                    print(
                        f"[cv-probe] hover_mode=pixel_servo (2D pixel {med_px[0]:.0f},{med_px[1]:.0f} "
                        f"z_est={args.hover_z_est_m}m) — bad 3D extrinsics, using pinhole",
                        flush=True,
                    )
                elif cv_pack is not None:
                    delta_user, tip_user, hover_user = cv_pack
                    hover_mode_used = "cv3d"
                    print("[cv-probe] hover_mode=cv3d (full backproj)", flush=True)
                else:
                    delta_user = None

                if delta_user is not None:
                    run_ik = True
                    print(f"[cv-probe] home tip (user m): {tip_user.round(3).tolist()}", flush=True)
                    if np.all(np.isfinite(cam_user)):
                        print(f"[cv-probe] home CV 3D (user m): {cam_user.round(3).tolist()} "
                              f"(ignored when pixel_servo)", flush=True)
                    print(f"[cv-probe] hover target (user m): {hover_user.round(3).tolist()}", flush=True)
                    print(f"[cv-probe] ik_relative --offsets (user m): {delta_user.round(3).tolist()}",
                          flush=True)

                    (trial_dir / "ik_hover_plan.json").write_text(
                        json.dumps({
                            "hover_mode": hover_mode_used,
                            "pixel_median": list(med_px),
                            "tip_user_m": tip_user.tolist(),
                            "cube_user_m": cam_user.tolist() if np.all(np.isfinite(cam_user)) else None,
                            "hover_user_m": hover_user.tolist(),
                            "offset_user_m": delta_user.tolist(),
                            "command": (
                                f"python -m toolset.kinematics.ik_relative --port {args.port} "
                                f"--offsets {delta_user[0]:.5f} {delta_user[1]:.5f} {delta_user[2]:.5f} "
                                f"--speed {args.hover_speed} --no-return_to_start --no-go_home"
                            ),
                        }, indent=2),
                        encoding="utf-8",
                    )

            if run_ik and delta_user is not None:
                robot.disconnect()
                robot = None
                time.sleep(0.5)

                if args.ik_dry_run:
                    rc = run_ik_relative_offsets(
                        port=args.port, camera_index=args.camera_index,
                        delta_user=delta_user, speed=args.hover_speed, dry_run=True,
                    )
                    ik_ok = "dry_run" if rc.returncode == 0 else f"dry_run_fail_{rc.returncode}"
                elif args.ik_staged:
                    ok, msg = run_ik_relative_hover_staged(
                        port=args.port, camera_index=args.camera_index,
                        delta_user=delta_user, speed=args.hover_speed,
                    )
                    ik_ok = msg if ok else msg
                else:
                    rc = run_ik_relative_offsets(
                        port=args.port, camera_index=args.camera_index,
                        delta_user=delta_user, speed=args.hover_speed,
                    )
                    ik_ok = "ok" if rc.returncode == 0 else f"exit_{rc.returncode}"

                print(f"[cv-probe] ik_relative result: {ik_ok}", flush=True)

                robot = make_robot_from_config(cfg)
                robot.connect()
                time.sleep(0.3)

                ik_moved = (
                    args.ik_dry_run
                    or "OK" in str(ik_ok)
                    or ik_ok == "ok"
                ) and "fail" not in str(ik_ok).lower()
                if ik_moved and not args.ik_dry_run:
                    cam_hover_urdf, log_hover = capture_at_home(
                        robot, localizer, mcfg, fk,
                        color=args.color, n_frames=args.n_frames, period=period,
                        trial_dir=trial_dir, save_frames=True,
                        camera_key=args.camera_key,
                        camera_index=args.camera_index,
                        subdir="hover",
                        show_preview=args.preview,
                        read_bgr=read_bgr,
                    )
                    if cam_hover_urdf:
                        hover_user = urdf_xyz_to_user_est(np.median(np.stack(cam_hover_urdf), axis=0))
                        print(f"[cv-probe] hover CV (user m): {hover_user.round(3).tolist()}", flush=True)
            elif use_hover:
                st: dict[str, int] = {}
                for fr in log_home:
                    st[fr.get("status", "?")] = st.get(fr.get("status", "?"), 0) + 1
                print(
                    "[cv-probe] no hover plan — skip ik_relative\n"
                    f"         frame statuses: {st}\n"
                    f"         need red pixel in view (inspect frame_00_overlay.png)",
                    flush=True,
                )
                ik_ok = "no_cv"

            if robot is not None:
                print("[cv-probe] return home ...", flush=True)
                ramp_to_target_deg(robot, home_deg)
                robot.disconnect()

            motor_cols = (
                [f"{home_motor_deg[i]:.2f}" for i in range(6)] if home_motor_deg else [""] * 6
            )
            ep_rec: dict = {
                "episode": ep,
                "trial_dir": f"trial_{ep:03d}",
                "home_motor_deg": home_motor_deg,
                "timestamp": time.time(),
            }
            if args.map_only:
                pu = pv = du_px = dv_px = float("nan")
                if med_px is not None:
                    pu, pv = float(med_px[0]), float(med_px[1])
                    du_px, dv_px = pu - cx_img, pv - cy_img
                csv_w.writerow([
                    ep, args.color, n_pix, f"{pu:.1f}", f"{pv:.1f}",
                    f"{du_px:+.1f}", f"{dv_px:+.1f}", *motor_cols,
                ])
                ep_rec["n_pixel_frames"] = n_pix
                if med_px is not None:
                    ep_rec["home_pixel"] = [pu, pv]
                    ep_rec["pixel_offset_px"] = [du_px, dv_px]
            else:
                du = delta_user if delta_user is not None else [np.nan, np.nan, np.nan]
                csv_w.writerow([
                    ep, args.color, len(cam_home_urdf), len(cam_hover_urdf),
                    f"{cam_user[0]:+.4f}", f"{cam_user[1]:+.4f}", f"{cam_user[2]:+.4f}",
                    f"{hover_user[0]:+.4f}", f"{hover_user[1]:+.4f}", f"{hover_user[2]:+.4f}",
                    f"{du[0]:+.4f}", f"{du[1]:+.4f}", f"{du[2]:+.4f}",
                    ik_ok, *motor_cols,
                ])
                ep_rec.update({
                    "home_frames_detected": len(cam_home_urdf),
                    "hover_frames_detected": len(cam_hover_urdf),
                    "cam_xyz_user_m_untrusted": cam_user.tolist(),
                    "hover_cam_xyz_user_m_untrusted": hover_user.tolist(),
                    "ik_offset_user_m": None if delta_user is None else delta_user.tolist(),
                    "ik_hover_ok": ik_ok,
                })
                final = hover_user if np.all(np.isfinite(hover_user)) else cam_user
                if np.all(np.isfinite(final)):
                    yaml_trials[f"trial_{ep:03d}"] = {
                        "cube_xy_user_m": [round(float(final[0]), 4), round(float(final[1]), 4)],
                        "cube_xyz_user_m": [round(float(v), 4) for v in final],
                    }
            csv_f.flush()
            meta["trials"].append(ep_rec)
            with open(out_dir / "session.json", "w") as f:
                json.dump(meta, f, indent=2, default=_np_to_json)

        if not args.map_only:
            with open(out_dir / "cube_positions.yaml", "w") as f:
                yaml.dump({
                    "session_dir": str(out_dir),
                    "color": args.color,
                    "method": "cv_then_ik_relative_offsets",
                    "hover_m": args.hover_m,
                    "trials": yaml_trials,
                }, f, default_flow_style=False)

    except KeyboardInterrupt:
        print("\n[cv-probe] interrupted.", flush=True)
    finally:
        if direct_cam is not None:
            direct_cam.release()
        cv2.destroyAllWindows()
        csv_f.close()
        write_cv_report(out_dir, meta)
        with open(out_dir / "session.json", "w") as f:
            json.dump(meta, f, indent=2, default=_np_to_json)
        print(f"\n[cv-probe] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
