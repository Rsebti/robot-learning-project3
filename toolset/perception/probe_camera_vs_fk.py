"""
probe_camera_vs_fk.py - spatial mapping via home view + grasp FK (no cal required).

Per trial: at home record red pixel (u,v) + joints; at grasp record FK tip = cube pose.
Cube must stay still between home and grasp. Dense trials build a pixel->position map
(see probe_space_map.py); do not trust backproj xyz until extrinsics are refit.

  --map_only   trusted fields only (pixel, motors, fk); skips 3D backproj in logs

Outputs under deploy/_snaps/probe_<ts>/:
  calibrations.json   K, T_cam_in_wrist, motor offsets, HSV, home, URDF paths
  session.json        structured trial log
  trials.csv          flat table
  session_report.txt  human-readable summary + per-trial notes
  trial_NNN/home/     frame_XX.png, frame_XX_mask.png, frame_XX_overlay.png
  trial_NNN/grasp/    frame.png (+ mask/overlay if detection attempted)

Usage:
    python -m toolset.perception.probe_camera_vs_fk `
        --port COM3 --color yellow --n_trials 20 --save_frames --map_only

    # More trials in same session (yellow map):
    python -m toolset.perception.probe_camera_vs_fk `
        --map_only --save_frames --color yellow `
        --session_dir deploy/_snaps/probe_1779205245 --n_trials 6

    # Each map_only trial prints: MAP estimate (kNN) -> grasp FK -> est vs meas -> coverage hints
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
PROJECT_ROOT = DEPLOY.parent
KIN_CONFIG_DIR = PROJECT_ROOT / "toolset" / "configs" / "kinematics"
LEGACY_CALIB = PROJECT_ROOT / "toolset" / "configs" / "camera_calibration.yaml"
HSV_CONFIG_PATH = PROJECT_ROOT / "toolset" / "perception" / "hsv_config.yaml"
MOTOR_OFFSETS_PATH = KIN_CONFIG_DIR / "motor_offsets.yaml"

if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

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

from homes import get_home_deg                                                  # noqa: E402
from robot_calibration import LOCAL_CALIBRATION_DIR, make_so101_follower_config  # noqa: E402
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig      # noqa: E402
from toolset.perception.cube_localization import WRIST_CAM_HEIGHT, WRIST_CAM_WIDTH  # noqa: E402
from lerobot.robots.utils import make_robot_from_config                         # noqa: E402

from toolset.kinematics.config import KinematicsConfig                          # noqa: E402
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig                  # noqa: E402
from toolset.kinematics.urdf_fk import SO101FK                                  # noqa: E402
from toolset.perception.cube_localization import CubeLocalizer                  # noqa: E402
from toolset.perception.estimate_cube_xy import load_hsv_ranges, loosen_hsv     # noqa: E402
from toolset.perception.probe_map_guidance import (                           # noqa: E402
    load_samples_from_mapping_csv,
    print_post_grasp_block,
    print_pre_grasp_block,
)

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

# LeRobot observation keys — prefer base_camera (wrist USB cam on this robot).
# Do NOT put generic "wrist" first: it can bind to the wrong device / index 0.
_WRIST_IMAGE_KEYS = (
    "observation.images.base_camera",
    "base_camera",
    "observation.images.wrist",
    "wrist",
    "images.wrist",
)


def log_robot_cameras(cfg, label: str = "robot") -> None:
    """Print which OpenCV index each configured camera uses."""
    cams = getattr(cfg, "cameras", None) or {}
    if not cams:
        print(f"[{label}] no cameras in robot config (LeRobot may default to index 0)", flush=True)
        return
    for name, cam in cams.items():
        idx = getattr(cam, "index_or_path", getattr(cam, "index", "?"))
        print(f"[{label}] camera {name!r} -> OpenCV index_or_path={idx!r}", flush=True)


def ramp_to_target_deg(
    robot,
    target_deg: np.ndarray,
    fps: int = 30,
    max_deg_per_step: float = 0.7,
    max_total_s: float = 25.0,
) -> bool:
    obs = robot.get_observation()
    q_cmd = np.array([float(obs[f"{n}.pos"]) for n in MOTOR_NAMES], dtype=float)
    period = 1.0 / fps
    start_t = time.time()
    while True:
        err = target_deg - q_cmd
        if np.max(np.abs(err)) < 0.5:
            return True
        if time.time() - start_t > max_total_s:
            return False
        step = np.clip(err, -max_deg_per_step, max_deg_per_step)
        q_cmd = q_cmd + step
        action = {f"{n}.pos": float(q_cmd[i]) for i, n in enumerate(MOTOR_NAMES)}
        t0 = time.time()
        robot.send_action(action)
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def _np_to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def _calibration_sources() -> dict:
    intr_src = "none"
    if (KIN_CONFIG_DIR / "camera_intrinsics.npz").exists():
        intr_src = str(KIN_CONFIG_DIR / "camera_intrinsics.npz")
    elif LEGACY_CALIB.exists():
        intr_src = str(LEGACY_CALIB)
    extr_src = "none"
    if (KIN_CONFIG_DIR / "camera_in_wrist.npy").exists():
        extr_src = str(KIN_CONFIG_DIR / "camera_in_wrist.npy")
    elif LEGACY_CALIB.exists():
        extr_src = str(LEGACY_CALIB)
    return {"intrinsics_source": intr_src, "extrinsics_source": extr_src}


def build_calibration_bundle(
    *,
    args: argparse.Namespace,
    kcfg: KinematicsConfig,
    mcfg: MotorToUrdfConfig,
    localizer: CubeLocalizer,
    home_deg: np.ndarray,
) -> dict:
    hsv_for_color = localizer.hsv_ranges.get(args.color)
    legacy_meta = {}
    if LEGACY_CALIB.exists():
        with open(LEGACY_CALIB) as f:
            legacy_meta = yaml.safe_load(f) or {}
    motor_offsets = {}
    if MOTOR_OFFSETS_PATH.exists():
        with open(MOTOR_OFFSETS_PATH) as f:
            motor_offsets = yaml.safe_load(f) or {}
    return {
        "session": {
            "port": args.port,
            "camera_index": args.camera_index,
            "camera_key_hint": args.camera_key,
            "home_pose": args.home_pose,
            "target_color": args.color,
            "n_obs_frames": args.n_obs_frames,
            "obs_hz": args.obs_hz,
        },
        "home_deg": home_deg.tolist(),
        "urdf_path": str(kcfg.urdf_path),
        "gripper_tip_offset_m": list(kcfg.gripper_tip_offset_m),
        "table_z_m": float(kcfg.table_z_m),
        "cube_half_height_m": float(kcfg.cube_half_height_m),
        "target_z_m": float(localizer.target_z),
        "camera": {
            "K": localizer.K.tolist(),
            "dist": localizer.dist.tolist(),
            "T_cam_in_wrist": localizer.T_cam_in_wrist.tolist(),
            **_calibration_sources(),
            "legacy_calibrated_from": legacy_meta.get("calibrated_from"),
            "legacy_mean_err_cm": legacy_meta.get("mean_err_cm"),
        },
        "motor_to_urdf": motor_offsets,
        "hsv_config_path": str(HSV_CONFIG_PATH) if HSV_CONFIG_PATH.exists() else None,
        "hsv_ranges_for_color": {
            str(args.color): [
                {"lo": list(lo), "hi": list(hi)}
                for lo, hi in (hsv_for_color or [])
            ],
        },
        "all_hsv_colors": sorted(localizer.hsv_ranges.keys()),
    }


def write_calibration_bundle(out_dir: Path, bundle: dict) -> None:
    path = out_dir / "calibrations.json"
    with open(path, "w") as f:
        json.dump(bundle, f, indent=2, default=_np_to_json)
  # snapshot copies for offline review
    for src_name, src in (
        ("camera_calibration.yaml", LEGACY_CALIB),
        ("camera_intrinsics.yaml", KIN_CONFIG_DIR / "camera_intrinsics.yaml"),
        ("camera_intrinsics.npz", KIN_CONFIG_DIR / "camera_intrinsics.npz"),
        ("motor_offsets.yaml", MOTOR_OFFSETS_PATH),
        ("hsv_config.yaml", HSV_CONFIG_PATH),
    ):
        if src.is_file():
            shutil.copy2(src, out_dir / src_name)


def extract_wrist_bgr(
    obs: dict,
    camera_key: str | None = None,
    *,
    camera_index: int | None = None,
) -> tuple[np.ndarray | None, str]:
    """Return (BGR uint8 image, key_used) or (None, reason)."""
    if camera_key:
        keys = (camera_key, *_WRIST_IMAGE_KEYS)
    else:
        keys = _WRIST_IMAGE_KEYS

    for key in keys:
        if key not in obs:
            continue
        img = obs[key]
        if img is None:
            continue
        if hasattr(img, "cpu"):
            img = img.cpu().numpy()
        arr = np.asarray(img)
        if arr.size == 0:
            continue
        if arr.dtype != np.uint8:
            arr = (
                (arr * 255).clip(0, 255).astype(np.uint8)
                if float(arr.max()) <= 1.0
                else arr.astype(np.uint8)
            )
        if arr.ndim == 2:
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[-1] == 4:
            bgr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if camera_index is not None:
            meta = f"key={key!r} index_cfg={camera_index}"
        else:
            meta = f"key={key!r}"
        return bgr, meta

    # Last resort: only base_camera-like keys (never random "wrist" stubs on index 0).
    for key, val in obs.items():
        if not isinstance(key, str):
            continue
        kl = key.lower()
        if "base_camera" not in kl and "images.base" not in kl:
            continue
        if hasattr(val, "shape") and len(getattr(val, "shape", ())) >= 2:
            bgr, _ = extract_wrist_bgr({key: val}, camera_key=key, camera_index=camera_index)
            if bgr is not None:
                return bgr, key
    sample = [k for k in obs if isinstance(k, str) and ("camera" in k.lower() or "image" in k.lower())][:20]
    return None, f"no image in obs (tried {list(keys)}; image-like keys: {sample})"


def detection_preview_bgr(
    bgr: np.ndarray,
    mask: np.ndarray | None,
    pixel: tuple[float, float] | None,
    *,
    label: str = "",
) -> np.ndarray:
    """BGR image with green mask + centroid for live preview."""
    import cv2 as _cv2

    disp = bgr.copy()
    if mask is not None and mask.size:
        green = np.zeros_like(disp)
        green[:, :, 1] = mask
        disp = _cv2.addWeighted(disp, 0.65, green, 0.35, 0)
    if pixel is not None:
        _cv2.circle(disp, (int(pixel[0]), int(pixel[1])), 12, (0, 0, 255), 2)
    if label:
        _cv2.putText(
            disp, label, (12, 28), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
    return disp


def preview_show_and_wait(
    window: str,
    bgr: np.ndarray,
    *,
    pause_ms: int = 2500,
    block_until_key: bool = False,
) -> None:
    """Show BGR frame; auto-advance after pause_ms unless block_until_key."""
    import cv2 as _cv2

    _cv2.imshow(window, bgr)
    if block_until_key or pause_ms <= 0:
        print(
            f"  [preview] Click the '{window}' window, then press any key ...",
            flush=True,
        )
        _cv2.waitKey(0)
    else:
        print(
            f"  [preview] '{window}' — continuing in {pause_ms} ms "
            f"(click window + key to advance now)",
            flush=True,
        )
        _cv2.waitKey(max(1, pause_ms))


def save_frame_bundle(
    out_path: Path,
    bgr: np.ndarray,
    mask: np.ndarray | None,
    pixel: tuple[float, float] | None,
    area: float,
) -> None:
    cv2.imwrite(str(out_path.with_suffix(".png")), bgr)
    if mask is not None:
        cv2.imwrite(str(out_path.parent / f"{out_path.name}_mask.png"), mask)
    overlay = bgr.copy()
    if mask is not None:
        green = np.zeros_like(overlay)
        green[:, :, 1] = mask
        overlay = cv2.addWeighted(overlay, 0.7, green, 0.3, 0)
    if pixel is not None:
        cv2.circle(overlay, (int(pixel[0]), int(pixel[1])), 8, (0, 0, 255), 2)
        cv2.putText(
            overlay, f"area={int(area)}", (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
    cv2.imwrite(str(out_path.parent / f"{out_path.name}_overlay.png"), overlay)


def capture_at_home(
    robot,
    localizer: CubeLocalizer,
    mcfg: MotorToUrdfConfig,
    fk: SO101FK,
    *,
    color: str,
    n_frames: int,
    period: float,
    trial_dir: Path,
    save_frames: bool,
    camera_key: str | None,
    camera_index: int | None = None,
    subdir: str = "home",
    show_preview: bool = False,
    read_bgr=None,
    map_only: bool = False,
) -> tuple[list[np.ndarray], list[dict]]:
    """Returns (successful base_xyz list in URDF, per-frame diagnostics).

    map_only: only HSV pixel + motor_deg (no backproj / base_xyz in frame log).
    """
    home_dir = trial_dir / subdir
    if save_frames:
        home_dir.mkdir(parents=True, exist_ok=True)

    cam_xyzs: list[np.ndarray] = []
    frame_log: list[dict] = []

    for fi in range(n_frames):
        t0 = time.time()
        obs = robot.get_observation()
        motor_deg = np.array([float(obs[f"{n}.pos"]) for n in MOTOR_NAMES], dtype=float)
        if read_bgr is not None:
            bgr = read_bgr()
            key_used = f"direct_opencv_index={camera_index}"
        else:
            bgr, key_used = extract_wrist_bgr(
                obs, camera_key=camera_key, camera_index=camera_index,
            )

        rec: dict = {
            "frame": fi,
            "image_key": key_used,
            "motor_deg": motor_deg.tolist(),
            "status": "pending",
        }

        if bgr is None:
            rec["status"] = "no_image"
            print(f"  [{fi}] {key_used}")
        else:
            pixel, mask, area = localizer.detect_pixel(bgr, color)
            det = None if map_only else localizer.locate(bgr, motor_deg, color)

            if save_frames:
                stem = home_dir / f"frame_{fi:02d}"
                save_frame_bundle(stem, bgr, mask, pixel, area)

            if pixel is None:
                rec["status"] = "no_pixel"
                rec["contour_area_px"] = float(area)
                if fi == 0 and area == 0:
                    print(
                        f"  [0] no {color!r} HSV match — cube in view? try --loosen_hsv or "
                        "toolset/perception/hsv_config.yaml",
                        flush=True,
                    )
            elif map_only:
                rec["status"] = "ok"
                rec["pixel"] = list(pixel)
                rec["contour_area_px"] = float(area)
            elif det is None:
                rec["pixel"] = list(pixel)
                rec["contour_area_px"] = float(area)
                urdf = mcfg.motor_to_urdf_rad(motor_deg)
                T_wrist = fk.fk(urdf, target="wrist")["T"]
                xyz = localizer.pixel_to_base(pixel, T_wrist)
                if xyz is not None:
                    rec["status"] = "ok"
                    rec["base_xyz_urdf"] = xyz.tolist()
                    cam_xyzs.append(xyz.copy())
                else:
                    rec["status"] = "no_backproj"
                    if fi == 0:
                        print(
                            "  [0] red pixel found but 3D backproj failed "
                            "(bad extrinsics / ray misses table?)",
                            flush=True,
                        )
            else:
                rec["status"] = "ok"
                rec["pixel"] = list(pixel)
                rec["contour_area_px"] = float(det.contour_area_px)
                rec["confidence"] = float(det.confidence)
                rec["base_xyz_urdf"] = det.base_xyz_m.tolist()
                cam_xyzs.append(det.base_xyz_m.copy())

            if bgr is not None and show_preview:
                import cv2 as _cv2

                label = f"{color} {rec['status']} area={int(area)}"
                disp = detection_preview_bgr(bgr, mask, pixel, label=label)
                _cv2.imshow("cv-probe wrist", disp)
                _cv2.waitKey(1)

            if fi == 0 and bgr is not None:
                mean_bgr = float(bgr.mean())
                if mean_bgr < 12.0:
                    print(
                        f"  [0] WARNING: frame nearly black (mean={mean_bgr:.1f}) — "
                        f"camera index={camera_index}; try --camera_fourcc MJPG, "
                        f"close other apps using the cam, or run with --camera_check",
                        flush=True,
                    )

            if fi == 0:
                shape_s = str(bgr.shape) if bgr is not None else "n/a"
                print(f"  [0] image_key={key_used!r} shape={shape_s} status={rec['status']}")
                if save_frames and bgr is not None:
                    meta_path = home_dir / "frame_00_camera_meta.txt"
                    meta_path.write_text(
                        f"image_key={key_used}\n"
                        f"opencv_index_config={camera_index}\n"
                        f"shape={bgr.shape}\n",
                        encoding="utf-8",
                    )

        frame_log.append(rec)
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)

    if save_frames:
        with open(home_dir / f"frames_{subdir}.json", "w") as f:
            json.dump(frame_log, f, indent=2)

    return cam_xyzs, frame_log


def write_session_report(out_dir: Path, session_meta: dict, color: str) -> None:
    map_only = bool(session_meta.get("map_only"))
    lines = [
        "probe_camera_vs_fk session report",
        f"out_dir: {out_dir}",
        f"mode: {'spatial_map (pixel+FK)' if map_only else 'compare cam vs FK'}",
        f"color: {color}",
        f"home_pose: {session_meta.get('home_pose')}",
        f"n_trials_planned: {session_meta.get('n_trials')}",
        f"n_trials_completed: {len(session_meta.get('trials', []))}",
        "",
        "Calibration files in this folder: calibrations.json + yaml snapshots.",
        "",
    ]
    trials = session_meta.get("trials", [])
    n_cam_ok = sum(1 for t in trials if t.get("frames_detected", 0) > 0)
    lines.append(f"Trials with >=1 camera detection: {n_cam_ok}/{len(trials)}")
    lines.append("")
    for t in trials:
        tri = t["trial"]
        fd = t.get("frames_detected", 0)
        fc = t.get("frames_captured", 0)
        st = t.get("home_frame_status_counts", {})
        if map_only:
            hp = t.get("home_pixel")
            off = t.get("pixel_offset_px")
            pix_s = f"pixel={hp} off={off}" if hp else "no_pixel"
            lines.append(
                f"Trial {tri:03d}: {pix_s}  "
                f"fk_user=({t['fk_xyz_user_m'][0]:+.3f}, {t['fk_xyz_user_m'][1]:+.3f}, "
                f"{t['fk_xyz_user_m'][2]:+.3f})"
            )
        else:
            lines.append(
                f"Trial {tri:03d}: cam_det={fd}/{fc}  "
                f"status_counts={st}  "
                f"fk_user=({t['fk_xyz_user_m'][0]:+.3f}, {t['fk_xyz_user_m'][1]:+.3f}, "
                f"{t['fk_xyz_user_m'][2]:+.3f})"
            )
        if fd == 0:
            lines.append(f"         -> inspect trial_{tri:03d}/home/frame_*_overlay.png for HSV")
    path = out_dir / "session_report.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM3")
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--camera_key", default="base_camera",
                   help="Observation key for RGB (default base_camera = wrist USB cam).")
    p.add_argument("--home_pose", default="eval1_rest")
    p.add_argument("--color", default="yellow")
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--n_obs_frames", type=int, default=10)
    p.add_argument("--obs_hz", type=float, default=5.0)
    p.add_argument("--save_frames", action="store_true",
                   help="Save every captured frame under trial_NNN/home/ and trial_NNN/grasp/.")
    p.add_argument("--no_initial_ramp", action="store_true")
    p.add_argument("--fk_target", default="gripper_tip",
                   choices=["gripper_tip", "gripper_frame", "wrist", "wrist_roll"],
                   help="FK frame recorded at grasp (default gripper_tip = centered on cube).")
    p.add_argument("--grasp_reference", default="tip_centered",
                   choices=["tip_centered", "side_ready", "gripper_frame", "wrist"],
                   help="tip_centered: symmetric jaw on cube center. side_ready: offset grasp — "
                        "uses gripper_frame FK (one jaw opens to the side).")
    p.add_argument("--map_only", action="store_true",
                   help="Mapping mode: save pixel + FK only (no 3D backproj in frames/CSV).")
    p.add_argument("--loosen_hsv", action="store_true",
                   help="Widen S/V lower bounds for --color (dim light / yellow).")
    p.add_argument("--k", type=int, default=3,
                   help="kNN neighbors for live map estimate each trial (map_only).")
    p.add_argument("--session_dir", type=Path, default=None,
                   help="Append trials to an existing probe_* session (resume).")
    p.add_argument("--start_trial", type=int, default=None,
                   help="First trial index when resuming (default: next after last in session.json).")
    args = p.parse_args()

    fk_target = args.fk_target
    if args.grasp_reference == "side_ready":
        fk_target = "gripper_frame"
    elif args.grasp_reference == "gripper_frame":
        fk_target = "gripper_frame"
    elif args.grasp_reference == "wrist":
        fk_target = "wrist"

    resume = args.session_dir is not None
    if resume:
        out_dir = args.session_dir.resolve()
        if not out_dir.is_dir():
            raise SystemExit(f"--session_dir not found: {out_dir}")
        session_path = out_dir / "session.json"
        if not session_path.is_file():
            raise SystemExit(f"No session.json in {out_dir}")
        session_meta = json.loads(session_path.read_text(encoding="utf-8"))
        session_meta["resumed_at"] = time.time()
        session_meta["n_trials"] = int(session_meta.get("n_trials", 0)) + args.n_trials
        existing_nums = [int(t["trial"]) for t in session_meta.get("trials", [])]
        start_trial = args.start_trial if args.start_trial is not None else (
            max(existing_nums) + 1 if existing_nums else 0
        )
        trial_indices = list(range(start_trial, start_trial + args.n_trials))
        print(f"[probe] RESUME {out_dir}  trials {trial_indices[0]:03d}..{trial_indices[-1]:03d} "
              f"({len(trial_indices)} new)")
    else:
        out_dir = DEPLOY / "_snaps" / f"probe_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)
        start_trial = 0
        trial_indices = list(range(args.n_trials))
        session_meta = {
            "mode": "space_map" if args.map_only else "probe_cam_fk",
            "out_dir": str(out_dir),
            "home_pose": args.home_pose,
            "color": args.color,
            "n_trials": args.n_trials,
            "n_obs_frames": args.n_obs_frames,
            "obs_hz": args.obs_hz,
            "port": args.port,
            "camera_index": args.camera_index,
            "camera_key": args.camera_key,
            "fk_target": fk_target,
            "grasp_reference": args.grasp_reference,
            "map_only": args.map_only,
            "started_at": time.time(),
            "trials": [],
        }

    cx_img = WRIST_CAM_WIDTH / 2.0
    cy_img = WRIST_CAM_HEIGHT / 2.0

    csv_path = out_dir / "trials.csv"
    csv_mode = "a" if resume and csv_path.is_file() else "w"
    csv_f = open(csv_path, csv_mode, newline="")
    csv_w = csv.writer(csv_f)
    if csv_mode == "w":
        if args.map_only:
            csv_w.writerow([
                "trial", "color", "n_pixel_frames", "pixel_u", "pixel_v", "du_px", "dv_px",
                "fk_x_user", "fk_y_user", "fk_z_user",
                "home_no_image", "home_no_pixel", "home_ok",
                *[f"home_{n}" for n in MOTOR_NAMES],
                *[f"grasp_{n}" for n in MOTOR_NAMES],
            ])
        else:
            csv_w.writerow([
                "trial", "color", "frames_captured", "frames_detected",
                "cam_x_user", "cam_y_user", "cam_z_user",
                "fk_x_user", "fk_y_user", "fk_z_user",
                "err_x_mm", "err_y_mm", "err_z_mm", "err_norm_mm",
                "home_no_image", "home_no_pixel", "home_no_backproj", "home_ok",
                *MOTOR_NAMES,
            ])
    map_csv_path = out_dir / "mapping_samples.csv"
    map_csv_f = None
    map_csv_w = None
    if args.map_only:
        map_mode = "a" if resume and map_csv_path.is_file() else "w"
        map_csv_f = open(map_csv_path, map_mode, newline="")
        map_csv_w = csv.writer(map_csv_f)
        if map_mode == "w":
            map_csv_w.writerow([
                "trial", "color", "pixel_u", "pixel_v", "du_px", "dv_px",
                "fk_x_user", "fk_y_user", "fk_z_user",
                *[f"home_{n}" for n in MOTOR_NAMES],
                *[f"grasp_{n}" for n in MOTOR_NAMES],
            ])
    print(f"[probe] session -> {out_dir}")
    if args.map_only:
        print("[probe] map_only: pixel + grasp FK only (no 3D backproj; offline: probe_space_map.py)")
        print("[probe] each trial: home photo -> MAP estimate -> grasp FK -> est vs meas -> mapping_samples.csv")

    validation_csv_path = out_dir / "trial_validation.csv"
    validation_csv_f = None
    validation_csv_w = None
    if args.map_only:
        val_mode = "a" if resume and validation_csv_path.is_file() else "w"
        validation_csv_f = open(validation_csv_path, val_mode, newline="", encoding="utf-8")
        validation_csv_w = csv.DictWriter(validation_csv_f, fieldnames=[
            "trial", "du_px", "dv_px", "est_x", "est_y", "est_z",
            "meas_x", "meas_y", "meas_z", "err_est_meas_xy_mm", "err_est_meas_z_mm",
            "in_map_hull", "nearest_prior_trial", "nearest_prior_px", "neighbor_trials",
        ])
        if val_mode == "w":
            validation_csv_w.writeheader()

    errors_xy_mm: list[float] = []

    home_deg = get_home_deg(args.home_pose)
    print(f"[probe] home {args.home_pose!r}: {home_deg.round(1).tolist()}")

    cfg = make_so101_follower_config(
        args.port,
        cameras={"base_camera": OpenCVCameraConfig(
            index_or_path=args.camera_index, fps=30,
            width=WRIST_CAM_WIDTH, height=WRIST_CAM_HEIGHT)},
        calibration_dir=LOCAL_CALIBRATION_DIR,
        use_degrees=True,
    )
    log_robot_cameras(cfg, "probe")
    print(f"[probe] observation key {args.camera_key!r} @ OpenCV index {args.camera_index}",
          flush=True)
    robot = make_robot_from_config(cfg)
    print(f"[probe] connecting {args.port} ...")
    robot.connect()

    kcfg = KinematicsConfig.load()
    mcfg = MotorToUrdfConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)
    hsv_ranges = load_hsv_ranges(HSV_CONFIG_PATH)
    if args.loosen_hsv:
        hsv_ranges = loosen_hsv(hsv_ranges, args.color)
        print(f"[probe] loosen_hsv enabled for {args.color!r}", flush=True)
    localizer = CubeLocalizer(kcfg=kcfg, mcfg=mcfg, fk=fk, hsv_ranges=hsv_ranges)

    calib_path = out_dir / "calibrations.json"
    if not (resume and calib_path.is_file()):
        calib_bundle = build_calibration_bundle(
            args=args, kcfg=kcfg, mcfg=mcfg, localizer=localizer, home_deg=home_deg,
        )
        write_calibration_bundle(out_dir, calib_bundle)
        print(f"[probe] calibrations -> {calib_path}")
    else:
        print(f"[probe] reusing {calib_path}")

    # Log observation keys once
    obs0 = robot.get_observation()
    img_keys = [k for k in obs0 if isinstance(k, str) and (
        "wrist" in k.lower() or "camera" in k.lower() or "image" in k.lower()
    )]
    session_meta["observation_image_keys"] = img_keys
    print(f"[probe] observation image-like keys: {img_keys}")

    try:
        if not args.no_initial_ramp:
            print("[probe] initial ramp to home ...")
            ramp_to_target_deg(robot, home_deg)
            time.sleep(0.5)

        period = 1.0 / max(0.1, args.obs_hz)

        n_new = len(trial_indices)
        for i_t, trial in enumerate(trial_indices):
            trial_dir = out_dir / f"trial_{trial:03d}"
            print(f"\n========== Trial {trial:03d}  ({i_t + 1}/{n_new} new) ==========")
            input("Place the cube, then press Enter. ")

            print("[ramp] home ...")
            ramp_to_target_deg(robot, home_deg)
            time.sleep(0.5)

            print(f"[cam] {args.n_obs_frames} frames -> {trial_dir / 'home' if args.save_frames else '(not saving)'}")
            cam_xyzs, frame_log = capture_at_home(
                robot, localizer, mcfg, fk,
                color=args.color,
                n_frames=args.n_obs_frames,
                period=period,
                trial_dir=trial_dir,
                save_frames=args.save_frames,
                camera_key=args.camera_key,
                camera_index=args.camera_index,
                map_only=args.map_only,
            )

            status_counts: dict[str, int] = {}
            for fr in frame_log:
                st = fr.get("status", "?")
                status_counts[st] = status_counts.get(st, 0) + 1
            frames_captured = sum(
                1 for fr in frame_log if fr.get("status") not in ("no_image", "pending")
            )

            pixels_ok = [
                np.asarray(fr["pixel"], dtype=float)
                for fr in frame_log if fr.get("pixel") is not None
            ]
            home_pixel = (
                np.median(np.stack(pixels_ok), axis=0) if pixels_ok else None
            )

            if args.map_only:
                if home_pixel is not None:
                    print(
                        f"[map] pixel median ({home_pixel[0]:.0f},{home_pixel[1]:.0f}) "
                        f"offset ({home_pixel[0]-cx_img:+.0f},{home_pixel[1]-cy_img:+.0f}) px",
                    )
                else:
                    print(f"[map] WARNING: no pixel (counts={status_counts})")
                cam_x_user = cam_y_user = cam_z_user = float("nan")
            elif not cam_xyzs:
                print(f"[cam] WARNING: {args.color!r} not detected "
                      f"(counts={status_counts}).")
                cam_x_user = cam_y_user = cam_z_user = float("nan")
            else:
                cam_med = np.median(np.stack(cam_xyzs), axis=0)
                cam_x_user = float(-cam_med[1])
                cam_y_user = float(+cam_med[0])
                cam_z_user = float(+cam_med[2])
                print(f"[cam] median n={len(cam_xyzs)} -> "
                      f"user ({cam_x_user:+.3f}, {cam_y_user:+.3f}, {cam_z_user:+.3f}) m")

            est_info = None
            pu = pv = du = dv = float("nan")
            if args.map_only and home_pixel is not None:
                pu, pv = float(home_pixel[0]), float(home_pixel[1])
                du, dv = pu - cx_img, pv - cy_img
                prior_samples = load_samples_from_mapping_csv(
                    map_csv_path, session_dir=out_dir,
                )
                est_info = print_pre_grasp_block(
                    trial=trial,
                    du=du,
                    dv=dv,
                    pu=pu,
                    pv=pv,
                    samples=prior_samples,
                    k=args.k,
                )
            elif args.map_only:
                print("\n[map] No pixel at home — skip map estimate; fix detect before trusting row.")

            print("\n>>> GRASP PROBE — Enter -> torque off -> side grasp on cube -> Enter.")
            if est_info is not None:
                print(">>> (measured FK will be compared to MAP estimate above)")
            input(">>> Support arm, Enter to disable torque. ")
            robot.bus.disable_torque()
            input(">>> Perfect grasp on cube center, Enter to record. ")

            obs = robot.get_observation()
            motor_deg = np.array([float(obs[f"{n}.pos"]) for n in MOTOR_NAMES], dtype=float)
            urdf_rad = mcfg.motor_to_urdf_rad(motor_deg)
            fk_urdf = fk.fk(urdf_rad, target=fk_target)["position"]
            fk_x_user = float(-fk_urdf[1])
            fk_y_user = float(+fk_urdf[0])
            fk_z_user = float(+fk_urdf[2])
            print(f"[fk] user ({fk_x_user:+.3f}, {fk_y_user:+.3f}, {fk_z_user:+.3f}) m")

            measured_fk = np.array([fk_x_user, fk_y_user, fk_z_user], dtype=float)
            err_est_xy = float("nan")
            if args.map_only and home_pixel is not None:
                err_est_xy = print_post_grasp_block(
                    trial=trial,
                    est_info=est_info,
                    measured=measured_fk,
                    errors_xy_mm=errors_xy_mm,
                )
                if np.isfinite(err_est_xy):
                    errors_xy_mm.append(err_est_xy)
                if validation_csv_w is not None:
                    est = est_info["est_xyz_user_m"] if est_info else [np.nan] * 3
                    validation_csv_w.writerow({
                        "trial": trial,
                        "du_px": f"{du:+.1f}",
                        "dv_px": f"{dv:+.1f}",
                        "est_x": f"{est[0]:+.4f}" if est_info else "nan",
                        "est_y": f"{est[1]:+.4f}" if est_info else "nan",
                        "est_z": f"{est[2]:+.4f}" if est_info else "nan",
                        "meas_x": f"{fk_x_user:+.4f}",
                        "meas_y": f"{fk_y_user:+.4f}",
                        "meas_z": f"{fk_z_user:+.4f}",
                        "err_est_meas_xy_mm": f"{err_est_xy:.1f}" if np.isfinite(err_est_xy) else "nan",
                        "err_est_meas_z_mm": (
                            f"{(est[2]-fk_z_user)*1000:+.1f}" if est_info else "nan"
                        ),
                        "in_map_hull": est_info.get("in_map_hull") if est_info else "",
                        "nearest_prior_trial": est_info.get("nearest_train_trial") if est_info else "",
                        "nearest_prior_px": (
                            f"{est_info['nearest_train_dist_px']:.1f}" if est_info else ""
                        ),
                        "neighbor_trials": str(est_info.get("neighbor_trials", "")) if est_info else "",
                    })
                    validation_csv_f.flush()
                (trial_dir / "trial_validation.json").write_text(
                    json.dumps({
                        "trial": trial,
                        "du_px": du, "dv_px": dv,
                        "est_info": est_info,
                        "measured_fk_xyz_user_m": measured_fk.tolist(),
                        "err_est_meas_xy_mm": err_est_xy,
                    }, indent=2, default=_np_to_json),
                    encoding="utf-8",
                )

            grasp_dir = trial_dir / "grasp"
            if args.save_frames:
                grasp_dir.mkdir(parents=True, exist_ok=True)
                bgr, key_used = extract_wrist_bgr(
                    obs, camera_key=args.camera_key, camera_index=args.camera_index,
                )
                if bgr is not None:
                    px, mask, area = localizer.detect_pixel(bgr, args.color)
                    save_frame_bundle(grasp_dir / "frame", bgr, mask, px, area)
                    with open(grasp_dir / "frame_meta.json", "w") as gf:
                        json.dump({
                            "image_key": key_used,
                            "motor_deg": motor_deg.tolist(),
                            "urdf_rad": urdf_rad.tolist(),
                            "fk_xyz_user_m": [fk_x_user, fk_y_user, fk_z_user],
                            "fk_target": fk_target,
                            "grasp_reference": args.grasp_reference,
                            "detect_pixel": list(px) if px else None,
                        }, gf, indent=2)

            home_motors = [
                np.asarray(fr["motor_deg"], dtype=float)
                for fr in frame_log
                if fr.get("motor_deg") is not None
            ]
            home_motor_deg = (
                np.median(np.stack(home_motors), axis=0).tolist()
                if home_motors else None
            )

            err_x = err_y = err_z = err_norm = float("nan")
            if not args.map_only and cam_xyzs:
                err_x = (cam_x_user - fk_x_user) * 1000
                err_y = (cam_y_user - fk_y_user) * 1000
                err_z = (cam_z_user - fk_z_user) * 1000
                err_norm = float(np.linalg.norm([err_x, err_y, err_z]))
                print(f"[err] mm: x={err_x:+.1f} y={err_y:+.1f} z={err_z:+.1f} ||={err_norm:.1f}")

            if args.map_only:
                if home_pixel is None:
                    pu = pv = du = dv = float("nan")
                csv_w.writerow([
                    trial, args.color, len(pixels_ok), f"{pu:.1f}", f"{pv:.1f}",
                    f"{du:+.1f}", f"{dv:+.1f}",
                    f"{fk_x_user:+.4f}", f"{fk_y_user:+.4f}", f"{fk_z_user:+.4f}",
                    status_counts.get("no_image", 0),
                    status_counts.get("no_pixel", 0),
                    status_counts.get("ok", 0),
                    *[f"{v:.2f}" for v in (home_motor_deg or [np.nan]*6)],
                    *[f"{v:.2f}" for v in motor_deg],
                ])
                if map_csv_w is not None and home_pixel is not None and home_motor_deg:
                    map_csv_w.writerow([
                        trial, args.color, f"{pu:.1f}", f"{pv:.1f}", f"{du:+.1f}", f"{dv:+.1f}",
                        f"{fk_x_user:+.4f}", f"{fk_y_user:+.4f}", f"{fk_z_user:+.4f}",
                        *[f"{v:.2f}" for v in home_motor_deg],
                        *[f"{v:.2f}" for v in motor_deg],
                    ])
            else:
                csv_w.writerow([
                    trial, args.color, frames_captured, len(cam_xyzs),
                    f"{cam_x_user:+.4f}", f"{cam_y_user:+.4f}", f"{cam_z_user:+.4f}",
                    f"{fk_x_user:+.4f}", f"{fk_y_user:+.4f}", f"{fk_z_user:+.4f}",
                    f"{err_x:+.1f}" if cam_xyzs else "nan",
                    f"{err_y:+.1f}" if cam_xyzs else "nan",
                    f"{err_z:+.1f}" if cam_xyzs else "nan",
                    f"{err_norm:.1f}" if cam_xyzs else "nan",
                    status_counts.get("no_image", 0),
                    status_counts.get("no_pixel", 0),
                    status_counts.get("no_backproj", 0),
                    status_counts.get("ok", 0),
                    *[f"{v:.2f}" for v in motor_deg],
                ])
            csv_f.flush()
            if map_csv_f is not None:
                map_csv_f.flush()

            trial_rec = {
                "trial": trial,
                "map_est_preview": est_info if args.map_only else None,
                "err_est_meas_xy_mm": err_est_xy if args.map_only else None,
                "trial_dir": str(trial_dir.relative_to(out_dir)),
                "frames_captured": frames_captured,
                "frames_detected": len(pixels_ok) if args.map_only else len(cam_xyzs),
                "home_frame_status_counts": status_counts,
                "home_motor_deg": home_motor_deg,
                "home_pixel": home_pixel.tolist() if home_pixel is not None else None,
                "pixel_offset_px": (
                    [float(home_pixel[0]) - cx_img, float(home_pixel[1]) - cy_img]
                    if home_pixel is not None else None
                ),
                "fk_xyz_user_m": [fk_x_user, fk_y_user, fk_z_user],
                "fk_target": fk_target,
                "grasp_reference": args.grasp_reference,
                "grasp_motor_deg": motor_deg.tolist(),
                "urdf_rad": urdf_rad.tolist(),
                "timestamp": time.time(),
            }
            if not args.map_only:
                trial_rec["cam_xyz_user_m"] = [cam_x_user, cam_y_user, cam_z_user]
                trial_rec["err_mm"] = [err_x, err_y, err_z] if cam_xyzs else None
                trial_rec["err_norm_mm"] = err_norm if cam_xyzs else None
            session_meta["trials"].append(trial_rec)
            with open(out_dir / "session.json", "w") as mf:
                json.dump(session_meta, mf, indent=2, default=_np_to_json)
            write_session_report(out_dir, session_meta, args.color)

            try:
                robot.bus.enable_torque()
            except Exception as e:
                print(f"[robot] enable torque: {e}")
            ramp_to_target_deg(robot, home_deg)

    except KeyboardInterrupt:
        print("\n[probe] interrupted.")
    finally:
        csv_f.close()
        if map_csv_f is not None:
            map_csv_f.close()
        if validation_csv_f is not None:
            validation_csv_f.close()
        write_session_report(out_dir, session_meta, args.color)
        with open(out_dir / "session.json", "w") as mf:
            json.dump(session_meta, mf, indent=2, default=_np_to_json)
        try:
            robot.bus.enable_torque()
        except Exception:
            pass
        robot.disconnect()
        done_files = "trials.csv"
        if args.map_only:
            done_files += "\n  mapping_samples.csv"
        print(f"\n[probe] done.\n  {out_dir}\n  {done_files}\n  calibrations.json\n  session_report.txt")

        rows = list(csv.DictReader(open(csv_path)))
        if not args.map_only:
            errs = [
                float(r["err_norm_mm"]) for r in rows
                if r.get("err_norm_mm") not in ("", "nan", "NaN")
            ]
            if errs:
                import statistics as st
                print(f"\n=== {len(errs)} trials with camera+FK ({args.color}) ===")
                print(f"  ||cam-fk|| mean={st.mean(errs):.1f}  median={st.median(errs):.1f} mm")
        else:
            n_px = sum(
                1 for r in rows
                if r.get("pixel_u") and str(r.get("pixel_u", "")).lower() != "nan"
            )
            n_map = 0
            map_path = out_dir / "mapping_samples.csv"
            if map_path.is_file():
                n_map = sum(1 for _ in csv.DictReader(open(map_path)))
            print(f"\n=== space_map: {n_px}/{len(rows)} trials with pixel "
                  f"({n_map} in mapping_samples.csv) ===")
            if errors_xy_mm:
                import statistics as st
                print(f"  est vs measured xy: n={len(errors_xy_mm)}  "
                      f"median={st.median(errors_xy_mm):.1f} mm  "
                      f"max={max(errors_xy_mm):.1f} mm")
                print(f"  see {out_dir / 'trial_validation.csv'}")
            print("  Offline: python -m toolset.perception.probe_map_data "
                  f"--session_dir {out_dir}")
            print("  HTML: python -m toolset.perception.probe_map_visual_report "
                  f"--map_session_dir {out_dir}")


if __name__ == "__main__":
    main()
