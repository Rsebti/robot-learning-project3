"""
cube_localization.py - locate cubes in robot base frame from wrist-cam
frames using URDF FK + chessboard-calibrated camera + hand-eye extrinsics.

Pipeline per frame:
  1. motor_deg -> URDF rad via motor_to_urdf
  2. URDF rad -> T_wrist_in_base via FK (target="wrist")
  3. T_cam_in_base = T_wrist_in_base @ T_cam_in_wrist (loaded from YAML)
  4. HSV mask for target color in image
  5. Largest contour centroid -> 2D pixel
  6. Undistort pixel using camera K + dist
  7. Backproject pixel ray, intersect with table plane
       (z_target = TABLE_Z + CUBE_HALF_HEIGHT)
  8. Return cube position in base frame, or None on failure

The per-frame estimate jitters at the few-mm level; for static scenes,
use `locate_cube_in_episode` which medianizes over a window of frames.

Compatibility shim: when the new calibration files don't exist yet, this
module falls back to the legacy `configs/camera_calibration.yaml` so old
deploy scripts keep working.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from toolset.kinematics.config import KinematicsConfig
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig
from toolset.kinematics.urdf_fk import SO101FK

PROJECT_ROOT = Path(__file__).parent.parent.parent
KIN_CONFIG_DIR = PROJECT_ROOT / "toolset" / "configs" / "kinematics"
LEGACY_CALIB = PROJECT_ROOT / "toolset" / "configs" / "camera_calibration.yaml"
HSV_CONFIG_PATH = PROJECT_ROOT / "toolset" / "perception" / "hsv_config.yaml"

# Deploy stream size (640x480 matches manual OpenCV color scripts on this robot).
# Intrinsics in camera_intrinsics.npz are scaled from native calib resolution if needed.
WRIST_CAM_WIDTH = 640
WRIST_CAM_HEIGHT = 480


# HSV ranges expressed in OpenCV's [0..179, 0..255, 0..255] convention. We
# load these from hsv_config.yaml at runtime so they can be tuned without
# touching code. The DEFAULT_HSV below mirrors the values that the eval-2
# deploy used.
DEFAULT_HSV: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "yellow": [((22, 100, 100), (35, 255, 255))],
    "orange": [((4, 120, 80), (22, 255, 255))],
    "red":    [((0, 120, 60), (4, 255, 255)),
               ((172, 120, 60), (180, 255, 255))],
    "blue":   [((100, 120, 60), (130, 255, 255))],
    "green":  [((40, 80, 60), (85, 255, 255))],
    "violet": [((130, 80, 60), (170, 255, 255))],
}


def _load_hsv() -> dict:
    if HSV_CONFIG_PATH.exists():
        with open(HSV_CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
        ranges = {}
        for color, rs in data.items():
            ranges[color] = [
                (tuple(lo), tuple(hi)) for lo, hi in rs
            ]
        return ranges
    return DEFAULT_HSV


def _intrinsics_source_label() -> str:
    new = KIN_CONFIG_DIR / "camera_intrinsics.npz"
    if new.exists():
        return str(new)
    if LEGACY_CALIB.exists():
        return str(LEGACY_CALIB)
    return "none"


def load_intrinsics_image_size() -> tuple[int, int] | None:
    """Return (width, height) from camera_intrinsics.npz, or None if missing."""
    new = KIN_CONFIG_DIR / "camera_intrinsics.npz"
    if not new.exists():
        return None
    d = np.load(new)
    if "image_size" not in d:
        return None
    w, h = np.asarray(d["image_size"], dtype=int).reshape(-1)[:2]
    return int(w), int(h)


def log_intrinsics_camera_size_match(tag: str = "cv") -> None:
    """Log whether npz image_size matches WRIST_CAM_WIDTH/HEIGHT deploy config."""
    npz_size = load_intrinsics_image_size()
    cam = (WRIST_CAM_WIDTH, WRIST_CAM_HEIGHT)
    if npz_size is None:
        print(f"[{tag}] WARN: no image_size in camera_intrinsics.npz", flush=True)
    elif npz_size != cam:
        print(
            f"[{tag}] intrinsics scaled {npz_size} -> {cam[0]}x{cam[1]} deploy stream",
            flush=True,
        )
    else:
        print(
            f"[{tag}] image_size OK: {cam[0]}x{cam[1]} (npz matches camera config)",
            flush=True,
        )


def _scale_K_to_stream(K: np.ndarray, native_wh: tuple[int, int]) -> np.ndarray:
    """Scale intrinsics K from calibration resolution to WRIST_CAM_WIDTH/HEIGHT."""
    nw, nh = native_wh
    tw, th = WRIST_CAM_WIDTH, WRIST_CAM_HEIGHT
    if (nw, nh) == (tw, th):
        return K
    sx, sy = tw / nw, th / nh
    Ks = K.copy()
    Ks[0, 0] *= sx
    Ks[1, 1] *= sy
    Ks[0, 2] *= sx
    Ks[1, 2] *= sy
    return Ks


def _load_intrinsics() -> tuple[np.ndarray, np.ndarray] | None:
    """Try new (chessboard) calibration first, then legacy. Return (K, dist) or None."""
    new = KIN_CONFIG_DIR / "camera_intrinsics.npz"
    if new.exists():
        d = np.load(new)
        K = np.asarray(d["K"], dtype=float)
        dist = np.asarray(d["dist"], dtype=float).reshape(-1)
        native = load_intrinsics_image_size()
        if native is not None:
            K = _scale_K_to_stream(K, native)
        return K, dist
    if LEGACY_CALIB.exists():
        with open(LEGACY_CALIB) as f:
            d = yaml.safe_load(f) or {}
        K = np.array(d["K"], dtype=float)
        return K, np.zeros(5, dtype=float)
    return None


def _load_T_cam_in_wrist() -> np.ndarray | None:
    new = KIN_CONFIG_DIR / "camera_in_wrist.npy"
    if new.exists():
        return np.load(new)
    if LEGACY_CALIB.exists():
        with open(LEGACY_CALIB) as f:
            d = yaml.safe_load(f) or {}
        if "T_cam_in_wrist" in d:
            return np.array(d["T_cam_in_wrist"], dtype=float)
    return None


@dataclass
class CubeDetection:
    color: str
    pixel: tuple[float, float]
    contour_area_px: float
    base_xyz_m: np.ndarray
    confidence: float        # in [0, 1]; based on contour area / image size


class CubeLocalizer:
    def __init__(
        self,
        kcfg: KinematicsConfig | None = None,
        mcfg: MotorToUrdfConfig | None = None,
        fk: SO101FK | None = None,
        hsv_ranges: dict | None = None,
    ):
        self.kcfg = kcfg or KinematicsConfig.load()
        self.mcfg = mcfg or MotorToUrdfConfig.load()
        self.fk = fk or SO101FK(
            self.kcfg.urdf_path,
            gripper_tip_offset=self.kcfg.gripper_tip_offset_m,
        )
        intr = _load_intrinsics()
        if intr is None:
            raise FileNotFoundError(
                "No camera intrinsics found. Run "
                "calibration_scripts/calibrate_camera_chessboard.py or "
                "toolset.calibration_scripts.import_checkerboard_intrinsics, or "
                "ensure legacy toolset/configs/camera_calibration.yaml exists."
            )
        self.K, self.dist = intr
        self.intrinsics_source = _intrinsics_source_label()
        T = _load_T_cam_in_wrist()
        if T is None:
            raise FileNotFoundError(
                "No T_cam_in_wrist found. Run calibrate_hand_eye.py "
                "or ensure legacy camera_calibration.yaml has T_cam_in_wrist."
            )
        self.T_cam_in_wrist = T
        self.hsv_ranges = hsv_ranges or _load_hsv()
        self.target_z = self.kcfg.table_z_m + self.kcfg.cube_half_height_m

    # ---- pure HSV pixel detection -------------------------------------
    def detect_pixel(
        self, bgr: np.ndarray, color: str,
    ) -> tuple[tuple[float, float] | None, np.ndarray | None, float]:
        if color not in self.hsv_ranges:
            raise KeyError(f"Unknown color '{color}'; known: {list(self.hsv_ranges)}")
        from toolset.perception.color_mask import mask_color_blob

        px, mask, area = mask_color_blob(
            bgr, self.hsv_ranges[color],
            bright_only=True,
            max_area_frac=0.02,
        )
        if px is None:
            return None, mask, area
        return px, mask, area

    # ---- pixel + camera pose -> base xyz ------------------------------
    def pixel_to_base(
        self,
        pixel: tuple[float, float],
        T_wrist_in_base: np.ndarray,
        z_target: float | None = None,
    ) -> np.ndarray | None:
        if z_target is None:
            z_target = self.target_z
        # Undistort pixel
        pt = np.array([[[float(pixel[0]), float(pixel[1])]]], dtype=np.float32)
        pt_u = cv2.undistortPoints(pt, self.K, self.dist, P=self.K)
        px_u, py_u = float(pt_u[0, 0, 0]), float(pt_u[0, 0, 1])
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        ray_cam = np.array([(px_u - cx) / fx, (py_u - cy) / fy, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)

        # Camera pose in base frame
        T_cam_in_base = T_wrist_in_base @ self.T_cam_in_wrist
        cam_origin = T_cam_in_base[:3, 3]
        ray_base = T_cam_in_base[:3, :3] @ ray_cam

        if abs(ray_base[2]) < 1e-8:
            return None
        t = (z_target - cam_origin[2]) / ray_base[2]
        if t < 0:
            return None  # behind the camera
        return cam_origin + t * ray_base

    # ---- one-shot: image + motor angles -> cube xyz -------------------
    def locate(
        self,
        bgr_image: np.ndarray,
        motor_deg: np.ndarray,
        color: str,
    ) -> CubeDetection | None:
        pixel, mask, area = self.detect_pixel(bgr_image, color)
        if pixel is None:
            return None
        urdf = self.mcfg.motor_to_urdf_rad(motor_deg)
        T_wrist = self.fk.fk(urdf, target="wrist")["T"]
        xyz_base = self.pixel_to_base(pixel, T_wrist)
        if xyz_base is None:
            return None
        h, w = bgr_image.shape[:2]
        conf = float(np.clip(area / (h * w * 0.02), 0.0, 1.0))
        return CubeDetection(
            color=color, pixel=pixel, contour_area_px=area,
            base_xyz_m=xyz_base, confidence=conf,
        )

    # ---- multi-frame median for static scenes -------------------------
    def locate_in_episode(
        self,
        frames: list[dict],
        color: str,
        max_frames: int | None = 60,
    ) -> tuple[np.ndarray, list[CubeDetection]] | None:
        """frames: list of {"bgr": ndarray, "motor_deg": [..6..]}.

        Returns (median_xyz, per_frame_detections) or None if too few
        successful detections.
        """
        dets: list[CubeDetection] = []
        for fr in frames[: max_frames or len(frames)]:
            d = self.locate(fr["bgr"], np.asarray(fr["motor_deg"]), color)
            if d is not None:
                dets.append(d)
        if len(dets) < 5:
            return None
        xs = np.stack([d.base_xyz_m for d in dets])
        return np.median(xs, axis=0), dets
