"""
refined_cube_detect.py — bright-top HSV + wall-line square fit (probe pipeline).

Same chain as probe_remask_session → probe_square_postprocess:
  learned_hsv (session) → mask_bright_top → fit_square_to_mask → centroid on kept_mask.

Use for live approach and for rebuilding mapping pixels from saved probe frames.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from toolset.perception.color_mask import mask_bright_top
from toolset.perception.cube_localization import DEFAULT_HSV, _load_hsv
from toolset.perception.estimate_cube_xy import load_hsv_ranges
from toolset.perception.mask_square_fit import SquareFitResult, fit_square_to_mask
from toolset.perception.probe_space_map import WRIST_CAM_HEIGHT, WRIST_CAM_WIDTH


@dataclass
class RefinedDetection:
    pixel: tuple[float, float] | None
    bright_mask: np.ndarray | None
    final_mask: np.ndarray | None
    area_px: float
    fit: SquareFitResult | None
    method: str


def load_session_hsv(session_dir: Path | None) -> dict | None:
    if session_dir is None:
        return None
    path = session_dir / "learned_hsv.yaml"
    return load_hsv_ranges(path)


def hsv_ranges_for(
    color: str,
    *,
    session_dir: Path | None = None,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    session = load_session_hsv(session_dir)
    if session and color in session:
        return session[color]
    global_hsv = _load_hsv()
    if color in global_hsv:
        return global_hsv[color]
    if color in DEFAULT_HSV:
        return DEFAULT_HSV[color]
    raise KeyError(f"Unknown color '{color}'")


def detect_refined_pixel(
    bgr: np.ndarray,
    color: str,
    *,
    session_dir: Path | None = None,
    hsv_ranges: list[tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
    v_floor: int | None = None,
    seed_xy: tuple[float, float] | None = None,
) -> RefinedDetection:
    """Bright-top mask + wall/color square refinement; centroid on kept_mask."""
    ranges = hsv_ranges if hsv_ranges is not None else hsv_ranges_for(color, session_dir=session_dir)
    px, bright_mask, area = mask_bright_top(
        bgr, ranges, v_floor=v_floor, seed_xy=seed_xy,
    )
    if bright_mask is None or not np.any(bright_mask):
        return RefinedDetection(None, bright_mask, None, 0.0, None, "no_bright_mask")

    fit = fit_square_to_mask(bgr, bright_mask)
    if fit.centroid is not None:
        px = fit.centroid
    elif px is None:
        return RefinedDetection(
            None, bright_mask, fit.kept_mask, area, fit, fit.method or "no_centroid",
        )

    final = fit.kept_mask if fit.kept_mask is not None and np.any(fit.kept_mask) else bright_mask
    area_f = float(np.count_nonzero(final)) if final is not None else area
    return RefinedDetection(
        (float(px[0]), float(px[1])),
        bright_mask,
        final,
        area_f,
        fit,
        fit.method or "bright_only",
    )


def refined_pixels_for_trial(session_dir: Path, trial: int) -> list[tuple[float, float]]:
    """Centroids from frame_*_square.json under trial_NNN/home/."""
    home = session_dir / f"trial_{trial:03d}" / "home"
    if not home.is_dir():
        return []
    pixels: list[tuple[float, float]] = []
    for sj in sorted(home.glob("frame_*_square.json")):
        try:
            meta = json.loads(sj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        c = meta.get("centroid_xy")
        if c and len(c) >= 2:
            pixels.append((float(c[0]), float(c[1])))
    return pixels


def refined_pixel_median_for_trial(session_dir: Path, trial: int) -> tuple[float, float] | None:
    pixels = refined_pixels_for_trial(session_dir, trial)
    if not pixels:
        return None
    med = np.median(np.asarray(pixels), axis=0)
    return float(med[0]), float(med[1])


def recompute_pixel_on_frame(
    session_dir: Path,
    trial: int,
    *,
    frame_stem: str = "frame_00",
    color: str | None = None,
) -> tuple[float, float] | None:
    """Re-run bright-top + square fit on a saved probe frame."""
    home = session_dir / f"trial_{trial:03d}" / "home"
    png = home / f"{frame_stem}.png"
    if not png.is_file():
        return None
    if color is None:
        meta_path = session_dir / "session.json"
        if meta_path.is_file():
            color = json.loads(meta_path.read_text(encoding="utf-8")).get("color", "yellow")
        else:
            color = "yellow"
    bgr = cv2.imread(str(png))
    if bgr is None:
        return None
    det = detect_refined_pixel(bgr, color, session_dir=session_dir)
    return det.pixel


def pixel_to_du_dv(u: float, v: float) -> tuple[float, float]:
    cx, cy = WRIST_CAM_WIDTH / 2.0, WRIST_CAM_HEIGHT / 2.0
    return u - cx, v - cy
