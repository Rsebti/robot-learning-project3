"""Blob-only cube masking — no square / wall cutting."""
from __future__ import annotations

import cv2
import numpy as np

from toolset.perception.color_mask import mask_bright_top, mask_color_blob


def keep_largest_blob(mask: np.ndarray) -> np.ndarray:
    """Drop pixels outside the main cube blob (remove HSV noise islands)."""
    if mask is None or not np.any(mask):
        return mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    main = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [main], -1, 255, -1)
    return out


def detect_cube_mask(
    bgr: np.ndarray,
    ranges: list[tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    mode: str = "bright_top",
    v_floor: int | None = None,
    seed_xy: tuple[float, float] | None = None,
    trim_outside_blob: bool = True,
    close_kernel: int = 0,
    close_iters: int = 0,
) -> tuple[tuple[float, float] | None, np.ndarray | None, float, str]:
    """
    Returns (centroid_xy, mask, area_px, method).

    ``bright_top`` — default; bright top face, single filled contour (recommended).
    ``full_blob`` — looser HSV + connected component (more of cube sides).
    """
    if mode == "bright_top":
        px, mask, area = mask_bright_top(
            bgr, ranges, v_floor=v_floor, seed_xy=seed_xy,
        )
        method = "bright_top"
    elif mode == "full_blob":
        px, mask, area = mask_color_blob(
            bgr, ranges,
            bright_only=False,
            v_floor=v_floor,
            seed_xy=seed_xy,
            close_kernel=close_kernel or 9,
            close_iters=close_iters or 2,
        )
        method = "full_blob"
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if mask is None or not np.any(mask):
        return px, mask, area, f"{method}_empty"

    if trim_outside_blob:
        mask = keep_largest_blob(mask)
        if px is not None:
            M = cv2.moments(mask)
            if M["m00"] > 0:
                px = (float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"]))
        area = float(np.count_nonzero(mask))
        method = f"{method}_blob_only"

    return px, mask, area, method


def draw_mask_overlay(
    bgr: np.ndarray,
    mask: np.ndarray | None,
    *,
    pixel: tuple[float, float] | None = None,
    area: float = 0.0,
    label: str = "",
) -> np.ndarray:
    out = bgr.copy()
    if mask is not None and np.any(mask):
        green = np.zeros_like(out)
        green[:, :, 1] = mask
        out = cv2.addWeighted(out, 0.55, green, 0.45, 0)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(out, cnts, -1, (0, 255, 255), 2)
    if pixel is not None:
        cv2.circle(out, (int(pixel[0]), int(pixel[1])), 7, (0, 0, 255), 2)
    txt = label or f"area={int(area)}"
    cv2.putText(out, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out
