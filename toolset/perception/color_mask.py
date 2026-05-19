"""HSV masks for cube detection — bright top face only, or full blob (legacy)."""
from __future__ import annotations

import cv2
import numpy as np


def in_range_union(hsv: np.ndarray, ranges: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        m = cv2.inRange(
            hsv,
            np.array(lo, dtype=np.uint8),
            np.array(hi, dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask, m)
    return mask


def bright_range_from_band(
    lo: tuple[int, ...],
    hi: tuple[int, ...],
    *,
    v_floor: int | None = None,
    s_floor: int | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Tight band: brightest yellow (top face), not sides/shadows."""
    vf = int(v_floor if v_floor is not None else max(int(lo[2]), 110))
    sf = int(s_floor if s_floor is not None else max(int(lo[1]), 70))
    return (
        (int(lo[0]), min(255, sf), min(255, vf)),
        (int(hi[0]), int(hi[1]), int(hi[2])),
    )


def _pick_contour(
    contours: list,
    h: int,
    w: int,
    *,
    min_area: float,
    max_area_frac: float,
    seed_xy: tuple[float, float] | None,
    hsv: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float]:
    best_c = None
    best_score = -1.0
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area_frac * h * w:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        if cy > h * 0.82:
            continue
        if seed_xy is not None:
            dist = (cx - seed_xy[0]) ** 2 + (cy - seed_xy[1]) ** 2
            inside = cv2.pointPolygonTest(c, seed_xy, False) >= 0
            score = (1e6 if inside else 0) - dist
        elif hsv is not None:
            tmp = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(tmp, [c], -1, 255, -1)
            mean_v = float(hsv[:, :, 2][tmp > 0].mean()) if np.any(tmp) else 0
            score = mean_v * 1000 + a * 0.01
        else:
            score = a
        if score > best_score:
            best_score = score
            best_c = c
    if best_c is None:
        return None, 0.0
    return best_c, float(cv2.contourArea(best_c))


def mask_bright_top(
    bgr: np.ndarray,
    ranges: list[tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    v_floor: int | None = None,
    exclude_bottom_frac: float = 0.10,
    min_area: float = 25.0,
    max_area_frac: float = 0.02,
    seed_xy: tuple[float, float] | None = None,
) -> tuple[tuple[float, float] | None, np.ndarray, float]:
    """Only the bright yellow top — no fill into back face or shadows."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo, hi = ranges[0]
    lo_b, hi_b = bright_range_from_band(lo, hi, v_floor=v_floor)
    mask = cv2.inRange(hsv, np.array(lo_b, dtype=np.uint8), np.array(hi_b, dtype=np.uint8))

    if exclude_bottom_frac > 0:
        mask[int(h * (1.0 - exclude_bottom_frac)):, :] = 0

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask, 0.0

    chosen, area = _pick_contour(
        contours, h, w,
        min_area=min_area,
        max_area_frac=max_area_frac,
        seed_xy=seed_xy,
        hsv=hsv,
    )
    if chosen is None:
        return None, mask, 0.0

    mask_out = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_out, [chosen], -1, 255, -1)
    M = cv2.moments(chosen)
    px = float(M["m10"] / M["m00"])
    py = float(M["m01"] / M["m00"])
    return (px, py), mask_out, area


def mask_color_blob(
    bgr: np.ndarray,
    ranges: list[tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    bright_only: bool = True,
    v_floor: int | None = None,
    close_kernel: int = 9,
    close_iters: int = 2,
    exclude_bottom_frac: float = 0.10,
    min_area: float = 40.0,
    max_area_frac: float = 0.035,
    seed_xy: tuple[float, float] | None = None,
) -> tuple[tuple[float, float] | None, np.ndarray, float]:
    if bright_only:
        return mask_bright_top(
            bgr, ranges,
            v_floor=v_floor,
            exclude_bottom_frac=exclude_bottom_frac,
            min_area=min_area,
            max_area_frac=min(max_area_frac, 0.02),
            seed_xy=seed_xy,
        )

    # Legacy: loose + connected component (fills sides/shadows)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_loose = in_range_union(hsv, ranges)
    if exclude_bottom_frac > 0:
        mask_loose[int(h * (1.0 - exclude_bottom_frac)):, :] = 0

    lo, hi = ranges[0]
    lo_c = (int(lo[0]), min(255, int(lo[1]) + 30), min(255, int(lo[2]) + 55))
    mask_core = cv2.inRange(hsv, np.array(lo_c, dtype=np.uint8), np.array(hi, dtype=np.uint8))

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_loose = cv2.morphologyEx(mask_loose, cv2.MORPH_OPEN, k3, iterations=1)
    if close_kernel > 0 and close_iters > 0:
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask_loose = cv2.morphologyEx(mask_loose, cv2.MORPH_CLOSE, kc, iterations=close_iters)

    contours_c, _ = cv2.findContours(mask_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sx = sy = None
    if seed_xy is not None:
        sx, sy = float(seed_xy[0]), float(seed_xy[1])
    else:
        chosen, _ = _pick_contour(
            list(contours_c), h, w,
            min_area=min_area, max_area_frac=max_area_frac, seed_xy=None, hsv=hsv,
        )
        if chosen is not None:
            M = cv2.moments(chosen)
            sx, sy = M["m10"] / M["m00"], M["m01"] / M["m00"]

    if sx is None:
        contours_l, _ = cv2.findContours(mask_loose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_l:
            return None, mask_loose, 0.0
        largest = max(contours_l, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < min_area:
            return None, mask_loose, area
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None, mask_loose, area
        mask_out = mask_loose
        return (float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])), mask_out, area

    ix = int(np.clip(round(sx), 0, w - 1))
    iy = int(np.clip(round(sy), 0, h - 1))
    nlab, labels = cv2.connectedComponents(mask_loose)
    label = int(labels[iy, ix])
    if label == 0:
        ys, xs = np.where(mask_loose > 0)
        if len(xs) == 0:
            return None, mask_loose, 0.0
        j = int(np.argmin((xs - ix) ** 2 + (ys - iy) ** 2))
        label = int(labels[ys[j], xs[j]])
        if label == 0:
            return None, mask_loose, 0.0
    mask_out = np.where(labels == label, 255, 0).astype(np.uint8)
    area = float(np.count_nonzero(mask_out))
    if area < min_area or area > max_area_frac * h * w * 1.5:
        return None, mask_out, area
    contours, _ = cv2.findContours(mask_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask_out, area
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    M = cv2.moments(largest)
    return (float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])), mask_out, area
