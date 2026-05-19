"""
mask_square_fit.py — top face from wall line crossovers; partial quads OK; cut by top color.

No expansion / no forced full square: region = intersection of half-planes from
detected edge lines. Pixels to cut = blob minus that region, plus outliers that
match the top-face HSV least vs the bright majority.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SquareFitResult:
    lines: list[tuple[float, float, float, float]] = field(default_factory=list)  # x1,y1,x2,y2
    corners: np.ndarray | None = None  # (N, 2) N<=4 visible intersections
    wall_mask: np.ndarray | None = None  # blob clipped by wall half-planes (may be partial)
    cut_mask: np.ndarray | None = None  # suggested cuts
    kept_mask: np.ndarray | None = None
    centroid: tuple[float, float] | None = None
    area_blob: float = 0.0
    area_kept: float = 0.0
    area_cut: float = 0.0
    method: str = "none"
    majority_hsv: list[float] | None = None


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _line_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.arctan2(y2 - y1, x2 - x1))


def _abc(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float]:
    a, b = y1 - y2, x2 - x1
    c = x1 * y2 - x2 * y1
    n = np.hypot(a, b) + 1e-9
    return a / n, b / n, c / n


def _intersect(
    l1: tuple[float, float, float],
    l2: tuple[float, float, float],
) -> tuple[float, float] | None:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-6:
        return None
    return (b1 * c2 - b2 * c1) / d, (c1 * a2 - c2 * a1) / d


def _extend_segment(x1: float, y1: float, x2: float, y2: float, h: int, w: int) -> tuple[int, int, int, int]:
    """Clip line segment to image bounds (draw full visible wall)."""
    pts = []
    for t in np.linspace(-max(h, w), max(h, w), 50):
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        if 0 <= x < w and 0 <= y < h:
            pts.append((int(x), int(y)))
    if len(pts) < 2:
        return int(x1), int(y1), int(x2), int(y2)
    return pts[0][0], pts[0][1], pts[-1][0], pts[-1][1]


def _detect_wall_segments(
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    centroid: tuple[float, float],
) -> list[tuple[float, float, float, float]]:
    h, w = mask.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 45, 130)
    band = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), 2)
    edges = cv2.bitwise_and(edges, band)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=24, minLineLength=22, maxLineGap=10)
    if lines is None:
        return []

    cx, cy = centroid
    segs: list[tuple[float, float, float, float, float, float]] = []
    for ln in lines.reshape(-1, 4):
        x1, y1, x2, y2 = [float(v) for v in ln]
        L = np.hypot(x2 - x1, y2 - y1)
        if L < 18:
            continue
        mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        if np.hypot(mx - cx, my - cy) > max(h, w) * 0.45:
            continue
        ang = _line_angle(x1, y1, x2, y2)
        segs.append((ang, x1, y1, x2, y2, L))

    if len(segs) < 2:
        return [(s[1], s[2], s[3], s[4]) for s in segs]

    base = segs[int(np.argmax([s[5] for s in segs]))][0]

    def bucket(a: float) -> int:
        d0 = abs(np.arctan2(np.sin(a - base), np.cos(a - base)))
        d1 = abs(np.arctan2(np.sin(a - base - np.pi / 2), np.cos(a - base - np.pi / 2)))
        return 0 if d0 <= d1 else 1

    g0 = [s for s in segs if bucket(s[0]) == 0]
    g1 = [s for s in segs if bucket(s[0]) == 1]

    def pick_two(group: list) -> list[tuple[float, float, float, float]]:
        if not group:
            return []
        abcs = [(s, _abc(s[1], s[2], s[3], s[4])) for s in group]
        scored = []
        for s, (a, b, c) in abcs:
            d = a * cx + b * cy + c
            scored.append((d, s))
        scored.sort(key=lambda t: t[0])
        out = []
        if len(scored) >= 2:
            out.append(scored[0][1])
            out.append(scored[-1][1])
        elif len(scored) == 1:
            out.append(scored[0][1])
        return [(s[1], s[2], s[3], s[4]) for s in out]

    walls = pick_two(g0) + pick_two(g1)
    return walls[:4]


def _corners_from_walls(
    walls: list[tuple[float, float, float, float]],
    h: int,
    w: int,
    *,
    near: tuple[float, float],
    max_dist: float,
) -> np.ndarray:
    abcs = [_abc(*w) for w in walls]
    corners = []
    for i in range(len(abcs)):
        for j in range(i + 1, len(abcs)):
            pt = _intersect(abcs[i], abcs[j])
            if pt is None:
                continue
            x, y = pt
            if not (-w * 0.1 <= x < w * 1.1 and -h * 0.1 <= y < h * 1.1):
                continue
            if np.hypot(x - near[0], y - near[1]) <= max_dist:
                corners.append([x, y])
    if not corners:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(corners, dtype=np.float32)


def _half_plane_mask(
    h: int,
    w: int,
    walls: list[tuple[float, float, float, float]],
    centroid: tuple[float, float],
) -> np.ndarray:
    """Keep pixels on the blob-centroid side of each wall line."""
    if not walls:
        return np.ones((h, w), dtype=np.uint8) * 255

    ys, xs = np.mgrid[0:h, 0:w]
    keep = np.ones((h, w), dtype=bool)
    cx, cy = centroid
    for x1, y1, x2, y2 in walls:
        a, b, c = _abc(x1, y1, x2, y2)
        side_c = a * cx + b * cy + c
        val = a * xs + b * ys + c
        if side_c >= 0:
            keep &= val >= -2.0
        else:
            keep &= val <= 2.0
    out = np.zeros((h, w), dtype=np.uint8)
    out[keep] = 255
    return out


def _majority_hsv(hsv: np.ndarray, mask: np.ndarray, centroid: tuple[float, float]) -> np.ndarray:
    """Bright core = majority top color."""
    v = hsv[:, :, 2]
    core = (mask > 0) & (v >= np.percentile(v[mask > 0], 60))
    if np.count_nonzero(core) < 15:
        core = mask > 0
    return np.median(hsv[core], axis=0)


def _color_cut_mask(
    hsv: np.ndarray,
    blob: np.ndarray,
    majority: np.ndarray,
    *,
    dh: float = 12.0,
    ds: float = 55.0,
    dv: float = 45.0,
) -> np.ndarray:
    """Cut blob pixels least like the top-face majority (neighbor vote via global ref)."""
    h_ch = hsv[:, :, 0].astype(np.float32)
    s_ch = hsv[:, :, 1].astype(np.float32)
    v_ch = hsv[:, :, 2].astype(np.float32)
    mh, ms, mv = majority
    dh_arr = np.minimum(np.abs(h_ch - mh), 180 - np.abs(h_ch - mh))
    dist = (dh_arr / dh) ** 2 + ((s_ch - ms) / ds) ** 2 + ((v_ch - mv) / dv) ** 2
    cut = (blob > 0) & (dist > 1.0)
    return cut.astype(np.uint8) * 255


def fit_square_to_mask(
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    force_equal_square: bool = False,  # ignored — no forced square
    refine_edges: bool = True,
) -> SquareFitResult:
    empty = SquareFitResult()
    if mask is None or not np.any(mask):
        return empty

    cnt = _largest_contour(mask)
    if cnt is None:
        return empty

    area_blob = float(cv2.contourArea(cnt))
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return empty
    centroid = (float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"]))

    h, w = mask.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    majority = _majority_hsv(hsv, mask, centroid)

    walls = _detect_wall_segments(bgr, mask, centroid=centroid) if refine_edges else []
    max_d = max(h, w) * 0.55
    corners = _corners_from_walls(walls, h, w, near=centroid, max_dist=max_d)

    hp = _half_plane_mask(h, w, walls, centroid)
    wall_clip = cv2.bitwise_and(mask, hp)

    color_cut = _color_cut_mask(hsv, mask, majority)
    # Cut: outside walls OR color outlier on blob; never add pixels outside original blob
    cut = cv2.bitwise_and(mask, cv2.bitwise_or(cv2.bitwise_xor(mask, wall_clip), color_cut))
    kept = cv2.bitwise_and(mask, cv2.bitwise_not(cut))

    method = f"walls={len(walls)} corners={len(corners)}"
    if not walls:
        method = "color_only"

    c2 = None
    if np.count_nonzero(kept) > 0:
        M2 = cv2.moments(kept)
        if M2["m00"] > 0:
            c2 = (float(M2["m10"] / M2["m00"]), float(M2["m01"] / M2["m00"]))

    return SquareFitResult(
        lines=walls,
        corners=corners if len(corners) else None,
        wall_mask=wall_clip,
        cut_mask=cut,
        kept_mask=kept,
        centroid=c2 or centroid,
        area_blob=area_blob,
        area_kept=float(np.count_nonzero(kept)),
        area_cut=float(np.count_nonzero(cut)),
        method=method,
        majority_hsv=[float(x) for x in majority],
    )


def draw_square_overlay(
    bgr: np.ndarray,
    mask: np.ndarray | None,
    fit: SquareFitResult,
) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]

    if fit.cut_mask is not None:
        mag = np.zeros_like(out)
        mag[:, :, 2] = fit.cut_mask
        out = cv2.addWeighted(out, 0.78, mag, 0.4, 0)

    if fit.kept_mask is not None:
        grn = np.zeros_like(out)
        grn[:, :, 1] = fit.kept_mask
        out = cv2.addWeighted(out, 0.72, grn, 0.22, 0)

    for x1, y1, x2, y2 in fit.lines:
        ex1, ey1, ex2, ey2 = _extend_segment(x1, y1, x2, y2, h, w)
        cv2.line(out, (ex1, ey1), (ex2, ey2), (255, 200, 0), 2, cv2.LINE_AA)

    if fit.corners is not None and len(fit.corners):
        for i, p in enumerate(fit.corners):
            pt = (int(p[0]), int(p[1]))
            cv2.circle(out, pt, 6, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(out, str(i), (pt[0] + 5, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if fit.corners is not None and len(fit.corners) >= 2:
        hull = cv2.convexHull(fit.corners.astype(np.float32))
        cv2.polylines(out, [hull.astype(np.int32)], len(fit.corners) >= 3,
                      (255, 200, 0), 1, cv2.LINE_AA)

    if fit.centroid is not None:
        c = (int(fit.centroid[0]), int(fit.centroid[1]))
        cv2.circle(out, c, 5, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    lbl = f"{fit.method} cut={int(fit.area_cut)} kept={int(fit.area_kept)}"
    cv2.putText(out, lbl, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return out


def draw_final_verification_overlay(
    bgr: np.ndarray,
    final_mask: np.ndarray | None,
    *,
    centroid: tuple[float, float] | None = None,
    area: float | None = None,
) -> np.ndarray:
    """Raw frame with only the final kept mask superposed (verification view)."""
    out = bgr.copy()
    if final_mask is None or not np.any(final_mask):
        cv2.putText(out, "FINAL: empty", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out

    tint = np.zeros_like(out)
    tint[:, :, 1] = final_mask
    out = cv2.addWeighted(out, 0.5, tint, 0.5, 0)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    if centroid is not None:
        c = (int(centroid[0]), int(centroid[1]))
        cv2.circle(out, c, 6, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(out, c, 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    a = int(area) if area is not None else int(np.count_nonzero(final_mask))
    cv2.putText(
        out, f"FINAL mask  area={a}px", (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, lineType=cv2.LINE_AA,
    )
    return out
