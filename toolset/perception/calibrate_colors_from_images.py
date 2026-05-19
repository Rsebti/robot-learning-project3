"""
calibrate_colors_from_images.py - learn HSV ranges from one or more
top-down photos of all 6 cubes.

Approach (no clicking required):

  1. Mask "is a cube" pixels via HSV: S >= sat_min (rejects whites/greys/blacks).
  2. Find connected components in that mask; keep those whose pixel-count is
     in the cube range (default 200..6000 px for a 480p frame).
  3. Each component is one cube. Within that component, keep ONLY the
     top face: take pixels whose V is at least the (1 - top_face_pct) quantile
     of the component's V channel. This trims dark side faces.
  4. Collect (H, S, V) samples for that cube. Median H -> assign to a known
     colour (red < orange < yellow < green < blue < violet, with red wrap
     around 180).
  5. Aggregate per colour across all input images, output an HSV range
     [(H_lo, S_lo, V_lo), (H_hi, S_hi, V_hi)] using percentile padding.

Usage:
    python -m toolset.perception.calibrate_colors_from_images \\
        deploy/_snaps/snap_1779103504_000.png deploy/_snaps/snap_1779103555_000.png

    # with a viz overlay PNG next to each input
    python -m toolset.perception.calibrate_colors_from_images <imgs> --viz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

# Expected colors in increasing-hue order (after handling red wrap).
# Red has the lowest "effective hue" because we unwrap it into [-N..0].
EXPECTED_COLORS_BY_HUE = ["red", "orange", "yellow", "green", "blue", "violet"]

# Approximate hue centres on OpenCV's 0..179 scale (for assignment fallback).
HUE_CENTERS = {
    "red":    0,
    "orange": 12,
    "yellow": 28,
    "green":  60,
    "blue":   105,
    "violet": 140,
}


def _normalize_red_hue(h: np.ndarray) -> np.ndarray:
    """Map H values near 180 to be just below 0, so red samples cluster.
    Threshold raised to 168 so violet (typically ~140-155) is NOT wrapped
    into red space, which would shift every cube one color slot."""
    h = h.astype(np.int32)
    h[h > 168] -= 180
    return h


def _per_cube_top_face_samples(
    bgr: np.ndarray,
    sat_min: int,
    val_min: int,
    min_area: int,
    max_area: int,
    top_face_pct: float,
) -> tuple[list[dict], np.ndarray]:
    """Find each cube blob, return per-blob HSV samples from its top face.

    Returns (cubes, sat_mask) where sat_mask is the global S/V binary mask
    BEFORE area filtering, for visualization.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    mask = ((S >= sat_min) & (V >= val_min)).astype(np.uint8) * 255
    # gentle clean
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    global_mask_for_viz = mask.copy()

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8,
    )

    cubes = []
    for lab in range(1, n_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue
        x, y, w, h = (
            int(stats[lab, cv2.CC_STAT_LEFT]),
            int(stats[lab, cv2.CC_STAT_TOP]),
            int(stats[lab, cv2.CC_STAT_WIDTH]),
            int(stats[lab, cv2.CC_STAT_HEIGHT]),
        )
        comp_mask = (labels == lab)
        # top face = brightest pixels within the component
        v_in = V[comp_mask]
        v_thr = int(np.quantile(v_in, 1.0 - top_face_pct))
        top_face_mask = comp_mask & (V >= v_thr)
        # filter to pixels safely interior (erode top_face_mask a touch to
        # avoid mixing with the side faces near the cube's silhouette)
        tf_u8 = top_face_mask.astype(np.uint8) * 255
        tf_u8 = cv2.erode(tf_u8, k, iterations=1)
        if int(tf_u8.sum() / 255) < max(20, area // 12):
            # erosion ate everything; fall back to the brightest 25%
            top_face_mask = comp_mask & (V >= int(np.quantile(v_in, 0.75)))
        else:
            top_face_mask = tf_u8.astype(bool)
        h_samples = H[top_face_mask].astype(np.int32)
        s_samples = S[top_face_mask].astype(np.int32)
        v_samples = V[top_face_mask].astype(np.int32)
        # red wrap: collect both raw and wrap-normalized
        h_norm = _normalize_red_hue(h_samples.copy())
        cubes.append({
            "centroid": (float(centroids[lab, 0]), float(centroids[lab, 1])),
            "bbox": (x, y, w, h),
            "area": area,
            "top_face_pixels": int(top_face_mask.sum()),
            "top_face_mask": top_face_mask,
            "blob_mask": comp_mask,
            "h": h_samples,
            "h_norm": h_norm,
            "s": s_samples,
            "v": v_samples,
        })
    return cubes, global_mask_for_viz


def _assign_colors(cubes: list[dict]) -> list[tuple[dict, str]]:
    """Sort cubes by median normalized hue, assign to EXPECTED_COLORS_BY_HUE.

    If we have exactly 6 cubes, assigning by sorted-hue gives the canonical
    mapping. With fewer/more cubes we fall back to nearest-center labelling.
    """
    if not cubes:
        return []
    med_h_norm = np.array([float(np.median(c["h_norm"])) for c in cubes])
    if len(cubes) == 6:
        order = np.argsort(med_h_norm)
        return [(cubes[order[i]], EXPECTED_COLORS_BY_HUE[i]) for i in range(6)]
    # nearest-center fallback (using raw hue, handling red wrap as two centres)
    out = []
    for c in cubes:
        h_raw = float(np.median(c["h"]))
        # consider distance to each canonical center, with red also at 180
        dists = {
            name: min(abs(h_raw - center), abs(h_raw - (center + 180)))
            for name, center in HUE_CENTERS.items()
        }
        name = min(dists, key=dists.get)
        out.append((c, name))
    return out


def _hsv_range_for_color(
    cubes_for_color: list[dict], color: str,
    pad_h: int = 3, pad_s: int = 30, pad_v: int = 30,
    h_pct: tuple[float, float] = (0.05, 0.95),
    s_pct: tuple[float, float] = (0.05, 0.95),
    v_pct: tuple[float, float] = (0.05, 0.95),
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Return one or two (low, high) HSV bounds. Returns two ranges for red
    when the samples span the wrap (some near 0, some near 180)."""
    h_all = np.concatenate([c["h"] for c in cubes_for_color]).astype(np.int32)
    s_all = np.concatenate([c["s"] for c in cubes_for_color]).astype(np.int32)
    v_all = np.concatenate([c["v"] for c in cubes_for_color]).astype(np.int32)

    s_lo = max(0,   int(np.quantile(s_all, s_pct[0])) - pad_s)
    s_hi = min(255, int(np.quantile(s_all, s_pct[1])) + pad_s)
    v_lo = max(0,   int(np.quantile(v_all, v_pct[0])) - pad_v)
    v_hi = min(255, int(np.quantile(v_all, v_pct[1])) + pad_v)

    # Red wrap detection
    low_count = int((h_all < 10).sum())
    high_count = int((h_all > 170).sum())
    if color == "red" and low_count > 0 and high_count > 0:
        lows = h_all[h_all < 90]
        highs = h_all[h_all >= 90]
        lo_lo = max(0,   int(np.quantile(lows,  h_pct[0])) - pad_h)
        lo_hi = min(179, int(np.quantile(lows,  h_pct[1])) + pad_h)
        hi_lo = max(0,   int(np.quantile(highs, h_pct[0])) - pad_h)
        hi_hi = min(179, int(np.quantile(highs, h_pct[1])) + pad_h)
        return [
            ((lo_lo, s_lo, v_lo), (lo_hi, s_hi, v_hi)),
            ((hi_lo, s_lo, v_lo), (hi_hi, s_hi, v_hi)),
        ]
    h_lo = max(0,   int(np.quantile(h_all, h_pct[0])) - pad_h)
    h_hi = min(179, int(np.quantile(h_all, h_pct[1])) + pad_h)
    return [((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))]


# BGR display colors per assigned label (used to paint sample masks).
_LABEL_BGR = {
    "red":    (40, 40, 255),
    "orange": (40, 140, 255),
    "yellow": (40, 230, 240),
    "green":  (60, 200, 60),
    "blue":   (220, 80, 40),
    "violet": (200, 60, 180),
}


def _annotate_with_boxes(bgr: np.ndarray, labeled_cubes: list[tuple[dict, str]]) -> np.ndarray:
    vis = bgr.copy()
    for c, name in labeled_cubes:
        x, y, w, h = c["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            vis, name, (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
        )
        med_h = int(np.median(c["h"]))
        med_s = int(np.median(c["s"]))
        med_v = int(np.median(c["v"]))
        cv2.putText(
            vis, f"H={med_h} S={med_s} V={med_v}",
            (x, min(bgr.shape[0] - 4, y + h + 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA,
        )
    return vis


def _paint_top_face_samples(
    bgr: np.ndarray, labeled_cubes: list[tuple[dict, str]],
) -> np.ndarray:
    """Dim the original, overlay the top-face sample masks in label color."""
    dim = (bgr.astype(np.int32) * 0.35).clip(0, 255).astype(np.uint8)
    out = dim.copy()
    for c, name in labeled_cubes:
        m = c["top_face_mask"]
        color = np.array(_LABEL_BGR.get(name, (255, 255, 255)), dtype=np.uint8)
        out[m] = color
        x, y, w, h = c["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 1)
        cv2.putText(
            out, name, (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            out, f"n={c['top_face_pixels']}",
            (x, min(bgr.shape[0] - 4, y + h + 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA,
        )
    return out


def _paint_blob_mask(
    bgr_shape: tuple[int, int, int], labeled_cubes: list[tuple[dict, str]],
) -> np.ndarray:
    """Black bg; each kept blob filled in its assigned color."""
    out = np.zeros(bgr_shape, dtype=np.uint8)
    for c, name in labeled_cubes:
        m = c["blob_mask"]
        color = np.array(_LABEL_BGR.get(name, (200, 200, 200)), dtype=np.uint8)
        out[m] = color
    return out


def _viz_overlay(
    bgr: np.ndarray,
    labeled_cubes: list[tuple[dict, str]],
    sat_mask: np.ndarray,
    out_path: Path,
    out_path_samples: Path | None = None,
):
    """Write a 4-panel viz:

      TL: original + bounding boxes + HSV medians per cube
      TR: S>=sat_min binary mask (pre-area-filter) -- shows EVERYTHING the
          script considered as a candidate blob
      BL: kept blobs (post area filter) painted in assigned color
      BR: TOP-FACE sample pixels in each cube, painted in label color over
          a dimmed original (shows exactly which pixels seeded the HSV stats)
    """
    h, w = bgr.shape[:2]
    panel_box = _annotate_with_boxes(bgr, labeled_cubes)
    panel_sat = cv2.cvtColor(sat_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel_sat, "S>=sat_min mask", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    panel_blobs = _paint_blob_mask(bgr.shape, labeled_cubes)
    cv2.putText(panel_blobs, "kept blobs (area filter)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    panel_samples = _paint_top_face_samples(bgr, labeled_cubes)
    cv2.putText(panel_samples, "top-face SAMPLES used for HSV stats", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    top = np.hstack([panel_box, panel_sat])
    bot = np.hstack([panel_blobs, panel_samples])
    grid = np.vstack([top, bot])
    cv2.imwrite(str(out_path), grid)

    if out_path_samples is not None:
        cv2.imwrite(str(out_path_samples), panel_samples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+", type=Path,
                   help="Top-down photos of all cubes (one or more).")
    p.add_argument("--sat_min", type=int, default=60,
                   help="Saturation threshold to keep cube pixels.")
    p.add_argument("--val_min", type=int, default=40,
                   help="Value threshold to drop very dark pixels.")
    p.add_argument("--min_area", type=int, default=200)
    p.add_argument("--max_area", type=int, default=6000)
    p.add_argument("--top_face_pct", type=float, default=0.5,
                   help="Within each cube blob, keep the top this fraction "
                        "of pixels by Value (the top face).")
    p.add_argument("--out_yaml", type=Path,
                   default=Path(__file__).resolve().parent / "hsv_config.yaml")
    p.add_argument("--viz", action="store_true",
                   help="Save annotated PNG next to each input.")
    args = p.parse_args()

    # Per-color sample accumulator
    by_color: dict[str, list[dict]] = {}

    for img_path in args.images:
        if not img_path.exists():
            print(f"[calib] WARNING: not found, skipping: {img_path}")
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[calib] WARNING: failed to load: {img_path}")
            continue
        print(f"\n[calib] {img_path}")
        cubes, sat_mask = _per_cube_top_face_samples(
            bgr, args.sat_min, args.val_min,
            args.min_area, args.max_area, args.top_face_pct,
        )
        print(f"  found {len(cubes)} cube blob(s) (target: 6)")
        labeled = _assign_colors(cubes)
        for c, name in labeled:
            med_h = int(np.median(c["h"]))
            med_s = int(np.median(c["s"]))
            med_v = int(np.median(c["v"]))
            print(f"    blob @ ({c['centroid'][0]:.0f}, {c['centroid'][1]:.0f}) "
                  f"area={c['area']:>4d}  top_face_px={c['top_face_pixels']:>4d}  "
                  f"H={med_h:>3d} S={med_s:>3d} V={med_v:>3d}  -> {name}")
            by_color.setdefault(name, []).append(c)

        if args.viz:
            out_grid = img_path.with_name(img_path.stem + "_calib_overlay.png")
            out_samples = img_path.with_name(img_path.stem + "_samples.png")
            _viz_overlay(bgr, labeled, sat_mask, out_grid, out_samples)
            print(f"  viz grid    -> {out_grid}")
            print(f"  viz samples -> {out_samples}")

    print("\n[calib] per-color HSV ranges (OpenCV [0..179, 0..255, 0..255]):")
    out_ranges: dict[str, list] = {}
    for color in EXPECTED_COLORS_BY_HUE:
        if color not in by_color:
            print(f"  {color:>7s}  NO SAMPLES (skipped)")
            continue
        ranges = _hsv_range_for_color(by_color[color], color)
        out_ranges[color] = [[list(lo), list(hi)] for (lo, hi) in ranges]
        for lo, hi in ranges:
            print(f"  {color:>7s}  H=[{lo[0]:>3d}, {hi[0]:>3d}]  "
                  f"S=[{lo[1]:>3d}, {hi[1]:>3d}]  V=[{lo[2]:>3d}, {hi[2]:>3d}]")

    args.out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_yaml, "w") as f:
        yaml.safe_dump(out_ranges, f, default_flow_style=False, sort_keys=False)
    print(f"\n[calib] wrote {args.out_yaml}")


if __name__ == "__main__":
    main()
