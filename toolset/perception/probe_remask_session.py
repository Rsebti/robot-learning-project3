"""
probe_remask_session.py — learn yellow/red HSV from probe frames (incl. shadows) and re-save masks.

Uses known centroids from session.json / frames_home.json when available; otherwise a loose
color pass to find the cube blob. Writes learned_hsv.yaml in the session folder and
overwrites frame_*_mask.png / frame_*_overlay.png under each trial/home/.

Usage:
    python -m toolset.perception.probe_remask_session `
        --session_dir deploy/_snaps/probe_1779205245

    # learn only (print ranges, no image writes)
    python -m toolset.perception.probe_remask_session `
        --session_dir deploy/_snaps/probe_1779205245 --learn_only

    # also patch repo hsv_config.yaml (optional)
    python -m toolset.perception.probe_remask_session `
        --session_dir deploy/_snaps/probe_1779205245 --write_global
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolset.perception.color_mask import mask_color_blob
from toolset.perception.estimate_cube_xy import load_hsv_ranges, loosen_hsv
from toolset.perception.probe_camera_vs_fk import save_frame_bundle

HSV_CONFIG_PATH = PROJECT_ROOT / "toolset" / "perception" / "hsv_config.yaml"

# Very loose seed to find cube when no centroid was saved
SEED_LO = (15, 35, 35)
SEED_HI = (40, 255, 255)


def _mask_detect(
    bgr: np.ndarray,
    ranges: list[tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    v_floor: int | None = None,
    seed_xy: tuple[float, float] | None = None,
    max_area_frac: float = 0.02,
) -> tuple[tuple[float, float] | None, np.ndarray, float]:
    return mask_color_blob(
        bgr, ranges,
        bright_only=True,
        v_floor=v_floor,
        seed_xy=seed_xy,
        max_area_frac=max_area_frac,
    )


def _seed_centroid(bgr: np.ndarray) -> tuple[float, float] | None:
    px, _, area = _mask_detect(bgr, [(SEED_LO, SEED_HI)], max_area_frac=0.03)
    if px is None or area < 80:
        return None
    return px


def _sample_hsv_patch(hsv: np.ndarray, px: float, py: float, radius: int = 45) -> np.ndarray:
    h, w = hsv.shape[:2]
    x0, x1 = max(0, int(px) - radius), min(w, int(px) + radius + 1)
    y0, y1 = max(0, int(py) - radius), min(h, int(py) + radius + 1)
    patch = hsv[y0:y1, x0:x1].reshape(-1, 3)
    return patch


def _sample_hsv_on_mask(hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sel = mask > 0
    if not np.any(sel):
        return np.empty((0, 3), dtype=np.uint8)
    return hsv[sel]


def learn_ranges_from_session(
    session_dir: Path,
    color: str,
    *,
    patch_radius: int = 50,
) -> list[tuple[list[int], list[int]]]:
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    hsv_all: list[np.ndarray] = []

    for t in meta.get("trials", []):
        tri = int(t["trial"])
        trial_dir = session_dir / f"trial_{tri:03d}" / "home"
        if not trial_dir.is_dir():
            continue

        frames_path = trial_dir / "frames_home.json"
        pixels: list[tuple[float, float]] = []
        if frames_path.is_file():
            for fr in json.loads(frames_path.read_text(encoding="utf-8")):
                if fr.get("pixel") is not None:
                    pixels.append((float(fr["pixel"][0]), float(fr["pixel"][1])))
        if t.get("home_pixel") is not None:
            pixels.append((float(t["home_pixel"][0]), float(t["home_pixel"][1])))

        pngs = sorted(trial_dir.glob("frame_*.png"))
        pngs = [p for p in pngs if "_mask" not in p.name and "_overlay" not in p.name]
        if not pngs:
            continue

        med_px = np.median(pixels, axis=0) if pixels else None
        for png in pngs[:3]:  # first few frames per trial
            bgr = cv2.imread(str(png))
            if bgr is None:
                continue
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            px, py = None, None
            if med_px is not None:
                px, py = float(med_px[0]), float(med_px[1])
            else:
                seed = _seed_centroid(bgr)
                if seed is None:
                    continue
                px, py = seed

            patch = _sample_hsv_patch(hsv, px, py, radius=patch_radius)
            if len(patch) < 20:
                continue
            hsv_all.append(patch)

    if not hsv_all:
        base = load_hsv_ranges(HSV_CONFIG_PATH) or {}
        widened = loosen_hsv(base, color, s_delta=50, v_delta=60)
        return [[list(lo), list(hi)] for lo, hi in widened[color]]

    samples = np.vstack(hsv_all)
    if color == "yellow":
        samples = samples[(samples[:, 0] >= 12) & (samples[:, 0] <= 45)]
    if len(samples) < 20:
        base = load_hsv_ranges(HSV_CONFIG_PATH) or {}
        widened = loosen_hsv(base, color, s_delta=30, v_delta=35)
        return [[list(lo), list(hi)] for lo, hi in widened[color]]

    h = samples[:, 0].astype(float)
    s = samples[:, 1].astype(float)
    v = samples[:, 2].astype(float)

    h_lo = float(np.percentile(h, 5)) - 3
    h_hi = float(np.percentile(h, 95)) + 3
    if color == "yellow":
        h_lo = max(12, min(35, h_lo))
        h_hi = min(42, max(24, h_hi))
    s_lo = max(70, float(np.percentile(s, 50)) - 15)
    s_hi = 255
    v_lo = max(105, float(np.percentile(v, 72)) - 8)  # bright top only
    v_hi = 255

    h_lo = int(np.clip(h_lo, 0, 179))
    h_hi = int(np.clip(h_hi, 0, 179))
    lo = [h_lo, int(s_lo), int(v_lo)]
    hi = [h_hi, int(s_hi), int(v_hi)]
    return [[lo, hi]]


def ranges_to_yaml_pairs(ranges: list) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    out = []
    for band in ranges:
        lo, hi = band[0], band[1]
        out.append((tuple(int(x) for x in lo), tuple(int(x) for x in hi)))
    return out


def remask_session(
    session_dir: Path,
    color: str,
    ranges: list[tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    v_floor: int | None,
    max_area_frac: float,
    full_blob: bool,
) -> tuple[int, int]:
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    trial_seed: dict[int, tuple[float, float] | None] = {}
    for t in meta.get("trials", []):
        tri = int(t["trial"])
        if t.get("home_pixel") is not None:
            trial_seed[tri] = (float(t["home_pixel"][0]), float(t["home_pixel"][1]))
        else:
            trial_seed[tri] = None

    n_img = 0
    n_hit = 0
    for home_dir in sorted(session_dir.glob("trial_*/home")):
        tri = int(home_dir.parent.name.split("_")[1])
        seed = trial_seed.get(tri)
        pngs = sorted(home_dir.glob("frame_*.png"))
        pngs = [p for p in pngs if "_mask" not in p.name and "_overlay" not in p.name]
        for png in pngs:
            bgr = cv2.imread(str(png))
            if bgr is None:
                continue
            n_img += 1
            if full_blob:
                from toolset.perception.color_mask import mask_color_blob as _mcb
                px, mask, area = _mcb(
                    bgr, ranges, bright_only=False,
                    seed_xy=seed, max_area_frac=max_area_frac,
                )
            else:
                px, mask, area = _mask_detect(
                    bgr, ranges, v_floor=v_floor,
                    seed_xy=seed, max_area_frac=max_area_frac,
                )
            stem = png.with_suffix("")
            save_frame_bundle(stem, bgr, mask, px, area)
            if px is not None:
                n_hit += 1
    return n_img, n_hit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session_dir", type=Path, required=True)
    p.add_argument("--color", default=None, help="Override color (default from session.json)")
    p.add_argument("--learn_only", action="store_true")
    p.add_argument("--write_global", action="store_true",
                   help="Also update toolset/perception/hsv_config.yaml")
    p.add_argument("--v_floor", type=int, default=None,
                   help="Min V for bright top (default: from learned band, usually ~110+)")
    p.add_argument("--full_blob", action="store_true",
                   help="Legacy loose mask + fill (not recommended)")
    p.add_argument("--max_area_frac", type=float, default=0.02,
                   help="Max blob area fraction (default 2%%)")
    args = p.parse_args()

    session_dir = args.session_dir.resolve()
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    color = args.color or meta.get("color", "yellow")

    print(f"[remask] session {session_dir}  color={color!r}")
    learned = learn_ranges_from_session(session_dir, color)
    learned_path = session_dir / "learned_hsv.yaml"
    with open(learned_path, "w", encoding="utf-8") as f:
        f.write(f"# learned from {session_dir.name}\n")
        yaml.safe_dump({color: learned}, f, sort_keys=False, default_flow_style=None)

    lo, hi = learned[0][0], learned[0][1]
    print(f"[remask] learned {color} HSV lo={lo} hi={hi}")
    print(f"[remask] wrote {learned_path}")

    if args.write_global:
        cfg = {}
        if HSV_CONFIG_PATH.is_file():
            cfg = yaml.safe_load(HSV_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        cfg[color] = learned
        lines = [
            "# HSV ranges (OpenCV: H 0-179, S/V 0-255).",
            f"# {color} band updated by probe_remask_session from {session_dir.name}",
            "",
        ]
        for cname, bands in cfg.items():
            lines.append(f"{cname}:")
            for lo, hi in bands:
                lines.append(f"  - [[{lo[0]}, {lo[1]}, {lo[2]}], [{hi[0]}, {hi[1]}, {hi[2]}]]")
            lines.append("")
        HSV_CONFIG_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"[remask] updated {HSV_CONFIG_PATH}")

    if args.learn_only:
        return

    ranges = ranges_to_yaml_pairs(learned)
    if args.full_blob:
        print("[remask] WARNING: --full_blob fills sides/shadows; default is bright top only")
    n_img, n_hit = remask_session(
        session_dir, color, ranges,
        v_floor=args.v_floor,
        max_area_frac=args.max_area_frac,
        full_blob=args.full_blob,
    )
    print(f"[remask] rewrote masks/overlays: {n_hit}/{n_img} frames with detection")


if __name__ == "__main__":
    main()
