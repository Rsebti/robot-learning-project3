#!/usr/bin/env python3
"""
cube_mask_batch.py — mask cube(s) in a folder of images (any color).

Detection: bright-top HSV (default, matches current good results).
Post-process: keep **main blob only** — drops pixels **outside** the cube mask.
Does **not** run square / wall cutting (see probe_square_postprocess for that).

Outputs under --out_dir:
  masks/<name>_mask.png
  overlays/<name>_overlay.png
  results.csv
  gallery.html

Examples:
  cd C:\\Users\\hugod\\project3
  conda activate trim

  # Yellow cube, global HSV
  python -m toolset.perception.cube_mask_batch `
      --image_dir deploy\\_snaps\\my_cube_frames `
      --color yellow

  # Learn HSV from your images first, then mask
  python -m toolset.perception.cube_mask_batch `
      --image_dir deploy\\_snaps\\my_cube_frames `
      --color yellow --learn_hsv

  # Custom HSV yaml (from calibrate_colors_from_images or hand-edited)
  python -m toolset.perception.cube_mask_batch `
      --image_dir deploy\\_snaps\\my_cube_frames `
      --color red --hsv_yaml deploy\\_snaps\\my_hsv.yaml

  # Looser mask (more cube sides), still blob-only trim
  python -m toolset.perception.cube_mask_batch `
      --image_dir deploy\\_snaps\\my_cube_frames `
      --color green --mode full_blob
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolset.perception.blob_mask_utils import detect_cube_mask, draw_mask_overlay
from toolset.perception.cube_localization import DEFAULT_HSV, _load_hsv
from toolset.perception.estimate_cube_xy import load_hsv_ranges
from toolset.perception.color_mask import mask_color_blob

HSV_CONFIG = PROJECT_ROOT / "toolset" / "perception" / "hsv_config.yaml"


def _list_images(image_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    skip = ("_mask", "_overlay", "_square", "_wall", "_final", "gallery", "analysis")
    return sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in exts and not any(s in p.name for s in skip)
    )


def _ranges_for_color(
    color: str,
    *,
    hsv_yaml: Path | None,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    if hsv_yaml and hsv_yaml.is_file():
        data = load_hsv_ranges(hsv_yaml)
        if data and color in data:
            return data[color]
    global_hsv = _load_hsv()
    if color in global_hsv:
        return global_hsv[color]
    if color in DEFAULT_HSV:
        return DEFAULT_HSV[color]
    raise KeyError(
        f"Color '{color}' not in --hsv_yaml or {HSV_CONFIG}. "
        f"Use --learn_hsv or calibrate_colors_from_images."
    )


def learn_hsv_from_images(
    images: list[Path],
    color: str,
    *,
    out_yaml: Path,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Learn one color band from largest blob samples across images."""
    hsv_samples: list[np.ndarray] = []
    seed_lo = (0, 25, 25)
    seed_hi = (179, 255, 255)

    for img_path in images[: min(30, len(images))]:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        px, mask, _ = mask_color_blob(
            bgr, [(seed_lo, seed_hi)],
            bright_only=False,
            max_area_frac=0.08,
        )
        if mask is None or not np.any(mask):
            continue
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv_samples.append(hsv[mask > 0])

    if not hsv_samples:
        raise RuntimeError("Could not find any cube blob to learn HSV — check images/lighting.")

    all_px = np.concatenate(hsv_samples, axis=0)
    h = all_px[:, 0].astype(np.float32)
    s = all_px[:, 1]
    v = all_px[:, 2]
    h_med = float(np.median(h))
    h_spread = max(8.0, float(np.percentile(np.abs(h - h_med), 90)) + 4.0)
    lo = [
        int(max(0, round(h_med - h_spread))),
        int(max(0, np.percentile(s, 5) - 15)),
        int(max(0, np.percentile(v, 5) - 25)),
    ]
    hi = [
        int(min(179, round(h_med + h_spread))),
        255,
        255,
    ]
    ranges = [(tuple(lo), tuple(hi))]
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump({color: [[list(lo), list(hi)]]}, f, default_flow_style=False)
    print(f"[learn_hsv] {color}: lo={lo} hi={hi} -> {out_yaml}")
    return ranges


def write_gallery(rows: list[dict], out_path: Path, *, color: str, mode: str) -> None:
    parts = [
        f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Cube masks — {color}</title>
<style>
body {{ font-family: system-ui,sans-serif; background:#1a1a1a; color:#eee; margin:16px; }}
h1 {{ font-size:1.2rem; }}
.row {{ display:flex; flex-wrap:wrap; gap:12px; margin:16px 0; }}
.cell img {{ max-width:360px; border:1px solid #555; }}
.ok {{ color:#8f8; }} .bad {{ color:#f88; }}
</style></head><body>
<h1>Cube mask batch — color=<span class="ok">{color}</span> mode={mode}</h1>
<p>Green = mask on cube. Cyan = blob outline. No square/wall cut.</p>
""",
    ]
    for r in rows:
        name = r["image"]
        ok = r["detected"]
        cls = "ok" if ok else "bad"
        parts.append(f'<h2 class="{cls}">{name}</h2><div class="row">')
        parts.append(
            f'<figure class="cell"><img src="../{r["raw_rel"]}"><figcaption>raw</figcaption></figure>'
        )
        if ok:
            parts.append(
                f'<figure class="cell"><img src="../{r["mask_rel"]}"><figcaption>mask</figcaption></figure>'
            )
            parts.append(
                f'<figure class="cell"><img src="../{r["overlay_rel"]}"><figcaption>overlay</figcaption></figure>'
            )
        parts.append("</div>")
    parts.append("</body></html>")
    out_path.write_text("".join(parts), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image_dir", type=Path, required=True, help="Folder of camera frames")
    p.add_argument("--out_dir", type=Path, default=None, help="Default: <image_dir>/cube_masks")
    p.add_argument("--color", required=True,
                   choices=["red", "orange", "yellow", "green", "blue", "violet"])
    p.add_argument("--hsv_yaml", type=Path, default=None, help="learned_hsv.yaml or calibrate output")
    p.add_argument("--learn_hsv", action="store_true",
                   help="Learn HSV from images, write <out_dir>/learned_hsv.yaml")
    p.add_argument("--mode", choices=["bright_top", "full_blob"], default="bright_top",
                   help="bright_top=current good detector; full_blob=more sides")
    p.add_argument("--no_trim_outside", action="store_true",
                   help="Disable largest-blob trim (not recommended)")
    p.add_argument("--v_floor", type=int, default=None, help="Min V for bright top (e.g. 20)")
    args = p.parse_args()

    image_dir = args.image_dir.resolve()
    out_dir = (args.out_dir or image_dir / "cube_masks").resolve()
    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    images = _list_images(image_dir)
    if not images:
        print(f"[cube_mask] no images in {image_dir}")
        return 1

    hsv_yaml = args.hsv_yaml
    if args.learn_hsv:
        learned = out_dir / "learned_hsv.yaml"
        ranges = learn_hsv_from_images(images, args.color, out_yaml=learned)
        hsv_yaml = learned
    else:
        ranges = _ranges_for_color(args.color, hsv_yaml=hsv_yaml)

    rows = []
    n_ok = 0
    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        px, mask, area, method = detect_cube_mask(
            bgr, ranges,
            mode=args.mode,
            v_floor=args.v_floor,
            trim_outside_blob=not args.no_trim_outside,
        )
        stem = img_path.stem
        raw_copy = out_dir / "raw" / f"{stem}.png"
        raw_copy.parent.mkdir(parents=True, exist_ok=True)
        if not raw_copy.is_file():
            cv2.imwrite(str(raw_copy), bgr)

        detected = mask is not None and np.any(mask)
        mask_rel = f"masks/{stem}_mask.png"
        overlay_rel = f"overlays/{stem}_overlay.png"
        if detected:
            cv2.imwrite(str(masks_dir / f"{stem}_mask.png"), mask)
            ov = draw_mask_overlay(bgr, mask, pixel=px, area=area, label=method)
            cv2.imwrite(str(overlays_dir / f"{stem}_overlay.png"), ov)
            n_ok += 1
            print(f"  OK  {stem}  area={int(area)}  {method}")
        else:
            print(f"  MISS {stem}")

        rows.append({
            "image": stem,
            "detected": detected,
            "area_px": int(area) if detected else 0,
            "centroid_u": f"{px[0]:.1f}" if px else "",
            "centroid_v": f"{px[1]:.1f}" if px else "",
            "method": method,
            "raw_rel": f"raw/{stem}.png",
            "mask_rel": mask_rel if detected else "",
            "overlay_rel": overlay_rel if detected else "",
        })

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = {
        "color": args.color,
        "mode": args.mode,
        "hsv_yaml": str(hsv_yaml) if hsv_yaml else str(HSV_CONFIG),
        "n_images": len(images),
        "n_detected": n_ok,
        "trim_outside_blob": not args.no_trim_outside,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_gallery(rows, out_dir / "gallery.html", color=args.color, mode=args.mode)

    print(f"\n[cube_mask] {n_ok}/{len(images)} detected")
    print(f"[cube_mask] gallery -> {out_dir / 'gallery.html'}")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
