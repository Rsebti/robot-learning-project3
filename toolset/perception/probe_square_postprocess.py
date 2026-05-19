"""
probe_square_postprocess.py — fit square top face from bright masks; refresh gallery.html.

Usage:
    python -m toolset.perception.probe_square_postprocess `
        --session_dir deploy/_snaps/probe_1779205245

Writes per frame (alongside frame_XX.png):
  frame_XX_final_mask.png       — final kept mask (binary)
  frame_XX_final_overlay.png    — raw + final mask superposed (verification)
  frame_XX_wall_mask.png        — blob clipped by wall lines (partial OK)
  frame_XX_square_overlay.png   — walls, corners, cut suggestions
  frame_XX_square.json          — metadata

Regenerates <session_dir>/gallery.html with square column.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolset.perception.mask_square_fit import (
    draw_final_verification_overlay,
    draw_square_overlay,
    fit_square_to_mask,
)


def process_session(session_dir: Path, *, frame_suffix: str = "frame") -> dict:
    stats = {"n_frames": 0, "n_ok": 0, "trials": {}}
    for home_dir in sorted(session_dir.glob("trial_*/home")):
        tri = home_dir.parent.name
        n_ok = 0
        pngs = sorted(home_dir.glob(f"{frame_suffix}_*.png"))
        pngs = [p for p in pngs if "_mask" not in p.name and "_overlay" not in p.name
                and "_square" not in p.name and "_wall" not in p.name
                and "_final" not in p.name]
        for png in pngs:
            stem = png.stem
            mask_path = home_dir / f"{stem}_mask.png"
            if not mask_path.is_file():
                continue
            bgr = cv2.imread(str(png))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if bgr is None or mask is None:
                continue
            stats["n_frames"] += 1
            fit = fit_square_to_mask(bgr, mask)
            if fit.kept_mask is None and fit.cut_mask is None:
                continue
            n_ok += 1
            stats["n_ok"] += 1
            if fit.wall_mask is not None:
                cv2.imwrite(str(home_dir / f"{stem}_wall_mask.png"), fit.wall_mask)
            if fit.kept_mask is not None:
                cv2.imwrite(str(home_dir / f"{stem}_final_mask.png"), fit.kept_mask)
                final_vis = draw_final_verification_overlay(
                    bgr, fit.kept_mask,
                    centroid=fit.centroid,
                    area=fit.area_kept,
                )
                cv2.imwrite(str(home_dir / f"{stem}_final_overlay.png"), final_vis)
            overlay = draw_square_overlay(bgr, mask, fit)
            cv2.imwrite(str(home_dir / f"{stem}_square_overlay.png"), overlay)
            meta = {
                "wall_lines_xyxy": [list(map(float, ln)) for ln in fit.lines],
                "corners_xy": fit.corners.tolist() if fit.corners is not None and len(fit.corners) else [],
                "majority_hsv": fit.majority_hsv,
                "centroid_xy": list(fit.centroid) if fit.centroid else None,
                "method": fit.method,
                "area_blob": fit.area_blob,
                "area_kept": fit.area_kept,
                "area_cut": fit.area_cut,
            }
            (home_dir / f"{stem}_square.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8",
            )
        stats["trials"][tri] = {"frames": len(pngs), "square_ok": n_ok}
    return stats


def write_gallery(session_dir: Path) -> None:
    session_json = session_dir / "session.json"
    trials_info: dict[int, dict] = {}
    color = "?"
    if session_json.is_file():
        meta = json.loads(session_json.read_text(encoding="utf-8"))
        color = meta.get("color", "?")
        for t in meta.get("trials", []):
            trials_info[int(t["trial"])] = t

    mapping_trials = set()
    map_csv = session_dir / "mapping_samples.csv"
    if map_csv.is_file():
        import csv
        with open(map_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    mapping_trials.add(int(row["trial"]))
                except (KeyError, ValueError):
                    pass

    rows_html = []
    for tri_dir in sorted(session_dir.glob("trial_*")):
        tri = int(tri_dir.name.split("_")[1])
        home = tri_dir / "home"
        if not home.is_dir():
            continue
        tmeta = trials_info.get(tri, {})
        hp = tmeta.get("home_pixel")
        in_map = tri in mapping_trials
        if hp is None and not (home / "frame_00_square.json").is_file():
            status_cls = "bad"
            status = "no pixel"
        elif in_map:
            status_cls = "ok"
            status = "in map"
        else:
            status_cls = "ok"
            status = "square ok"
        rows_html.append(
            f'  <h2 class="{status_cls}">{tri_dir.name} — {status}</h2>\n'
            f'  <div class="row">\n'
            f'    <figure class="cell"><img src="{tri_dir.name}/home/frame_00.png">'
            f"<figcaption>raw</figcaption></figure>\n"
            f'    <figure class="cell"><img src="{tri_dir.name}/home/frame_00_mask.png">'
            f"<figcaption>bright mask</figcaption></figure>\n"
            f'    <figure class="cell"><img src="{tri_dir.name}/home/frame_00_overlay.png">'
            f"<figcaption>mask overlay</figcaption></figure>\n"
            f'    <figure class="cell"><img src="{tri_dir.name}/home/frame_00_square_overlay.png">'
            f"<figcaption>walls · cut suggest</figcaption></figure>\n"
            f'    <figure class="cell"><img src="{tri_dir.name}/home/frame_00_final_mask.png">'
            f"<figcaption>final mask</figcaption></figure>\n"
            f'    <figure class="cell verify"><img src="{tri_dir.name}/home/frame_00_final_overlay.png">'
            f"<figcaption><b>verify</b> final on cube</figcaption></figure>\n"
            f"  </div>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{session_dir.name} — masks & square fit</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 16px; background: #1a1a1a; color: #eee; }}
    h1 {{ font-size: 1.2rem; }}
    h2 {{ font-size: 1rem; margin-top: 2rem; border-bottom: 1px solid #444; padding-bottom: 4px; }}
    .ok {{ color: #8f8; }}
    .bad {{ color: #f88; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 24px; }}
    .cell {{ text-align: center; }}
    .cell img {{ max-width: 280px; height: auto; border: 1px solid #555; background: #000; }}
    .cell.verify img {{ border: 2px solid #6f6; max-width: 320px; }}
    .cell figcaption {{ font-size: 0.72rem; color: #aaa; max-width: 300px; }}
    .legend {{ font-size: 0.85rem; color: #bbb; margin: 8px 0 16px; }}
  </style>
</head>
<body>
  <h1>{session_dir.name} ({color})</h1>
  <p class="legend">
    <b>Wall fit:</b> straight Hough walls, corners at line crossovers (partial top OK).
    No forced full square. <span style="color:#8f8">green</span> = kept;
    <span style="color:#f8f">magenta</span> = suggested cut.
    Last column = <b>final mask</b> on raw (verification).
  </p>
{"".join(rows_html)}
</body>
</html>
"""
    out = session_dir / "gallery.html"
    out.write_text(html, encoding="utf-8")
    print(f"[square] gallery -> {out}")


def rebuild_final_verification(session_dir: Path) -> int:
  """Write final_mask + final_overlay only (uses bright mask + wall fit)."""
  n = 0
  for home_dir in sorted(session_dir.glob("trial_*/home")):
    for png in sorted(home_dir.glob("frame_*.png")):
      if any(x in png.name for x in ("_mask", "_overlay", "_square", "_wall", "_final")):
        continue
      stem = png.stem
      mask_path = home_dir / f"{stem}_mask.png"
      if not mask_path.is_file():
        continue
      bgr = cv2.imread(str(png))
      mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
      if bgr is None or mask is None:
        continue
      fit = fit_square_to_mask(bgr, mask)
      if fit.kept_mask is None:
        continue
      cv2.imwrite(str(home_dir / f"{stem}_final_mask.png"), fit.kept_mask)
      cv2.imwrite(
        str(home_dir / f"{stem}_final_overlay.png"),
        draw_final_verification_overlay(
          bgr, fit.kept_mask, centroid=fit.centroid, area=fit.area_kept,
        ),
      )
      n += 1
  return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session_dir", type=Path, required=True)
    p.add_argument("--gallery_only", action="store_true",
                   help="Only rebuild gallery.html (PNGs already exist)")
    p.add_argument("--final_only", action="store_true",
                   help="Rebuild final_mask/final_overlay from existing square.json + raw")
    args = p.parse_args()

    session_dir = args.session_dir.resolve()
    if args.final_only:
        n = rebuild_final_verification(session_dir)
        print(f"[square] final verification overlays: {n} frames")
    elif not args.gallery_only:
        stats = process_session(session_dir)
        print(f"[square] fitted {stats['n_ok']}/{stats['n_frames']} frames")
    write_gallery(session_dir)


if __name__ == "__main__":
    main()
