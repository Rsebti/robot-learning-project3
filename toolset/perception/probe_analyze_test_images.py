"""
probe_analyze_test_images.py — analyze new home images against a probe map session.

For each test image (or live capture batch):
  1. Refined mask detect (session learned_hsv + bright-top + square fit)
  2. kNN map query -> estimated fk_xyz_user_m
  3. Compare to training probes from --map_session_dir

Writes under --out_dir (default: <image_dir>/analysis or deploy/_snaps/test_probe_<ts>):
  results.csv
  analysis_report.txt
  overlays/frame_XX_overlay.png
  gallery.html

Usage:
    # 10 images you saved at home (cube at one new pose, 10 frames OK)
    python -m toolset.perception.probe_analyze_test_images `
        --map_session_dir deploy/_snaps/probe_1779205245 `
        --image_dir deploy/_snaps/my_test_cube

    # Live: ramp home, grab 10 frames
    python -m toolset.perception.probe_analyze_test_images `
        --map_session_dir deploy/_snaps/probe_1779205245 `
        --capture --port COM3 --camera_index 0 --n_frames 10
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolset.perception.mask_square_fit import draw_final_verification_overlay
from toolset.perception.probe_map_data import load_mapping_rows
from toolset.perception.probe_space_map import (
    WRIST_CAM_HEIGHT,
    WRIST_CAM_WIDTH,
    build_index,
    load_samples,
    query,
)
from toolset.perception.refined_cube_detect import detect_refined_pixel, pixel_to_du_dv


def _list_images(image_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in exts]
    skip = ("_mask", "_overlay", "_square", "_wall", "_final", "analysis")
    return [p for p in files if not any(s in p.name for s in skip)]


def _hull_area_xy(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _point_in_hull_2d(pt: np.ndarray, hull_pts: np.ndarray) -> bool:
    """Ray casting; hull_pts ordered along hull."""
    x, y = float(pt[0]), float(pt[1])
    n = len(hull_pts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = hull_pts[i]
        xj, yj = hull_pts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _du_dv_hull(train_feats: np.ndarray) -> np.ndarray | None:
    if len(train_feats) < 3:
        return None
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(train_feats)
        return train_feats[hull.vertices]
    except Exception:
        return None


def _capture_frames(
    *,
    port: str,
    camera_index: int,
    home_pose: str,
    n_frames: int,
    out_dir: Path,
) -> list[Path]:
    deploy = PROJECT_ROOT / "deploy"
    if str(deploy) not in sys.path:
        sys.path.insert(0, str(deploy))
    from homes import get_home_deg
    from lerobot.robots.utils import make_robot_from_config
    from robot_calibration import LOCAL_CALIBRATION_DIR, make_so101_follower_config
    from wrist_camera_config import make_wrist_opencv_camera_config
    from toolset.perception.probe_camera_vs_fk import ramp_to_target_deg

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_so101_follower_config(
        port,
        cameras={"base_camera": make_wrist_opencv_camera_config(camera_index)},
        calibration_dir=LOCAL_CALIBRATION_DIR,
        use_degrees=True,
    )
    robot = make_robot_from_config(cfg)
    robot.connect()
    ramp_to_target_deg(robot, get_home_deg(home_pose))
    time.sleep(0.4)
    paths: list[Path] = []
    try:
        for i in range(n_frames):
            obs = robot.get_observation()
            bgr = obs.get("observation.images.base_camera")
            if bgr is None:
                for k, v in obs.items():
                    if "base_camera" in str(k) and hasattr(v, "shape"):
                        bgr = v
                        break
            if hasattr(bgr, "cpu"):
                bgr = bgr.cpu().numpy()
            if bgr.ndim == 3 and bgr.shape[0] == 3:
                bgr = np.transpose(bgr, (1, 2, 0))
            if bgr.dtype != np.uint8:
                bgr = (np.clip(bgr, 0, 1) * 255).astype(np.uint8)
            if bgr.shape[2] == 3:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
            p = out_dir / f"frame_{i:02d}.png"
            cv2.imwrite(str(p), bgr)
            paths.append(p)
            time.sleep(0.15)
    finally:
        robot.disconnect()
    return paths


def analyze_images(
    *,
    map_session_dir: Path,
    image_paths: list[Path],
    out_dir: Path,
    color: str,
    k: int,
    label: str,
) -> list[dict]:
    map_session_dir = map_session_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    samples = load_samples(map_session_dir, use_refined=True)
    if not samples:
        raise ValueError(f"No mapping samples in {map_session_dir}")
    index_path = map_session_dir / "space_map_index_refined.yaml"
    if not index_path.is_file():
        index = build_index(samples)
        with open(index_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(index, f, sort_keys=False)
    else:
        index = yaml.safe_load(index_path.read_text(encoding="utf-8"))

    train_rows = load_mapping_rows(map_session_dir, use_refined=True)
    train_feats = np.array([[r["du"], r["dv"]] for r in train_rows], dtype=float)
    train_fks = np.stack([r["fk"] for r in train_rows], axis=0)
    hull = _du_dv_hull(train_feats)

    results: list[dict] = []
    for i, img_path in enumerate(image_paths):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            results.append({"frame": img_path.name, "status": "read_fail"})
            continue

        det = detect_refined_pixel(bgr, color, session_dir=map_session_dir)
        row: dict = {
            "frame": img_path.name,
            "label": label,
            "status": "ok" if det.pixel else "no_detect",
        }
        if det.pixel is None:
            results.append(row)
            continue

        u, v = det.pixel
        du, dv = pixel_to_du_dv(u, v)
        q = query(index, du, dv, k=k)
        xyz = np.asarray(q["fk_xyz_user_m"], dtype=float)

        dists_px = np.linalg.norm(train_feats - np.array([du, dv]), axis=1)
        j_near = int(np.argmin(dists_px))
        near_fk = train_fks[j_near]
        err_mm = float(np.linalg.norm((xyz - near_fk)[:2]) * 1000.0)
        in_hull = False
        if hull is not None:
            in_hull = _point_in_hull_2d(np.array([du, dv]), hull)
        else:
            in_hull = (
                train_feats[:, 0].min() <= du <= train_feats[:, 0].max()
                and train_feats[:, 1].min() <= dv <= train_feats[:, 1].max()
            )

        row.update({
            "pixel_u": round(u, 1),
            "pixel_v": round(v, 1),
            "du_px": round(du, 1),
            "dv_px": round(dv, 1),
            "est_x_m": round(float(xyz[0]), 4),
            "est_y_m": round(float(xyz[1]), 4),
            "est_z_m": round(float(xyz[2]), 4),
            "neighbor_trials": str(q["neighbor_trials"]),
            "neighbor_dist_px": str([round(x, 1) for x in q["neighbor_dist_px"]]),
            "nearest_train_trial": int(train_rows[j_near]["trial"]),
            "nearest_train_dist_px": round(float(dists_px[j_near]), 1),
            "nearest_train_x_m": round(float(near_fk[0]), 4),
            "nearest_train_y_m": round(float(near_fk[1]), 4),
            "in_map_hull": in_hull,
            "xy_vs_nearest_train_mm": round(err_mm, 1),
            "detect_method": det.method,
        })
        results.append(row)

        if det.final_mask is not None:
            vis = draw_final_verification_overlay(bgr, det.final_mask, centroid=det.pixel)
            cv2.circle(vis, (int(u), int(v)), 8, (0, 255, 255), 2)
            cv2.putText(
                vis,
                f"est ({xyz[0]:+.2f},{xyz[1]:+.2f})m",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(overlay_dir / f"{img_path.stem}_overlay.png"), vis)

    ok = [r for r in results if r.get("status") == "ok"]
    _write_csv(out_dir / "results.csv", ok)
    _write_report(
        out_dir / "analysis_report.txt",
        map_session_dir=map_session_dir,
        train_rows=train_rows,
        results=results,
        ok=ok,
        label=label,
    )
    _write_gallery(out_dir, ok)
    return results


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_report(
    path: Path,
    *,
    map_session_dir: Path,
    train_rows: list[dict],
    results: list[dict],
    ok: list[dict],
    label: str,
) -> None:
    lines = [
        "probe test image analysis",
        f"map_session: {map_session_dir}",
        f"test_label: {label}",
        f"n_images: {len(results)}  detected: {len(ok)}",
        "",
        "=== Training probes (previous FK at grasp, user m) ===",
    ]
    for r in train_rows:
        fk = r["fk"]
        lines.append(
            f"  trial {r['trial']:03d}  du={r['du']:+.0f} dv={r['dv']:+.0f}  "
            f"fk=({fk[0]:+.3f},{fk[1]:+.3f},{fk[2]:+.3f})"
        )
    lines.append("")
    if not ok:
        lines.append("No detections — check color / lighting / learned_hsv.yaml")
    else:
        xs = np.array([r["est_x_m"] for r in ok], dtype=float)
        ys = np.array([r["est_y_m"] for r in ok], dtype=float)
        zs = np.array([r["est_z_m"] for r in ok], dtype=float)
        lines.extend([
            "=== Test batch (map estimate from new images) ===",
            f"  median xyz: ({np.median(xs):+.3f}, {np.median(ys):+.3f}, {np.median(zs):+.3f})",
            f"  std  xyz:  ({xs.std():.3f}, {ys.std():.3f}, {zs.std():.3f})",
            f"  spread xy: {np.std(np.hypot(xs - xs.mean(), ys - ys.mean())) * 100:.1f} mm (batch repeatability)",
            "",
            "=== Per frame ===",
        ])
        for r in ok:
            lines.append(
                f"  {r['frame']}  du={r['du_px']:+.0f} dv={r['dv_px']:+.0f}  "
                f"est=({r['est_x_m']:+.3f},{r['est_y_m']:+.3f},{r['est_z_m']:+.3f})  "
                f"hull={'yes' if r['in_map_hull'] else 'NO'}  "
                f"nn=trial{r['nearest_train_trial']} {r['nearest_train_dist_px']:.0f}px  "
                f"xy_err_vs_nn={r['xy_vs_nearest_train_mm']:.0f}mm"
            )
        outside = [r for r in ok if not r["in_map_hull"]]
        if outside:
            lines.append("")
            lines.append(f"  WARNING: {len(outside)} frame(s) outside training du-dv hull — extrapolation.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gallery(out_dir: Path, ok: list[dict]) -> None:
    overlay_dir = out_dir / "overlays"
    if not overlay_dir.is_dir():
        return
    rows_html = []
    for r in ok:
        stem = Path(r["frame"]).stem
        ov = overlay_dir / f"{stem}_overlay.png"
        if not ov.is_file():
            continue
        rel = f"overlays/{ov.name}"
        rows_html.append(
            f"<tr><td>{r['frame']}</td>"
            f"<td>({r['est_x_m']:+.3f}, {r['est_y_m']:+.3f}, {r['est_z_m']:+.3f})</td>"
            f"<td><img src='{rel}' width='320'></td></tr>"
        )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>test probe analysis</title></head>
<body>
<h1>Test probe analysis</h1>
<table border="1" cellpadding="4">
<tr><th>frame</th><th>est xyz (m)</th><th>overlay</th></tr>
{''.join(rows_html)}
</table>
</body></html>"""
    (out_dir / "gallery.html").write_text(html, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map_session_dir", type=Path, required=True,
                   help="Probe session with mapping_samples.csv (training map).")
    p.add_argument("--image_dir", type=Path, default=None,
                   help="Folder of home images (png/jpg).")
    p.add_argument("--capture", action="store_true",
                   help="Capture n_frames at home instead of image_dir.")
    p.add_argument("--port", default="COM3")
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--home_pose", default="eval1_rest")
    p.add_argument("--n_frames", type=int, default=10)
    p.add_argument("--color", default=None,
                   help="Cube color (default: from map session.json).")
    p.add_argument("--k", type=int, default=3, help="kNN for map query.")
    p.add_argument("--label", default="test",
                   help="Tag in report (e.g. new_pose_A).")
    p.add_argument("--out_dir", type=Path, default=None)
    args = p.parse_args()

    map_session = args.map_session_dir.resolve()
    meta_path = map_session / "session.json"
    color = args.color
    if color is None and meta_path.is_file():
        color = json.loads(meta_path.read_text(encoding="utf-8")).get("color", "yellow")
    color = color or "yellow"

    if args.capture:
        ts = int(time.time())
        image_dir = PROJECT_ROOT / "deploy" / "_snaps" / f"test_probe_{ts}"
        print(f"[test] capturing {args.n_frames} frames -> {image_dir}")
        image_paths = _capture_frames(
            port=args.port,
            camera_index=args.camera_index,
            home_pose=args.home_pose,
            n_frames=args.n_frames,
            out_dir=image_dir,
        )
        out_dir = args.out_dir or (image_dir / "analysis")
    else:
        if args.image_dir is None:
            p.error("Provide --image_dir or --capture")
        image_dir = args.image_dir.resolve()
        image_paths = _list_images(image_dir)
        if not image_paths:
            raise SystemExit(f"No images in {image_dir}")
        out_dir = args.out_dir or (image_dir / "analysis")

    print(f"[test] {len(image_paths)} images  map={map_session}  color={color}")
    analyze_images(
        map_session_dir=map_session,
        image_paths=image_paths,
        out_dir=out_dir.resolve(),
        color=color,
        k=args.k,
        label=args.label,
    )
    print(f"[test] wrote {out_dir / 'results.csv'}")
    print(f"[test] wrote {out_dir / 'analysis_report.txt'}")
    print(f"[test] gallery -> {out_dir / 'gallery.html'}")


if __name__ == "__main__":
    main()
