"""
probe_space_map.py — lookup cube position from dense probe mapping (no extrinsics).

Trained from mapping_samples.csv (--map_only probe sessions):
  feature = pixel offset from image center (du, dv) at home
  target  = FK grasp xyz in user frame (trusted)

Usage:
    python -m toolset.perception.probe_space_map `
        --session_dir deploy/_snaps/probe_XXXXX --build

    python -m toolset.perception.probe_space_map `
        --session_dir deploy/_snaps/probe_XXXXX --du -40 --dv 12

    python -m toolset.perception.probe_space_map `
        --session_dir deploy/_snaps/probe_XXXXX --pixel 350 260 --k 5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WRIST_CAM_WIDTH = 640
WRIST_CAM_HEIGHT = 480


def load_samples(session_dir: Path) -> list[dict]:
    path = session_dir / "mapping_samples.csv"
    if path.is_file():
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        out = []
        for r in rows:
            try:
                out.append({
                    "trial": int(r["trial"]),
                    "du": float(r["du_px"]),
                    "dv": float(r["dv_px"]),
                    "fk": np.array([
                        float(r["fk_x_user"]),
                        float(r["fk_y_user"]),
                        float(r["fk_z_user"]),
                    ]),
                })
            except (KeyError, ValueError):
                continue
        if out:
            return out

    meta = json.loads((session_dir / "session.json").read_text())
    samples = []
    for t in meta.get("trials", []):
        off = t.get("pixel_offset_px")
        fk = t.get("fk_xyz_user_m")
        if off is None or fk is None:
            continue
        if not np.all(np.isfinite(off + fk)):
            continue
        samples.append({
            "trial": int(t["trial"]),
            "du": float(off[0]),
            "dv": float(off[1]),
            "fk": np.asarray(fk, dtype=float),
        })
    return samples


def build_index(samples: list[dict]) -> dict:
    feats = np.array([[s["du"], s["dv"]] for s in samples], dtype=float)
    fks = np.stack([s["fk"] for s in samples], axis=0)
    return {
        "n": len(samples),
        "features_du_dv": feats.tolist(),
        "fk_xyz_user_m": fks.tolist(),
        "trials": [s["trial"] for s in samples],
        "image_size": [WRIST_CAM_WIDTH, WRIST_CAM_HEIGHT],
        "cx_cy": [WRIST_CAM_WIDTH / 2.0, WRIST_CAM_HEIGHT / 2.0],
    }


def query(index: dict, du: float, dv: float, k: int = 3) -> dict:
    feats = np.asarray(index["features_du_dv"], dtype=float)
    fks = np.asarray(index["fk_xyz_user_m"], dtype=float)
    q = np.array([du, dv])
    d = np.linalg.norm(feats - q, axis=1)
    k = min(k, len(d))
    idx = np.argsort(d)[:k]
    w = 1.0 / (d[idx] + 1e-6)
    w /= w.sum()
    est = (fks[idx] * w[:, None]).sum(axis=0)
    return {
        "query_du_dv": [du, dv],
        "k": k,
        "neighbor_trials": [int(index["trials"][i]) for i in idx],
        "neighbor_dist_px": [float(d[i]) for i in idx],
        "fk_xyz_user_m": est.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session_dir", type=Path, required=True)
    p.add_argument("--build", action="store_true", help="Write space_map_index.yaml")
    p.add_argument("--k", type=int, default=3, help="k nearest neighbors (default 3)")
    p.add_argument("--du", type=float, help="Pixel offset u from center")
    p.add_argument("--dv", type=float, help="Pixel offset v from center")
    p.add_argument("--pixel", nargs=2, type=float, metavar=("U", "V"),
                   help="Raw pixel; converts to du,dv using 640x480 center")
    args = p.parse_args()

    session_dir = args.session_dir.resolve()
    samples = load_samples(session_dir)
    if not samples:
        raise SystemExit(
            f"No mapping samples in {session_dir}. Re-run probe with --map_only --save_frames.",
        )

    index_path = session_dir / "space_map_index.yaml"
    if args.build or not index_path.is_file():
        index = build_index(samples)
        with open(index_path, "w") as f:
            yaml.safe_dump(index, f, sort_keys=False)
        print(f"[map] built index n={index['n']} -> {index_path}")

    index = yaml.safe_load(index_path.read_text())

    if args.pixel is not None:
        cx, cy = WRIST_CAM_WIDTH / 2.0, WRIST_CAM_HEIGHT / 2.0
        du = float(args.pixel[0]) - cx
        dv = float(args.pixel[1]) - cy
    elif args.du is not None and args.dv is not None:
        du, dv = args.du, args.dv
    else:
        print(f"[map] index ready ({index['n']} samples). Pass --du/--dv or --pixel to query.")
        return

    result = query(index, du, dv, k=args.k)
    xyz = result["fk_xyz_user_m"]
    print(
        f"[map] du={du:+.0f} dv={dv:+.0f} px  k={result['k']}  "
        f"neighbors={result['neighbor_trials']}  dist={ [round(x,1) for x in result['neighbor_dist_px']] }",
    )
    print(f"[map] estimated fk_xyz_user_m: [{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}]")


if __name__ == "__main__":
    main()
