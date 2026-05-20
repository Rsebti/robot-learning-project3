"""Load probe mapping rows (pixel, FK, home/grasp motors) from a session folder."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from toolset.perception.probe_space_map import (
    WRIST_CAM_HEIGHT,
    WRIST_CAM_WIDTH,
    build_index,
    load_samples,
    query,
)
from toolset.perception.refined_cube_detect import (
    pixel_to_du_dv,
    refined_pixel_median_for_trial,
    recompute_pixel_on_frame,
)

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

EXCLUDED_YAML = "excluded_trials.yaml"


def load_excluded_trials(session_dir: Path) -> set[int]:
    """Trial indices listed in session excluded_trials.yaml (not used in map)."""
    path = session_dir / EXCLUDED_YAML
    if not path.is_file():
        return set()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("excluded", data if isinstance(data, list) else [])
    return {int(x) for x in raw}


def save_excluded_trials(session_dir: Path, excluded: set[int], notes: dict[int, str] | None = None) -> Path:
    path = session_dir / EXCLUDED_YAML
    existing: dict = {}
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    note_map = {int(k): str(v) for k, v in (existing.get("notes") or {}).items()}
    if notes:
        note_map.update({int(k): str(v) for k, v in notes.items()})
    note_map = {k: v for k, v in note_map.items() if k in excluded}
    payload = {"excluded": sorted(int(x) for x in excluded), "notes": note_map}
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def trial_quality_flags(row: dict) -> list[str]:
    """Heuristics for bad grasp / hand-in-frame review (not auto-excluded)."""
    flags: list[str] = []
    fk = row["fk"]
    if fk[2] > 0.06:
        flags.append(f"FK z={fk[2]:.3f} m (expected ~0.02)")
    hm, gm = row.get("home_motor_deg"), row.get("grasp_motor_deg")
    if hm is not None and gm is not None:
        d = float(np.linalg.norm(np.asarray(hm[:5], float) - np.asarray(gm[:5], float)))
        if d < 4.0:
            flags.append("grasp joints ≈ home (likely no grasp)")
    return flags


def apply_refined_pixels(rows: list[dict], session_dir: Path) -> list[dict]:
    """Overwrite pixel_u/v and du/dv from square.json / live re-fit (FK unchanged)."""
    out = []
    for r in rows:
        tri = int(r["trial"])
        px = refined_pixel_median_for_trial(session_dir, tri)
        if px is None:
            px = recompute_pixel_on_frame(session_dir, tri)
        row = dict(r)
        if px is not None:
            du, dv = pixel_to_du_dv(px[0], px[1])
            row["pixel_u"] = px[0]
            row["pixel_v"] = px[1]
            row["du"] = du
            row["dv"] = dv
            row["pixel_source"] = "refined_mask"
        else:
            row["pixel_source"] = "raw"
        out.append(row)
    return out


def load_mapping_rows(
    session_dir: Path,
    *,
    use_refined: bool = True,
    include_excluded: bool = False,
) -> list[dict]:
    """Full rows from mapping_samples_refined.csv, mapping_samples.csv, or session.json."""
    refined_path = session_dir / "mapping_samples_refined.csv"
    if use_refined and refined_path.is_file():
        path = refined_path
    else:
        path = session_dir / "mapping_samples.csv"
    rows: list[dict] = []
    if path.is_file():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append({
                        "trial": int(r["trial"]),
                        "color": r.get("color", ""),
                        "pixel_u": float(r["pixel_u"]),
                        "pixel_v": float(r["pixel_v"]),
                        "du": float(r["du_px"]),
                        "dv": float(r["dv_px"]),
                        "fk": np.array([
                            float(r["fk_x_user"]),
                            float(r["fk_y_user"]),
                            float(r["fk_z_user"]),
                        ]),
                        "home_motor_deg": [
                            float(r[f"home_{n}"]) for n in MOTOR_NAMES
                        ],
                        "grasp_motor_deg": [
                            float(r[f"grasp_{n}"]) for n in MOTOR_NAMES
                        ],
                    })
                except (KeyError, ValueError):
                    continue
        if rows:
            if use_refined and path.name != "mapping_samples_refined.csv":
                rows = apply_refined_pixels(rows, session_dir)
            if not include_excluded:
                excluded = load_excluded_trials(session_dir)
                rows = [r for r in rows if int(r["trial"]) not in excluded]
            return rows

    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    for t in meta.get("trials", []):
        off = t.get("pixel_offset_px")
        fk = t.get("fk_xyz_user_m")
        if off is None or fk is None:
            continue
        rows.append({
            "trial": int(t["trial"]),
            "color": meta.get("color", ""),
            "pixel_u": float(t["home_pixel"][0]) if t.get("home_pixel") else float("nan"),
            "pixel_v": float(t["home_pixel"][1]) if t.get("home_pixel") else float("nan"),
            "du": float(off[0]),
            "dv": float(off[1]),
            "fk": np.asarray(fk, dtype=float),
            "home_motor_deg": t.get("home_motor_deg"),
            "grasp_motor_deg": t.get("grasp_motor_deg"),
        })
    if rows and use_refined:
        rows = apply_refined_pixels(rows, session_dir)
    if rows and not include_excluded:
        excluded = load_excluded_trials(session_dir)
        rows = [r for r in rows if int(r["trial"]) not in excluded]
    return rows


def write_refined_mapping_csv(session_dir: Path) -> Path:
    """Write mapping_samples_refined.csv (refined pixels, same FK/motors)."""
    rows = load_mapping_rows(session_dir, use_refined=False, include_excluded=True)
    if not rows:
        raise ValueError(f"No mapping rows in {session_dir}")
    excluded = load_excluded_trials(session_dir)
    rows = [r for r in rows if int(r["trial"]) not in excluded]
    if not rows:
        raise ValueError(f"No rows left after exclusions {sorted(excluded)}")
    refined = apply_refined_pixels(rows, session_dir)
    out_path = session_dir / "mapping_samples_refined.csv"
    fieldnames = [
        "trial", "color", "pixel_u", "pixel_v", "du_px", "dv_px",
        "fk_x_user", "fk_y_user", "fk_z_user",
        *(f"home_{n}" for n in MOTOR_NAMES),
        *(f"grasp_{n}" for n in MOTOR_NAMES),
        "pixel_source",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in refined:
            w.writerow({
                "trial": r["trial"],
                "color": r.get("color", ""),
                "pixel_u": f"{r['pixel_u']:.2f}",
                "pixel_v": f"{r['pixel_v']:.2f}",
                "du_px": f"{r['du']:+.1f}",
                "dv_px": f"{r['dv']:+.1f}",
                "fk_x_user": f"{r['fk'][0]:+.4f}",
                "fk_y_user": f"{r['fk'][1]:+.4f}",
                "fk_z_user": f"{r['fk'][2]:+.4f}",
                **{f"home_{n}": r["home_motor_deg"][i] for i, n in enumerate(MOTOR_NAMES)},
                **{f"grasp_{n}": r["grasp_motor_deg"][i] for i, n in enumerate(MOTOR_NAMES)},
                "pixel_source": r.get("pixel_source", "refined_mask"),
            })
    for stale in ("space_map_index.yaml", "space_map_index_refined.yaml"):
        p = session_dir / stale
        if p.is_file():
            p.unlink()
    return out_path


def estimate_xyz_from_pixel(
    session_dir: Path,
    du: float,
    dv: float,
    *,
    k: int = 3,
    use_refined: bool = True,
) -> dict:
    samples = load_samples(session_dir, use_refined=use_refined)
    if not samples:
        raise ValueError(f"No mapping samples in {session_dir}")
    index_path = session_dir / (
        "space_map_index_refined.yaml" if use_refined else "space_map_index.yaml"
    )
    if not index_path.is_file():
        index = build_index(samples)
        import yaml
        with open(index_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(index, f, sort_keys=False)
    else:
        import yaml
        index = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    q = query(index, du, dv, k=k)
    rows = load_mapping_rows(session_dir, use_refined=use_refined)
    trial_map = {r["trial"]: r for r in rows}
    neighbors = []
    for tri in q["neighbor_trials"]:
        r = trial_map.get(tri)
        if r is not None:
            neighbors.append(r)
    return {
        "fk_xyz_user_m": np.asarray(q["fk_xyz_user_m"], dtype=float),
        "neighbor_trials": q["neighbor_trials"],
        "neighbor_dist_px": q["neighbor_dist_px"],
        "neighbors": neighbors,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild mapping_samples_refined.csv from final masks.")
    p.add_argument("--session_dir", type=Path, required=True)
    p.add_argument("--exclude", type=int, action="append", default=[],
                   help="Add trial(s) to excluded_trials.yaml and rebuild refined CSV.")
    args = p.parse_args()
    session_dir = args.session_dir.resolve()
    if args.exclude:
        excl = load_excluded_trials(session_dir) | set(args.exclude)
        save_excluded_trials(session_dir, excl)
        print(f"[probe_map_data] excluded trials: {sorted(excl)}")
    out = write_refined_mapping_csv(session_dir)
    rows = load_mapping_rows(session_dir, use_refined=True)
    excluded = load_excluded_trials(session_dir)
    print(f"[probe_map_data] wrote {out} ({len(rows)} active rows, refined pixels)")
    if excluded:
        print(f"[probe_map_data] excluded from map: {sorted(excluded)}")
    for r in rows:
        print(f"  trial {r['trial']:03d}  du={r['du']:+.0f} dv={r['dv']:+.0f}  "
              f"({r.get('pixel_source', '?')})")


if __name__ == "__main__":
    main()
