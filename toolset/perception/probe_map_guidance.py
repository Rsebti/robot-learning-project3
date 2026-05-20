"""
probe_map_guidance.py — per-trial map estimate, coverage hints, est vs measured validation.

Used during map_only probe sessions so each rollout shows:
  - kNN estimate after home photo (before grasp)
  - whether (du,dv) is inside training hull / needs more samples
  - est vs measured FK after grasp
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from toolset.perception.probe_map_data import load_excluded_trials
from toolset.perception.probe_space_map import build_index, query


def load_samples_from_mapping_csv(
    csv_path: Path,
    *,
    session_dir: Path | None = None,
) -> list[dict]:
    """Rows from mapping_samples.csv (in-progress session)."""
    if not csv_path.is_file():
        return []
    excluded = load_excluded_trials(session_dir) if session_dir else set()
    out: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                tri = int(r["trial"])
                if tri in excluded:
                    continue
                out.append({
                    "trial": tri,
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
    return out


def _du_dv_hull(feats: np.ndarray) -> np.ndarray | None:
    if len(feats) < 3:
        return None
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(feats)
        return feats[hull.vertices]
    except Exception:
        return None


def _point_in_hull_2d(pt: np.ndarray, hull_pts: np.ndarray) -> bool:
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


def query_live_map(samples: list[dict], du: float, dv: float, *, k: int = 3) -> dict | None:
    if not samples:
        return None
    index = build_index(samples)
    return query(index, du, dv, k=min(k, len(samples)))


def nearest_sample(samples: list[dict], du: float, dv: float) -> tuple[dict, float]:
    best = samples[0]
    best_d = float("inf")
    q = np.array([du, dv])
    for s in samples:
        d = float(np.linalg.norm(np.array([s["du"], s["dv"]]) - q))
        if d < best_d:
            best_d = d
            best = s
    return best, best_d


def coverage_hints(samples: list[dict], du: float, dv: float) -> list[str]:
    """Where more probes help vs region looks covered."""
    hints: list[str] = []
    if not samples:
        hints.append("First map anchor — no prior estimate (grasp FK becomes truth here).")
        return hints

    feats = np.array([[s["du"], s["dv"]] for s in samples], dtype=float)
    fks = np.stack([s["fk"] for s in samples], axis=0)
    hull = _du_dv_hull(feats)
    q = np.array([du, dv])
    _, near_px = nearest_sample(samples, du, dv)

    if hull is not None:
        inside = _point_in_hull_2d(q, hull)
        if inside:
            hints.append(f"Inside image hull ({len(samples)} probes) — interpolation OK if grasp is good.")
        else:
            hints.append(
                f"OUTSIDE hull — extrapolating ({near_px:.0f} px to nearest). "
                "Add probes along the edge toward this spot.",
            )
    else:
        hints.append(f"Only {len(samples)} probe(s) — need ≥3 spread spots before hull is meaningful.")

    if near_px > 80:
        hints.append(f"Far from nearest probe ({near_px:.0f} px) — add intermediate samples.")

    ys = fks[:, 1]
    y_span = float(ys.max() - ys.min())
    if y_span < 0.08 and len(samples) >= 3:
        hints.append(
            f"Y coverage narrow ({y_span*100:.0f} cm FK span) — place cubes more forward/back on table.",
        )

    du_span = float(feats[:, 0].max() - feats[:, 0].min())
    dv_span = float(feats[:, 1].max() - feats[:, 1].min())
    if du_span < 120 or dv_span < 120:
        hints.append(
            f"Image spread du={du_span:.0f} dv={dv_span:.0f} px — spread cubes left/right and up/down in view.",
        )

    return hints


def convergence_hints(
    errors_xy_mm: list[float],
    *,
    inside_hull: bool | None,
) -> list[str]:
    if len(errors_xy_mm) < 2:
        return ["Need ≥2 validated probes to judge convergence."]
    recent = errors_xy_mm[-3:]
    med = float(np.median(recent))
    if med < 25 and inside_hull is True:
        return [f"Recent est≈meas xy median {med:.0f} mm in hull — map may be converging here."]
    if med > 50:
        return [f"Recent est≈meas xy median {med:.0f} mm — large; check grasp or add probes nearby."]
    return [f"Recent est≈meas xy median {med:.0f} mm — keep filling gaps if outside hull."]


def print_pre_grasp_block(
    *,
    trial: int,
    du: float,
    dv: float,
    pu: float,
    pv: float,
    samples: list[dict],
    k: int = 3,
) -> dict | None:
    """After home photo, before grasp — show MAP estimate + coverage."""
    print("\n" + "=" * 62)
    print(f"  TRIAL {trial:03d} — MAP ESTIMATE (from {len(samples)} prior probe(s))")
    print("=" * 62)
    print(f"  home pixel (u,v): ({pu:.0f}, {pv:.0f})   offset (du,dv): ({du:+.0f}, {dv:+.0f}) px")

    q = query_live_map(samples, du, dv, k=k)
    if q is None:
        print("  (no prior samples — grasp FK will be first anchor)")
        print("=" * 62)
        return None

    est = np.asarray(q["fk_xyz_user_m"], dtype=float)
    near, near_px = nearest_sample(samples, du, dv)
    nf = near["fk"]
    feats = np.array([[s["du"], s["dv"]] for s in samples], dtype=float)
    hull = _du_dv_hull(feats)
    inside = _point_in_hull_2d(np.array([du, dv]), hull) if hull is not None else None

    print(f"  kNN neighbors:     trials {q['neighbor_trials']}  dist_px {q['neighbor_dist_px']}")
    print(f"  MAP estimate:      ({est[0]:+.3f}, {est[1]:+.3f}, {est[2]:+.3f}) m  (user frame)")
    print(f"  nearest prior:     trial {near['trial']:03d}  ({near_px:.0f} px)")
    print(f"                     ({nf[0]:+.3f}, {nf[1]:+.3f}, {nf[2]:+.3f}) m")
    for line in coverage_hints(samples, du, dv):
        print(f"  · {line}")
    print("=" * 62)
    print("  Next: grasp probe — measured FK will be compared to estimate above.")
    return {
        "est_xyz_user_m": est.tolist(),
        "neighbor_trials": q["neighbor_trials"],
        "nearest_train_trial": int(near["trial"]),
        "nearest_train_dist_px": float(near_px),
        "in_map_hull": inside,
    }


def print_post_grasp_block(
    *,
    trial: int,
    est_info: dict | None,
    measured: np.ndarray,
    errors_xy_mm: list[float],
) -> float:
    """After grasp FK recorded."""
    print("\n" + "=" * 62)
    print(f"  TRIAL {trial:03d} — VALIDATION (estimate vs measured grasp FK)")
    print("=" * 62)
    print(f"  measured FK:       ({measured[0]:+.3f}, {measured[1]:+.3f}, {measured[2]:+.3f}) m")

    err_xy = float("nan")
    if est_info is not None:
        est = np.asarray(est_info["est_xyz_user_m"], dtype=float)
        err_xy = float(np.linalg.norm(est[:2] - measured[:2]) * 1000)
        err_z = float((est[2] - measured[2]) * 1000)
        print(f"  MAP estimate was:  ({est[0]:+.3f}, {est[1]:+.3f}, {est[2]:+.3f}) m")
        print(f"  err est − meas:    xy {err_xy:.1f} mm   z {err_z:+.1f} mm")
        if err_xy < 30:
            print("  >> Good match — this row strengthens the map here.")
        elif err_xy < 50:
            print("  >> Moderate error — usable rough XY; Y may still be off.")
        else:
            print("  >> Large XY error — consider excluding trial or re-grasping.")
        inside = est_info.get("in_map_hull")
        for line in convergence_hints(errors_xy_mm + [err_xy], inside_hull=inside):
            print(f"  · {line}")
    else:
        print("  (no prior estimate — first anchor added to map)")

    print("=" * 62)
    print("  Row appended to mapping_samples.csv for this trial.")
    return err_xy
