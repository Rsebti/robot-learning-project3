"""
probe_map_feasibility.py — sanity-check probe map before running approach on the robot.

Reads mapping_samples.csv, prints table-space coverage, workspace checks, and
the staged approach targets (hover / side / down) for each trial and on a grid.

Usage:
    python -m toolset.perception.probe_map_feasibility `
        --session_dir deploy/_snaps/probe_1779205245

    python -m toolset.perception.probe_map_feasibility `
        --session_dir deploy/_snaps/probe_1779205245 `
        --du -100 --dv 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolset.kinematics.config import KinematicsConfig
from toolset.perception.approach_plan import (
    plan_approach_waypoints,
    user_xyz_to_urdf,
)
from toolset.perception.probe_map_data import (
    estimate_xyz_from_pixel,
    load_mapping_rows,
    write_refined_mapping_csv,
)
from toolset.perception.refined_cube_detect import pixel_to_du_dv


def _hull_area_xy(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    pts = points[:, :2]
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session_dir", type=Path, required=True)
    p.add_argument("--du", type=float, default=None)
    p.add_argument("--dv", type=float, default=None)
    p.add_argument("--hover_above_m", type=float, default=0.03)
    p.add_argument("--side_offset_m", type=float, default=0.03)
    p.add_argument("--descend_m", type=float, default=0.01)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--raw_map", action="store_true",
                   help="Skip refined masks; use mapping_samples.csv pixels only.")
    args = p.parse_args()

    session_dir = args.session_dir.resolve()
    kcfg = KinematicsConfig.load()
    use_refined = not args.raw_map
    if use_refined:
        write_refined_mapping_csv(session_dir)
    rows_raw = load_mapping_rows(session_dir, use_refined=False)
    rows = load_mapping_rows(session_dir, use_refined=use_refined)
    if not rows:
        raise SystemExit(f"No mapping rows in {session_dir}")

    fks = np.stack([r["fk"] for r in rows], axis=0)
    dus = np.array([r["du"] for r in rows])
    dvs = np.array([r["dv"] for r in rows])

    lines = [
        "probe map feasibility report",
        f"session: {session_dir}",
        f"n_samples: {len(rows)}",
        f"pixel source: {'refined masks (bright-top + wall fit)' if use_refined else 'raw probe capture'}",
        "",
    ]
    if use_refined and rows_raw:
        lines.append("=== Raw vs refined pixel (FK unchanged) ===")
        raw_by_trial = {r["trial"]: r for r in rows_raw}
        for r in rows:
            raw = raw_by_trial.get(r["trial"])
            if raw is None:
                continue
            dpx = float(np.hypot(r["pixel_u"] - raw["pixel_u"], r["pixel_v"] - raw["pixel_v"]))
            lines.append(
                f"  trial {r['trial']:03d}  raw du={raw['du']:+.0f} dv={raw['dv']:+.0f}  "
                f"refined du={r['du']:+.0f} dv={r['dv']:+.0f}  shift={dpx:.1f} px"
            )
        lines.append("")

    lines.extend([
        "=== Table positions from probe FK (user m: right+, forward+, up+) ===",
        f"  x: min={fks[:,0].min():+.3f}  max={fks[:,0].max():+.3f}  "
        f"span={fks[:,0].max()-fks[:,0].min():.3f}",
        f"  y: min={fks[:,1].min():+.3f}  max={fks[:,1].max():+.3f}  "
        f"span={fks[:,1].max()-fks[:,1].min():.3f}",
        f"  z: min={fks[:,2].min():+.3f}  max={fks[:,2].max():+.3f}  "
        f"mean={fks[:,2].mean():+.3f}  std={fks[:,2].std():.3f}",
        f"  map hull area (XY): ~{_hull_area_xy(fks) * 1e4:.0f} cm²",
        "",
        f"  pixel du: [{dus.min():+.0f}, {dus.max():+.0f}]  "
        f"dv: [{dvs.min():+.0f}, {dvs.max():+.0f}]",
        "",
        "=== Per-trial: map center + approach stages (cube at FK xy, z from table) ===",
        "",
    ])

    n_ws_ok = 0
    for r in rows:
        grasp = r.get("grasp_motor_deg")
        if grasp is None:
            continue
        plan = plan_approach_waypoints(
            kcfg,
            cube_xy_user=r["fk"][:2],
            grasp_motor_deg=grasp,
            hover_above_m=args.hover_above_m,
            side_offset_m=args.side_offset_m,
            descend_m=args.descend_m,
        )
        ws_flags = []
        for name, pt in [
            ("hover", plan["hover_user"]),
            ("side", plan["side_user"]),
            ("down", plan["down_user"]),
        ]:
            ok = kcfg.in_workspace(user_xyz_to_urdf(pt))
            ws_flags.append(f"{name}:{'ok' if ok else 'OOW'}")
        if all(kcfg.in_workspace(user_xyz_to_urdf(p)) for p in [
            plan["hover_user"], plan["side_user"], plan["down_user"],
        ]):
            n_ws_ok += 1
        lines.append(
            f"trial {r['trial']:03d}  du={r['du']:+.0f} dv={r['dv']:+.0f}  "
            f"fk=({r['fk'][0]:+.3f},{r['fk'][1]:+.3f},{r['fk'][2]:+.3f})"
        )
        lines.append(
            f"         hover=({plan['hover_user'][0]:+.3f},{plan['hover_user'][1]:+.3f},"
            f"{plan['hover_user'][2]:+.3f})  "
            f"side=({plan['side_user'][0]:+.3f},{plan['side_user'][1]:+.3f})  "
            f"down_z={plan['down_user'][2]:+.3f}  "
            f"{' '.join(ws_flags)}"
        )

    lines.extend([
        "",
        f"workspace: {n_ws_ok}/{len(rows)} trials all stages in workspace",
        "",
    ])

    if len(rows) >= 2:
        dists = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                dists.append(float(np.linalg.norm(fks[i, :2] - fks[j, :2])))
        lines.append(
            f"XY spacing between samples: min={min(dists)*100:.1f} cm  "
            f"median={np.median(dists)*100:.1f} cm  max={max(dists)*100:.1f} cm"
        )
        if min(dists) > 0.08:
            lines.append("  WARNING: sparse map (>8 cm nearest neighbor) — probe more trials.")
        elif min(dists) < 0.03:
            lines.append("  NOTE: some samples very close — redundant but fine.")
        lines.append("")

    if args.du is not None and args.dv is not None:
        est = estimate_xyz_from_pixel(
            session_dir, args.du, args.dv, k=args.k, use_refined=use_refined,
        )
        xyz = est["fk_xyz_user_m"]
        grasp = est["neighbors"][0]["grasp_motor_deg"] if est["neighbors"] else rows[0]["grasp_motor_deg"]
        plan = plan_approach_waypoints(
            kcfg, cube_xy_user=xyz[:2], grasp_motor_deg=grasp,
            hover_above_m=args.hover_above_m,
            side_offset_m=args.side_offset_m,
            descend_m=args.descend_m,
        )
        lines.extend([
            "=== Query point ===",
            f"  du={args.du:+.0f} dv={args.dv:+.0f}",
            f"  neighbors={est['neighbor_trials']}  dist_px={[round(x,1) for x in est['neighbor_dist_px']]}",
            f"  est fk=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f})",
            f"  hover=({plan['hover_user'][0]:+.3f},{plan['hover_user'][1]:+.3f},{plan['hover_user'][2]:+.3f})",
            f"  side_xy=({plan['side_user'][0]:+.3f},{plan['side_user'][1]:+.3f})",
            f"  down_z={plan['down_user'][2]:+.3f}",
            "",
        ])

    report = session_dir / "map_feasibility_report.txt"
    text = "\n".join(lines) + "\n"
    report.write_text(text, encoding="utf-8")
    print(text)
    print(f"[feasibility] wrote {report}")


if __name__ == "__main__":
    main()
