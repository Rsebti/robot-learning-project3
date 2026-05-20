#!/usr/bin/env python3
"""
annotate_grasp_fk.py — per-episode cube pose from quasi-static grasp FK.

Dataset default: Rsebti/projet3_demos_v1 (LeRobot v3, 6 motor deg + wrist video).

Assumption (real2sim-lite):
  At grasp, the arm is quasi-static and the gripper tip FK position maps to the
  cube: XY = tip XY, Z = tip_z - cube_half_height (tip resting on top face).

Outputs under --output_dir:
  grasp_cube_annotations.csv   one row per episode
  episodes.json                episode metadata + joint trajectories for replay
  annotation_report.md         summary stats

Usage:
  cd project3
  python -m sim.demo_replay.annotate_grasp_fk --download
  python -m sim.demo_replay.annotate_grasp_fk
  python -m sim.demo_replay.annotate_grasp_fk --repo_id Rsebti/projet3_demos_v1 --output_dir outputs/datasets/projet3_demos_v1_grasp_fk
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]

from sim.demo_replay.grasp_fk_common import (  # noqa: E402
    MOTOR_NAMES,
    cube_center_user_from_tip,
    cube_pose_urdf_world,
    episode_record_for_isaac,
    hf_snapshot_dir,
    load_parquet_dataset,
    make_fk_stack,
    pick_grasp_frame_rollout,
    stack_state_column,
    tip_user_from_motor_deg,
    user_xyz_to_urdf,
)


DEFAULT_REPO = "Rsebti/projet3_demos_v1"
DEFAULT_OUTPUT = _REPO_ROOT / "outputs/datasets/projet3_demos_v1_grasp_fk"


def _download_dataset(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=repo_id, repo_type="dataset")
    print(f"[annotate] downloaded / cached at {path}")
    return Path(path)


def _find_snapshot(repo_id: str, root: Path | None) -> Path:
    if root is not None:
        root = Path(root)
        if (root / "meta" / "info.json").exists() or list((root / "data").glob("chunk-*/*.parquet")):
            return root
        raise FileNotFoundError(f"--dataset_root {root} is not a LeRobot dataset root.")
    try:
        return hf_snapshot_dir(repo_id)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No local cache for {repo_id}. Re-run with --download."
        ) from exc


def annotate_episode(
    ep_df: pd.DataFrame,
    ep_index: int,
    *,
    kcfg,
    fk,
    mcfg,
    fps: float,
    max_arm_vel_deg_s: float,
    min_static_frames: int,
    gripper_closed_max_deg: float,
    cube_z_mode: str,
) -> dict:
    obs = stack_state_column(ep_df["observation.state"])
    act = stack_state_column(ep_df["action"]) if "action" in ep_df.columns else obs.copy()

    fi_local, diag = pick_grasp_frame_rollout(
        obs,
        fps=fps,
        max_arm_vel_deg_s=max_arm_vel_deg_s,
        min_static_frames=min_static_frames,
        gripper_closed_max_deg=gripper_closed_max_deg,
    )
    frame_index = int(ep_df["frame_index"].iloc[fi_local])
    grasp_motor = obs[fi_local].tolist()
    tip_user = tip_user_from_motor_deg(grasp_motor, fk, mcfg)
    cube_user = cube_center_user_from_tip(tip_user, kcfg, z_mode=cube_z_mode)
    cube_urdf = user_xyz_to_urdf(cube_user)
    table_z_expected = kcfg.table_z_m + kcfg.cube_half_height_m
    z_err_mm = (cube_user[2] - table_z_expected) * 1000.0

    in_ws = kcfg.in_workspace(cube_urdf)

    return {
        "episode_index": int(ep_index),
        "n_frames": int(len(ep_df)),
        "grasp_frame_index": frame_index,
        "grasp_frame_local": int(fi_local),
        "grasp_pick_reason": diag["reason"],
        "grasp_max_arm_vel_deg_s": float(diag["max_arm_vel_deg_s"]),
        "grasp_gripper_deg": float(diag["gripper_deg"]),
        "grasp_static_run_len": int(diag.get("static_run_len", 0)),
        "grasp_motor_deg": [float(x) for x in grasp_motor],
        "tip_xyz_user_m": [float(x) for x in tip_user],
        "cube_xyz_user_m": [float(x) for x in cube_user],
        "cube_xyz_urdf_m": [float(x) for x in cube_urdf],
        "cube_pose_urdf_world": cube_pose_urdf_world(cube_user),
        "cube_z_mode": cube_z_mode,
        "table_z_expected_user_m": float(table_z_expected),
        "cube_z_err_vs_table_mm": float(z_err_mm),
        "in_workspace": bool(in_ws),
        "trajectory_observation_state_deg": obs.tolist(),
        "trajectory_action_deg": act.tolist(),
        "trajectory_frame_index": ep_df["frame_index"].astype(int).tolist(),
        "trajectory_timestamp_s": ep_df["timestamp"].astype(float).tolist()
        if "timestamp" in ep_df.columns
        else None,
    }


def write_report(rows: list[dict], out_dir: Path, repo_id: str, meta: dict) -> None:
    lines = [
        f"# Grasp FK annotations — `{repo_id}`",
        "",
        f"- Episodes: **{len(rows)}**",
        f"- FPS: **{meta.get('fps', '?')}**",
        f"- Cube Z mode: **{meta.get('cube_z_mode', '?')}**",
        "",
        "## Summary",
        "",
    ]
    if not rows:
        lines.append("_No episodes._")
    else:
        z_err = [r["cube_z_err_vs_table_mm"] for r in rows]
        in_ws = sum(1 for r in rows if r["in_workspace"])
        reasons = {}
        for r in rows:
            reasons[r["grasp_pick_reason"]] = reasons.get(r["grasp_pick_reason"], 0) + 1
        lines.extend([
            f"- In workspace: **{in_ws}/{len(rows)}**",
            f"- cube_z err vs table (mm): median **{np.median(z_err):.1f}**, "
            f"max **{np.max(np.abs(z_err)):.1f}**",
            f"- Grasp pick reasons: `{reasons}`",
            "",
            "## Columns (CSV)",
            "",
            "`cube_xyz_user_m` / `cube_pose_urdf_world` → spawn cube in Isaac replay.",
            "`grasp_motor_deg` → FK audit; same frame used for cube XY.",
            "",
        ])
    (out_dir / "annotation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", default=DEFAULT_REPO)
    p.add_argument("--dataset_root", type=Path, default=None,
                   help="Local LeRobot dataset root (else HF cache / --download).")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--download", action="store_true", help="snapshot_download from HF if missing.")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max_arm_vel_deg_s", type=float, default=8.0,
                   help="Arm quasi-static threshold (max joint speed deg/s).")
    p.add_argument("--min_static_frames", type=int, default=8,
                   help="Min consecutive quasi-static frames for a 'clamped' hold (~0.27s @ 30Hz).")
    p.add_argument("--gripper_closed_max_deg", type=float, default=35.0,
                   help="Gripper pos (deg) must be <= this during the hold.")
    p.add_argument("--cube_z_mode", choices=["tip_minus_half", "tip_z", "table_plus_half"],
                   default="table_plus_half",
                   help="Pick-place demos: table_plus_half; probe top-grasp: tip_minus_half.")
    p.add_argument("--max_episodes", type=int, default=None)
    args = p.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        snapshot = _download_dataset(args.repo_id)
    else:
        snapshot = _find_snapshot(args.repo_id, args.dataset_root)

    print(f"[annotate] repo={args.repo_id}")
    print(f"[annotate] snapshot={snapshot}")

    info_path = snapshot / "meta" / "info.json"
    meta = {}
    if info_path.exists():
        meta = json.loads(info_path.read_text(encoding="utf-8"))
        args.fps = float(meta.get("fps", args.fps))

    print("[annotate] loading parquet ...")
    df = load_parquet_dataset(snapshot)
    kcfg, fk, mcfg = make_fk_stack()

    eps = sorted(df["episode_index"].unique().tolist())
    if args.max_episodes is not None:
        eps = eps[: args.max_episodes]

    rows: list[dict] = []
    episodes_json: dict[str, dict] = {}

    for ep in eps:
        ep_df = df[df["episode_index"] == ep].sort_values("frame_index")
        row = annotate_episode(
            ep_df,
            int(ep),
            kcfg=kcfg,
            fk=fk,
            mcfg=mcfg,
            fps=args.fps,
            max_arm_vel_deg_s=args.max_arm_vel_deg_s,
            min_static_frames=args.min_static_frames,
            gripper_closed_max_deg=args.gripper_closed_max_deg,
            cube_z_mode=args.cube_z_mode,
        )
        rows.append(row)
        episodes_json[str(ep)] = episode_record_for_isaac(row)
        print(
            f"  ep {ep:3d}  frames={row['n_frames']:4d}  grasp_f={row['grasp_frame_index']:4d}  "
            f"cube_user=({row['cube_xyz_user_m'][0]:+.3f}, {row['cube_xyz_user_m'][1]:+.3f}, "
            f"{row['cube_xyz_user_m'][2]:+.3f})  z_err={row['cube_z_err_vs_table_mm']:+.1f}mm  "
            f"{row['grasp_pick_reason']}",
        )

    csv_rows = []
    for r in rows:
        flat = {k: v for k, v in r.items() if not k.startswith("trajectory_")}
        for i, name in enumerate(MOTOR_NAMES):
            flat[f"grasp_{name}_deg"] = r["grasp_motor_deg"][i]
        flat["tip_x_user_m"] = r["tip_xyz_user_m"][0]
        flat["tip_y_user_m"] = r["tip_xyz_user_m"][1]
        flat["tip_z_user_m"] = r["tip_xyz_user_m"][2]
        flat["cube_x_user_m"] = r["cube_xyz_user_m"][0]
        flat["cube_y_user_m"] = r["cube_xyz_user_m"][1]
        flat["cube_z_user_m"] = r["cube_xyz_user_m"][2]
        csv_rows.append(flat)

    csv_path = out_dir / "grasp_cube_annotations.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    ep_path = out_dir / "episodes.json"
    ep_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo_id": args.repo_id,
                "snapshot": str(snapshot),
                "fps": args.fps,
                "cube_z_mode": args.cube_z_mode,
                "grasp_selection": "static_hold_most_closed_last_frame",
                "episodes": episodes_json,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    meta_out = {
        "fps": args.fps,
        "cube_z_mode": args.cube_z_mode,
        "min_static_frames": args.min_static_frames,
    }
    write_report(rows, out_dir, args.repo_id, meta_out)

    print(f"\n[annotate] wrote {csv_path}")
    print(f"[annotate] wrote {ep_path}")
    print(f"[annotate] wrote {out_dir / 'annotation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
