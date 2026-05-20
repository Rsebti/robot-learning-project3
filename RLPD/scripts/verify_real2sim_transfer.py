#!/usr/bin/env python3
"""
verify_real2sim_transfer.py — QA real teleop vs Isaac replay.

Reads ``RLPD/data/episodes.json`` (pre-extracted). No FK annotation scripts needed.

CPU (real wrist frames + HTML):
  python RLPD/scripts/verify_real2sim_transfer.py --skip_sim --episodes 0 1 2 3 4

CUDA (Isaac kinematic check):
  python RLPD/scripts/verify_real2sim_transfer.py --episodes 0 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_RLPD = Path(__file__).resolve().parents[1]
_PROJECT = _RLPD.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

_DEFAULT_ANN = _RLPD / "data" / "episodes.json"


def _rgb_to_uint8_hwc(img) -> np.ndarray | None:
    if img is None:
        return None
    arr = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return arr


def _global_frame_index(episodes_meta: dict, ep_index: int, frame_local: int) -> int:
    """Map (episode, local frame) → flat LeRobot dataset index."""
    offset = 0
    for ep in sorted(int(k) for k in episodes_meta):
        if ep == ep_index:
            return offset + min(frame_local, episodes_meta[str(ep)]["n_frames"] - 1)
        offset += int(episodes_meta[str(ep)]["n_frames"])
    raise KeyError(f"episode {ep_index} not in annotations")


def load_real_grasp_image(
    repo_id: str,
    ep_index: int,
    frame_local: int,
    *,
    episodes_meta: dict | None = None,
) -> np.ndarray | None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        return None
    ds = LeRobotDataset(repo_id)
    if episodes_meta is not None:
        idx = _global_frame_index(episodes_meta, ep_index, frame_local)
    elif hasattr(ds, "episode_data_index") and ds.episode_data_index is not None:
        i0 = int(ds.episode_data_index["from"][ep_index])
        i1 = int(ds.episode_data_index["to"][ep_index])
        idx = min(i0 + frame_local, i1 - 1)
    else:
        return None
    return _rgb_to_uint8_hwc(ds[idx]["observation.images.wrist"])


def run_isaac_metrics(annotations: Path, ep_index: int, *, task: str, subsample: int) -> dict:
    import gymnasium as gym
    from isaaclab.app import AppLauncher
    from isaaclab_tasks.utils import parse_env_cfg

    import sim.eval2  # noqa: F401
    _script_dir = Path(__file__).resolve().parent
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))
    from isaac_replay_common import set_cube_pose, set_robot_qpos_deg, tip_user_from_motor_deg
    from toolset.kinematics.ik_relative import urdf_xyz_to_user
    from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig
    from toolset.kinematics.config import KinematicsConfig
    from toolset.kinematics.urdf_fk import SO101FK

    data = json.loads(annotations.read_text(encoding="utf-8"))
    ep = data["episodes"][str(ep_index)]
    traj = ep["trajectory_observation_state_deg"][::subsample]
    grasp_local = int(ep["grasp_frame_local"]) // subsample

    kcfg = KinematicsConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)
    mcfg = MotorToUrdfConfig.load()

    launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = launcher.app

    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=1)
    env = gym.make(task, cfg=env_cfg)
    base = env.unwrapped

    env.reset(seed=ep_index + 3000)
    set_cube_pose(base, ep["cube_pose_urdf_world"])
    for _ in range(20):
        base.sim.step()
        base.scene.update(dt=base.physics_dt)

    tip_errors_mm = []
    for motor_deg in traj:
        set_robot_qpos_deg(base, motor_deg)
        base.sim.step()
        base.scene.update(dt=base.physics_dt)
        tip_user = tip_user_from_motor_deg(motor_deg)
        q_urdf = mcfg.motor_to_urdf_rad(np.asarray(motor_deg, dtype=float))
        tip_sim_user = urdf_xyz_to_user(fk.fk(q_urdf, target="gripper_tip")["position"])
        tip_errors_mm.append(float(np.linalg.norm(tip_user - tip_sim_user) * 1000.0))

    g = min(grasp_local, len(tip_errors_mm) - 1)
    env.close()
    simulation_app.close()

    return {
        "episode_index": ep_index,
        "tcp_err_mm_median": float(np.median(tip_errors_mm)),
        "tcp_err_mm_max": float(np.max(tip_errors_mm)),
        "tcp_err_mm_at_grasp": float(tip_errors_mm[g]),
    }


def build_gallery(verify_dir: Path, rows: list[dict], tcp_good_mm: float) -> None:
    parts = []
    for r in rows:
        ep = r["episode_index"]
        ok = r.get("transfer_ok", False)
        color = "#2d6a4f" if ok else "#9b2226"
        parts.append(f"""
<div class="card" style="border-color:{color}">
<h3>ep {ep} <span>{'GOOD' if ok else 'BAD'}</span></h3>
<p>tcp@grasp={r.get('tcp_err_mm_at_grasp', 'n/a')} mm |
   hold={r.get('grasp_static_run_len','?')} fr |
   grip={r.get('grasp_gripper_deg','?')} deg</p>
<p>cube user m: {r.get('cube_xyz_user_m')}</p>
<img src="ep_{ep:03d}/real_at_grasp.png" width="280"/>
<img src="ep_{ep:03d}/sim_at_grasp.png" width="280"/>
</div>""")
    (verify_dir / "gallery.html").write_text(
        f"<!DOCTYPE html><html><body style='background:#111;color:#eee'>"
        f"<h1>RLPD real2sim Isaac QA</h1><p>GOOD if tcp@grasp &lt; {tcp_good_mm} mm</p>{''.join(parts)}</body></html>",
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, default=_DEFAULT_ANN)
    p.add_argument("--episodes", type=int, nargs="*")
    p.add_argument("--task", default="Isaac-SquintNative-Place-Replay-v0")
    p.add_argument("--subsample", type=int, default=3)
    p.add_argument("--tcp_good_mm", type=float, default=30.0)
    p.add_argument("--skip_sim", action="store_true")
    args = p.parse_args()

    import cv2
    import pandas as pd

    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    repo_id = data.get("repo_id", "Rsebti/projet3_demos_v1")
    verify_dir = _RLPD / "data" / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    ep_list = sorted(int(k) for k in data["episodes"]) if not args.episodes else args.episodes

    summary = []
    for ep in ep_list:
        meta = data["episodes"][str(ep)]
        ep_out = verify_dir / f"ep_{ep:03d}"
        ep_out.mkdir(parents=True, exist_ok=True)
        real = load_real_grasp_image(
            repo_id, ep, int(meta["grasp_frame_local"]), episodes_meta=data["episodes"]
        )
        if real is not None:
            cv2.imwrite(str(ep_out / "real_at_grasp.png"), cv2.cvtColor(real, cv2.COLOR_RGB2BGR))

        row = {
            "episode_index": ep,
            "grasp_gripper_deg": meta.get("grasp_gripper_deg"),
            "grasp_static_run_len": meta.get("grasp_static_run_len"),
            "cube_xyz_user_m": meta["cube_xyz_user_m"],
            "grasp_pick_reason": meta.get("grasp_pick_reason"),
        }
        if not args.skip_sim:
            try:
                row.update(run_isaac_metrics(args.annotations, ep, task=args.task, subsample=args.subsample))
            except Exception as exc:
                row["sim_error"] = str(exc)
                print(f"[verify] sim ep {ep} failed: {exc}")
        row["transfer_ok"] = row.get("tcp_err_mm_at_grasp", 999.0) <= args.tcp_good_mm
        (ep_out / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        summary.append(row)
        print(f"  ep {ep}: tcp={row.get('tcp_err_mm_at_grasp','skip')}  {'OK' if row.get('transfer_ok') else 'BAD'}")

    pd.DataFrame(summary).to_csv(verify_dir / "transfer_summary.csv", index=False)
    build_gallery(verify_dir, summary, args.tcp_good_mm)
    print(f"[verify] gallery: {verify_dir / 'gallery.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
