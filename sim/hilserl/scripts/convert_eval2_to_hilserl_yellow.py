#!/usr/bin/env python3
"""Convert yellow-target episodes from osammotg1/projet3-eval2-v1-tom-hugo
(joint-space ACT/SmolVLA dataset) into HIL-SERL EE-space demo format.

This is the CHEAP PROBE — it produces a single combined parquet at:
    outputs/datasets/projet3-hilserl-yellow-v1-converted/yellow_ee_frames.parquet

The probe answers two questions before we invest in a full HF dataset rebuild:
  1. Does FK on the recorded joint trajectories produce sensible EE poses?
     (smooth, within the SO-101 workspace, gripper opens/closes at the right moments)
  2. Are the resulting EE deltas within the `end_effector_step_sizes: 0.02 m`
     bound we set in env_config_so101.json? (If many deltas exceed it,
     HIL-SERL's IK will clamp them and the demos will look choppy.)

If the probe looks good, the next step is to wrap this output as a proper
LeRobot v3 dataset (videos + meta) — that's a separate, heavier task.
If the probe looks bad, we re-record fresh demos at the robot.

Run:
    /Users/admin/miniforge3/bin/python3.13 \
        sim/hilserl/scripts/convert_eval2_to_hilserl_yellow.py
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
URDF_PATH = REPO_ROOT / "sim/hilserl/assets/so101/urdf/so_arm101.urdf"

SOURCE_DATASET = "osammotg1/projet3-eval2-v1-tom-hugo"
SOURCE_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / f"datasets--{SOURCE_DATASET.replace('/', '--')}"
    / "snapshots"
)
TARGET_TASK_INDEX = 0  # "Pick yellow block..."
SOURCE_FPS = 30

OUTPUT_DIR = REPO_ROOT / "outputs/datasets/projet3-hilserl-yellow-v1-converted"
OUTPUT_PARQUET = OUTPUT_DIR / "yellow_ee_frames.parquet"
OUTPUT_REPORT = OUTPUT_DIR / "conversion_report.md"

# The env_config_so101.json clamp — for the diagnostic table only.
EE_STEP_BOUND_M = 0.02


def find_latest_snapshot() -> Path:
    snaps = sorted(SOURCE_SNAPSHOT.glob("*"), key=lambda p: p.stat().st_mtime)
    if not snaps:
        raise FileNotFoundError(
            f"No snapshot under {SOURCE_SNAPSHOT}. Run snapshot_download first."
        )
    return snaps[-1]


def load_yellow_frames(snapshot: Path) -> pd.DataFrame:
    """Read all data parquets, return only rows for task_index == TARGET_TASK_INDEX."""
    parquet_files = sorted(glob.glob(str(snapshot / "data/chunk-*/*.parquet")))
    frames = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        yel = df[df["task_index"] == TARGET_TASK_INDEX]
        if len(yel) > 0:
            frames.append(yel)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["episode_index", "frame_index"]
    ).reset_index(drop=True)


def convert_episode(
    ep_df: pd.DataFrame,
    rk,  # RobotKinematics instance
) -> pd.DataFrame:
    """Convert one episode's joint-space frames into EE-space frames.

    Returns a dataframe with the new schema:
      - episode_index, frame_index, timestamp (copied)
      - observation.state_joint (6,)       — joint positions in degrees (unchanged)
      - observation.joint_velocity (6,)    — finite diff, degrees/sec
      - observation.ee_pose (3,)           — xyz in meters via FK
      - action_ee_delta (3,)               — ee_pose[t+1] - ee_pose[t], meters
      - action_gripper_delta (1,)          — gripper[t+1] - gripper[t]
      - source_action (6,)                 — original joint targets, kept for audit
    The LAST frame of each episode is DROPPED (no t+1 to diff against).
    """
    if len(ep_df) < 2:
        return pd.DataFrame()

    obs_state = np.stack(ep_df["observation.state"].values).astype(np.float64)
    src_action = np.stack(ep_df["action"].values).astype(np.float64)
    # obs_state shape: (T, 6), columns [shoulder_pan, lift, elbow, wrist_flex, wrist_roll, gripper] in DEGREES.
    # lerobot.model.kinematics.RobotKinematics.forward_kinematics(joint_pos_deg) does the deg->rad
    # conversion internally. Feed the raw degree values straight through.
    # The gripper joint barely affects gripper_frame_link FK (which is the wrist tip, not the jaw
    # tip), so zeroing it has no effect on EE position; we keep the recorded value for completeness.
    q_all = obs_state  # (T, 6) DEGREES, float64 — placo wrapper handles the conversion

    # FK each frame to get the EE xyz.
    ee_xyz = np.array([rk.forward_kinematics(q)[:3, 3] for q in q_all])  # (T, 3) meters

    # EE delta (action target for the SAC actor in EE-space).
    ee_delta = ee_xyz[1:] - ee_xyz[:-1]  # (T-1, 3)

    # Joint velocity (extra observation column).
    dt = 1.0 / SOURCE_FPS
    joint_vel = (obs_state[1:] - obs_state[:-1]) / dt  # (T-1, 6) deg/sec

    # Gripper delta (continuous).
    gripper_delta = obs_state[1:, 5:6] - obs_state[:-1, 5:6]  # (T-1, 1)

    out = pd.DataFrame({
        "episode_index": ep_df["episode_index"].values[:-1],
        "frame_index": ep_df["frame_index"].values[:-1],
        "timestamp": ep_df["timestamp"].values[:-1],
        "task_index": ep_df["task_index"].values[:-1],
        # New schema:
        "observation.state_joint": list(obs_state[:-1]),
        "observation.joint_velocity": list(joint_vel),
        "observation.ee_pose": list(ee_xyz[:-1]),
        "action_ee_delta": list(ee_delta),
        "action_gripper_delta": list(gripper_delta.squeeze(axis=-1)),
        # Audit columns:
        "source_action": list(src_action[:-1]),
        "source_obs_state": list(obs_state[:-1]),
    })
    return out


def main() -> int:
    print(f"Source dataset: {SOURCE_DATASET}")
    snapshot = find_latest_snapshot()
    print(f"Snapshot dir:   {snapshot}")

    print("\n[1/4] Loading yellow-target frames from parquets...")
    yellow_df = load_yellow_frames(snapshot)
    yellow_eps = sorted(yellow_df["episode_index"].unique().tolist())
    print(f"   episodes: {len(yellow_eps)}  ({yellow_eps})")
    print(f"   frames:   {len(yellow_df)}")

    print("\n[2/4] Loading placo kinematics solver...")
    # Late import to avoid placo's startup warnings polluting the report header.
    from lerobot.model.kinematics import RobotKinematics  # noqa: E402
    rk = RobotKinematics(urdf_path=str(URDF_PATH), target_frame_name="gripper_frame_link")
    print(f"   joints: {rk.joint_names}")

    print("\n[3/4] Per-episode FK + delta conversion...")
    converted_episodes: list[pd.DataFrame] = []
    for ep_idx in yellow_eps:
        ep_df = yellow_df[yellow_df["episode_index"] == ep_idx].reset_index(drop=True)
        out = convert_episode(ep_df, rk)
        if not out.empty:
            converted_episodes.append(out)
            print(f"   ep {ep_idx:3d}: {len(ep_df):4d} src frames → {len(out):4d} converted")
    converted = pd.concat(converted_episodes, ignore_index=True)
    print(f"   total converted frames: {len(converted)}")

    print(f"\n[4/4] Writing output to {OUTPUT_PARQUET} ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    converted.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"   wrote {OUTPUT_PARQUET.stat().st_size // 1024} KB")

    # --- Diagnostic report --------------------------------------------------
    print("\n=== Sanity report ===")
    ee_xyz_all = np.stack(converted["observation.ee_pose"].values)
    ee_delta_all = np.stack(converted["action_ee_delta"].values)
    grip_delta_all = converted["action_gripper_delta"].values

    ee_range = (ee_xyz_all.min(axis=0), ee_xyz_all.max(axis=0))
    delta_abs = np.abs(ee_delta_all)
    over_bound_pct = 100.0 * (delta_abs.max(axis=1) > EE_STEP_BOUND_M).mean()

    report_lines = [
        f"# Conversion report — yellow → EE space\n",
        f"Source: `{SOURCE_DATASET}` (snapshot `{snapshot.name[:12]}`)\n",
        f"Frames converted: **{len(converted)}** across **{len(yellow_eps)}** episodes\n",
        f"\n## EE pose range (meters, robot frame)\n",
        f"- X: min={ee_range[0][0]:+.3f}  max={ee_range[1][0]:+.3f}\n",
        f"- Y: min={ee_range[0][1]:+.3f}  max={ee_range[1][1]:+.3f}\n",
        f"- Z: min={ee_range[0][2]:+.3f}  max={ee_range[1][2]:+.3f}\n",
        f"\n## EE deltas (meters per 30-fps frame)\n",
        f"- |dx| mean={delta_abs[:,0].mean():.4f}  max={delta_abs[:,0].max():.4f}\n",
        f"- |dy| mean={delta_abs[:,1].mean():.4f}  max={delta_abs[:,1].max():.4f}\n",
        f"- |dz| mean={delta_abs[:,2].mean():.4f}  max={delta_abs[:,2].max():.4f}\n",
        f"- Frames where max-axis delta > {EE_STEP_BOUND_M} m: **{over_bound_pct:.1f}%**\n",
        f"\n## Gripper deltas (raw units per 30-fps frame)\n",
        f"- min={grip_delta_all.min():+.3f}  mean={grip_delta_all.mean():+.4f}  max={grip_delta_all.max():+.3f}\n",
        f"\n## Verdict knobs\n",
        f"- ✅ healthy if EE range fits inside the find_joint_limits bounds we'll measure tomorrow\n",
        f"- ⚠️ if `> {EE_STEP_BOUND_M} m` percentage is HIGH (>30%), SAC will be cloning saturated/clamped actions; consider raising end_effector_step_sizes to 0.025 or 0.03 in env_config\n",
        f"- ⚠️ if gripper delta is mostly 0 (mean near 0, max small), the gripper signal is too weak to learn — re-record\n",
    ]
    OUTPUT_REPORT.write_text("".join(report_lines))
    print("\n".join(line.rstrip() for line in report_lines))
    print(f"\n   Report saved to {OUTPUT_REPORT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
