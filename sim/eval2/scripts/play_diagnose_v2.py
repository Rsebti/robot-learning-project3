"""Enhanced diagnostic PLAY script — v2.

Extends play_diagnose.py with:
  1. Per-step reward decomposition (raw values + weighted total).
  2. Action command logging every N steps (raw_action, gripper, target vs actual).
  3. Grasp diagnostic events at GRASP_FIRE / grasp_LOST (rel vel, per-jaw dist, EE orientation).
  4. Goal pose tracking every M steps.
  5. Physics alarms (qdot > 1.5× limit, tip below table).
  6. Phase state machine per step (PRE_REACH/NEAR_CUBE/GRASPED/LIFTING/AT_GOAL/FAILURE).
  7. JSONL event log + human-readable summary file.
  8. Aggregate stats: phase reach rates, reward source decomposition, action saturation,
     grasp survival probabilities.

Preserved from v1: FIRST_CLOSE, GRASP_FIRE, LIFT, grasp_LOST, TIP_BELOW_TABLE, TIMEOUT,
plus continuous trace every N sim steps.

Output paths:
  ./play_logs/<task>_<timestamp>.jsonl       — machine-readable event log
  ./play_logs/<task>_<timestamp>.summary.txt — human-readable aggregate

Usage
-----
.. code-block:: powershell

    python -m sim.eval2.scripts.play_diagnose_v2 `
      --task Isaac-LeIsaac-SO101-Lift-Visual-V210-Play-v0 `
      --num_envs 4 --num_episodes 20 --enable_cameras
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort:skip

parser = argparse.ArgumentParser(description="Enhanced diagnostic play.")
parser.add_argument("--task", type=str, default="Isaac-LeIsaac-SO101-Lift-Visual-V29-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--trace_env", type=int, default=0)
parser.add_argument("--trace_every", type=int, default=5)
parser.add_argument("--action_log_every", type=int, default=5)
parser.add_argument("--goal_log_every", type=int, default=10)
parser.add_argument("--qdot_limit", type=float, default=10.0,
                    help="Joint velocity limit (rad/s). Alarm fires above 1.5×.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Late imports
import json
import os
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import combine_frame_transforms, quat_apply
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import sim.eval2  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


# =============================================================================
# Phase state machine
# =============================================================================

PHASE_PRE_REACH = "PRE_REACH"
PHASE_NEAR_CUBE = "NEAR_CUBE"
PHASE_GRASPED = "GRASPED"
PHASE_LIFTING = "LIFTING"
PHASE_AT_GOAL = "AT_GOAL"
PHASE_FAILURE = "FAILURE"

# Order matters — index = "depth" reached (higher = more progress).
PHASE_ORDER = [PHASE_PRE_REACH, PHASE_NEAR_CUBE, PHASE_GRASPED,
               PHASE_LIFTING, PHASE_AT_GOAL]


def classify_phase(prev_max_phase, gripper_cube_dist, is_grasped, is_lifted, is_at_goal):
    """Compute the deepest phase reached so far (forward-only)."""
    if is_at_goal:
        return PHASE_AT_GOAL
    if is_lifted and is_grasped:
        return _max_phase(prev_max_phase, PHASE_LIFTING)
    if is_grasped:
        return _max_phase(prev_max_phase, PHASE_GRASPED)
    if gripper_cube_dist < 0.08:
        return _max_phase(prev_max_phase, PHASE_NEAR_CUBE)
    return prev_max_phase


def _max_phase(a, b):
    return a if PHASE_ORDER.index(a) >= PHASE_ORDER.index(b) else b


# =============================================================================
# Predicates (match RewardsCfg V285+/V210)
# =============================================================================


def grasp_pred(jaw_cube_dist, gripper_q, diff_th=0.04, grip_th=0.35):
    return (jaw_cube_dist < diff_th) & (gripper_q < grip_th)


def lift_pred(cube_z_w, base_z_w, h_th=0.08):
    return (cube_z_w - base_z_w) > h_th


def goal_pred(cube_goal_dist, d_th=0.05):
    return cube_goal_dist < d_th


# =============================================================================
# Cube bounding box query (USD-level, runtime)
# =============================================================================


def query_cube_bbox(env_unwrapped) -> dict:
    """Return cube physical dimensions by querying USD prim bbox."""
    out = {"method": None, "size_xyz_m": None, "info": None}
    try:
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        # The cube prim path under the LeIsaac scene.
        prim_path = "/World/envs/env_0/Scene/cube"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            # Try alt paths
            for alt in ("/World/envs/env_0/cube",
                        "/World/envs/env_0/Scene/red_cube",
                        "/World/envs/env_0/Object"):
                prim = stage.GetPrimAtPath(alt)
                if prim.IsValid():
                    prim_path = alt
                    break
        if prim.IsValid():
            bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
            bbox = bbox_cache.ComputeWorldBound(prim)
            r = bbox.GetRange()
            size = r.GetSize()  # Gf.Vec3d
            out["method"] = "USD_BBOX"
            out["size_xyz_m"] = (float(size[0]), float(size[1]), float(size[2]))
            out["info"] = f"prim_path={prim_path}"
        else:
            out["info"] = "cube prim not found at expected paths"
    except Exception as e:
        out["info"] = f"USD bbox query failed: {type(e).__name__}: {e}"
    return out


# =============================================================================
# Quaternion helper for EE orientation
# =============================================================================


def quat_to_axis_z(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Apply quaternion to the z-axis unit vector → returns gripper-axis in world."""
    z_local = torch.zeros_like(quat_wxyz[..., 1:])
    z_local[..., 2] = 1.0
    return quat_apply(quat_wxyz, z_local)


# =============================================================================
# Main
# =============================================================================


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ---- Output paths ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_clean = task_name.replace("/", "_").replace(":", "_")
    out_dir = Path("play_logs")
    out_dir.mkdir(exist_ok=True)
    jsonl_path = out_dir / f"{task_clean}_{ts}.jsonl"
    summary_path = out_dir / f"{task_clean}_{ts}.summary.txt"
    print(f"[INFO] JSONL log     : {jsonl_path}")
    print(f"[INFO] Summary file  : {summary_path}")
    jsonl_f = open(jsonl_path, "w", encoding="utf-8")

    def log_jsonl(d):
        jsonl_f.write(json.dumps(d, default=str) + "\n")

    def log_event(d, also_print=True):
        log_jsonl(d)
        if also_print:
            t = d.get("type", "?")
            ep = d.get("ep_id", "")
            env_i = d.get("env", "")
            step = d.get("step", "")
            extra = " ".join(f"{k}={v}" for k, v in d.items()
                             if k not in {"type", "ep_id", "env", "step", "ts"})
            print(f"[{t} ep#{ep}/env{env_i} t={step}] {extra}")

    # ---- Resolve checkpoint ----
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    # ---- Env + policy ----
    env_cfg.log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_unwrapped = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env_unwrapped.device)

    # Scene handles
    cube: RigidObject = env_unwrapped.scene["cube"]
    robot: Articulation = env_unwrapped.scene["robot"]
    ee_frame = env_unwrapped.scene["ee_frame"]
    base_idx = robot.find_bodies("base")[0][0]
    gripper_body_idx = robot.find_bodies("gripper")[0][0]
    gripper_joint_idx = robot.joint_names.index("gripper")
    num_envs = args_cli.num_envs

    # ---- Scene info ----
    cube_info = query_cube_bbox(env_unwrapped)
    joint_pos_limits = robot.data.joint_pos_limits[0].cpu().tolist()
    scene_info = {
        "type": "SCENE_INFO",
        "task": task_name,
        "checkpoint": os.path.basename(resume_path),
        "num_envs": num_envs,
        "num_episodes_target": args_cli.num_episodes,
        "cube_bbox": cube_info,
        "joint_names": list(robot.joint_names),
        "joint_pos_limits": joint_pos_limits,
        "qdot_limit_used_for_alarm": args_cli.qdot_limit,
        "qdot_alarm_threshold": args_cli.qdot_limit * 1.5,
    }
    log_event(scene_info)

    print(f"\n=== SCENE INFO ===")
    print(f"  cube_bbox    : {cube_info}")
    print(f"  joints       : {robot.joint_names}")
    print(f"  qdot alarm   : > {args_cli.qdot_limit * 1.5} rad/s\n")

    # ---- Per-env state ----
    ep_step = torch.zeros(num_envs, dtype=torch.int32)
    ep_id = torch.arange(num_envs, dtype=torch.int32)
    first_close = torch.full((num_envs,), -1, dtype=torch.int32)
    first_grasp = torch.full((num_envs,), -1, dtype=torch.int32)
    first_lift = torch.full((num_envs,), -1, dtype=torch.int32)
    first_tip_below = torch.full((num_envs,), -1, dtype=torch.int32)
    grasp_count = torch.zeros(num_envs, dtype=torch.int32)
    grasp_steps = torch.zeros(num_envs, dtype=torch.int32)
    was_grasped = torch.zeros(num_envs, dtype=torch.bool)
    grasp_start_step = torch.full((num_envs,), -1, dtype=torch.int32)
    grasp_durations: list[int] = []  # all grasp durations across all episodes
    min_cube_goal = torch.full((num_envs,), 999.0)
    min_jaw_cube = torch.full((num_envs,), 999.0)
    min_gripper_z = torch.full((num_envs,), 999.0)
    min_jaw_z = torch.full((num_envs,), 999.0)
    tip_below_steps = torch.zeros(num_envs, dtype=torch.int32)
    max_qdot_ep = torch.zeros(num_envs)
    sum_qdot_ep = torch.zeros(num_envs)
    qdot_count = torch.zeros(num_envs, dtype=torch.int32)
    qdot_alarm_count = torch.zeros(num_envs, dtype=torch.int32)
    action_saturation_count = torch.zeros(num_envs, dtype=torch.int32)
    max_cube_speed_ep = torch.zeros(num_envs)
    current_phase = [PHASE_PRE_REACH] * num_envs
    deepest_phase = [PHASE_PRE_REACH] * num_envs
    reward_term_sums: dict[str, torch.Tensor] = {}  # term_name -> (num_envs,) cumulative

    # ---- Posture / trajectory diagnostics (V2.12 addition) ----
    # Detect "snake-like" crawling vs "human-like" top-down approach.
    #
    # Definitions:
    #   gripper_down_score = -ee_axis_z_world.z
    #     +1.0 = gripper z-axis points straight DOWN (perfect top-down approach)
    #      0.0 = gripper z-axis horizontal
    #     -1.0 = gripper points straight UP
    #
    #   table_slide_steps  = number of steps where jaw_z_world < TABLE_TOP + 2cm.
    #     High value (sustained) = bras "rampe" sur la table (snake)
    #     Low value = approche par le haut, descente puis grasp
    #
    #   min_jaw_z_pre_grasp = lowest jaw_z_world reached BEFORE first GRASP_FIRE.
    #     Low (<= TABLE_TOP + 2cm) = jaw scraping the table during approach
    #     High (>= 0.10 m) = jaw stays well above table during approach
    #
    #   gripper_down_at_grasp = snapshot of gripper_down_score at first GRASP_FIRE.
    #     >= 0.7 = approached cube from above (good for sim-to-real)
    #     <= 0.3 = approached horizontally (snake; bad for sim-to-real)
    TABLE_TOP_Z_W = 0.0415  # measured empirically (audit_scene.py)
    SLIDE_MARGIN = 0.02     # 2 cm above table top counts as "sliding"
    DOWN_SCORE_THRESHOLD = 0.5  # gripper-down score above this = "pointing down"

    gripper_down_steps = torch.zeros(num_envs, dtype=torch.int32)   # n steps with gripper_down > 0.5
    table_slide_steps = torch.zeros(num_envs, dtype=torch.int32)    # n steps with jaw_z < TABLE+2cm
    min_jaw_z_pre_grasp = torch.full((num_envs,), 999.0)            # min jaw_z BEFORE first grasp
    gripper_down_score_sum = torch.zeros(num_envs)                  # for mean computation
    gripper_down_score_count = torch.zeros(num_envs, dtype=torch.int32)
    gripper_down_at_grasp = torch.full((num_envs,), float("nan"))   # snapshot at first GRASP_FIRE
    jaw_z_at_grasp = torch.full((num_envs,), float("nan"))          # snapshot at first GRASP_FIRE

    # Aggregates
    completed = 0
    summary = {
        "success": 0, "dropped": 0, "timeout": 0,
        "ttg": [], "ttsuccess": [], "ttdrop": [],
        "grasp_dur_per_ep": [], "grasp_count_per_ep": [],
        "min_cube_goal": [], "tip_below_steps": [],
        "min_gripper_z": [], "min_jaw_z": [],
        "max_qdot_per_ep": [], "mean_qdot_per_ep": [],
        "qdot_alarm_count_per_ep": [], "action_sat_count_per_ep": [],
        "max_cube_speed_per_ep": [],
        "deepest_phase_per_ep": [],
        "ep_with_tip_below": 0,
        # V2.12 posture diagnostics
        "gripper_down_steps_per_ep": [],     # n steps with gripper pointing down (>0.5)
        "table_slide_steps_per_ep": [],      # n steps with jaw_z near table
        "min_jaw_z_pre_grasp_per_ep": [],    # min jaw_z before first grasp
        "mean_gripper_down_per_ep": [],      # avg gripper-down score over the episode
        "gripper_down_at_grasp_per_ep": [],  # snapshot at first GRASP_FIRE (NaN if no grasp)
        "jaw_z_at_grasp_per_ep": [],         # jaw_z at first GRASP_FIRE
    }
    drop_modes = {"grasp_then_slip": 0, "bump_without_grasp": 0,
                  "tip_below_table": 0, "other": 0}
    phase_reach_count = {p: 0 for p in PHASE_ORDER + [PHASE_FAILURE]}

    # ---- Reward tracking via _episode_sums snapshotting ----
    # Snapshot _episode_sums BEFORE env.step every iteration. After env.step,
    # the sums for done envs are zeroed by auto-reset, so we use the pre-step
    # snapshot as the final cumulative reward for that episode (missing only
    # the very last step's per-term reward, negligible for aggregate stats).
    active_reward_terms: list[str] = []
    try:
        rew_mgr_ref = env_unwrapped.reward_manager
        active_reward_terms = list(rew_mgr_ref.active_terms)
    except Exception:
        pass
    prev_episode_sums: dict[str, torch.Tensor] = {
        t: torch.zeros(num_envs) for t in active_reward_terms
    }
    reward_term_total: dict[str, float] = {t: 0.0 for t in active_reward_terms}

    # ---- Per-episode records (full breakdown at end) ----
    per_episode_records: list[dict] = []

    # ---- Initial conditions (cube and goal positions at episode start) ----
    initial_cube_pos = torch.zeros(num_envs, 3)              # world frame
    initial_goal_pos_b = torch.zeros(num_envs, 3)            # robot root frame
    initial_robot_pos = torch.zeros(num_envs, 3)             # world frame, captured once

    # ---- Action history accumulator (for percentile histogram) ----
    action_history: list[torch.Tensor] = []

    # ---- Grasp lifecycle records (per GRASP_FIRE event) ----
    grasp_records: list[dict] = []

    # ---- Main loop ----
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    while simulation_app.is_running() and completed < args_cli.num_episodes:
        # Snapshot per-term episode_sums BEFORE env.step. After env.step the
        # sums for done envs are zeroed by auto-reset, so this snapshot holds
        # the cumulative reward through the previous iteration.
        try:
            for term in active_reward_terms:
                prev_episode_sums[term] = rew_mgr_ref._episode_sums[term].cpu().clone()
        except Exception:
            pass

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, tuple):
                obs = obs[0]

        # Accumulate raw action values for percentile histogram (B, action_dim)
        action_history.append(actions.cpu().clone())

        # ---- Per-step state ----
        cube_w = cube.data.root_pos_w
        base_z_w = robot.data.body_pos_w[:, base_idx, 2]
        gripper_q = robot.data.joint_pos[:, gripper_joint_idx]
        gripper_w = ee_frame.data.target_pos_w[:, 0, :]
        jaw_w = ee_frame.data.target_pos_w[:, 1, :]
        gripper_quat_w = ee_frame.data.target_quat_w[:, 0, :]  # gripper orientation
        jaw_cube = torch.norm(jaw_w - cube_w, dim=-1)
        gripper_cube = torch.norm(gripper_w - cube_w, dim=-1)
        joint_pos = robot.data.joint_pos
        joint_pos_target = robot.data.joint_pos_target
        joint_vel = robot.data.joint_vel
        abs_qdot = joint_vel.abs()
        max_qdot_now, max_qdot_idx = abs_qdot.max(dim=-1)

        # Goal in world frame
        try:
            cmd = env_unwrapped.command_manager.get_command("object_pose")
            goal_b = cmd[:, :3]
            goal_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, goal_b)
            cube_goal = torch.norm(cube_w - goal_w, dim=-1)
            ee_goal = torch.norm(gripper_w - goal_w, dim=-1)
        except Exception:
            cube_goal = torch.full_like(jaw_cube, float("nan"))
            ee_goal = torch.full_like(jaw_cube, float("nan"))
            goal_w = torch.zeros_like(cube_w)

        # Cube velocity
        try:
            cube_lin_vel = cube.data.root_lin_vel_w
            cube_speed = cube_lin_vel.norm(dim=-1)
        except AttributeError:
            cube_lin_vel = torch.zeros_like(cube_w)
            cube_speed = torch.zeros(num_envs, device=cube_w.device)

        # Predicates
        is_grasped = grasp_pred(jaw_cube, gripper_q)
        is_lifted = lift_pred(cube_w[:, 2], base_z_w)
        is_at_goal = goal_pred(cube_goal)

        # Tip-below
        tip_below = (gripper_w[:, 2] < 0.03) | (jaw_w[:, 2] < 0.03)
        action_is_close = gripper_q < 0.25

        # NOTE: per-term reward decomposition is now tracked via the
        # `prev_episode_sums` snapshot (taken at the top of the loop, BEFORE
        # env.step). At episode termination we use that snapshot as the final
        # cumulative reward for each term — see the termination handler below.

        # Action saturation: rsl_rl clips actions to ±clip_actions (default 1.0).
        # We approximate by checking if the policy output (which we don't have
        # easily here) was at limit. Use joint_pos_target jumping by max as proxy.
        # Simpler: compute fraction of action dims at clip_actions:
        try:
            clip_a = float(agent_cfg.clip_actions) if agent_cfg.clip_actions is not None else 1.0
        except Exception:
            clip_a = 1.0
        # actions tensor from policy (B, action_dim)
        actions_clipped = (actions.abs() >= clip_a * 0.99)
        per_env_action_sat = actions_clipped.any(dim=-1).cpu()  # any dim saturated

        # Update aggregates
        min_cube_goal = torch.minimum(min_cube_goal, cube_goal.cpu())
        min_jaw_cube = torch.minimum(min_jaw_cube, jaw_cube.cpu())
        min_gripper_z = torch.minimum(min_gripper_z, gripper_w[:, 2].cpu())
        min_jaw_z = torch.minimum(min_jaw_z, jaw_w[:, 2].cpu())
        max_qdot_ep = torch.maximum(max_qdot_ep, max_qdot_now.cpu())
        sum_qdot_ep += max_qdot_now.cpu()
        qdot_count += 1
        max_cube_speed_ep = torch.maximum(max_cube_speed_ep, cube_speed.cpu())

        # V2.12 posture diagnostics — per-step
        # Gripper z-axis in world: positive z component = pointing up,
        # negative z component = pointing down. We define
        # gripper_down_score = -axis_z_world[2], so:
        #   +1.0 = perfectly down (palm above cube, fingers down)
        #    0.0 = horizontal (palm sideways)
        #   -1.0 = perfectly up (jaw above gripper body)
        ee_axis_z_world_step = quat_to_axis_z(gripper_quat_w)        # (B, 3)
        gripper_down_score_step = (-ee_axis_z_world_step[:, 2]).cpu()  # (B,)
        gripper_is_down_step = (gripper_down_score_step > DOWN_SCORE_THRESHOLD).int()
        gripper_down_score_sum += gripper_down_score_step
        gripper_down_score_count += 1
        gripper_down_steps += gripper_is_down_step

        # Table-slide detection: jaw close to (or below) table top
        jaw_z_cpu = jaw_w[:, 2].cpu()
        table_slide_now = (jaw_z_cpu < (TABLE_TOP_Z_W + SLIDE_MARGIN)).int()
        table_slide_steps += table_slide_now

        # min_jaw_z BEFORE first GRASP_FIRE (track only when no grasp yet)
        no_grasp_yet = torch.tensor(
            [first_grasp[i] < 0 for i in range(num_envs)], dtype=torch.bool
        )
        if no_grasp_yet.any():
            min_jaw_z_pre_grasp = torch.where(
                no_grasp_yet,
                torch.minimum(min_jaw_z_pre_grasp, jaw_z_cpu),
                min_jaw_z_pre_grasp,
            )

        qdot_alarm_now = (max_qdot_now > args_cli.qdot_limit * 1.5).cpu()
        qdot_alarm_count += qdot_alarm_now.int()
        action_saturation_count += per_env_action_sat.int()

        # ---- Continuous trace for trace_env ----
        ti = args_cli.trace_env
        if 0 <= ti < num_envs and (int(ep_step[ti]) % args_cli.trace_every == 0):
            log_jsonl({
                "type": "TRACE", "ts": time.time(), "env": ti,
                "ep_id": int(ep_id[ti]), "step": int(ep_step[ti]),
                "cube_w": cube_w[ti].cpu().tolist(),
                "gripper_w": gripper_w[ti].cpu().tolist(),
                "jaw_w": jaw_w[ti].cpu().tolist(),
                "jaw_cube_dist": float(jaw_cube[ti]),
                "qdot": joint_vel[ti].cpu().tolist(),
                "qdot_max": float(max_qdot_now[ti]),
                "cube_speed": float(cube_speed[ti]),
                "phase": current_phase[ti],
            })

        # ---- Action log every action_log_every steps ----
        if 0 <= ti < num_envs and (int(ep_step[ti]) % args_cli.action_log_every == 0):
            log_jsonl({
                "type": "ACTION", "ts": time.time(), "env": ti,
                "ep_id": int(ep_id[ti]), "step": int(ep_step[ti]),
                "raw_action": actions[ti].cpu().tolist(),
                "joint_pos_target": joint_pos_target[ti].cpu().tolist(),
                "joint_pos_actual": joint_pos[ti].cpu().tolist(),
                "gripper_action_close": bool(action_is_close[ti].item()),
                "any_action_saturated": bool(per_env_action_sat[ti].item()),
            })

        # ---- Goal log every goal_log_every steps ----
        if 0 <= ti < num_envs and (int(ep_step[ti]) % args_cli.goal_log_every == 0):
            log_jsonl({
                "type": "GOAL", "ts": time.time(), "env": ti,
                "ep_id": int(ep_id[ti]), "step": int(ep_step[ti]),
                "goal_w": goal_w[ti].cpu().tolist(),
                "cube_goal_dist": float(cube_goal[ti]),
                "ee_goal_dist": float(ee_goal[ti]),
            })

        # ---- Per-env event detection + phase update ----
        for i in range(num_envs):
            # Capture initial conditions on the very first step of each episode
            if int(ep_step[i]) == 0:
                initial_cube_pos[i] = cube_w[i].cpu()
                try:
                    initial_goal_pos_b[i] = goal_b[i].cpu()
                except Exception:
                    pass
                initial_robot_pos[i] = robot.data.root_pos_w[i].cpu()
            ep_step[i] += 1
            cx, cy, cz = cube_w[i, 0].item(), cube_w[i, 1].item(), cube_w[i, 2].item()
            this_step = int(ep_step[i])
            ep_iid = int(ep_id[i])

            # Phase update
            new_phase = classify_phase(deepest_phase[i],
                                       float(gripper_cube[i]),
                                       bool(is_grasped[i].item()),
                                       bool(is_lifted[i].item()),
                                       bool(is_at_goal[i].item()))
            if new_phase != deepest_phase[i]:
                log_event({
                    "type": "PHASE_TRANSITION", "ts": time.time(), "env": i,
                    "ep_id": ep_iid, "step": this_step,
                    "from": deepest_phase[i], "to": new_phase,
                    "cube_w": [cx, cy, cz],
                })
                deepest_phase[i] = new_phase
            current_phase[i] = new_phase

            # Physics alarms
            if qdot_alarm_now[i]:
                log_jsonl({
                    "type": "ALARM_QDOT", "ts": time.time(), "env": i,
                    "ep_id": ep_iid, "step": this_step,
                    "qdot_max": float(max_qdot_now[i]),
                    "joint": robot.joint_names[int(max_qdot_idx[i])],
                    "limit": args_cli.qdot_limit,
                })

            # FIRST_CLOSE
            if action_is_close[i] and first_close[i] < 0:
                first_close[i] = this_step
                log_event({
                    "type": "FIRST_CLOSE", "ts": time.time(), "env": i,
                    "ep_id": ep_iid, "step": this_step,
                    "d_grip_cube": float(gripper_cube[i]),
                    "d_jaw_cube": float(jaw_cube[i]),
                    "cube_w": [cx, cy, cz],
                })

            # GRASP_FIRE
            if is_grasped[i]:
                grasp_steps[i] += 1
                if not was_grasped[i]:
                    grasp_count[i] += 1
                    grasp_start_step[i] = this_step
                    if first_grasp[i] < 0:
                        first_grasp[i] = this_step
                    # Detailed grasp diagnostic
                    rel_vel = (cube_lin_vel[i] - torch.zeros(3, device=cube_w.device)).cpu().tolist()
                    ee_axis_z = quat_to_axis_z(gripper_quat_w[i:i+1])[0].cpu().tolist()
                    cube_speed_at_fire = float(cube_speed[i])
                    gripper_qdot_at_fire = float(joint_vel[i, gripper_joint_idx])
                    jaw_cube_at_fire = float(jaw_cube[i])
                    # V2.12 posture snapshot at first GRASP_FIRE only
                    if torch.isnan(gripper_down_at_grasp[i]):
                        gripper_down_at_grasp[i] = -ee_axis_z[2]   # +1=down, -1=up
                        jaw_z_at_grasp[i] = float(jaw_w[i, 2])
                    log_event({
                        "type": "GRASP_FIRE", "ts": time.time(), "env": i,
                        "ep_id": ep_iid, "step": this_step,
                        "d_jaw_cube": jaw_cube_at_fire,
                        "d_grip_cube": float(gripper_cube[i]),
                        "gripper_q": float(gripper_q[i]),
                        "gripper_qdot": gripper_qdot_at_fire,
                        "cube_w": [cx, cy, cz],
                        "cube_vel_w": cube_lin_vel[i].cpu().tolist(),
                        "cube_speed": cube_speed_at_fire,
                        "rel_vel_grip_cube": rel_vel,
                        "ee_axis_z_world": ee_axis_z,
                        "phase": deepest_phase[i],
                    })
                    # Lifecycle record (closed at grasp_LOST or episode end)
                    grasp_records.append({
                        "env": i, "ep_id": ep_iid,
                        "grasp_idx": int(grasp_count[i]),
                        "fire_step": this_step,
                        "cube_speed_at_fire": cube_speed_at_fire,
                        "gripper_qdot_at_fire": gripper_qdot_at_fire,
                        "jaw_cube_at_fire": jaw_cube_at_fire,
                        "lost_step": None,
                        "lost_d_jaw": None,
                        "duration_steps": None,
                    })
                was_grasped[i] = True
            else:
                if was_grasped[i]:
                    dur = this_step - int(grasp_start_step[i])
                    grasp_durations.append(dur)
                    lost_d_jaw = float(jaw_cube[i])
                    log_event({
                        "type": "grasp_LOST", "ts": time.time(), "env": i,
                        "ep_id": ep_iid, "step": this_step,
                        "duration_steps": dur,
                        "d_jaw_cube": lost_d_jaw,
                        "cube_w": [cx, cy, cz],
                        "cube_vel_w": cube_lin_vel[i].cpu().tolist(),
                    })
                    # Close the most recent unclosed lifecycle record for this env
                    for rec in reversed(grasp_records):
                        if rec["env"] == i and rec["lost_step"] is None:
                            rec["lost_step"] = this_step
                            rec["lost_d_jaw"] = lost_d_jaw
                            rec["duration_steps"] = dur
                            break
                was_grasped[i] = False
                grasp_start_step[i] = -1

            # LIFT
            if is_lifted[i] and first_lift[i] < 0:
                first_lift[i] = this_step
                log_event({
                    "type": "LIFT", "ts": time.time(), "env": i,
                    "ep_id": ep_iid, "step": this_step,
                    "cube_z_rel_base": float(cz - base_z_w[i].item()),
                    "cube_w": [cx, cy, cz],
                })

            # TIP_BELOW_TABLE
            if tip_below[i]:
                tip_below_steps[i] += 1
                if first_tip_below[i] < 0:
                    first_tip_below[i] = this_step
                    log_event({
                        "type": "TIP_BELOW_TABLE", "ts": time.time(), "env": i,
                        "ep_id": ep_iid, "step": this_step,
                        "gripper_z": float(gripper_w[i, 2]),
                        "jaw_z": float(jaw_w[i, 2]),
                        "cube_w": [cx, cy, cz],
                    })

        # ---- Episode terminations ----
        if dones.any():
            # Read per-term termination flags BEFORE iterating done_ids — they
            # survive auto-reset (the manager records them at the termination
            # step). This is the only reliable way to distinguish SUCCESS from
            # TIMEOUT after auto-reset has wiped the cube position.
            term_mgr = env_unwrapped.termination_manager
            term_flags = {}
            for tname in ("success", "cube_dropped", "time_out"):
                try:
                    if tname in term_mgr.active_terms:
                        # Try public API first
                        try:
                            term_flags[tname] = term_mgr.get_term(tname).cpu()
                        except (AttributeError, KeyError):
                            # Fallback to private buffer
                            term_flags[tname] = term_mgr._term_dones[tname].cpu()
                except Exception:
                    pass

            done_ids = torch.nonzero(dones, as_tuple=True)[0]
            for i in done_ids.cpu().tolist():
                step = int(ep_step[i])
                ep_iid = int(ep_id[i])
                ep_min_cube_goal = float(min_cube_goal[i])
                last_cube_z = cube_w[i, 2].item()

                # Classify via termination_manager flags (authoritative)
                term_success = bool(term_flags.get("success", torch.zeros(num_envs, dtype=torch.bool))[i])
                term_dropped = bool(term_flags.get("cube_dropped", torch.zeros(num_envs, dtype=torch.bool))[i])
                term_timeout = bool(term_flags.get("time_out", torch.zeros(num_envs, dtype=torch.bool))[i])

                if term_success:
                    outcome = "SUCCESS"
                    summary["success"] += 1
                    summary["ttsuccess"].append(step)
                    deepest_phase[i] = PHASE_AT_GOAL
                elif term_dropped:
                    outcome = "DROPPED"
                    summary["dropped"] += 1
                    summary["ttdrop"].append(step)
                    if first_tip_below[i] >= 0 and first_tip_below[i] <= step - 2:
                        drop_modes["tip_below_table"] += 1
                    elif first_grasp[i] >= 0:
                        drop_modes["grasp_then_slip"] += 1
                    elif first_close[i] >= 0:
                        drop_modes["bump_without_grasp"] += 1
                    else:
                        drop_modes["other"] += 1
                elif term_timeout:
                    outcome = "TIMEOUT"
                    summary["timeout"] += 1
                else:
                    # Fallback: term flags missing → use heuristics
                    if ep_min_cube_goal < 0.05:
                        outcome = "SUCCESS"
                        summary["success"] += 1
                        summary["ttsuccess"].append(step)
                        deepest_phase[i] = PHASE_AT_GOAL
                    elif last_cube_z < 0.04:
                        outcome = "DROPPED"
                        summary["dropped"] += 1
                        summary["ttdrop"].append(step)
                        drop_modes["other"] += 1
                    else:
                        outcome = "TIMEOUT"
                        summary["timeout"] += 1

                # If outcome != AT_GOAL we mark FAILURE in deepest if appropriate
                if outcome != "SUCCESS":
                    # The deepest reached is whatever we tracked, plus FAILURE
                    # marker only if no progress was made (still PRE_REACH).
                    pass

                # Phase reach: count this episode reached up to deepest_phase[i]
                deepest = deepest_phase[i]
                # mark FAILURE explicitly if outcome != SUCCESS and deepest is shallow
                phase_reach_count[deepest] += 1
                if outcome != "SUCCESS":
                    phase_reach_count[PHASE_FAILURE] += 1

                # Aggregate
                if first_grasp[i] >= 0:
                    summary["ttg"].append(int(first_grasp[i]))
                summary["grasp_dur_per_ep"].append(int(grasp_steps[i]))
                summary["grasp_count_per_ep"].append(int(grasp_count[i]))
                summary["min_cube_goal"].append(float(min_cube_goal[i]))
                summary["tip_below_steps"].append(int(tip_below_steps[i]))
                summary["min_gripper_z"].append(float(min_gripper_z[i]))
                summary["min_jaw_z"].append(float(min_jaw_z[i]))
                summary["max_qdot_per_ep"].append(float(max_qdot_ep[i]))
                mean_qdot = float(sum_qdot_ep[i]) / max(int(qdot_count[i]), 1)
                summary["mean_qdot_per_ep"].append(mean_qdot)
                summary["qdot_alarm_count_per_ep"].append(int(qdot_alarm_count[i]))
                summary["action_sat_count_per_ep"].append(int(action_saturation_count[i]))
                summary["max_cube_speed_per_ep"].append(float(max_cube_speed_ep[i]))
                summary["deepest_phase_per_ep"].append(deepest)
                if first_tip_below[i] >= 0:
                    summary["ep_with_tip_below"] += 1
                # V2.12 posture diagnostics
                summary["gripper_down_steps_per_ep"].append(int(gripper_down_steps[i]))
                summary["table_slide_steps_per_ep"].append(int(table_slide_steps[i]))
                summary["min_jaw_z_pre_grasp_per_ep"].append(
                    float(min_jaw_z_pre_grasp[i]) if min_jaw_z_pre_grasp[i] < 999 else float("nan")
                )
                count = max(int(gripper_down_score_count[i]), 1)
                summary["mean_gripper_down_per_ep"].append(
                    float(gripper_down_score_sum[i]) / count
                )
                summary["gripper_down_at_grasp_per_ep"].append(
                    float(gripper_down_at_grasp[i])
                )
                summary["jaw_z_at_grasp_per_ep"].append(
                    float(jaw_z_at_grasp[i])
                )

                # Per-term reward totals for this episode (from pre-step snapshot,
                # which is cumulative through the iteration BEFORE termination —
                # missing only the final step's per-term reward, ~negligible).
                ep_term_totals = {}
                for tn in active_reward_terms:
                    snap = prev_episode_sums.get(tn)
                    val = float(snap[i]) if snap is not None else 0.0
                    ep_term_totals[tn] = val
                    reward_term_total[tn] = reward_term_total.get(tn, 0.0) + val

                # Per-episode record (printed in detail at end)
                per_episode_records.append({
                    "ep_id": ep_iid,
                    "outcome": outcome,
                    "step": step,
                    "first_close": int(first_close[i]),
                    "first_grasp": int(first_grasp[i]),
                    "first_lift": int(first_lift[i]),
                    "first_tip_below": int(first_tip_below[i]),
                    "grasp_count": int(grasp_count[i]),
                    "grasp_steps_total": int(grasp_steps[i]),
                    "tip_below_steps": int(tip_below_steps[i]),
                    "min_d_goal": float(min_cube_goal[i]),
                    "qdot_max": float(max_qdot_ep[i]),
                    "max_cube_speed": float(max_cube_speed_ep[i]),
                    "deepest_phase": deepest_phase[i],
                    "initial_cube_w": initial_cube_pos[i].tolist(),
                    "initial_goal_b": initial_goal_pos_b[i].tolist(),
                    "initial_robot_w": initial_robot_pos[i].tolist(),
                })

                log_event({
                    "type": "EPISODE_END", "ts": time.time(), "env": i,
                    "ep_id": ep_iid, "step": step, "outcome": outcome,
                    "deepest_phase": deepest,
                    "first_close": int(first_close[i]),
                    "first_grasp": int(first_grasp[i]),
                    "first_lift": int(first_lift[i]),
                    "first_tip_below": int(first_tip_below[i]),
                    "grasp_count": int(grasp_count[i]),
                    "grasp_steps_total": int(grasp_steps[i]),
                    "tip_below_steps": int(tip_below_steps[i]),
                    "min_cube_goal": float(min_cube_goal[i]),
                    "min_gripper_z": float(min_gripper_z[i]),
                    "min_jaw_z": float(min_jaw_z[i]),
                    "max_qdot": float(max_qdot_ep[i]),
                    "mean_qdot_max": mean_qdot,
                    "qdot_alarm_count": int(qdot_alarm_count[i]),
                    "action_sat_count": int(action_saturation_count[i]),
                    "max_cube_speed": float(max_cube_speed_ep[i]),
                    "reward_term_totals": ep_term_totals,
                })
                completed += 1

                # Reset trackers for this env slot
                ep_step[i] = 0
                first_close[i] = -1
                first_grasp[i] = -1
                first_lift[i] = -1
                first_tip_below[i] = -1
                grasp_count[i] = 0
                grasp_steps[i] = 0
                tip_below_steps[i] = 0
                was_grasped[i] = False
                grasp_start_step[i] = -1
                min_cube_goal[i] = 999.0
                min_jaw_cube[i] = 999.0
                min_gripper_z[i] = 999.0
                min_jaw_z[i] = 999.0
                max_qdot_ep[i] = 0.0
                sum_qdot_ep[i] = 0.0
                qdot_count[i] = 0
                qdot_alarm_count[i] = 0
                action_saturation_count[i] = 0
                max_cube_speed_ep[i] = 0.0
                current_phase[i] = PHASE_PRE_REACH
                deepest_phase[i] = PHASE_PRE_REACH
                # V2.12 posture trackers reset
                gripper_down_steps[i] = 0
                table_slide_steps[i] = 0
                min_jaw_z_pre_grasp[i] = 999.0
                gripper_down_score_sum[i] = 0.0
                gripper_down_score_count[i] = 0
                gripper_down_at_grasp[i] = float("nan")
                jaw_z_at_grasp[i] = float("nan")
                # prev_episode_sums is read fresh from the manager each iter,
                # no per-env reset needed here.
                ep_id[i] = num_envs + completed

    # ===========================================================
    # Aggregate summary
    # ===========================================================
    lines = []
    lines.append(f"=== DIAGNOSTIC PLAY V2 SUMMARY ===")
    lines.append(f"task         : {task_name}")
    lines.append(f"checkpoint   : {os.path.basename(resume_path)}")
    lines.append(f"num_envs     : {num_envs}")
    lines.append(f"num_episodes : {completed}")
    lines.append("")
    lines.append("=== Cube physical dimensions ===")
    lines.append(f"  {cube_info}")
    lines.append("")
    lines.append("=== Outcomes ===")
    lines.append(f"  SUCCESS  : {summary['success']:3d} ({100*summary['success']/max(completed,1):.0f}%)")
    lines.append(f"  DROPPED  : {summary['dropped']:3d} ({100*summary['dropped']/max(completed,1):.0f}%)")
    lines.append(f"  TIMEOUT  : {summary['timeout']:3d} ({100*summary['timeout']/max(completed,1):.0f}%)")
    lines.append("")
    lines.append("=== Phase reach rates (% of episodes that reached AT LEAST this phase) ===")
    # Compute reach rate per phase: episodes that reached deepest >= phase.
    for phase in PHASE_ORDER:
        reached = sum(1 for d in summary["deepest_phase_per_ep"]
                      if PHASE_ORDER.index(d) >= PHASE_ORDER.index(phase))
        lines.append(f"  {phase:12s} : {reached:3d}/{completed} ({100*reached/max(completed,1):.0f}%)")
    lines.append("")
    lines.append("=== Timings ===")
    if summary["ttg"]:
        ttg = torch.tensor(summary["ttg"], dtype=torch.float)
        lines.append(f"  Time to first grasp   : mean={ttg.mean():.1f} std={ttg.std():.1f} steps  (n={len(summary['ttg'])})")
    if summary["ttsuccess"]:
        tts = torch.tensor(summary["ttsuccess"], dtype=torch.float)
        lines.append(f"  Time to success       : mean={tts.mean():.1f} std={tts.std():.1f} steps  (n={len(summary['ttsuccess'])})")
    if summary["ttdrop"]:
        ttd = torch.tensor(summary["ttdrop"], dtype=torch.float)
        lines.append(f"  Time to drop          : mean={ttd.mean():.1f} std={ttd.std():.1f} steps  (n={len(summary['ttdrop'])})")
    lines.append("")
    lines.append("=== Grasp survival ===")
    if grasp_durations:
        gd = torch.tensor(grasp_durations, dtype=torch.float)
        lines.append(f"  Total grasp events recorded : {len(grasp_durations)}")
        lines.append(f"  Grasp duration mean / std   : {gd.mean():.2f} / {gd.std():.2f} steps")
        # Survival probabilities
        for thresh in [1, 3, 5, 10, 20, 50]:
            survived = (gd >= thresh).sum().item()
            lines.append(f"  P(grasp lasts >= {thresh:2d} steps)  : {100*survived/len(grasp_durations):.1f}%")
    else:
        lines.append("  No grasp events recorded.")
    lines.append("")
    lines.append("=== Action saturation ===")
    if summary["action_sat_count_per_ep"]:
        a = torch.tensor(summary["action_sat_count_per_ep"], dtype=torch.float)
        lines.append(f"  Steps with any action dim at limit / ep : mean={a.mean():.1f}  max={a.max():.0f}")
    if summary["qdot_alarm_count_per_ep"]:
        q = torch.tensor(summary["qdot_alarm_count_per_ep"], dtype=torch.float)
        lines.append(f"  qdot ALARM steps (>{args_cli.qdot_limit*1.5:.1f} rad/s) / ep : mean={q.mean():.1f}  max={q.max():.0f}")
    if summary["max_qdot_per_ep"]:
        mq = torch.tensor(summary["max_qdot_per_ep"], dtype=torch.float)
        lines.append(f"  Max |qdot| / ep      : mean={mq.mean():.2f}  max={mq.max():.2f} rad/s")
    if summary["mean_qdot_per_ep"]:
        mnq = torch.tensor(summary["mean_qdot_per_ep"], dtype=torch.float)
        lines.append(f"  Mean |qdot|max / step / ep (smoothness) : mean={mnq.mean():.2f}")
    lines.append("")
    lines.append("=== Cube/tip kinematics ===")
    if summary["max_cube_speed_per_ep"]:
        mcs = torch.tensor(summary["max_cube_speed_per_ep"], dtype=torch.float)
        lines.append(f"  Max cube speed / ep   : mean={mcs.mean():.2f}  max={mcs.max():.2f} m/s")
    lines.append(f"  Episodes with tip below table : {summary['ep_with_tip_below']}/{completed}"
                 f" ({100*summary['ep_with_tip_below']/max(completed,1):.0f}%)")
    if summary["min_jaw_z"]:
        mjz = torch.tensor(summary["min_jaw_z"], dtype=torch.float)
        lines.append(f"  Min jaw.z / ep (world)        : mean={mjz.mean():+.3f}  min={mjz.min():+.3f}")
    lines.append("")

    # ---- POSTURE / TRAJECTORY DIAGNOSIS (V2.12 addition) ----
    lines.append("=== Posture / trajectory diagnosis (snake vs human) ===")
    if summary["mean_gripper_down_per_ep"]:
        mgd = torch.tensor(summary["mean_gripper_down_per_ep"], dtype=torch.float)
        gds = torch.tensor(summary["gripper_down_steps_per_ep"], dtype=torch.float)
        tss = torch.tensor(summary["table_slide_steps_per_ep"], dtype=torch.float)
        mjzpg = torch.tensor(
            [v for v in summary["min_jaw_z_pre_grasp_per_ep"]
             if not (isinstance(v, float) and v != v)],
            dtype=torch.float,
        )
        gdag_raw = summary["gripper_down_at_grasp_per_ep"]
        gdag = torch.tensor(
            [v for v in gdag_raw if not (isinstance(v, float) and v != v)],
            dtype=torch.float,
        )
        jzg_raw = summary["jaw_z_at_grasp_per_ep"]
        jzg = torch.tensor(
            [v for v in jzg_raw if not (isinstance(v, float) and v != v)],
            dtype=torch.float,
        )
        max_steps = max(int(s) for s in summary["grasp_dur_per_ep"]) if summary["grasp_dur_per_ep"] else 150

        lines.append(
            f"  Mean gripper-down score / ep     : "
            f"mean={mgd.mean():+.3f}  min={mgd.min():+.3f}  max={mgd.max():+.3f}"
        )
        lines.append(f"    (+1=down/human  0=horizontal  -1=up)")
        lines.append(
            f"  Steps with gripper down (>{DOWN_SCORE_THRESHOLD}) / ep : "
            f"mean={gds.mean():.1f} / 150  ({100*gds.mean()/150:.0f}%)"
        )
        lines.append(
            f"  Table slide steps / ep (jaw_z<{TABLE_TOP_Z_W+SLIDE_MARGIN:.2f}m) : "
            f"mean={tss.mean():.1f} / 150  ({100*tss.mean()/150:.0f}%)"
        )
        if mjzpg.numel() > 0:
            lines.append(
                f"  Min jaw_z BEFORE first grasp / ep  : "
                f"mean={mjzpg.mean():+.3f}m  min={mjzpg.min():+.3f}m"
            )
        else:
            lines.append("  Min jaw_z BEFORE first grasp / ep  : (no episodes had pre-grasp data)")
        if gdag.numel() > 0:
            lines.append(
                f"  Gripper-down score AT first grasp  : "
                f"mean={gdag.mean():+.3f}  ({gdag.numel()}/{completed} eps had grasp)"
            )
            lines.append(
                f"  Jaw z AT first grasp              : "
                f"mean={jzg.mean():+.3f}m  min={jzg.min():+.3f}m  max={jzg.max():+.3f}m"
            )
        else:
            lines.append("  Gripper-down score AT first grasp  : N/A (no grasps)")

        # ---- Verdict ----
        avg_down = float(mgd.mean())
        avg_slide_pct = float(100 * tss.mean() / 150)
        if gdag.numel() > 0:
            avg_grasp_down = float(gdag.mean())
        else:
            avg_grasp_down = float("nan")

        lines.append("")
        lines.append("  TRAJECTORY VERDICT:")
        if avg_down > 0.7 and avg_slide_pct < 10:
            lines.append("    HUMAN-LIKE (gripper points down, no table sliding) — sim-to-real OK")
        elif avg_down < 0.3 and avg_slide_pct > 30:
            lines.append("    SNAKE-LIKE (gripper horizontal + significant table sliding) — sim-to-real RISKY")
            lines.append("      → policy reaches cube by sliding the gripper along the table.")
            lines.append("      → friction model in sim is forgiving; real table will not be.")
            lines.append("      → consider: add 'wrist_pointing_down' bonus reward, or restrict")
            lines.append("        approach via initial pose, or use a curriculum that rewards")
            lines.append("        top-down approach early.")
        else:
            lines.append(f"    MIXED (avg_down={avg_down:+.2f}, slide={avg_slide_pct:.0f}%)")
            lines.append("      → neither pure snake nor pure human. Mostly horizontal but")
            lines.append("        without consistent table sliding.")
    lines.append("")

    lines.append("=== Reward source decomposition (cumulative across all episodes) ===")
    if reward_term_total:
        pos_total = sum(v for v in reward_term_total.values() if v > 0) or 1.0
        neg_total = sum(v for v in reward_term_total.values() if v < 0) or -1.0
        sorted_pos = sorted([(t, v) for t, v in reward_term_total.items() if v >= 0],
                            key=lambda kv: -kv[1])
        sorted_neg = sorted([(t, v) for t, v in reward_term_total.items() if v < 0],
                            key=lambda kv: kv[1])
        lines.append(f"  Positive reward total : {pos_total:+10.2f}")
        for tn, v in sorted_pos:
            pct = 100 * v / pos_total
            lines.append(f"    {tn:38s} : {v:+10.2f}  ({pct:5.1f}% of positive)")
        lines.append(f"  Negative reward total : {neg_total:+10.2f}")
        for tn, v in sorted_neg:
            pct = 100 * v / abs(neg_total)
            lines.append(f"    {tn:38s} : {v:+10.2f}  (-{pct:5.1f}% of |negative|)")
        net = pos_total + neg_total
        lines.append(f"  NET cumulative reward : {net:+10.2f}")
    else:
        lines.append("  (no reward terms tracked — reward_manager._episode_sums unavailable)")
    lines.append("")

    lines.append("=== Drop mode breakdown ===")
    for k, v in drop_modes.items():
        lines.append(f"  {k:25s} : {v}")
    lines.append("")

    # ---- Per-episode breakdown ----
    lines.append("=== Per-episode breakdown ===")
    for r in per_episode_records:
        lines.append(
            f"  Ep {r['ep_id']:3d}: {r['outcome']:7s} | "
            f"first_grasp={r['first_grasp']:>3d} | first_lift={r['first_lift']:>3d} | "
            f"grasp_count={r['grasp_count']} | "
            f"min_d_goal={r['min_d_goal']:.3f} | "
            f"qdot_max={r['qdot_max']:6.2f} | "
            f"phase={r['deepest_phase']}"
        )
    lines.append("")

    # ---- Spawn condition vs outcome correlation ----
    lines.append("=== Spawn condition vs outcome correlation ===")

    def _bucket(value, low_thresh, high_thresh):
        if value < low_thresh:
            return "near"
        if value < high_thresh:
            return "mid"
        return "far"

    if per_episode_records:
        # Cube spawn distance to robot (world frame translated)
        # Using initial_robot_pos and initial_cube_w
        rate_by_bucket = {}
        for r in per_episode_records:
            cube0 = torch.tensor(r["initial_cube_w"])
            rob0 = torch.tensor(r["initial_robot_w"])
            d = float((cube0 - rob0).norm())
            b = _bucket(d, 0.15, 0.20)
            rate_by_bucket.setdefault(("cube_robot_dist", b), [0, 0])
            rate_by_bucket[("cube_robot_dist", b)][0] += 1
            if r["outcome"] == "SUCCESS":
                rate_by_bucket[("cube_robot_dist", b)][1] += 1

        # Goal distance to cube (in robot root frame for goal, world for cube — approx)
        for r in per_episode_records:
            cube0_b = torch.tensor(r["initial_cube_w"]) - torch.tensor(r["initial_robot_w"])
            goal_b = torch.tensor(r["initial_goal_b"])
            d = float((goal_b - cube0_b).norm())
            b = _bucket(d, 0.15, 0.25)
            rate_by_bucket.setdefault(("goal_cube_dist", b), [0, 0])
            rate_by_bucket[("goal_cube_dist", b)][0] += 1
            if r["outcome"] == "SUCCESS":
                rate_by_bucket[("goal_cube_dist", b)][1] += 1

        # Goal height (z component of goal in robot frame)
        for r in per_episode_records:
            gz = float(torch.tensor(r["initial_goal_b"])[2])
            b = "low" if gz < 0.13 else ("mid" if gz < 0.17 else "high")
            rate_by_bucket.setdefault(("goal_z", b), [0, 0])
            rate_by_bucket[("goal_z", b)][0] += 1
            if r["outcome"] == "SUCCESS":
                rate_by_bucket[("goal_z", b)][1] += 1

        # Print
        for category in ("cube_robot_dist", "goal_cube_dist", "goal_z"):
            buckets = [b for (cat, b) in rate_by_bucket.keys() if cat == category]
            if not buckets:
                continue
            lines.append(f"  by {category}:")
            order = ["near", "mid", "far"] if category != "goal_z" else ["low", "mid", "high"]
            for b in order:
                key = (category, b)
                if key not in rate_by_bucket:
                    continue
                n_total, n_succ = rate_by_bucket[key]
                pct = 100 * n_succ / max(n_total, 1)
                lines.append(f"    {b:6s} : {n_succ}/{n_total} success ({pct:.0f}%)")
    lines.append("")

    # ---- Spawn variability check ----
    lines.append("=== Spawn variability check (across all episodes) ===")
    if per_episode_records:
        cube_world = torch.tensor([r["initial_cube_w"] for r in per_episode_records])
        robot_world = torch.tensor([r["initial_robot_w"] for r in per_episode_records])
        cube_rel = cube_world - robot_world  # world axes, translated to robot base
        goal_b = torch.tensor([r["initial_goal_b"] for r in per_episode_records])

        def _stat_axis(label, t):
            mn, mx, mean, std = float(t.min()), float(t.max()), float(t.mean()), float(t.std())
            range_ = mx - mn
            verdict = "CONSTANT" if std < 0.003 else "ACTIVE"
            return (
                f"    {label:1s} : min={mn:+.4f}  max={mx:+.4f}  mean={mean:+.4f}  "
                f"std={std:.4f}  range={range_*100:5.1f} cm  [{verdict}]"
            )

        lines.append("  Initial cube position (world frame, translated to robot base):")
        for i, ax in enumerate("xyz"):
            lines.append(_stat_axis(ax, cube_rel[:, i]))
        lines.append("  Initial goal position (robot root frame, what the policy sees):")
        for i, ax in enumerate("xyz"):
            lines.append(_stat_axis(ax, goal_b[:, i]))

        # Pairwise distances between consecutive initial cubes — if these are
        # all near 0, every reset puts cube back in the same spot.
        if len(per_episode_records) > 1:
            diffs = []
            for j in range(1, len(per_episode_records)):
                d = float((cube_rel[j] - cube_rel[j - 1]).norm())
                diffs.append(d)
            d_t = torch.tensor(diffs)
            lines.append(f"  Consecutive-spawn delta (cube): "
                         f"mean={d_t.mean()*100:.2f} cm  max={d_t.max()*100:.2f} cm")
            n_zero = int((d_t < 0.005).sum().item())
            lines.append(f"    Episodes with cube spawn delta < 5 mm vs prev : {n_zero}/{len(diffs)}")
    else:
        lines.append("  (no episodes recorded)")
    lines.append("")

    # ---- Action histogram (per joint percentiles) ----
    lines.append("=== Action histogram (per joint percentiles, raw policy output) ===")
    if action_history:
        all_a = torch.cat(action_history, dim=0)  # (total_steps × num_envs, action_dim)
        action_dim = all_a.shape[-1]
        # Try to label dims — arm joints + gripper (binary).
        labels = list(robot.joint_names) + ["?"] * max(0, action_dim - len(robot.joint_names))
        labels = labels[:action_dim]
        lines.append(f"  Total samples : {all_a.shape[0]}")
        lines.append(f"  {'dim':4s} {'name':18s} {'p05':>7s} {'p25':>7s} {'p50':>7s} {'p75':>7s} {'p95':>7s}")
        for d in range(action_dim):
            col = all_a[:, d]
            p05, p25, p50, p75, p95 = (
                torch.quantile(col, torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95]))
            ).tolist()
            sat_flag = " ⚠SAT" if max(abs(p05), abs(p95)) > 2.0 else ""
            lines.append(
                f"  [{d}]  {labels[d]:18s} "
                f"{p05:+7.3f} {p25:+7.3f} {p50:+7.3f} {p75:+7.3f} {p95:+7.3f}{sat_flag}"
            )
    else:
        lines.append("  (no actions recorded)")
    lines.append("")

    # ---- Grasp lifecycle analysis ----
    lines.append("=== Grasp lifecycle analysis ===")
    if grasp_records:
        n = len(grasp_records)
        cs = torch.tensor([r["cube_speed_at_fire"] for r in grasp_records])
        gq = torch.tensor([r["gripper_qdot_at_fire"] for r in grasp_records])
        jc = torch.tensor([r["jaw_cube_at_fire"] for r in grasp_records])
        lines.append(f"  Total GRASP_FIRE events : {n}")
        lines.append(f"  Mean cube speed at GRASP_FIRE     : {cs.mean():.3f} m/s  "
                     f"(p50={cs.median():.3f}, max={cs.max():.3f})")
        lines.append(f"    → high values = bras frappe le cube avant grasp")
        lines.append(f"  Mean gripper qdot at GRASP_FIRE   : {gq.mean():+.3f} rad/s  "
                     f"(closing transient; negative = closing)")
        lines.append(f"  Mean d_jaw at GRASP_FIRE          : {jc.mean():.4f} m  (threshold = 0.04)")
        # Grasp_LOST distribution
        lost = [r for r in grasp_records if r["lost_step"] is not None]
        lines.append(f"  Grasps that ended in grasp_LOST   : {len(lost)} / {n}")
        if lost:
            lj = torch.tensor([r["lost_d_jaw"] for r in lost])
            ld = torch.tensor([r["duration_steps"] for r in lost], dtype=torch.float)
            lines.append(f"    d_jaw at grasp_LOST   mean : {lj.mean():.4f} m  max : {lj.max():.4f} m")
            lines.append(f"    duration_steps        mean : {ld.mean():.2f}  max : {ld.max():.0f}")
            # Slip vs ejection: small lost_d_jaw = cube barely escaped (slip),
            # large = cube ejected far (likely thrown by arm motion).
            n_slip = int((lj < 0.10).sum().item())
            n_eject = int((lj >= 0.10).sum().item())
            lines.append(f"    slip vs ejection : slip={n_slip} (d<0.10)  ejection={n_eject} (d≥0.10)")
    else:
        lines.append("  (no grasp events recorded)")
    lines.append("")
    lines.append(f"=== Output files ===")
    lines.append(f"  JSONL events  : {jsonl_path}")
    lines.append(f"  Summary file  : {summary_path}")
    lines.append("")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    log_event({"type": "RUN_END", "summary": summary_text}, also_print=False)
    jsonl_f.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
