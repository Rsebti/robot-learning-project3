"""Play with diagnostic logging — visual + per-event stdout trace.

Loads a V2.9 (or V2.x) checkpoint, runs the policy with rendering, and
logs key events to stdout so we can analyze what the policy is doing
even without watching the visual ourselves.

Per step (only on event): first-close, grasp-fire, grasp-lost, lift,
drop, success. Per episode: outcome + timing summary. At the end:
aggregate stats over all completed episodes.

Usage
-----
.. code-block:: powershell

    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    .\\.venv\\Scripts\\Activate.ps1
    python -m sim.eval2.scripts.play_diagnose `
        --task Isaac-LeIsaac-SO101-Lift-Visual-V29-Play-v0 `
        --num_envs 4 --num_episodes 20 --enable_cameras

Add ``--checkpoint <path>`` to pin a specific iter, otherwise loads the
most recent under ``logs/rsl_rl/lift_v2_9/<latest_run>/``.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort:skip

parser = argparse.ArgumentParser(description="Diagnostic play of an RSL-RL checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-LeIsaac-SO101-Lift-Visual-V29-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=20,
                    help="Stop after this many episodes (across all envs) have terminated.")
parser.add_argument("--trace_env", type=int, default=0,
                    help="Index of env to log continuous per-step trajectory (cube/jaw position, "
                         "joint velocities). Default 0. Set to -1 to disable continuous tracing "
                         "(events still logged for all envs).")
parser.add_argument("--trace_every", type=int, default=1,
                    help="Log a trajectory line every N steps for the trace_env (default 1).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Late imports — Isaac Lab needs AppLauncher live first.
import os
import time
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import sim.eval2  # noqa: F401 — registers our tasks
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


# ---- Event detection helpers ----------------------------------------------


def _grasp_predicate(jaw_cube_dist: torch.Tensor, gripper_q: torch.Tensor,
                     diff_th: float = 0.04, grip_th: float = 0.35) -> torch.Tensor:
    """Same predicate as `cube_grasped` in our RewardsCfg."""
    return (jaw_cube_dist < diff_th) & (gripper_q < grip_th)


def _lift_predicate(cube_z_world: torch.Tensor, base_z_world: torch.Tensor,
                    height_th: float = 0.08) -> torch.Tensor:
    """Same as `cube_lifted_above_base(threshold=0.08)` in V2.8.5+ rewards."""
    return (cube_z_world - base_z_world) > height_th


# ---------------------------------------------------------------------------


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )
    print(f"[INFO] log_root_path: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env_cfg.log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_unwrapped = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env_unwrapped.device)

    # Scene handles for diagnostics.
    cube: RigidObject = env_unwrapped.scene["cube"]
    robot: Articulation = env_unwrapped.scene["robot"]
    ee_frame = env_unwrapped.scene["ee_frame"]
    base_idx = robot.find_bodies("base")[0][0]
    gripper_joint_idx = robot.joint_names.index("gripper")

    num_envs = args_cli.num_envs
    target_episodes = args_cli.num_episodes

    # Per-env episode state tracking.
    ep_step = torch.zeros(num_envs, dtype=torch.int32)          # current step in episode
    first_close = torch.full((num_envs,), -1, dtype=torch.int32)  # first step where action=close
    first_grasp = torch.full((num_envs,), -1, dtype=torch.int32)
    first_lift = torch.full((num_envs,), -1, dtype=torch.int32)
    first_tip_below = torch.full((num_envs,), -1, dtype=torch.int32)  # first step gripper/jaw under table
    grasp_count = torch.zeros(num_envs, dtype=torch.int32)      # # of grasp events (count slips)
    grasp_steps = torch.zeros(num_envs, dtype=torch.int32)      # total steps grasped
    was_grasped = torch.zeros(num_envs, dtype=torch.bool)
    min_cube_goal = torch.full((num_envs,), 999.0)              # min cube→goal distance seen
    min_jaw_cube = torch.full((num_envs,), 999.0)
    min_gripper_z = torch.full((num_envs,), 999.0)              # min height of gripper target
    min_jaw_z = torch.full((num_envs,), 999.0)                  # min height of jaw target
    tip_below_steps = torch.zeros(num_envs, dtype=torch.int32)  # cumulative steps tip was under table
    max_qdot_ep = torch.zeros(num_envs)                         # max |joint_vel| seen this ep
    sum_qdot_ep = torch.zeros(num_envs)                         # sum |joint_vel|max for mean
    qdot_count = torch.zeros(num_envs, dtype=torch.int32)       # # steps for averaging qdot
    max_cube_speed_ep = torch.zeros(num_envs)                   # max cube linear speed (slip indicator)
    ep_id = torch.arange(num_envs, dtype=torch.int32)           # global episode id per env

    # Threshold for "tip is below the table top". Empirically the table top
    # sits at ~0.04 in world z (cube z=0.0615 minus cube half-extent 0.02).
    # Anything below 0.03 means gripper/jaw has clipped through the table.
    TIP_BELOW_THRESHOLD = 0.03

    # Aggregates over all completed episodes.
    completed = 0
    summary = {"success": 0, "dropped": 0, "timeout": 0,
               "ttg": [], "ttsuccess": [], "ttdrop": [],
               "grasp_dur": [], "grasp_count_per_ep": [],
               "min_cube_goal": [],
               "tip_below_steps": [], "min_gripper_z": [], "min_jaw_z": [],
               "ep_with_tip_below": 0,
               "max_qdot_per_ep": [], "mean_qdot_per_ep": [],
               "max_cube_speed_per_ep": []}
    drop_modes = {"grasp_then_slip": 0, "bump_without_grasp": 0,
                  "tip_below_table": 0, "other": 0}

    print(f"\n=== DIAGNOSTIC PLAY ===")
    print(f"task         : {task_name}")
    print(f"checkpoint   : {os.path.basename(resume_path)}")
    print(f"num_envs     : {num_envs}")
    print(f"num_episodes : {target_episodes}\n")

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    while simulation_app.is_running() and completed < target_episodes:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, tuple):
                obs = obs[0]

        # ---- per-step diagnostics ----
        cube_w = cube.data.root_pos_w           # (B, 3)
        base_z_w = robot.data.body_pos_w[:, base_idx, 2]
        gripper_q = robot.data.joint_pos[:, gripper_joint_idx]   # (B,)
        gripper_w = ee_frame.data.target_pos_w[:, 0, :]          # gripper target
        jaw_w = ee_frame.data.target_pos_w[:, 1, :]              # jaw target
        jaw_cube = torch.norm(jaw_w - cube_w, dim=-1)            # (B,)
        gripper_cube = torch.norm(gripper_w - cube_w, dim=-1)

        # goal in world frame (from command manager)
        try:
            cmd = env_unwrapped.command_manager.get_command("object_pose")
            goal_b = cmd[:, :3]
            goal_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, goal_b)
            cube_goal = torch.norm(cube_w - goal_w, dim=-1)
        except Exception:
            cube_goal = torch.full_like(jaw_cube, float("nan"))

        is_grasped_now = _grasp_predicate(jaw_cube, gripper_q)
        is_lifted_now = _lift_predicate(cube_w[:, 2], base_z_w)

        # Last action's gripper value (binary action result); read joint target instead.
        # We approximate "policy chose close" = gripper_q decreasing toward 0.
        # We know binary action open=0.5 / close=0.0. So we use joint target proxy = gripper_q < 0.25.
        action_is_close = gripper_q < 0.25

        # Tip-below-table check on either gripper target or jaw target.
        gripper_z = gripper_w[:, 2]
        jaw_z = jaw_w[:, 2]
        tip_below_now = (gripper_z < TIP_BELOW_THRESHOLD) | (jaw_z < TIP_BELOW_THRESHOLD)

        # Joint velocities (rad/s for revolute joints).
        joint_vel = robot.data.joint_vel                          # (B, J)
        abs_qdot = joint_vel.abs()
        max_qdot_now = abs_qdot.max(dim=-1).values                # (B,) — max joint speed
        max_qdot_idx = abs_qdot.max(dim=-1).indices               # which joint is fastest

        # Cube linear velocity (slip indicator when grasped).
        try:
            cube_lin_vel = cube.data.root_lin_vel_w               # (B, 3)
            cube_speed = cube_lin_vel.norm(dim=-1)
        except AttributeError:
            cube_speed = torch.zeros(num_envs, device=cube_w.device)

        # Update min trackers
        min_cube_goal = torch.minimum(min_cube_goal, cube_goal.cpu())
        min_jaw_cube = torch.minimum(min_jaw_cube, jaw_cube.cpu())
        min_gripper_z = torch.minimum(min_gripper_z, gripper_z.cpu())
        min_jaw_z = torch.minimum(min_jaw_z, jaw_z.cpu())
        max_qdot_ep = torch.maximum(max_qdot_ep, max_qdot_now.cpu())
        sum_qdot_ep += max_qdot_now.cpu()
        qdot_count += 1
        max_cube_speed_ep = torch.maximum(max_cube_speed_ep, cube_speed.cpu())

        # ---- Continuous trajectory log for the traced env ----
        trace_i = args_cli.trace_env
        if 0 <= trace_i < num_envs and (int(ep_step[trace_i]) % args_cli.trace_every == 0):
            cx, cy, cz = cube_w[trace_i, 0].item(), cube_w[trace_i, 1].item(), cube_w[trace_i, 2].item()
            jx, jy, jz = jaw_w[trace_i, 0].item(), jaw_w[trace_i, 1].item(), jaw_w[trace_i, 2].item()
            qdot_str = " ".join(f"{q:+.2f}" for q in joint_vel[trace_i].cpu().tolist())
            joint_max = robot.joint_names[int(max_qdot_idx[trace_i])]
            print(f"[trace env{trace_i} t={int(ep_step[trace_i]):3d}] "
                  f"cube=({cx:+.3f},{cy:+.3f},{cz:+.3f}) "
                  f"jaw=({jx:+.3f},{jy:+.3f},{jz:+.3f}) "
                  f"|jaw-cube|={jaw_cube[trace_i].item():.3f} "
                  f"qdot=[{qdot_str}] "
                  f"|qdot|max={max_qdot_now[trace_i].item():.2f}({joint_max}) "
                  f"cube_v={cube_speed[trace_i].item():.2f}")

        # Update grasp count + duration (per env)
        for i in range(num_envs):
            ep_step[i] += 1
            cx, cy, cz = cube_w[i, 0].item(), cube_w[i, 1].item(), cube_w[i, 2].item()
            if action_is_close[i] and first_close[i] < 0:
                first_close[i] = ep_step[i]
                print(f"[Ep#{ep_id[i].item():3d}/env{i}] step {int(ep_step[i]):3d}: "
                      f"FIRST_CLOSE  d_grip={gripper_cube[i].item():.3f} "
                      f"d_jaw_cube={jaw_cube[i].item():.3f}  "
                      f"cube=({cx:+.3f}, {cy:+.3f}, {cz:+.3f})")
            if is_grasped_now[i]:
                grasp_steps[i] += 1
                if not was_grasped[i]:
                    grasp_count[i] += 1
                    if first_grasp[i] < 0:
                        first_grasp[i] = ep_step[i]
                        print(f"[Ep#{ep_id[i].item():3d}/env{i}] step {int(ep_step[i]):3d}: "
                              f"GRASP_FIRE  d_jaw={jaw_cube[i].item():.3f} "
                              f"grip_q={gripper_q[i].item():.3f}  "
                              f"cube=({cx:+.3f}, {cy:+.3f}, {cz:+.3f})")
                was_grasped[i] = True
            else:
                if was_grasped[i]:
                    print(f"[Ep#{ep_id[i].item():3d}/env{i}] step {int(ep_step[i]):3d}: "
                          f"grasp_LOST after {int(grasp_steps[i])} step(s) "
                          f"(d_jaw={jaw_cube[i].item():.3f})  "
                          f"cube=({cx:+.3f}, {cy:+.3f}, {cz:+.3f})")
                was_grasped[i] = False
            if is_lifted_now[i] and first_lift[i] < 0:
                first_lift[i] = ep_step[i]
                print(f"[Ep#{ep_id[i].item():3d}/env{i}] step {int(ep_step[i]):3d}: "
                      f"LIFT  cube_z_rel={cz-base_z_w[i].item():+.3f}  "
                      f"cube=({cx:+.3f}, {cy:+.3f}, {cz:+.3f})")
            if tip_below_now[i]:
                tip_below_steps[i] += 1
                if first_tip_below[i] < 0:
                    first_tip_below[i] = ep_step[i]
                    print(f"[Ep#{ep_id[i].item():3d}/env{i}] step {int(ep_step[i]):3d}: "
                          f"⚠ TIP_BELOW_TABLE  gripper_z={gripper_z[i].item():+.3f} "
                          f"jaw_z={jaw_z[i].item():+.3f}  "
                          f"cube=({cx:+.3f}, {cy:+.3f}, {cz:+.3f})")

        # ---- Handle episode terminations ----
        if dones.any():
            done_ids = torch.nonzero(dones, as_tuple=True)[0]
            for i in done_ids.cpu().tolist():
                step = int(ep_step[i])
                # Determine outcome
                last_cube_x = cube_w[i, 0].item()
                last_cube_y = cube_w[i, 1].item()
                last_cube_z = cube_w[i, 2].item()
                last_cube_goal = cube_goal[i].item()
                # NOTE: env auto-resets on done, so cube_w/cube_goal here may
                # already reflect the NEW episode's spawn. We use the per-episode
                # MIN tracker for success (catches cube briefly at goal even if
                # it bounced away by the time we read state).
                ep_min_cube_goal = float(min_cube_goal[i])
                if ep_min_cube_goal < 0.05:
                    # Cube was within 5cm of goal at some point → success
                    outcome = "SUCCESS"
                    summary["success"] += 1
                    summary["ttsuccess"].append(step)
                elif last_cube_z < 0.04:
                    outcome = "DROPPED"
                    summary["dropped"] += 1
                    summary["ttdrop"].append(step)
                    # Classify drop mode
                    if first_tip_below[i] >= 0 and first_tip_below[i] <= step - 2:
                        drop_modes["tip_below_table"] += 1
                    elif first_grasp[i] >= 0:
                        drop_modes["grasp_then_slip"] += 1
                    elif first_close[i] >= 0:
                        drop_modes["bump_without_grasp"] += 1
                    else:
                        drop_modes["other"] += 1
                else:
                    outcome = "TIMEOUT"
                    summary["timeout"] += 1

                if first_grasp[i] >= 0:
                    summary["ttg"].append(int(first_grasp[i]))
                summary["grasp_dur"].append(int(grasp_steps[i]))
                summary["grasp_count_per_ep"].append(int(grasp_count[i]))
                summary["min_cube_goal"].append(float(min_cube_goal[i]))
                summary["tip_below_steps"].append(int(tip_below_steps[i]))
                summary["min_gripper_z"].append(float(min_gripper_z[i]))
                summary["min_jaw_z"].append(float(min_jaw_z[i]))
                summary["max_qdot_per_ep"].append(float(max_qdot_ep[i]))
                mean_qdot_this_ep = float(sum_qdot_ep[i]) / max(int(qdot_count[i]), 1)
                summary["mean_qdot_per_ep"].append(mean_qdot_this_ep)
                summary["max_cube_speed_per_ep"].append(float(max_cube_speed_ep[i]))
                if first_tip_below[i] >= 0:
                    summary["ep_with_tip_below"] += 1

                tip_marker = " ⚠TIP" if first_tip_below[i] >= 0 else ""
                print(
                    f"[Ep#{ep_id[i].item():3d}/env{i}] {outcome:7s}{tip_marker} step={step:3d}  "
                    f"first_close={int(first_close[i]):>3}  first_grasp={int(first_grasp[i]):>3}  "
                    f"first_lift={int(first_lift[i]):>3}  "
                    f"first_tip_below={int(first_tip_below[i]):>3}  "
                    f"grasp_count={int(grasp_count[i])} (sustained {int(grasp_steps[i])} steps)  "
                    f"tip_below_steps={int(tip_below_steps[i])}  "
                    f"min_grip_z={float(min_gripper_z[i]):+.3f}  min_jaw_z={float(min_jaw_z[i]):+.3f}  "
                    f"max_|qdot|={float(max_qdot_ep[i]):.2f}  mean_|qdot|max={mean_qdot_this_ep:.2f}  "
                    f"max_cube_v={float(max_cube_speed_ep[i]):.2f}  "
                    f"min_cube_goal={float(min_cube_goal[i]):.3f}  "
                    f"final_cube=({last_cube_x:+.3f}, {last_cube_y:+.3f}, {last_cube_z:+.3f})\n"
                )
                completed += 1

                # Reset trackers for this env (next episode will reuse this slot).
                ep_step[i] = 0
                first_close[i] = -1
                first_grasp[i] = -1
                first_lift[i] = -1
                first_tip_below[i] = -1
                grasp_count[i] = 0
                grasp_steps[i] = 0
                tip_below_steps[i] = 0
                was_grasped[i] = False
                min_cube_goal[i] = 999.0
                min_jaw_cube[i] = 999.0
                min_gripper_z[i] = 999.0
                min_jaw_z[i] = 999.0
                max_qdot_ep[i] = 0.0
                sum_qdot_ep[i] = 0.0
                qdot_count[i] = 0
                max_cube_speed_ep[i] = 0.0
                ep_id[i] = num_envs + completed  # unique ids

        if completed >= target_episodes:
            break

    # ---- Aggregate report ----
    print("\n" + "=" * 60)
    print(f"=== SUMMARY over {completed} completed episodes ===")
    print(f"  SUCCESS  : {summary['success']:3d} ({100*summary['success']/max(completed,1):.0f}%)")
    print(f"  DROPPED  : {summary['dropped']:3d} ({100*summary['dropped']/max(completed,1):.0f}%)")
    print(f"  TIMEOUT  : {summary['timeout']:3d} ({100*summary['timeout']/max(completed,1):.0f}%)")
    print()
    if summary["ttg"]:
        ttg = torch.tensor(summary["ttg"], dtype=torch.float)
        print(f"  Time to first grasp   : mean={ttg.mean():.1f} std={ttg.std():.1f} steps")
    if summary["ttsuccess"]:
        tts = torch.tensor(summary["ttsuccess"], dtype=torch.float)
        print(f"  Time to success       : mean={tts.mean():.1f} std={tts.std():.1f} steps")
    if summary["ttdrop"]:
        ttd = torch.tensor(summary["ttdrop"], dtype=torch.float)
        print(f"  Time to drop          : mean={ttd.mean():.1f} std={ttd.std():.1f} steps")
    gd = torch.tensor(summary["grasp_dur"], dtype=torch.float) if summary["grasp_dur"] else None
    if gd is not None:
        print(f"  Grasp duration / ep   : mean={gd.mean():.1f} std={gd.std():.1f} steps")
    gc = torch.tensor(summary["grasp_count_per_ep"], dtype=torch.float) if summary["grasp_count_per_ep"] else None
    if gc is not None:
        print(f"  # grasp events / ep   : mean={gc.mean():.2f} (1 = clean grasp, >1 = slips and re-grasps)")
    mcg = torch.tensor(summary["min_cube_goal"], dtype=torch.float) if summary["min_cube_goal"] else None
    if mcg is not None:
        print(f"  Min cube→goal / ep    : mean={mcg.mean():.3f} m  (success threshold = 0.05 m)")
    print()
    print(f"  Tip-below-table behavior (TIP_BELOW_THRESHOLD = {TIP_BELOW_THRESHOLD} m):")
    print(f"    Episodes with tip going below table : "
          f"{summary['ep_with_tip_below']:3d} / {completed} "
          f"({100*summary['ep_with_tip_below']/max(completed,1):.0f}%)")
    tbs = torch.tensor(summary["tip_below_steps"], dtype=torch.float) if summary["tip_below_steps"] else None
    if tbs is not None:
        print(f"    Steps below table / ep              : mean={tbs.mean():.1f}  max={tbs.max():.0f}")
    mgz = torch.tensor(summary["min_gripper_z"], dtype=torch.float) if summary["min_gripper_z"] else None
    if mgz is not None:
        print(f"    Min gripper.z / ep (world)          : mean={mgz.mean():+.3f}  min={mgz.min():+.3f}")
    mjz = torch.tensor(summary["min_jaw_z"], dtype=torch.float) if summary["min_jaw_z"] else None
    if mjz is not None:
        print(f"    Min jaw.z / ep (world)              : mean={mjz.mean():+.3f}  min={mjz.min():+.3f}")
    print(f"    (Reference: cube spawn z=+0.0615, table top ≈ +0.04, world floor = 0)")
    print()
    mq = torch.tensor(summary["max_qdot_per_ep"], dtype=torch.float) if summary["max_qdot_per_ep"] else None
    if mq is not None:
        print(f"  Joint velocity (rad/s):")
        print(f"    Max |qdot| / ep   : mean={mq.mean():.2f}  max={mq.max():.2f}")
    mnq = torch.tensor(summary["mean_qdot_per_ep"], dtype=torch.float) if summary["mean_qdot_per_ep"] else None
    if mnq is not None:
        print(f"    Mean |qdot|max/step / ep (smoothness): mean={mnq.mean():.2f}")
    mcs = torch.tensor(summary["max_cube_speed_per_ep"], dtype=torch.float) if summary["max_cube_speed_per_ep"] else None
    if mcs is not None:
        print(f"  Cube linear speed (m/s):")
        print(f"    Max cube speed / ep : mean={mcs.mean():.2f}  max={mcs.max():.2f}  "
              f"(high values during grasp = slip)")
    print()
    print(f"  Drop modes:")
    for k, v in drop_modes.items():
        print(f"    {k:25s} : {v}")
    print("=" * 60)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
