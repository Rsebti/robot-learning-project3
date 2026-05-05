"""Smoke test for the scripted pick controller (BC warmstart Step 2).

Loads the Eval 2 v1 Play env (50 envs by default, no observation noise) and
rolls out the ScriptedPickController for ``--num_episodes`` total episodes,
reporting the success rate. We need >=80% before bothering with the larger
data collection, otherwise the BC dataset would be too noisy for the actor
to learn from.

Usage (from the isaac_so_arm101 venv with this repo installed editable):

    uv run python -m sim.eval2.bc.run_scripted --num_episodes 100

Optional flags:
    --task                 Gym task ID (default Eval2-PickInClutter-Play-v1)
    --num_envs             Parallel envs (default 50)
    --no_headless          Show the Isaac Sim window
    --verbose_phases       Print phase histogram every 10 episodes

Pass criteria: success rate >= 80%. If lower:
    - Watch a rollout with --no_headless, see which phase fails.
    - Most likely culprits, in order: (a) DESCEND too aggressive (jaws hit
      the cube before closing), (b) wrist drifts off "down" orientation
      mid-trajectory and grasp misses, (c) ABOVE_BOWL approach can't reach
      the bowl from the LIFT pose due to 5-DOF singularity at the wrist.
    - Tune the constants at the top of scripted_controller.py and re-run.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# AppLauncher MUST be invoked before any torch / isaaclab import.
parser = argparse.ArgumentParser(description="Smoke test the scripted pick controller.")
parser.add_argument("--task", type=str, default="Eval2-PickInClutter-Play-v1")
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=100,
    help="Total episodes to roll out across all envs.",
)
parser.add_argument(
    "--no_headless",
    action="store_true",
    help="Open the Isaac Sim window. Slower but lets you watch the rollouts.",
)
parser.add_argument(
    "--verbose_phases",
    action="store_true",
    help="Print phase histogram every 10 episodes.",
)
parser.add_argument(
    "--debug_env0",
    action="store_true",
    help="Every 5 steps, print env 0's phase + target xyz + actual gripper_link + "
    "tip + target block + bowl. Slows down the run; use with --num_episodes 1-5.",
)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.no_headless

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main():
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    # Side-effect: registers our gym envs.
    import sim.eval2  # noqa: F401
    from sim.eval2.bc.scripted_controller import ScriptedPickController
    from sim.eval2.pick_in_clutter_env_cfg import BOWL_INNER_HALF

    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    # Override episode_length to fit our scripted phase budget. Sum of
    # PHASE_MAX_STEPS = 360 steps; at env_dt = 0.02 s -> 7.2 s. We give
    # 10 s for safety margin.
    env_cfg.episode_length_s = 10.0

    print(f"[scripted] gym.make({args.task!r}, num_envs={args.num_envs})", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()
    print(f"[scripted] env reset -> obs type={type(obs).__name__}", flush=True)

    # IMPORTANT: build the controller AFTER reset so default_joint_pos is
    # populated correctly on the underlying Articulation.
    controller = ScriptedPickController(env.unwrapped)
    print(
        f"[scripted] arm joints = {controller.arm_joint_names}  "
        f"(ids={controller.arm_joint_ids})",
        flush=True,
    )
    print(
        f"[scripted] ee body idx={controller.ee_body_idx} "
        f"jacobi idx={controller.ee_jacobi_idx}",
        flush=True,
    )

    n_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    # Per-env "did we succeed at any step this episode?" flag.
    # Auto-reset wipes the success state on env.step before we can read it,
    # so we OR-accumulate success_now across the episode and tally on done.
    episode_success_seen = torch.zeros(n_envs, dtype=torch.bool, device=device)
    n_episodes_done = 0
    n_successes = 0
    # Track which phase each env was in when its episode ended (success or not),
    # to spot where rollouts get stuck.
    end_phase_hist = torch.zeros(controller.NUM_PHASES, dtype=torch.long, device=device)
    success_phase_hist = torch.zeros(controller.NUM_PHASES, dtype=torch.long, device=device)

    print(
        f"[scripted] rollout starting | envs={n_envs} | target episodes={args.num_episodes}",
        flush=True,
    )

    step_idx = 0
    while n_episodes_done < args.num_episodes:
        # ---- Debug snapshot for env 0 ----
        if args.debug_env0 and step_idx % 5 == 0:
            scn = env.unwrapped.scene
            robot_a = scn["robot"]
            ee_body_idx_dbg = controller.ee_body_idx
            gripper_link_pos = robot_a.data.body_pose_w[0, ee_body_idx_dbg, :3].cpu().tolist()
            tip_pos = scn["ee_frame"].data.target_pos_w[0, 0, :].cpu().tolist()
            target_pos_w = controller._compute_target_xyz_world()[0].cpu().tolist()
            block_red_p = scn["block_red"].data.root_pos_w[0].cpu().tolist()
            block_blue_p = scn["block_blue"].data.root_pos_w[0].cpu().tolist()
            bowl_p = scn["bowl_floor"].data.root_pos_w[0].cpu().tolist()
            tgt_color = int(env.unwrapped.target_color[0].item())
            phase_idx = int(controller.phase[0].item())
            phase_name = controller.PHASE_NAMES[phase_idx]
            phase_step = int(controller.phase_step[0].item())
            print(
                f"[debug] step {step_idx:4d} env0 phase={phase_name}({phase_idx}) "
                f"step_in_phase={phase_step} target_color={'red' if tgt_color == 0 else 'blue'}",
                flush=True,
            )
            print(
                f"          target_xyz_world(gripper_link) = ({target_pos_w[0]:+.3f}, "
                f"{target_pos_w[1]:+.3f}, {target_pos_w[2]:+.3f})",
                flush=True,
            )
            print(
                f"          actual gripper_link            = ({gripper_link_pos[0]:+.3f}, "
                f"{gripper_link_pos[1]:+.3f}, {gripper_link_pos[2]:+.3f})",
                flush=True,
            )
            print(
                f"          actual tip (ee_frame)          = ({tip_pos[0]:+.3f}, "
                f"{tip_pos[1]:+.3f}, {tip_pos[2]:+.3f})",
                flush=True,
            )
            tgt_block_p = block_blue_p if tgt_color == 1 else block_red_p
            tgt_block_z = tgt_block_p[2]
            lifted_marker = " <-- LIFTED!" if tgt_block_z > 0.03 else ""
            print(
                f"          block_red                      = ({block_red_p[0]:+.3f}, "
                f"{block_red_p[1]:+.3f}, {block_red_p[2]:+.3f})  "
                f"block_blue=({block_blue_p[0]:+.3f},{block_blue_p[1]:+.3f},{block_blue_p[2]:+.3f}){lifted_marker}",
                flush=True,
            )
            # Joint state — to detect joint-limit saturation during DESCEND.
            j_pos = robot_a.data.joint_pos[0].cpu().tolist()
            j_vel = robot_a.data.joint_vel[0].cpu().tolist()
            j_names = robot_a.data.joint_names
            print(
                "          joint pos (rad): "
                + ", ".join(f"{n}={p:+.3f}" for n, p in zip(j_names, j_pos)),
                flush=True,
            )
            print(
                "          joint vel (rad/s): "
                + ", ".join(f"{n}={v:+.3f}" for n, v in zip(j_names, j_vel)),
                flush=True,
            )
            print(
                f"          bowl_floor                     = ({bowl_p[0]:+.3f}, "
                f"{bowl_p[1]:+.3f}, {bowl_p[2]:+.3f})",
                flush=True,
            )

        action = controller.compute_action()  # (N, 6)
        if args.debug_env0 and step_idx % 5 == 0:
            print(
                f"          action env0                    = "
                f"arm({action[0, 0].item():+.2f},{action[0, 1].item():+.2f},"
                f"{action[0, 2].item():+.2f},{action[0, 3].item():+.2f},"
                f"{action[0, 4].item():+.2f}) grip={action[0, 5].item():+.2f}",
                flush=True,
            )
        step_idx += 1

        # Compute success criterion at the CURRENT state (BEFORE step), same
        # criterion as the env's success termination + deploy/eval2_inference.py.
        scene = env.unwrapped.scene
        red = scene["block_red"]
        blue = scene["block_blue"]
        target_idx = env.unwrapped.target_color
        is_red = (target_idx == 0).unsqueeze(-1)
        target_pos_w = torch.where(is_red, red.data.root_pos_w, blue.data.root_pos_w)
        bowl_pos_w = scene["bowl_floor"].data.root_pos_w
        xy_dist = torch.norm(target_pos_w[:, :2] - bowl_pos_w[:, :2], dim=1)
        dz = target_pos_w[:, 2] - bowl_pos_w[:, 2]
        success_now = (xy_dist < BOWL_INNER_HALF) & (dz > -0.01) & (dz < 0.10)
        episode_success_seen |= success_now

        # Step
        obs, rew, terminated, truncated, info = env.step(action)
        dones = (terminated | truncated)

        # Tally per-env episode endings
        done_envs = dones.nonzero(as_tuple=False).flatten()
        if done_envs.numel() > 0:
            for env_idx in done_envs.tolist():
                phase_at_end = controller.phase[env_idx].item()
                end_phase_hist[phase_at_end] += 1
                if bool(episode_success_seen[env_idx]):
                    n_successes += 1
                    success_phase_hist[phase_at_end] += 1
                episode_success_seen[env_idx] = False
                n_episodes_done += 1
                if n_episodes_done >= args.num_episodes:
                    break
            # Reset controller state for envs whose episode just ended.
            controller.reset(done_envs)

            # Periodic logging.
            if n_episodes_done % 10 == 0 or n_episodes_done >= args.num_episodes:
                rate = n_successes / max(n_episodes_done, 1) * 100.0
                line = (
                    f"[scripted] {n_episodes_done:>4d}/{args.num_episodes} "
                    f"| success {n_successes}/{n_episodes_done} = {rate:.1f}%"
                )
                if args.verbose_phases:
                    histogram = " ".join(
                        f"{name}={int(end_phase_hist[i].item())}"
                        for i, name in enumerate(controller.PHASE_NAMES)
                    )
                    line += f"\n          end-phase counts: {histogram}"
                print(line, flush=True)

    # Final report
    rate = n_successes / max(n_episodes_done, 1) * 100.0
    print("\n" + "=" * 60)
    print(f"[scripted] FINAL: {n_successes}/{n_episodes_done} = {rate:.2f}% success rate")
    print(f"[scripted] PASS = {rate >= 80.0} (target = 80%)")

    print("\n[scripted] end-phase histogram (which phase the EE was in when episode ended):")
    for i, name in enumerate(controller.PHASE_NAMES):
        ended = int(end_phase_hist[i].item())
        succeeded = int(success_phase_hist[i].item())
        if ended > 0:
            print(f"           {name:<10}  ended={ended:4d}  succeeded={succeeded:4d}")

    env.close()


if __name__ == "__main__":
    import sys
    import traceback

    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        sys.stdout.flush()
        sys.stderr.flush()
        print("\n[scripted] !!! UNCAUGHT EXCEPTION !!!", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        simulation_app.close()
