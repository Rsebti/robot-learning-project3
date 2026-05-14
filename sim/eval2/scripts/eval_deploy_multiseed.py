"""Deploy a SquintActor checkpoint across N seeds and report per-seed outcome.

For each seed:
  1. Reset env (PLAY mode — random cube/bowl)
  2. Run policy for N steps (default 75 = Squint episode length)
  3. Track:
     - Did the goal cube ever leave the table? (lift detected: z > 0.04)
     - Did the goal cube end up in the bowl? (xy within bowl radius AND z near bowl floor)
     - Max contact force, closest tcp→cube distance
     - Final cube position vs initial
  4. Save a per-seed line and an aggregate summary.

Output: notes/_multiseed_eval.log + a small JSON summary.
"""
from __future__ import annotations
import argparse, json, math, os, sys
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=r"C:/Users/user/Downloads/ckpt (8).pt")
parser.add_argument("--task", default="Isaac-SquintNative-Place-Play-v0")
parser.add_argument("--n_steps", type=int, default=75)
parser.add_argument("--n_seeds", type=int, default=10)
parser.add_argument("--seed_start", type=int, default=0)
parser.add_argument("--lift_z_threshold", type=float, default=0.04,
                    help="Cube z above table for 'lifted' (table at z=0, cube half-size 0.01).")
parser.add_argument("--bowl_radius_xy", type=float, default=0.06,
                    help="Cube within this xy radius of bowl centre AND z<0.04 counts as placed.")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
args.num_envs = 1
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import sim.eval2  # noqa: F401, E402
from sim.eval2.policy import CNNEncoder, SquintActor  # noqa: E402


PALETTE = ("red", "blue", "green", "yellow", "purple", "orange")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    enc_sd, actor_sd = ckpt["encoder"], ckpt["actor"]
    n_act = int(actor_sd["action_scale"].shape[-1])
    n_state = int(actor_sd["proj.state_proj.0.weight"].shape[-1])

    env_cfg = parse_env_cfg(args.task, num_envs=1, device=str(device))
    try: env_cfg.recorders = None
    except Exception: pass
    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped

    encoder = CNNEncoder(n_obs=(16, 16, 3), device=device).eval()
    a_scale = actor_sd["action_scale"]
    a_bias = actor_sd["action_bias"]
    actor = SquintActor(
        n_rgb_repr=encoder.repr_dim, n_state=n_state, n_act=n_act,
        action_low=a_bias - a_scale, action_high=a_bias + a_scale,
        device=device,
    ).eval()
    encoder.load_state_dict(enc_sd)
    actor.load_state_dict(actor_sd)

    LOG_PATH = r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/_multiseed_eval.log"
    SUMMARY_PATH = r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/_multiseed_summary.json"
    log = open(LOG_PATH, "w", buffering=1)
    log.write(f"# ckpt = {args.ckpt}\n")
    log.write(f"# task = {args.task}\n")
    log.write(f"# n_steps = {args.n_steps}, n_seeds = {args.n_seeds}, seed_start = {args.seed_start}\n")
    log.write(f"# n_state = {n_state}, n_act = {n_act}\n")
    log.write(f"# lift_z >= {args.lift_z_threshold}, bowl_radius_xy = {args.bowl_radius_xy}\n")
    log.write("\n")
    log.write(f"{'seed':>4s}  {'goal':>6s}  {'distr':>6s}  {'cube0(xyz)':>26s}  "
              f"{'bowl(xyz)':>26s}  {'lifted':>6s}  {'placed':>6s}  "
              f"{'min_d':>7s}  {'max_z':>7s}  {'max_F':>7s}\n")
    log.write("-" * 140 + "\n")

    robot = base.scene["robot"]
    cube = base.scene["cube"]
    bowl = base.scene["bowl"]
    gi = robot.body_names.index("gripper")
    g_sensor = base.scene.sensors["gripper_contact"]
    j_sensor = base.scene.sensors["jaw_contact"]

    results = []

    with torch.no_grad():
        for k in range(args.n_seeds):
            seed = args.seed_start + k
            obs, _ = env.reset(seed=seed)

            # Episode metadata.
            try:
                goal_idx = int(base._goal_color_idx[0].item())
                distr_idx = int(base._distractor_color_idx[0].item())
            except Exception:
                goal_idx, distr_idx = -1, -1
            cube0 = cube.data.root_pos_w[0].cpu().tolist()
            bowl_p = bowl.data.root_pos_w[0].cpu().tolist()

            # Track per-step metrics.
            lifted = False
            placed = False
            min_d = float("inf")
            max_z = cube0[2]
            max_F = 0.0

            for s in range(args.n_steps):
                state = obs["policy"][:, :n_state]
                rgb = obs["rgb"]["rgb"] if isinstance(obs["rgb"], dict) else obs["rgb"]
                rgb_feat = encoder(rgb.to(device))
                action = actor.get_eval_action(rgb_feat, state.to(device))

                tcp = robot.data.body_pos_w[0, gi].cpu().tolist()
                cube_p = cube.data.root_pos_w[0].cpu().tolist()
                d = math.sqrt(sum((tcp[i] - cube_p[i]) ** 2 for i in range(3)))
                F_g = torch.linalg.norm(g_sensor.data.net_forces_w[0, 0]).item() if g_sensor.data.net_forces_w is not None else 0.0
                F_j = torch.linalg.norm(j_sensor.data.net_forces_w[0, 0]).item() if j_sensor.data.net_forces_w is not None else 0.0
                F = max(F_g, F_j)

                if d < min_d: min_d = d
                if cube_p[2] > max_z: max_z = cube_p[2]
                if F > max_F: max_F = F
                if cube_p[2] >= args.lift_z_threshold: lifted = True
                xy_to_bowl = math.sqrt((cube_p[0] - bowl_p[0]) ** 2 + (cube_p[1] - bowl_p[1]) ** 2)
                if xy_to_bowl < args.bowl_radius_xy and cube_p[2] < 0.04 and cube_p[2] > 0.005:
                    # cube within bowl xy and resting near bowl interior
                    placed = True

                obs, _, term, trunc, _ = env.step(action)
                if term.any() or trunc.any():
                    break

            line = (f"{seed:>4d}  "
                    f"{PALETTE[goal_idx] if goal_idx>=0 else '?':>6s}  "
                    f"{PALETTE[distr_idx] if distr_idx>=0 else '?':>6s}  "
                    f"({cube0[0]:+.3f},{cube0[1]:+.3f},{cube0[2]:+.3f})  "
                    f"({bowl_p[0]:+.3f},{bowl_p[1]:+.3f},{bowl_p[2]:+.3f})  "
                    f"{'Y' if lifted else '-':>6s}  "
                    f"{'Y' if placed else '-':>6s}  "
                    f"{min_d:>7.4f}  {max_z:>7.4f}  {max_F:>7.2f}\n")
            log.write(line)
            print(line.rstrip(), flush=True)
            results.append(dict(seed=seed, goal=goal_idx, distr=distr_idx,
                                cube0=cube0, bowl=bowl_p,
                                lifted=lifted, placed=placed,
                                min_d=min_d, max_z=max_z, max_F=max_F))

    n_lifted = sum(1 for r in results if r["lifted"])
    n_placed = sum(1 for r in results if r["placed"])
    log.write("\n")
    log.write(f"Aggregate: {n_lifted}/{args.n_seeds} ever lifted, "
              f"{n_placed}/{args.n_seeds} ever placed (cube near bowl while still low).\n")
    log.close()

    with open(SUMMARY_PATH, "w") as f:
        json.dump(dict(
            ckpt=args.ckpt, n_steps=args.n_steps, n_seeds=args.n_seeds,
            n_lifted=n_lifted, n_placed=n_placed, results=results,
        ), f, indent=2)

    print(f"\n[summary] lifted={n_lifted}/{args.n_seeds}  placed={n_placed}/{args.n_seeds}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
