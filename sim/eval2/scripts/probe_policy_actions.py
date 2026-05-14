"""Trace what the policy outputs each step during a deploy episode.

For each control step we print:
  - cube xyz, bowl xyz, gripper xyz
  - tcp -> cube distance
  - action emitted by policy (6 floats)
  - is_grasping force on gripper / jaw

This tells us whether the policy is silent (zero deltas — visual OOD),
mis-targeting (large but wrong-direction deltas), or actually working
(deltas drive gripper toward cube).
"""
from __future__ import annotations
import argparse, os
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=r"C:/Users/user/Downloads/ckpt (7).pt")
parser.add_argument("--task", default="Isaac-SquintNative-Place-Play-v0")
parser.add_argument("--n_steps", type=int, default=50)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# AppLauncher honors --headless from the CLI; default is GUI on.
args.enable_cameras = True
args.num_envs = 1
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import sim.eval2  # noqa: F401, E402
from sim.eval2.policy import CNNEncoder, SquintActor  # noqa: E402


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
    obs, _ = env.reset(seed=0)

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

    robot = base.scene["robot"]
    cube = base.scene["cube"]
    bowl = base.scene["bowl"]
    gi = robot.body_names.index("gripper")
    jaw_i = robot.body_names.index("jaw")
    g_sensor = base.scene.sensors["gripper_contact"]
    j_sensor = base.scene.sensors["jaw_contact"]

    LOG_PATH = r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/_probe_steps.log"
    log_f = open(LOG_PATH, "w", buffering=1)  # line-buffered

    # Goal color the policy thinks it's reaching for.
    palette = ["red", "blue", "green", "yellow", "purple", "orange"]
    goal_idx = int(base._goal_color_idx[0].item()) if hasattr(base, "_goal_color_idx") else -1
    distr_idx = int(base._distractor_color_idx[0].item()) if hasattr(base, "_distractor_color_idx") else -1
    log_f.write(f"# goal_color = {palette[goal_idx] if goal_idx >= 0 else '?'} ({goal_idx})\n")
    log_f.write(f"# distractor_color = {palette[distr_idx] if distr_idx >= 0 else '?'} ({distr_idx})\n")
    try:
        dp = base.scene["cube_distractor"].data.root_pos_w[0].cpu().tolist()
        log_f.write(f"# distractor_xyz = ({dp[0]:+.3f}, {dp[1]:+.3f}, {dp[2]:+.3f})\n")
    except Exception: pass
    try:
        bp = base.scene["bowl"].data.root_pos_w[0].cpu().tolist()
        log_f.write(f"# bowl_xyz       = ({bp[0]:+.3f}, {bp[1]:+.3f}, {bp[2]:+.3f})\n")
    except Exception: pass
    log_f.write(f"# goal_one_hot   = {obs['policy'][0, 12:18].cpu().tolist()}\n")

    header = (f"{'step':>4s}  {'tcp_xyz':>22s}  {'cube_xyz':>22s}  "
              f"{'d_tcp_cube':>10s}  {'action[arm,gripper]':>60s}  "
              f"{'F_g':>6s} {'F_j':>6s}\n")
    log_f.write(header)
    log_f.write("-" * 140 + "\n")
    print(header.rstrip())

    with torch.no_grad():
        for s in range(args.n_steps):
            state = obs["policy"][:, :n_state]
            rgb = obs["rgb"]["rgb"] if isinstance(obs["rgb"], dict) else obs["rgb"]
            rgb_feat = encoder(rgb.to(device))
            action = actor.get_eval_action(rgb_feat, state.to(device))
            a = action[0].cpu().tolist()

            tcp = robot.data.body_pos_w[0, gi].cpu().tolist()
            cube_p = cube.data.root_pos_w[0].cpu().tolist()
            d = ((tcp[0]-cube_p[0])**2 + (tcp[1]-cube_p[1])**2 + (tcp[2]-cube_p[2])**2) ** 0.5

            F_g = torch.linalg.norm(g_sensor.data.net_forces_w[0, 0]).item() if g_sensor.data.net_forces_w is not None else 0.0
            F_j = torch.linalg.norm(j_sensor.data.net_forces_w[0, 0]).item() if j_sensor.data.net_forces_w is not None else 0.0

            line = (f"{s:>4d}  "
                    f"({tcp[0]:+.3f},{tcp[1]:+.3f},{tcp[2]:+.3f})  "
                    f"({cube_p[0]:+.3f},{cube_p[1]:+.3f},{cube_p[2]:+.3f})  "
                    f"{d:>10.4f}  "
                    f"[{a[0]:+.4f},{a[1]:+.4f},{a[2]:+.4f},{a[3]:+.4f},{a[4]:+.4f},{a[5]:+.4f}]  "
                    f"{F_g:>5.2f}  {F_j:>5.2f}\n")
            log_f.write(line)
            print(line.rstrip(), flush=True)

            obs, _, term, trunc, _ = env.step(action)
            if term.any() or trunc.any():
                tail = f"[end] step {s}: term={term.item()} trunc={trunc.item()}\n"
                log_f.write(tail)
                print(tail.rstrip())
                break

    log_f.close()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
