"""GUI viewer: load a SquintActor checkpoint and run it through N episodes
back-to-back so you can watch the rollouts live.

Defaults to PLAY task (random cube/bowl per episode). Use --task to switch
to the canonical REPLAY env.

Launch:
    & "C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101\\.venv\\Scripts\\python.exe" \
        "C:\\Users\\user\\Desktop\\MA2\\robot-learning-project3\\sim\\eval2\\scripts\\view_deploy_multiseed.py" \
        --ckpt "C:\\Users\\user\\Downloads\\ckpt (8).pt" --n_episodes 10
"""
from __future__ import annotations

import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=r"C:/Users/user/Downloads/ckpt (8).pt")
parser.add_argument("--task", default="Isaac-SquintNative-Place-Play-v0")
parser.add_argument("--n_episodes", type=int, default=10)
parser.add_argument("--n_steps", type=int, default=75)
parser.add_argument("--seed_start", type=int, default=0)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
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
    a_scale = actor_sd["action_scale"]; a_bias = actor_sd["action_bias"]
    actor = SquintActor(
        n_rgb_repr=encoder.repr_dim, n_state=n_state, n_act=n_act,
        action_low=a_bias - a_scale, action_high=a_bias + a_scale, device=device,
    ).eval()
    encoder.load_state_dict(enc_sd); actor.load_state_dict(actor_sd)

    cube = base.scene["cube"]; bowl = base.scene["bowl"]

    print(f"\n{'ep':>3s}  {'seed':>4s}  {'goal':>6s}  "
          f"{'cube_xy':>16s}  {'bowl_xy':>16s}  {'max_z':>6s}  {'min_d':>6s}")
    print("-" * 80)

    with torch.no_grad():
        for k in range(args.n_episodes):
            seed = args.seed_start + k
            obs, _ = env.reset(seed=seed)
            try:
                gi = int(base._goal_color_idx[0].item())
            except Exception:
                gi = -1
            cube0 = cube.data.root_pos_w[0].cpu().tolist()
            bowl_p = bowl.data.root_pos_w[0].cpu().tolist()
            max_z = cube0[2]
            min_d = 1e9
            robot = base.scene["robot"]
            gpr = robot.body_names.index("gripper")

            for s in range(args.n_steps):
                state = obs["policy"][:, :n_state]
                rgb = obs["rgb"]["rgb"] if isinstance(obs["rgb"], dict) else obs["rgb"]
                rgb_feat = encoder(rgb.to(device))
                action = actor.get_eval_action(rgb_feat, state.to(device))

                tcp = robot.data.body_pos_w[0, gpr]
                cp = cube.data.root_pos_w[0]
                d = float(torch.linalg.norm(tcp - cp).item())
                z = float(cp[2].item())
                if d < min_d: min_d = d
                if z > max_z: max_z = z

                obs, _, term, trunc, _ = env.step(action)
                if term.any() or trunc.any():
                    break

            print(f"{k:>3d}  {seed:>4d}  "
                  f"{PALETTE[gi] if gi>=0 else '?':>6s}  "
                  f"({cube0[0]:+.3f},{cube0[1]:+.3f})  "
                  f"({bowl_p[0]:+.3f},{bowl_p[1]:+.3f})  "
                  f"{max_z:>6.3f}  {min_d:>6.3f}", flush=True)

    env.close()
    app.close()


if __name__ == "__main__":
    main()
