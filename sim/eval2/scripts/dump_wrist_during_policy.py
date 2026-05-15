"""Run the Squint policy for N steps, dump the wrist 128×128 RGB at given steps.

Shows what the CNN sees (before its internal 16×16 downsample) DURING a
policy rollout, not at home pose.
"""
from __future__ import annotations
import argparse, os
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str,
                    default=r"C:/Users/user/Downloads/ckpt (5).pt")
parser.add_argument("--task", type=str, default="Isaac-SquintNative-Place-Play-v0")
parser.add_argument("--out_dir", type=str,
                    default=r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes")
parser.add_argument("--steps_to_dump", type=str, default="10,30,60,120,200",
                    help="comma-sep control steps at which to save the wrist frame")
parser.add_argument("--max_steps", type=int, default=250)
parser.add_argument("--warmup_steps", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
args.num_envs = 1
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import sim.eval2  # noqa: F401,E402


def _weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None: m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.orthogonal_(m.weight, nn.init.calculate_gain("relu"))
        if m.bias is not None: m.bias.data.fill_(0.0)


class CNNEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, device=device), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=1, device=device), nn.ReLU(),
            nn.Flatten(),
        )
        self.repr_dim = 1024
        self.apply(_weight_init)
        self.conv = self.conv.to(memory_format=torch.channels_last)

    def forward(self, obs):
        obs = obs.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        obs = obs.float() / 255.0 - 0.5
        return self.conv(obs)


class Projection(nn.Module):
    def __init__(self, n_obs, n_state, device):
        super().__init__()
        self.repr_dim = 50 + 256
        self.rgb_proj = nn.Sequential(nn.Linear(n_obs, 50, device=device),
                                      nn.LayerNorm(50, device=device), nn.Tanh())
        self.state_proj = nn.Sequential(nn.Linear(n_state, 256, device=device),
                                        nn.LayerNorm(256, device=device), nn.ReLU())

    def forward(self, rgb, state):
        return torch.cat([self.rgb_proj(rgb), self.state_proj(state)], dim=-1)


class Actor(nn.Module):
    def __init__(self, n_state, n_act, device):
        super().__init__()
        h = 256
        self.proj = Projection(1024, n_state, device)
        self.fc = nn.Sequential(
            nn.Linear(self.proj.repr_dim, h, device=device), nn.LayerNorm(h, device=device), nn.ReLU(),
            nn.Linear(h, h, device=device), nn.LayerNorm(h, device=device), nn.ReLU(),
            nn.Linear(h, h, device=device), nn.LayerNorm(h, device=device), nn.ReLU(),
        )
        self.fc_mean = nn.Linear(h, n_act, device=device)
        self.fc_logstd = nn.Linear(h, n_act, device=device)
        self.register_buffer("action_scale", torch.ones(n_act, device=device))
        self.register_buffer("action_bias", torch.zeros(n_act, device=device))
        self.apply(_weight_init)

    def act(self, rgb_features, state):
        x = self.proj(rgb_features, state)
        x = self.fc(x)
        return torch.tanh(self.fc_mean(x)) * self.action_scale + self.action_bias


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    enc_sd, actor_sd = ckpt["encoder"], ckpt["actor"]
    n_act = int(actor_sd["action_scale"].shape[-1])
    n_state = int(actor_sd["proj.state_proj.0.weight"].shape[-1])
    print(f"[deploy] n_state={n_state}, n_act={n_act}")

    env_cfg = parse_env_cfg(args.task, device=str(device), num_envs=1)
    try: env_cfg.recorders = None
    except Exception: pass
    env = gym.make(args.task, cfg=env_cfg)

    encoder = CNNEncoder(device).eval()
    actor = Actor(n_state=n_state, n_act=n_act, device=device).eval()
    encoder.load_state_dict(enc_sd)
    actor.load_state_dict(actor_sd)

    obs, _ = env.reset(seed=args.seed)
    zero = torch.zeros(1, n_act, device=device)
    for _ in range(args.warmup_steps):
        obs, _, _, _, _ = env.step(zero)

    cam = env.unwrapped.scene.sensors["wrist"]
    targets = set(int(s) for s in args.steps_to_dump.split(","))
    saved = []
    from PIL import Image

    with torch.no_grad():
        for step in range(args.max_steps):
            state = obs["policy"][:, :n_state]
            rgb16 = obs["rgb"]["rgb"] if isinstance(obs["rgb"], dict) else obs["rgb"]
            rgb_feat = encoder(rgb16.to(device))
            act = actor.act(rgb_feat, state.to(device))
            obs, _, term, trunc, _ = env.step(act)

            if step in targets:
                rgb = cam.data.output["rgb"]
                if rgb.shape[-1] == 4: rgb = rgb[..., :3]
                p = os.path.join(args.out_dir, f"wrist_step{step:03d}.png")
                Image.fromarray(rgb[0].cpu().numpy()).save(p)
                saved.append(p)
                print(f"[dump] step {step:3d}: wrote {p}")

            if term.any() or trunc.any():
                print(f"[dump] episode ended at step {step} (term={term.item()}, trunc={trunc.item()})")
                break

    print(f"\n[dump] saved {len(saved)} frames in {args.out_dir}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
