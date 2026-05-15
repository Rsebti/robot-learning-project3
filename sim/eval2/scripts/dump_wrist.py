"""Dump the wrist cam 128×128 RGB (what the CNN sees before the 16×16 downsample)."""
from __future__ import annotations
import argparse, os
import torch  # noqa

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-SquintNative-Place-Play-v0")
parser.add_argument("--out", type=str,
                    default=r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/wrist_raw_128.png")
parser.add_argument("--settle_steps", type=int, default=10)
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


def main():
    env_cfg = parse_env_cfg(args.task, num_envs=1)
    try: env_cfg.recorders = None
    except Exception: pass
    env = gym.make(args.task, cfg=env_cfg)
    base_env = env.unwrapped
    env.reset(seed=0)
    n_act = env.action_space.shape[-1]
    zero = torch.zeros(1, n_act, device=base_env.device)
    for _ in range(args.settle_steps):
        env.step(zero)

    cam = base_env.scene.sensors["wrist"]
    rgb = cam.data.output["rgb"]
    if rgb.shape[-1] == 4: rgb = rgb[..., :3]
    from PIL import Image
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Image.fromarray(rgb[0].cpu().numpy()).save(args.out)
    cam_p = cam.data.pos_w[0].cpu().tolist()
    robot = base_env.scene["robot"]
    gi = robot.body_names.index("gripper")
    g_p = robot.data.body_pos_w[0, gi].cpu().tolist()
    print(f"[dump] wrote {args.out}  ({rgb.shape[1]}×{rgb.shape[2]} raw, FOV 71°)")
    print(f"[pose] cam world pos     = ({cam_p[0]:+.3f}, {cam_p[1]:+.3f}, {cam_p[2]:+.3f})")
    print(f"[pose] gripper world pos = ({g_p[0]:+.3f}, {g_p[1]:+.3f}, {g_p[2]:+.3f})")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
