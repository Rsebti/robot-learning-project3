"""Render the wrist cam at home pose with the cube teleported to each of
the spawn-box corners + center, to verify the cube is always visible in
the policy's wrist obs at the start of an episode.

Saves a 3x3 grid of 16x16 obs images (matching what the policy actually
reads) + a 3x3 grid of 128x128 full renders, and prints per-corner stats:
  - cube xy in robot frame
  - presence of red pixels (channel R > some threshold)
  - pixel count above threshold

Spawn box (from squint_events.py):
  center = (0.3, 0.0), half_size = SPAWN_BOX_HALF (0.125 currently)

Sample grid:
   (xmin, ymax) (xmid, ymax) (xmax, ymax)
   (xmin, ymid) (xmid, ymid) (xmax, ymid)
   (xmin, ymin) (xmid, ymin) (xmax, ymin)
"""
from __future__ import annotations

import argparse
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-SquintNative-Place-Play-v0")
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
args.num_envs = 1
app = AppLauncher(args).app

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import sim.eval2  # noqa: F401, E402
from sim.eval2.envs.squint_native.squint_events import (  # noqa: E402
    CUBE_SPAWN_BOX_CENTER as SPAWN_BOX_CENTER,
    CUBE_SPAWN_BOX_HALF as SPAWN_BOX_HALF,
)


OUT_DIR = r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes"


def _to_uint8_rgb(arr):
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5: arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:                arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def _make_grid(images, rows, cols, gap=4):
    h, w = images[0].shape[:2]
    grid = np.full((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3), 30, dtype=np.uint8)
    for k, img in enumerate(images):
        r, c = divmod(k, cols)
        y = r * (h + gap); x = c * (w + gap)
        grid[y:y+h, x:x+w] = img
    return grid


def main():
    env_cfg = parse_env_cfg(args.task, num_envs=1)
    try: env_cfg.recorders = None
    except Exception: pass
    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    obs, _ = env.reset(seed=0)

    # Settle 5 steps with zero action.
    n_act = base.action_manager.total_action_dim
    zero = torch.zeros(1, n_act, device=base.device)
    for _ in range(5):
        obs, *_ = env.step(zero)

    cube = base.scene["cube"]
    wrist = base.scene.sensors["wrist"]
    cx, cy = SPAWN_BOX_CENTER
    h = SPAWN_BOX_HALF

    # Force the cube to red so the visibility check has a clean colour
    # signal, regardless of the random goal_color sampled at reset.
    from sim.eval2.envs.squint_native.squint_events import _recolor_prim_diffuse
    _recolor_prim_diffuse(base, "/Cube/", (1.0, 0.0, 0.0))

    # 3x3 sample grid in xy.
    xs = [cx - h, cx, cx + h]
    ys = [cy + h, cy, cy - h]
    poses = [(x, y) for y in ys for x in xs]

    log_lines = [f"# spawn box: center={SPAWN_BOX_CENTER}, half={SPAWN_BOX_HALF}",
                 f"# sample grid (row-major, top-down y):", ""]
    obs_imgs, full_imgs = [], []
    for (x, y) in poses:
        # Teleport the cube.
        pose = torch.zeros(1, 7, device=base.device)
        pose[0, 0] = x; pose[0, 1] = y; pose[0, 2] = 0.010
        pose[0, 3] = 1.0   # qw=1
        cube.write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=base.device))
        cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=base.device),
                                         env_ids=torch.tensor([0], device=base.device))
        # Step zero so the renderer updates with the new pose.
        for _ in range(2):
            obs, *_ = env.step(zero)
        # Grab images.
        full = _to_uint8_rgb(wrist.data.output["rgb"].cpu().numpy())
        rgb_obs = obs["rgb"]["rgb"] if isinstance(obs["rgb"], dict) else obs["rgb"]
        obs_img = _to_uint8_rgb(rgb_obs.cpu().numpy())
        obs_imgs.append(obs_img)
        full_imgs.append(full)

        # Stats: red pixel count (R > 180 & G < 80 & B < 80) on full.
        r, g, b = full[..., 0], full[..., 1], full[..., 2]
        red_mask = (r > 180) & (g < 100) & (b < 100)
        red_px = int(red_mask.sum())
        log_lines.append(f"cube xy=({x:+.3f}, {y:+.3f})   "
                          f"full R/G/B={r.mean():.1f}/{g.mean():.1f}/{b.mean():.1f}  "
                          f"red_px={red_px}/{full.shape[0]*full.shape[1]}")

    # Save the grids.
    obs_grid = _make_grid(obs_imgs, 3, 3, gap=2)
    full_grid = _make_grid(full_imgs, 3, 3, gap=6)
    Image.fromarray(obs_grid).save(os.path.join(OUT_DIR, "_spawn_visibility_obs_grid.png"))
    Image.fromarray(full_grid).save(os.path.join(OUT_DIR, "_spawn_visibility_full_grid.png"))
    with open(os.path.join(OUT_DIR, "_spawn_visibility.log"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"[saved] {OUT_DIR}/_spawn_visibility_obs_grid.png")
    print(f"[saved] {OUT_DIR}/_spawn_visibility_full_grid.png")
    print(f"[saved] {OUT_DIR}/_spawn_visibility.log")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
