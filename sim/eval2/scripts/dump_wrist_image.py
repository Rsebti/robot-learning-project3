"""Save the wrist camera render at home pose to PNG, so we can eyeball what
the policy actually sees. Two outputs:
  notes/_wrist_cam_full.png   (raw camera resolution, e.g. 128x128 RGB)
  notes/_wrist_cam_obs.png    (16x16 downsampled, what the policy reads)
Optionally also writes obs/policy stats and goal-color one-hot.
"""
from __future__ import annotations

import argparse, os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-SquintNative-Place-Play-v0")
parser.add_argument("--settle_steps", type=int, default=5)
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
import sim.eval2  # noqa: F401,E402


OUT_FULL = r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/_wrist_cam_full.png"
OUT_OBS = r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/_wrist_cam_obs.png"


def _to_png(arr_uint8: np.ndarray, path: str) -> None:
    if arr_uint8.ndim == 4:
        arr_uint8 = arr_uint8[0]  # drop env dim
    if arr_uint8.shape[-1] == 4:
        arr_uint8 = arr_uint8[..., :3]
    Image.fromarray(arr_uint8.astype(np.uint8), mode="RGB").save(path)
    print(f"[saved] {path}  shape={arr_uint8.shape}")


def main():
    env_cfg = parse_env_cfg(args.task, num_envs=1)
    try: env_cfg.recorders = None
    except Exception: pass
    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    obs, _ = env.reset(seed=0)
    zero = torch.zeros(1, base.action_manager.total_action_dim, device=base.device)
    for _ in range(args.settle_steps):
        obs, *_ = env.step(zero)

    # Raw wrist render (from the camera sensor directly).
    wrist = base.scene.sensors["wrist"]
    full_rgb = wrist.data.output["rgb"]  # (N, H, W, 3 or 4) uint8
    full_np = full_rgb.cpu().numpy()
    _to_png(full_np, OUT_FULL)
    print(f"  full R/G/B mean = {full_np[0,...,0].mean():.1f} / {full_np[0,...,1].mean():.1f} / {full_np[0,...,2].mean():.1f}")

    # Policy obs RGB (16x16).
    rgb_obs = obs["rgb"]["rgb"] if isinstance(obs["rgb"], dict) else obs["rgb"]
    obs_np = rgb_obs.cpu().numpy()
    print(f"  obs RGB shape={obs_np.shape}, dtype={obs_np.dtype}")
    if obs_np.dtype != np.uint8:
        # Heuristic: if float, assume already in [0, 255] OR [0, 1]
        if obs_np.max() <= 1.5:
            obs_np = (obs_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            obs_np = obs_np.clip(0, 255).astype(np.uint8)
    _to_png(obs_np, OUT_OBS)
    print(f"  obs R/G/B mean = {obs_np[0,...,0].mean():.1f} / {obs_np[0,...,1].mean():.1f} / {obs_np[0,...,2].mean():.1f}")

    log_lines: list[str] = []
    def log(s: str) -> None:
        log_lines.append(s)
        print(s, flush=True)

    log(f"[full] R/G/B mean = {full_np[0,...,0].mean():.1f} / {full_np[0,...,1].mean():.1f} / {full_np[0,...,2].mean():.1f}")
    log(f"[obs]  R/G/B mean = {obs_np[0,...,0].mean():.1f} / {obs_np[0,...,1].mean():.1f} / {obs_np[0,...,2].mean():.1f}")
    log(f"[obs]  shape = {obs_np.shape}, dtype = {obs_np.dtype}")

    # Seg mask — let us know whether the cube is even in the cam FOV
    # (instance_segmentation_fast is enabled on the wrist cam).
    try:
        seg = wrist.data.output.get("instance_segmentation_fast", None)
        idx2labels = wrist.data.info[0].get("instance_segmentation_fast", {}) if wrist.data.info else {}
        if seg is not None:
            seg_np = seg.cpu().numpy()
            if seg_np.ndim == 4:
                seg_np = seg_np[0, ..., 0] if seg_np.shape[-1] == 1 else seg_np[0]
            elif seg_np.ndim == 3:
                seg_np = seg_np[0]
            unique = np.unique(seg_np)
            log(f"[seg] unique ids in wrist cam at home pose: {unique.tolist()}")
            log(f"[seg] idx2labels keys: {list(idx2labels.keys()) if isinstance(idx2labels, dict) else type(idx2labels)}")
            id_map = idx2labels.get("idToLabels", {}) if isinstance(idx2labels, dict) else {}
            for k, v in id_map.items():
                log(f"[seg]   {k} -> {v}")
            for uid in unique:
                px = int((seg_np == uid).sum())
                log(f"[seg]   id {uid}: {px} px")
    except Exception as e:
        log(f"[seg] dump failed: {e!r}")

    with open(r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/_wrist_dump.log", "w") as f:
        f.write("\n".join(log_lines))

    env.close()
    app.close()


if __name__ == "__main__":
    main()
