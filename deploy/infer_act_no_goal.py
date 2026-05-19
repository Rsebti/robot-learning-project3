"""
infer_act_no_goal.py - LeRobot ACT inference for checkpoints that have
NO observation.environment_state (single-task / no goal-conditioning).

Used for the eval2 dark-shadow / dark-noise / no-aug variants whose
config.json shows `observation.environment_state: None`.

Usage:
    python deploy/infer_act_no_goal.py `
        --policy_path osammotg1/projet3-act-eval2-dark-shadow `
        --follower_port COM3 --camera_index 1 `
        --episode_time_s 15 --no-home
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Windows MSMF -> DSHOW patch
_OriginalVideoCapture = cv2.VideoCapture


class _DShowVideoCapture(_OriginalVideoCapture):
    def __init__(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], int) and args[1] in (
            cv2.CAP_ANY, cv2.CAP_MSMF
        ):
            args = (args[0], cv2.CAP_DSHOW)
        elif len(args) == 1 and isinstance(args[0], int):
            args = (args[0], cv2.CAP_DSHOW)
        super().__init__(*args, **kwargs)


cv2.VideoCapture = _DShowVideoCapture

import sys                                                              # noqa: E402
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from homes import get_home_deg                                          # noqa: E402
from robot_calibration import make_so101_follower_config                # noqa: E402
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.robots.utils import make_robot_from_config                 # noqa: E402

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def _load_policy(policy_path: str, device: str):
    try:
        from lerobot.common.policies.act.modeling_act import ACTPolicy
    except ImportError:
        from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy = ACTPolicy.from_pretrained(policy_path).to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def _robot_obs_to_policy_obs(raw_obs: dict) -> dict:
    state = np.array([raw_obs[f"{m}.pos"] for m in MOTOR_NAMES], dtype=np.float32)
    img = raw_obs["wrist"]
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    return {
        "observation.state": state,
        "observation.images.wrist": img,
    }


def _action_to_robot(action_np: np.ndarray) -> dict:
    return {f"{name}.pos": float(action_np[i]) for i, name in enumerate(MOTOR_NAMES)}


def _ramp_to_home(robot, target_deg: np.ndarray, fps: int = 30,
                  max_deg_per_step: float = 0.7, max_total_s: float = 25.0):
    obs = robot.get_observation()
    q_cmd = np.array([float(obs[f"{n}.pos"]) for n in MOTOR_NAMES], dtype=float)
    period = 1.0 / fps
    t0_session = time.time()
    while True:
        err = target_deg - q_cmd
        if np.max(np.abs(err)) < 0.5:
            return
        if time.time() - t0_session > max_total_s:
            return
        q_cmd = q_cmd + np.clip(err, -max_deg_per_step, max_deg_per_step)
        action = {f"{n}.pos": float(q_cmd[i]) for i, n in enumerate(MOTOR_NAMES)}
        t0 = time.time()
        robot.send_action(action)
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy_path", required=True,
                   help="HF repo or local path to a no-env_state ACT ckpt.")
    p.add_argument("--follower_port", default="COM3")
    p.add_argument("--camera_index", type=int, default=1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--episode_time_s", type=float, default=15.0)
    p.add_argument("--num_episodes", type=int, default=1)
    p.add_argument("--reset_time_s", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--home", action=argparse.BooleanOptionalAction, default=True,
                   help="Ramp to --home_pose preset before each episode (default on).")
    p.add_argument("--home_pose", default="eval1_rest",
                   help="Named home preset in deploy/homes.py.")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--task_desc", default=None,
                   help="Optional task string passed to predict_action.")
    args = p.parse_args()

    print(f"[infer] policy: {args.policy_path}")
    print(f"[infer] device: {args.device}")
    print(f"[infer] Loading policy + processors ...")
    policy, preprocessor, postprocessor = _load_policy(args.policy_path, args.device)

    if args.dry_run:
        from lerobot.utils.control_utils import predict_action
        print("[infer] DRY RUN -- synthetic obs.")
        obs = {
            "observation.state":        np.zeros(6, dtype=np.float32),
            "observation.images.wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        policy.reset()
        a = predict_action(
            observation=obs, policy=policy, device=torch.device(args.device),
            preprocessor=preprocessor, postprocessor=postprocessor,
            use_amp=False, task=args.task_desc, robot_type="so101_follower",
        ).detach().cpu().numpy()
        if a.ndim == 2:
            a = a.squeeze(0)
        print(f"[infer] action shape={a.shape}  values={a.round(3).tolist()}")
        max_abs = float(np.max(np.abs(a)))
        if max_abs < 5.0:
            print("[infer] !! action looks NORMALIZED — do not run on robot.")
        elif max_abs > 200.0:
            print("[infer] !! action very large — check.")
        else:
            print("[infer] OK — plausible joint-degree range.")
        return

    cfg = make_so101_follower_config(
        args.follower_port,
        cameras={"wrist": OpenCVCameraConfig(
            index_or_path=args.camera_index, fps=args.fps,
            width=640, height=480,
        )},
        use_degrees=True,
    )
    robot = make_robot_from_config(cfg)
    print(f"[infer] Connecting to {args.follower_port} ...")
    robot.connect()

    home_deg = get_home_deg(args.home_pose) if args.home else None

    from lerobot.utils.control_utils import predict_action

    try:
        n_steps = int(args.episode_time_s * args.fps)
        period = 1.0 / args.fps
        for ep in range(args.num_episodes):
            print(f"\n[infer] === Episode {ep + 1}/{args.num_episodes} ===")
            if args.home:
                print(f"[home] ramp to {args.home_pose!r}")
                _ramp_to_home(robot, home_deg, fps=args.fps)
                time.sleep(0.5)
            policy.reset()
            prev_action = None
            for step in range(n_steps):
                t0 = time.time()
                raw_obs = robot.get_observation()
                obs = _robot_obs_to_policy_obs(raw_obs)
                action_t = predict_action(
                    observation=obs, policy=policy,
                    device=torch.device(args.device),
                    preprocessor=preprocessor, postprocessor=postprocessor,
                    use_amp=False, task=args.task_desc,
                    robot_type="so101_follower",
                )
                a = action_t.detach().cpu().numpy()
                if a.ndim == 2:
                    a = a.squeeze(0)
                if step == 0 and float(np.max(np.abs(a))) < 5.0:
                    raise RuntimeError(
                        f"First action looks NORMALIZED (max |a|={np.max(np.abs(a)):.3f}); "
                        f"refusing to drive robot."
                    )
                if prev_action is not None:
                    delta = a - prev_action
                    big = np.abs(delta) > 8.0
                    if big.any():
                        a = prev_action + np.clip(delta, -8.0, 8.0)
                robot.send_action(_action_to_robot(a))
                prev_action = a
                elapsed = time.time() - t0
                if elapsed < period:
                    time.sleep(period - elapsed)
                if step % args.fps == 0:
                    print(f"  step {step:4d}/{n_steps}  action={a.round(2).tolist()}")

            if ep + 1 < args.num_episodes:
                print(f"[infer] Reset {args.reset_time_s:.1f}s — re-stage cubes.")
                time.sleep(args.reset_time_s)
    finally:
        if args.home and home_deg is not None:
            print("\n[infer] returning to home before disconnect ...")
            _ramp_to_home(robot, home_deg, fps=args.fps)
        print("[infer] Disconnecting robot ...")
        robot.disconnect()
        print("[infer] Done")


if __name__ == "__main__":
    main()
