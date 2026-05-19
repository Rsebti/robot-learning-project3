"""
Shared LeRobot ACT deploy helpers (goal-conditioned environment_state).

Used by infer_eval1_act_nocube.py (8D), infer_eval2.py (8D), infer_eval1_act.py (10D).
Reference implementation: infer_eval1_act_nocube.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_DEPLOY = Path(__file__).resolve().parent
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))

from homes import DEFAULT_HOME_POSE, get_home_deg, list_home_poses  # noqa: E402
from robot_calibration import LOCAL_CALIBRATION_DIR, make_so101_follower_config  # noqa: E402

# --- Windows MSMF -> DSHOW patch ------------------------------------------
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
# --------------------------------------------------------------------------

COLORS = ["yellow", "orange", "red", "blue", "green", "violet"]
MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def build_env_state(
    color: str,
    bowl_x_m: float,
    bowl_y_m: float,
    *,
    cube_x_m: float | None = None,
    cube_y_m: float | None = None,
) -> np.ndarray:
    """8D: color one-hot + bowl xy. 10D: append cube xy."""
    if color not in COLORS:
        raise ValueError(f"color must be one of {COLORS}, got {color!r}")
    env_dim = 10 if cube_x_m is not None and cube_y_m is not None else 8
    vec = np.zeros(env_dim, dtype=np.float32)
    vec[COLORS.index(color)] = 1.0
    vec[6] = bowl_x_m
    vec[7] = bowl_y_m
    if env_dim == 10:
        vec[8] = cube_x_m
        vec[9] = cube_y_m
    return vec


def load_policy_and_processors(policy_path: str, device: str):
    try:
        from lerobot.common.policies.act.modeling_act import ACTPolicy
    except ImportError:
        from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy = ACTPolicy.from_pretrained(policy_path).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def make_robot(port: str, camera_index: int, fps: int, calibration_dir: Path | None = None):
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cal_dir = calibration_dir or LOCAL_CALIBRATION_DIR
    cfg = make_so101_follower_config(
        port,
        cameras={
            "wrist": OpenCVCameraConfig(
                index_or_path=camera_index, fps=fps, width=640, height=480,
            ),
        },
        calibration_dir=cal_dir,
        use_degrees=True,
    )
    print(f"[infer] calibration: {cal_dir / 'so101_follower.json'}", flush=True)
    robot = make_robot_from_config(cfg)
    robot.connect()
    return robot


def robot_obs_to_policy_obs(
    raw_obs: dict, env_state_np: np.ndarray, camera_key: str = "wrist",
) -> dict:
    state = np.array([raw_obs[f"{m}.pos"] for m in MOTOR_NAMES], dtype=np.float32)
    img = raw_obs[camera_key]
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    return {
        "observation.state": state,
        "observation.images.wrist": img,
        "observation.environment_state": env_state_np.astype(np.float32),
    }


def action_to_robot(action_np: np.ndarray) -> dict:
    return {f"{name}.pos": float(action_np[i]) for i, name in enumerate(MOTOR_NAMES)}


def read_joint_deg(robot) -> np.ndarray:
    raw = robot.get_observation()
    return np.array([raw[f"{m}.pos"] for m in MOTOR_NAMES], dtype=np.float32)


def ramp_to_home(
    robot,
    home_deg: np.ndarray,
    *,
    fps: int = 30,
    max_delta_deg: float = 8.0,
    tol_deg: float = 0.5,
    max_steps: int = 600,
) -> None:
    """Smooth ramp to ``home_deg`` using the same per-step delta cap as rollout."""
    target = np.asarray(home_deg, dtype=np.float32).reshape(len(MOTOR_NAMES))
    current = read_joint_deg(robot)
    for _ in range(max_steps):
        delta = target - current
        if float(np.max(np.abs(delta))) <= tol_deg:
            break
        current = current + np.clip(delta, -max_delta_deg, max_delta_deg)
        robot.send_action(action_to_robot(current))
        time.sleep(1.0 / fps)
    else:
        print("[home] warning: did not fully converge within max_steps", flush=True)

    final = read_joint_deg(robot)
    err = final - target
    print(f"[home] target (deg):  {target.round(2).tolist()}", flush=True)
    print(f"[home] reached (deg): {final.round(2).tolist()}", flush=True)
    print(
        f"[home] error (deg):    {err.round(2).tolist()}  "
        f"max={float(np.max(np.abs(err))):.1f}",
        flush=True,
    )


def maybe_home_robot(robot, args) -> None:
    """If ``args.home``, ramp to ``args.home_pose`` and optional countdown."""
    if not getattr(args, "home", True):
        print("[home] --no-home: skipping ramp; rollout starts from current pose.", flush=True)
        return

    pose_name = getattr(args, "home_pose", None) or DEFAULT_HOME_POSE
    home_deg = get_home_deg(pose_name)
    print(f"[home] Ramping to preset {pose_name!r} ...", flush=True)
    ramp_to_home(robot, home_deg, fps=args.fps)

    countdown = float(getattr(args, "home_countdown_s", 5.0))
    if countdown > 0:
        print(f"[home] HOMED. Rollout starts in {countdown:.0f}s (Ctrl+C to abort) ...", flush=True)
        steps = max(1, int(round(countdown)))
        for k in range(steps, 0, -1):
            print(f"  {k} ...", flush=True)
            time.sleep(countdown / steps)


def run_episode(
    robot,
    policy,
    preprocessor,
    postprocessor,
    env_state_np: np.ndarray,
    n_steps: int,
    fps: int,
    device: str,
    task_desc: str | None = None,
    max_delta_deg: float = 8.0,
    abort_if_normalized: bool = True,
):
    from lerobot.utils.control_utils import predict_action

    torch_device = torch.device(device)
    period = 1.0 / fps
    policy.reset()
    prev_action = None
    for step in range(n_steps):
        t0 = time.time()
        raw_obs = robot.get_observation()
        obs = robot_obs_to_policy_obs(raw_obs, env_state_np)

        action_t = predict_action(
            observation=obs,
            policy=policy,
            device=torch_device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=False,
            task=task_desc,
            robot_type="so101_follower",
        )
        action_np = action_t.detach().cpu().numpy()
        if action_np.ndim == 2:
            action_np = action_np.squeeze(0)

        if step == 0 and abort_if_normalized and float(np.max(np.abs(action_np))) < 5.0:
            raise RuntimeError(
                f"First action looks NORMALIZED (max |a|={np.max(np.abs(action_np)):.3f} < 5). "
                f"Refusing to drive robot. Values: {action_np.tolist()}."
            )

        if prev_action is not None:
            delta = action_np - prev_action
            big = np.abs(delta) > max_delta_deg
            if big.any():
                action_np = prev_action + np.clip(delta, -max_delta_deg, max_delta_deg)
                print(f"  step {step:4d}: clamped delta on joints {np.where(big)[0].tolist()}")

        robot.send_action(action_to_robot(action_np))
        prev_action = action_np

        elapsed = time.time() - t0
        if elapsed < period:
            time.sleep(period - elapsed)
        if step % fps == 0:
            print(f"  step {step:4d}/{n_steps}: action={action_np.round(2).tolist()}")


def dry_run_check(
    policy,
    preprocessor,
    postprocessor,
    env_state: np.ndarray,
    device: str,
    task_desc: str,
):
    from lerobot.utils.control_utils import predict_action

    print("\n[infer] DRY RUN -- no robot, synthetic obs.")
    obs = {
        "observation.state": np.zeros(6, dtype=np.float32),
        "observation.images.wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation.environment_state": env_state.astype(np.float32),
    }
    policy.reset()
    action_t = predict_action(
        observation=obs,
        policy=policy,
        device=torch.device(device),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=False,
        task=task_desc,
        robot_type="so101_follower",
    )
    action_np = action_t.detach().cpu().numpy()
    if action_np.ndim == 2:
        action_np = action_np.squeeze(0)
    print(f"[infer] action: shape={tuple(action_t.shape)}, values={action_np.round(3).tolist()}")
    max_abs = float(np.max(np.abs(action_np)))
    print(f"[infer] max |action| = {max_abs:.3f}")
    if max_abs < 5.0:
        print("[infer] !! WARNING: action looks NORMALIZED. DO NOT run on robot.")
    elif max_abs > 200.0:
        print("[infer] !! WARNING: action magnitude exceptionally large.")
    else:
        print("[infer] OK -- action magnitude in plausible joint-angle range.")


def add_home_cli_args(
    parser: argparse.ArgumentParser,
    *,
    default_home_pose: str = DEFAULT_HOME_POSE,
) -> None:
    parser.add_argument(
        "--home",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ramp to a rest pose from deploy/homes.py before each episode (default: on)",
    )
    parser.add_argument(
        "--home_pose",
        default=default_home_pose,
        choices=list_home_poses(),
        help="Named preset in deploy/homes.py",
    )
    parser.add_argument(
        "--home_countdown_s",
        type=float,
        default=5.0,
        help="Pause after homing before rollout (0 = skip countdown)",
    )


def add_act_cli_args(
    parser: argparse.ArgumentParser,
    *,
    default_home_pose: str = DEFAULT_HOME_POSE,
) -> None:
    add_home_cli_args(parser, default_home_pose=default_home_pose)
    parser.add_argument("--target_color", required=True, choices=COLORS)
    parser.add_argument("--bowl_x", type=float, required=True,
                        help="Bowl x in meters (right +, left -)")
    parser.add_argument("--bowl_y", type=float, required=True,
                        help="Bowl y in meters (forward +, back -)")
    parser.add_argument("--follower_port", default="COM3")
    parser.add_argument(
        "--calibration_dir",
        type=Path,
        default=LOCAL_CALIBRATION_DIR,
        help="Folder containing so101_follower.json (default: deploy/calibration)",
    )
    parser.add_argument("--camera_index", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode_time_s", type=float, default=15.0)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--reset_time_s", type=float, default=10.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load policy + run ONE inference with synthetic obs, no robot")


def run_act_rollouts(args, policy, preprocessor, postprocessor, env_state, task_desc: str):
    if args.dry_run:
        dry_run_check(policy, preprocessor, postprocessor, env_state, args.device, task_desc)
        return

    print(f"[infer] Connecting to follower on {args.follower_port} ...")
    robot = make_robot(
        args.follower_port, args.camera_index, args.fps,
        calibration_dir=args.calibration_dir,
    )
    home_deg = get_home_deg(getattr(args, "home_pose", None) or DEFAULT_HOME_POSE)
    try:
        n_steps = int(args.episode_time_s * args.fps)
        for ep in range(args.num_episodes):
            print(f"\n[infer] === Episode {ep+1}/{args.num_episodes} ===")
            maybe_home_robot(robot, args)
            run_episode(
                robot, policy, preprocessor, postprocessor,
                env_state, n_steps, args.fps, args.device, task_desc=task_desc,
            )
            if ep + 1 < args.num_episodes:
                print(f"[infer] Reset for {args.reset_time_s:.1f}s (manual scene reset)")
                time.sleep(args.reset_time_s)
    finally:
        if getattr(args, "home", True):
            print("\n[infer] Returning to home before disconnect ...", flush=True)
            ramp_to_home(robot, home_deg, fps=args.fps)
        print("\n[infer] Disconnecting robot ...")
        robot.disconnect()
        print("[infer] Done")
