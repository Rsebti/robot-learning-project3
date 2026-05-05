"""Eval 2 — keyboard teleop in Isaac Sim, records demos to disk.

Lets you drive the simulated SO-101 with the keyboard and save successful
pick-and-place runs as expert demos. The recorded demos feed BC pretrain
(ACT) → DAPG finetune.

Two modes selectable with --mode:
    joint     (default) you drive each of the 5 arm joints individually
              + gripper. Bypasses IK entirely — robust, no saturation
              surprises, but unintuitive (one key per joint axis).
    cartesian you drive the gripper TIP in xyz (3 keys for x, 3 for y,
              3 for z). The analytical IK chooses the joint angles.
              Much more natural for pick-and-place, BUT the IK can
              saturate near workspace edges — when it does, the robot
              just stops moving in that direction and you pick another.

Key bindings — JOINT mode (default)
-----------------------------------
    Joint deltas (top row = positive, bottom row = negative):
      Q / A   shoulder_pan      +/-
      W / S   shoulder_lift     +/-
      E / D   elbow_flex        +/-
      R / F   wrist_flex        +/-
      T / G   wrist_roll        +/-

    Gripper:
      SPACE   toggle open / close

    Episode:
      N       reset env (skip / next episode)
      M       save current episode as success and reset
      ESC     quit (without saving current episode)

Key bindings — CARTESIAN mode (--mode cartesian)
------------------------------------------------
    Tip xyz deltas in robot base frame:
      W / S   move tip +/- in x   (forward / backward)
      A / D   move tip +/- in y   (left / right)
      Q / E   move tip +/- in z   (up / down)

    Gripper orientation tilt (occasional fine-tune):
      R / F   tilt gripper        more vertical / more tilted back

    Gripper:
      SPACE   toggle open / close

    Episode:
      N       reset env (skip / next episode)
      M       save current episode as success and reset
      ESC     quit (without saving current episode)

Tips
----
- Hold a key to keep moving that joint. Step size is 0.02 rad/frame
  (~1.15 deg) by default — adjustable with --step_size.
- The grasp method to follow is documented in notes/teleop_protocol.md.
  Stick to it: top-down approach, vertical descent, close, lift,
  transport, drop. BC will copy whatever you demonstrate.
- Aim for 10-15s per demo. Slower demos make BC learn sluggish
  trajectories.

Usage
-----
    uv run python -m sim.eval2.scripts.teleop \\
        --task Eval2-PickInClutter-Play-v2 \\
        --enable_cameras \\
        --output_dir data/teleop_demos_eval2/

Each successful episode is saved as one ``.npz`` file:
    data/teleop_demos_eval2/episode_000.npz
    data/teleop_demos_eval2/episode_001.npz
    ...

The npz contains, for the saved episode:
    joint_pos          (T, 6) float32     — robot joint positions
    joint_vel          (T, 6) float32     — robot joint velocities
    action             (T, 6) float32     — commanded action
    image              (T, H, W, 3) uint8 — wrist cam RGB
    target_color       int                — 0=red, 1=blue
    bowl_xyz_b         (3,) float32       — bowl position in base frame
    block_red_xyz_b    (T, 3) float32     — GT (diagnostic only)
    block_blue_xyz_b   (T, 3) float32     — GT (diagnostic only)
"""

from __future__ import annotations

import argparse
from pathlib import Path

# AppLauncher must be invoked before any torch / isaaclab import.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    type=str,
    default="Eval2-PickInClutter-Play-v2",
    help="Gym task id. Must include the wrist camera (v2).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="data/teleop_demos_eval2",
    help="Directory to write episode_XXX.npz files into.",
)
parser.add_argument(
    "--mode",
    type=str,
    default="joint",
    choices=["joint", "cartesian"],
    help="joint = 1 key per joint, cartesian = 1 key per xyz axis.",
)
parser.add_argument(
    "--step_size",
    type=float,
    default=0.005,
    help=(
        "Step size per held-key frame in JOINT mode (radians). 0.005 = "
        "~0.29 deg/frame (~9-17 deg/s depending on FPS). Lower = slower."
    ),
)
parser.add_argument(
    "--cartesian_step_xyz",
    type=float,
    default=0.001,
    help=(
        "Step size for tip xyz in CARTESIAN mode (meters). 0.001 = "
        "1 mm/frame (~3-6 cm/s depending on FPS). Lower = slower."
    ),
)
parser.add_argument(
    "--cartesian_step_phi",
    type=float,
    default=0.005,
    help="Step size for gripper phi tilt in cartesian mode (radians).",
)
parser.add_argument(
    "--cartesian_phi_init",
    type=float,
    default=-1.7,
    help=(
        "Initial gripper phi (rad) in cartesian mode. -pi/2 = strict vertical, "
        "-1.7 = mostly down with slight back-tilt (default, gives wrist_flex "
        "headroom). Adjust mid-session with R/F."
    ),
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=300.0,
    help=(
        "Override episode_length_s on the env (default 300s = 5 min). "
        "The default training value is ~5-10 s, which is too short for "
        "human teleop — the env would auto-reset mid-demo. Use N/M to "
        "advance episodes manually."
    ),
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Always 1 for teleop. Argument kept for compatibility.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force UI on (we need keyboard input + visual feedback).
args.headless = False
# Force cameras on (we need to record the wrist cam image).
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main():
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab.utils.math import subtract_frame_transforms
    from isaaclab_tasks.utils import parse_env_cfg

    import sim.eval2  # noqa: F401  (registers gym envs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Find the next free episode index so we can resume across runs.
    existing = sorted(out_dir.glob("episode_*.npz"))
    episode_idx = (
        int(existing[-1].stem.split("_")[1]) + 1 if existing else 0
    )
    print(f"[teleop] Output: {out_dir.resolve()}")
    print(f"[teleop] Resuming at episode index {episode_idx}")

    # ---- env ----
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    # Override episode length so the env doesn't auto-reset mid-demo.
    # The user controls all resets manually via N (discard) and M (save).
    env_cfg.episode_length_s = args.episode_length_s
    print(f"[teleop] episode_length_s = {args.episode_length_s}s")
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()

    robot = env.unwrapped.scene["robot"]
    wrist_cam = env.unwrapped.scene["wrist_cam"]
    block_red = env.unwrapped.scene["block_red"]
    block_blue = env.unwrapped.scene["block_blue"]
    bowl_floor = env.unwrapped.scene["bowl_floor"]

    device = env.unwrapped.device

    # ---- joint indices ----
    arm_ids, arm_names = robot.find_joints(["shoulder_.*", "elbow_flex", "wrist_.*"])
    gripper_id = robot.find_joints(["gripper"])[0][0]
    print(f"[teleop] arm joints (in URDF order): {arm_names}")

    default_arm_pos = robot.data.default_joint_pos[:, arm_ids].clone()  # (1, 5)

    # ---- keyboard handler ----
    teleop = KeyboardTeleop(
        joint_step=args.step_size,
        xyz_step=args.cartesian_step_xyz,
        phi_step=args.cartesian_step_phi,
    )
    print_help_banner(args.mode)

    # ---- mode-specific setup ----
    use_cartesian = args.mode == "cartesian"
    if use_cartesian:
        # Load IK calibration. Even though sim/eval2/bc/ is archived,
        # the IK math + json calibration are still valid (test passes
        # at 0.0001 mm roundtrip). We use it here as a forward kinematics
        # tool to map (xyz_b, phi) -> URDF joint angles.
        from sim.eval2.bc.analytical_ik import analytical_ik_so101
        import json

        link_cfg_path = (
            Path(__file__).parent.parent / "bc" / "so101_link_lengths.json"
        )
        with open(link_cfg_path) as f:
            link_cfg = json.load(f)
        ik_params = dict(
            L1=float(link_cfg["L1"]),
            L2=float(link_cfg["L2"]),
            L3=float(link_cfg["L3"]),
            base_offset_z=float(link_cfg["base_offset_z"]),
            base_offset_r=float(link_cfg["base_offset_r"]),
            offset_theta2=float(link_cfg["offset_theta2"]),
            offset_theta3=float(link_cfg["offset_theta3"]),
            offset_theta4=float(link_cfg["offset_theta4"]),
        )
        # Initialize desired tip from the current ee_frame position (in
        # base frame), so the first frame doesn't jump anywhere.
        ee_frame = env.unwrapped.scene["ee_frame"]
        tip_w = ee_frame.data.target_pos_w[0, 0, :]  # (3,) world frame
        desired_tip_b = _world_to_base(tip_w, robot).clone()  # (3,)
        gripper_phi = float(args.cartesian_phi_init)
        # We won't update desired_arm_pos directly in cartesian mode — we
        # solve IK each step instead. But we keep last-known-good IK as
        # the desired_arm_pos for fallback when IK fails.
        desired_arm_pos = default_arm_pos.clone()
        last_good_arm_pos = default_arm_pos.clone()
    else:
        desired_arm_pos = default_arm_pos.clone()

    gripper_open = True  # initial state

    # ---- per-episode buffers ----
    def new_buffer():
        return {
            "joint_pos": [],
            "joint_vel": [],
            "action": [],
            "image": [],
            "block_red_xyz_b": [],
            "block_blue_xyz_b": [],
        }

    buf = new_buffer()
    target_color_at_reset = int(env.unwrapped.target_color[0].item())
    bowl_xyz_b_at_reset = _bowl_xyz_in_base(bowl_floor, robot)
    print(
        f"\n[teleop] === episode {episode_idx} === "
        f"target={'RED' if target_color_at_reset == 0 else 'BLUE'} "
        f"bowl_xyz_b=({bowl_xyz_b_at_reset[0]:+.3f}, "
        f"{bowl_xyz_b_at_reset[1]:+.3f}, {bowl_xyz_b_at_reset[2]:+.3f})"
    )

    # ---- main loop ----
    while simulation_app.is_running() and not teleop.event_quit:
        if use_cartesian:
            # Cartesian mode: keyboard -> tip xyz delta -> IK -> joint targets.
            xyz_delta = teleop.get_xyz_delta()
            phi_delta = teleop.get_phi_delta()
            if xyz_delta.any():
                desired_tip_b = desired_tip_b + torch.from_numpy(xyz_delta).to(
                    device=device, dtype=desired_tip_b.dtype
                )
            if phi_delta != 0.0:
                gripper_phi += phi_delta
                gripper_phi = max(-2.20, min(-1.0, gripper_phi))  # safety bounds

            # Solve IK for the current desired tip + phi.
            ik = analytical_ik_so101(
                desired_tip_b.unsqueeze(0),  # (1, 3)
                phi=gripper_phi,
                elbow_up=False,
                **ik_params,
            )
            wrist_ok = ik.joint_pos[0, 3].abs() < 1.55  # URDF soft = 1.658
            if ik.reachable[0] and wrist_ok:
                desired_arm_pos = ik.joint_pos.clone()
                last_good_arm_pos = desired_arm_pos.clone()
            else:
                # Unreachable / saturated. Block movement: revert the
                # desired_tip update and keep the last good joint config
                # so the robot freezes instead of doing weird things.
                if xyz_delta.any():
                    desired_tip_b = desired_tip_b - torch.from_numpy(xyz_delta).to(
                        device=device, dtype=desired_tip_b.dtype
                    )
                desired_arm_pos = last_good_arm_pos.clone()
        else:
            # Joint mode: keyboard -> direct joint deltas.
            delta = teleop.get_joint_delta()
            if delta.any():
                desired_arm_pos[0] += torch.from_numpy(delta).to(
                    device=device, dtype=desired_arm_pos.dtype
                )
                # Clamp to URDF soft limits to avoid solver explosion.
                soft_limits = robot.data.soft_joint_pos_limits[:, arm_ids]
                desired_arm_pos = torch.clamp(
                    desired_arm_pos, soft_limits[..., 0], soft_limits[..., 1]
                )

        # Toggle gripper state on Space (both modes).
        if teleop.consume_gripper_toggle():
            gripper_open = not gripper_open
            print(f"[teleop] gripper -> {'OPEN' if gripper_open else 'CLOSED'}")
            # Resync desired_tip_b to the actual tip. Why: during a
            # descent toward a cube, the user's desired_tip_b drifts
            # ahead of the actual robot tip (PD tracking lag). When the
            # gripper closes on the cube, the arm can no longer descend
            # but desired_tip_b is still pointing below the cube — the
            # IK keeps pushing the arm down against the table, which
            # ricochets backward. Resyncing eliminates this mismatch.
            if use_cartesian:
                tip_w = ee_frame.data.target_pos_w[0, 0, :]
                desired_tip_b = _world_to_base(tip_w, robot).clone()

        # Build action: arm_action = 2 * (desired - default), gripper +/-1.
        arm_action = 2.0 * (desired_arm_pos - default_arm_pos)  # (1, 5)
        gripper_action = torch.tensor(
            [[+1.0 if gripper_open else -1.0]], device=device
        )
        action = torch.cat([arm_action, gripper_action], dim=-1)  # (1, 6)

        # Step the env.
        obs, _, _, _, _ = env.step(action)

        # Force-refresh the camera (update_period > step_dt by default).
        wrist_cam.update(dt=1.0)
        rgb = wrist_cam.data.output.get("rgb")
        if rgb is None:
            continue  # camera not warmed up yet

        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        # rgb is (1, H, W, 3) uint8.

        # Record this step.
        buf["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy())
        buf["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy())
        buf["action"].append(action[0].cpu().numpy())
        buf["image"].append(rgb[0].cpu().numpy())
        buf["block_red_xyz_b"].append(
            _world_to_base(block_red.data.root_pos_w[0], robot).cpu().numpy()
        )
        buf["block_blue_xyz_b"].append(
            _world_to_base(block_blue.data.root_pos_w[0], robot).cpu().numpy()
        )

        # Episode controls.
        do_save = teleop.consume_save_flag()
        do_reset = teleop.consume_reset_flag()
        if do_save or do_reset:
            if do_save:
                _save_episode(
                    out_dir,
                    episode_idx,
                    buf,
                    target_color_at_reset,
                    bowl_xyz_b_at_reset,
                )
                episode_idx += 1
            else:
                print(f"[teleop] episode {episode_idx} discarded (reset).")
            buf, gripper_open, desired_arm_pos = _reset_session(
                env, robot, default_arm_pos
            )
            if use_cartesian:
                # Re-init desired_tip from the new home pose tip.
                tip_w = ee_frame.data.target_pos_w[0, 0, :]
                desired_tip_b = _world_to_base(tip_w, robot).clone()
                gripper_phi = float(args.cartesian_phi_init)
                last_good_arm_pos = default_arm_pos.clone()
            target_color_at_reset = int(env.unwrapped.target_color[0].item())
            bowl_xyz_b_at_reset = _bowl_xyz_in_base(bowl_floor, robot)
            print(
                f"\n[teleop] === episode {episode_idx} === "
                f"target={'RED' if target_color_at_reset == 0 else 'BLUE'} "
                f"bowl_xyz_b=({bowl_xyz_b_at_reset[0]:+.3f}, "
                f"{bowl_xyz_b_at_reset[1]:+.3f}, {bowl_xyz_b_at_reset[2]:+.3f})"
            )

    print(f"\n[teleop] quitting. {episode_idx} episodes saved in {out_dir.resolve()}")
    env.close()


def _world_to_base(p_w, robot):
    import torch
    from isaaclab.utils.math import subtract_frame_transforms

    base_pos_w = robot.data.root_state_w[0:1, :3]
    base_quat_w = robot.data.root_state_w[0:1, 3:7]
    p_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, p_w.unsqueeze(0))
    return p_b[0]


def _bowl_xyz_in_base(bowl_floor, robot):
    import numpy as np

    return _world_to_base(bowl_floor.data.root_pos_w[0], robot).cpu().numpy()


def _reset_session(env, robot, default_arm_pos):
    """Reset the env and the teleop session state. Returns fresh (buf,
    gripper_open, desired_arm_pos)."""
    env.reset()
    return (
        {
            "joint_pos": [],
            "joint_vel": [],
            "action": [],
            "image": [],
            "block_red_xyz_b": [],
            "block_blue_xyz_b": [],
        },
        True,  # gripper_open
        default_arm_pos.clone(),
    )


def _save_episode(out_dir, idx, buf, target_color, bowl_xyz_b):
    import numpy as np

    if not buf["joint_pos"]:
        print(f"[teleop] episode {idx} has 0 frames — not saving.")
        return
    out = out_dir / f"episode_{idx:03d}.npz"
    np.savez_compressed(
        out,
        joint_pos=np.array(buf["joint_pos"], dtype=np.float32),
        joint_vel=np.array(buf["joint_vel"], dtype=np.float32),
        action=np.array(buf["action"], dtype=np.float32),
        image=np.array(buf["image"], dtype=np.uint8),
        target_color=np.int32(target_color),
        bowl_xyz_b=np.array(bowl_xyz_b, dtype=np.float32),
        block_red_xyz_b=np.array(buf["block_red_xyz_b"], dtype=np.float32),
        block_blue_xyz_b=np.array(buf["block_blue_xyz_b"], dtype=np.float32),
    )
    n_steps = len(buf["joint_pos"])
    print(f"[teleop] saved {out.name} ({n_steps} steps)")


def print_help_banner(mode: str):
    if mode == "cartesian":
        banner = """
============================================================
 CARTESIAN teleop. Drive the gripper TIP, IK does the rest.

  RIGHT              tip avance     (+x)
  LEFT               tip recule     (-x)
  DOWN               tip va à droite
  UP                 tip va à gauche
  PAGE_DOWN          tip monte      (+z)
  PAGE_UP            tip descend    (-z)
  P / ;              gripper tilt (plus vertical / plus tilté)
  SPACE              toggle gripper open/close
  N                  reset (discard episode)
  M                  save episode (mark success) and reset
  ESC                quit

 If the robot stops moving in a direction, the IK saturated.
 Back off (opposite key) and pick a different path. Or adjust
 the tilt with P / ; for more reach in z.
============================================================
""".strip()
    else:
        banner = """
============================================================
 JOINT-space teleop. Drive each joint individually.

  LEFT / RIGHT       shoulder_pan   +/-   (rotate base L/R)
  UP / DOWN          shoulder_lift  +/-   (arm up/down)
  PAGE_UP / PAGE_DN  elbow_flex     +/-
  I / K              wrist_flex     +/-
  O / L              wrist_roll     +/-
  SPACE              toggle gripper
  N                  reset (discard episode)
  M                  save episode (mark success) and reset
  ESC                quit
============================================================
""".strip()
    print(banner)


class KeyboardTeleop:
    """Carb-input-based keyboard handler for both joint and cartesian modes.

    Tracks held keys (continuous motion while held) and edge-triggered
    events (Space / N / M / ESC = one-shot signals that consumers drain
    via consume_*).

    The same held-key set is interpreted differently by ``get_joint_delta``
    (5,) vs ``get_xyz_delta`` (3,) / ``get_phi_delta`` (scalar) — only one
    interpretation should be used at a time, picked by --mode in main().
    """

    # Key bindings deliberately avoid Q/W/E/R/T/F because Isaac Sim's
    # manipulator toolbar (Select / Move / Rotate / Scale icons on the
    # left of the viewport) hijacks those keys. The 4 arrow keys are
    # the primary navigation, plus PAGE_UP/PAGE_DOWN for the 3rd axis,
    # plus letter keys for additional DoF.

    # Joint-mode mapping:
    #   UP / DOWN     shoulder_lift  +/-   (arm up/down visually)
    #   LEFT / RIGHT  shoulder_pan   +/-   (arm rotate L/R)
    #   PAGE_UP / PAGE_DOWN elbow_flex +/-
    #   I / K         wrist_flex +/-
    #   O / L         wrist_roll +/-
    JOINT_DELTAS = {
        "LEFT": (0, +1.0),
        "RIGHT": (0, -1.0),
        "UP": (1, +1.0),
        "DOWN": (1, -1.0),
        "PAGE_UP": (2, +1.0),
        "PAGE_DOWN": (2, -1.0),
        "I": (3, +1.0),
        "K": (3, -1.0),
        "O": (4, +1.0),
        "L": (4, -1.0),
    }

    # Cartesian-mode mapping (custom convention from user perspective):
    #   RIGHT         tip avance         (+x)
    #   LEFT          tip recule         (-x)
    #   DOWN          tip va à droite    (-y, robot frame)
    #   UP            tip va à gauche    (+y, robot frame)
    #   PAGE_DOWN     tip monte          (+z)
    #   PAGE_UP       tip descend        (-z)
    XYZ_DELTAS = {
        "RIGHT": (0, +1.0),      # +x  (avance)
        "LEFT": (0, -1.0),       # -x  (recule)
        "DOWN": (1, -1.0),       # -y  (à droite)
        "UP": (1, +1.0),         # +y  (à gauche)
        "PAGE_DOWN": (2, +1.0),  # +z  (haut)
        "PAGE_UP": (2, -1.0),    # -z  (bas)
    }

    # Cartesian-mode phi tilt: P = more vertical, SEMICOLON = more tilted.
    PHI_DELTAS = {
        "P": +1.0,
        "SEMICOLON": -1.0,
    }

    def __init__(
        self,
        joint_step: float = 0.02,
        xyz_step: float = 0.005,
        phi_step: float = 0.02,
    ):
        import carb
        import omni

        self.joint_step = joint_step
        self.xyz_step = xyz_step
        self.phi_step = phi_step
        self.held_keys: set[str] = set()

        self._gripper_toggle_pending = False
        self.event_save_pending = False
        self.event_reset_pending = False
        self.event_quit = False

        self._app_window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._app_window.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

    # --- public — joint mode ---
    def get_joint_delta(self):
        """Returns (5,) numpy array of joint deltas (rad) for this frame."""
        import numpy as np

        delta = np.zeros(5, dtype=np.float32)
        for key in self.held_keys:
            if key in self.JOINT_DELTAS:
                joint_idx, sign = self.JOINT_DELTAS[key]
                delta[joint_idx] += sign * self.joint_step
        return delta

    # --- public — cartesian mode ---
    def get_xyz_delta(self):
        """Returns (3,) numpy array of tip xyz deltas (meters) for this frame."""
        import numpy as np

        delta = np.zeros(3, dtype=np.float32)
        for key in self.held_keys:
            if key in self.XYZ_DELTAS:
                idx, sign = self.XYZ_DELTAS[key]
                delta[idx] += sign * self.xyz_step
        return delta

    def get_phi_delta(self) -> float:
        """Returns scalar gripper phi delta (rad) for this frame."""
        delta = 0.0
        for key in self.held_keys:
            if key in self.PHI_DELTAS:
                delta += self.PHI_DELTAS[key] * self.phi_step
        return delta

    # --- public — events ---
    def consume_gripper_toggle(self) -> bool:
        flag = self._gripper_toggle_pending
        self._gripper_toggle_pending = False
        return flag

    def consume_save_flag(self) -> bool:
        flag = self.event_save_pending
        self.event_save_pending = False
        return flag

    def consume_reset_flag(self) -> bool:
        flag = self.event_reset_pending
        self.event_reset_pending = False
        return flag

    # All keys we care about (incl. event keys). Used to decide whether
    # to consume the event (block propagation to Isaac Sim's toolbar) or
    # let it pass through normally.
    _CONSUMED_KEYS = frozenset(
        list(JOINT_DELTAS.keys())
        + list(XYZ_DELTAS.keys())
        + list(PHI_DELTAS.keys())
        + ["SPACE", "N", "M", "ESCAPE"]
    )

    # --- internal ---
    def _on_keyboard_event(self, event, *args, **kwargs):
        import carb

        key = event.input.name
        et = event.type
        if et == carb.input.KeyboardEventType.KEY_PRESS:
            if key == "SPACE":
                self._gripper_toggle_pending = True
            elif key == "N":
                self.event_reset_pending = True
            elif key == "M":
                self.event_save_pending = True
            elif key == "ESCAPE":
                self.event_quit = True
            else:
                self.held_keys.add(key)
        elif et == carb.input.KeyboardEventType.KEY_RELEASE:
            self.held_keys.discard(key)
        # Carb convention: return False = event consumed (stop propagation
        # to other subscribers like Isaac Sim's manipulator toolbar).
        # Return False ONLY for keys we use, so the rest of the UI still
        # works (Ctrl+S to save scene, etc.).
        if key in self._CONSUMED_KEYS:
            return False
        return True


if __name__ == "__main__":
    import sys
    import traceback

    try:
        main()
    except Exception:
        print("\n=== teleop.py crashed — Python traceback follows ===")
        traceback.print_exc()
        print("=== end traceback ===\n")
        sys.exit(1)
    finally:
        simulation_app.close()
