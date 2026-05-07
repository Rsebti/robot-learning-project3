"""Smoke test for pytorch_kinematics on the SO-101 URDF.

Validates that:
    1. The URDF loads cleanly into a pk SerialChain.
    2. Forward kinematics on the home pose gives the expected gripper_frame_link
       position (~the values measured by sim/eval2/bc/measure_link_lengths.py:
       x ~ 0.222, y ~ 0.0, z ~ 0.070).
    3. Damped least squares IK round-trip: sample random joints -> FK gives
       a target pose -> IK from a perturbed initial guess recovers a pose
       that matches the FK target within a few mm.

Usage (from any venv with pytorch_kinematics + torch):
    python -m sim.eval2.bc.test_pytorch_kinematics

Pass criterion: max position error after roundtrip < 5 mm on >95% of cases.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# ------------- locate URDF -----------------------------------------------------
URDF_PATH = (
    Path("C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101")
    / "src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf"
)


def main() -> int:
    import pytorch_kinematics as pk

    if not URDF_PATH.exists():
        print(f"FAIL: URDF not found at {URDF_PATH}")
        return 1

    print(f"[pk] loading URDF: {URDF_PATH}")
    with open(URDF_PATH, "rb") as f:
        urdf_bytes = f.read()

    chain = pk.build_serial_chain_from_urdf(urdf_bytes, "gripper_frame_link")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    chain = chain.to(device=device, dtype=dtype)

    joint_names = chain.get_joint_parameter_names()
    print(f"[pk] joints in chain: {joint_names}")
    print(f"[pk] device: {device}")

    # ---- 1. FK on the home pose ---------------------------------------------
    # SO-101 default joint pose from SO_ARM101_CFG: (0, 0, 0, 1.57, 0). The
    # measure_link_lengths.py script (which steps the env) reads
    # gripper_frame_link at (0.222, 0.000, 0.070) in base frame.
    home = torch.tensor([[0.0, 0.0, 0.0, math.pi / 2, 0.0]], device=device, dtype=dtype)
    home_tf = chain.forward_kinematics(home)
    home_pos = home_tf.get_matrix()[:, :3, 3]
    print(f"\n[pk] FK at home pose:")
    print(f"     joints     = {home[0].tolist()}")
    print(f"     gripper xyz = ({home_pos[0, 0].item():+.4f}, "
          f"{home_pos[0, 1].item():+.4f}, {home_pos[0, 2].item():+.4f})")
    print( "     expected   = ( +0.222 ,  +0.000 ,  +0.070 )")
    err = torch.norm(home_pos[0] - torch.tensor([0.222, 0.0, 0.070], device=device, dtype=dtype)).item()
    print(f"     pos error  = {err * 1000:.2f} mm")

    if err > 0.01:
        print(f"WARNING: home FK position differs by > 10 mm from measured value.")

    # ---- 2. Roundtrip FK -> IK -> FK ---------------------------------------
    n = 200
    print(f"\n[pk] roundtrip test, n={n}")
    torch.manual_seed(42)
    # Sample random joints in a reasonable range. Constrain wrist_flex
    # near +pi/2 so the gripper is gripper-down-ish (avoids edge cases).
    q_true = torch.zeros(n, 5, device=device, dtype=dtype)
    q_true[:, 0] = torch.empty(n, device=device).uniform_(-0.8, 0.8)
    q_true[:, 1] = torch.empty(n, device=device).uniform_(-0.5, 0.5)
    q_true[:, 2] = torch.empty(n, device=device).uniform_(-0.5, 0.5)
    q_true[:, 3] = torch.empty(n, device=device).uniform_(1.0, 2.0)
    q_true[:, 4] = torch.empty(n, device=device).uniform_(-0.5, 0.5)

    tf_true = chain.forward_kinematics(q_true).get_matrix()  # (n, 4, 4)
    pos_true = tf_true[:, :3, 3]

    # Perturb initial guess by +/- 0.02 rad (~ 1 deg). This matches our
    # actual use case (incremental teleop / step-by-step scripted control)
    # where each new IK call starts from the previous frame's joints. Large
    # perturbations (eg 0.1 rad = 6 deg) sometimes land in a different DLS
    # basin and never recover, but in practice we don't make 6 deg jumps
    # between consecutive frames.
    q_init = q_true + torch.empty_like(q_true).uniform_(-0.02, 0.02)

    # Build pk InverseKinematics solver. The constructor only accepts
    # (num_retries, dof) for retry_configs (same init for all problems),
    # but solve() supports (num_problems, num_retries, dof) if we set
    # ``initial_config`` directly. We use that to seed each problem with
    # its own current joint pose — exactly what we need for teleop /
    # scripted control.
    # Pass a placeholder retry_configs to skip the uniform-sample branch
    # (which would error without joint_limits). We override initial_config
    # right after with our per-problem seeds.
    placeholder = torch.zeros(1, 5, device=device, dtype=dtype)
    # POSITION-ONLY IK: orientation_weight=0 lets DLS pick whatever wrist
    # orientation falls out, instead of struggling against the SO-101's
    # 5-DoF inability to track 6-DoF goals. This matches our use case
    # (top-down pick-and-place) where the gripper "should be" pointing
    # down but a few degrees of tilt is acceptable.
    ik = pk.PseudoInverseIK(
        chain,
        max_iterations=300,
        retry_configs=placeholder,
        early_stopping_any_converged=True,
        early_stopping_no_improvement="all",
        debug=False,
        lr=0.2,
        position_weight=1.0,
        orientation_weight=0.0,
        pos_tolerance=2e-3,    # 2 mm — fine for grasping a 2 cm cube
        rot_tolerance=1.0,     # don't even check rotation
    )
    # Per-problem seed: shape (n, 1, dof). num_retries is read from .shape[-2].
    ik.initial_config = q_init.unsqueeze(1)  # (n, 1, 5)
    ik.num_retries = 1
    sol = ik.solve(pk.Transform3d(matrix=tf_true))

    q_solved = sol.solutions[:, 0]  # (n, 5) — 1 retry, take retry 0
    converged = sol.converged_any if hasattr(sol, "converged_any") else None

    pos_recovered = chain.forward_kinematics(q_solved).get_matrix()[:, :3, 3]
    pos_err_mm = torch.norm(pos_recovered - pos_true, dim=-1) * 1000.0

    print(f"     mean pos error : {pos_err_mm.mean().item():.3f} mm")
    print(f"     p50  pos error : {pos_err_mm.median().item():.3f} mm")
    print(f"     p95  pos error : {torch.quantile(pos_err_mm, 0.95).item():.3f} mm")
    print(f"     max  pos error : {pos_err_mm.max().item():.3f} mm")
    if converged is not None:
        print(f"     pk-converged   : {converged.float().mean().item() * 100:.1f}%")

    pass_threshold_mm = 5.0
    frac_below = (pos_err_mm < pass_threshold_mm).float().mean().item()
    print(f"     fraction < {pass_threshold_mm} mm: {frac_below * 100:.1f}%")

    if frac_below > 0.95:
        print("\n[pk] PASS — pytorch_kinematics roundtrip < 5 mm on > 95 % of cases")
        return 0
    print("\n[pk] FAIL — too many cases above the 5 mm threshold. "
          "Try increasing max_iterations or lowering lr.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
