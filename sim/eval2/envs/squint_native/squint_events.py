"""Event terms — reset robot to start keyframe + spawn cube/bin coherently.

Replicates Squint's ``Place._initialize_episode`` (place.py:347):

1. Robot qpos = start keyframe + N(0, 0.02 rad) noise (per-joint independent)
2. Sample item xy in spawn_box around (0.3, 0) (size 20×20 cm)
3. Sample bin xy in same box, non-overlapping with item
4. Item z = item_half_size, bin z = bin_thickness/2
5. Random z-rotation for both item and bin

Bin is composed of 5 RigidObjects (floor + 4 walls). When the bin is
re-positioned, ALL 5 parts must move together — handled here by computing
the world bin center then writing each part's pose accordingly.

Also exposes a startup event ``make_robot_matte_black`` that walks every
Shader prim under the robot and overrides its OmniPBR material to a near
black matte finish (Squint trains its policy on a black robot).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Constants — extracted from Squint's place.py defaults
# ---------------------------------------------------------------------------

SPAWN_BOX_CENTER = (0.3, 0.0)        # robot frame xy
SPAWN_BOX_HALF = 0.1                 # 20×20 cm total
CUBE_HALF_SIZE = 0.01                # 2 cm side (user override; Squint mid was 0.0125)
BIN_HALF_X = 0.04                    # mid (0.035, 0.045)
BIN_HALF_Y = 0.05                    # mid (0.045, 0.055)
BIN_HALF_Z = 0.015                   # mid (0.012, 0.018)
BIN_THICKNESS = 0.005

# Bin-part offsets (LOCAL to bin center). Match the layout in
# squint_scene.py.  (x, y, z), in meters.
BIN_PART_LOCAL_OFFSETS = {
    "bin_floor": (0.0, 0.0, BIN_THICKNESS / 2),
    "bin_wall_pos_y": (0.0, +BIN_HALF_Y, BIN_HALF_Z),     # +y wall
    "bin_wall_neg_y": (0.0, -BIN_HALF_Y, BIN_HALF_Z),     # -y wall
    "bin_wall_pos_x": (+BIN_HALF_X, 0.0, BIN_HALF_Z),     # +x wall
    "bin_wall_neg_x": (-BIN_HALF_X, 0.0, BIN_HALF_Z),     # -x wall
}

# Min separation between cube and bin centers to avoid overlap.
ITEM_BIN_MIN_DIST = math.hypot(BIN_HALF_X, BIN_HALF_Y) + CUBE_HALF_SIZE + 0.01


# ---------------------------------------------------------------------------
# Event term functions
# ---------------------------------------------------------------------------


def make_robot_matte_black(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    rgb: tuple[float, float, float] = (0.02, 0.02, 0.02),
    robot_prim_substr: str = "/Robot/",
) -> None:
    """Walk all Shader prims whose path contains ``/Robot/`` and override
    material attributes for a matte-black look:

    - diffuse_color_constant            -> ``rgb``
    - metallic_constant                 -> 0.0
    - reflection_roughness_constant     -> 1.0
    - specular_level                    -> 0.0

    Squint's so101 USD has a pure black robot (its training policy never
    sees the LeIsaac chrome look) — visual sim2real wants the deploy view
    to look the same. Idempotent and safe to call as a ``startup`` event.
    """
    from pxr import Sdf, UsdShade
    stage = env.sim.stage
    if stage is None:
        return
    target_attrs = {
        "diffuse_color_constant": (Sdf.ValueTypeNames.Color3f, tuple(rgb)),
        "metallic_constant": (Sdf.ValueTypeNames.Float, 0.0),
        "reflection_roughness_constant": (Sdf.ValueTypeNames.Float, 1.0),
        "specular_level": (Sdf.ValueTypeNames.Float, 0.0),
    }
    affected_paths: list[str] = []
    n_overridden = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        path = prim.GetPath().pathString
        if robot_prim_substr not in path:
            continue
        shader = UsdShade.Shader(prim)
        if not shader:
            continue
        affected_paths.append(path)
        for attr_name, (sdf_type, value) in target_attrs.items():
            inp = shader.GetInput(attr_name)
            if not inp:
                inp = shader.CreateInput(attr_name, sdf_type)
            try:
                inp.Set(value)
                n_overridden += 1
            except Exception:
                pass
    # Log to a file so we can audit even when stdout is swallowed by Isaac.
    try:
        with open(r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/_matte_black.log", "w") as f:
            f.write(f"override count: {n_overridden}\n")
            f.write(f"prim_substr filter: {robot_prim_substr!r}\n")
            f.write(f"target rgb: {rgb}\n\n")
            f.write("affected shader prim paths:\n")
            for p in affected_paths:
                f.write(f"  {p}\n")
    except Exception:
        pass


def reset_robot_to_home(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    qpos_noise_std: float = 0.02,
) -> None:
    """Reset robot qpos to default (home) + Gaussian noise, qvel to zero.

    ALSO writes the same qpos as the joint position target so PhysX' PD
    controller starts with zero error — without this, the implicit PD
    drives toward whatever stale target was last set, which causes
    Isaac's qvel to remain non-zero after a few settle steps (Squint's
    SAPIEN env converges to qvel=0 in 5 steps; we want to match).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    n = env_ids.shape[0]
    home = robot.data.default_joint_pos[env_ids]
    noise = torch.randn_like(home) * qpos_noise_std
    qpos = home + noise
    qvel = torch.zeros_like(home)

    # Clamp to soft limits to avoid kicking the solver.
    lo = robot.data.soft_joint_pos_limits[env_ids, :, 0]
    hi = robot.data.soft_joint_pos_limits[env_ids, :, 1]
    qpos = torch.clamp(qpos, lo, hi)

    robot.write_joint_state_to_sim(position=qpos, velocity=qvel, env_ids=env_ids)


def _sample_xy_in_box(
    n: int,
    center: tuple[float, float],
    half_size: float,
    device: torch.device,
) -> torch.Tensor:
    """Uniform sample in a square box centered at ``center`` with half-extent ``half_size``."""
    cx, cy = center
    rand = torch.rand(n, 2, device=device) * 2.0 - 1.0  # [-1, +1]
    rand = rand * half_size
    rand[:, 0] += cx
    rand[:, 1] += cy
    return rand


def _random_z_quat(n: int, device: torch.device) -> torch.Tensor:
    """Random quaternion (wxyz) representing a rotation around world +Z."""
    yaw = (torch.rand(n, device=device) * 2.0 - 1.0) * math.pi  # [-π, +π]
    half = yaw / 2
    qw = torch.cos(half)
    qz = torch.sin(half)
    qx = torch.zeros_like(qw)
    qy = torch.zeros_like(qw)
    return torch.stack([qw, qx, qy, qz], dim=-1)


def reset_scene_squint(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    distractor_cfg: SceneEntityCfg | None = SceneEntityCfg("cube_distractor"),
    fixed_cube_xy: tuple[float, float] | None = None,
    fixed_bin_xy: tuple[float, float] | None = None,
    fixed_cube_yaw: float | None = None,
    fixed_bin_yaw: float | None = None,
) -> None:
    """Spawn the cube AND the 5 bin parts coherently.

    Cube and bin xy sampled in the spawn box around (0.3, 0). Resampled
    until they're at least ``ITEM_BIN_MIN_DIST`` apart (max 100 attempts).
    Both get a random z-rotation.

    For deploy / canonical audit replay we accept ``fixed_*`` overrides
    to pin the spawn locations and yaws — useful to align with Squint's
    seed=0 trajectory (cube at (+0.205, -0.066), bin at (+0.340, +0.060)).
    """
    device = env.device
    n = env_ids.shape[0]
    env_origins = env.scene.env_origins[env_ids]

    # ---- Sample non-overlapping cube + bin xy ----
    if fixed_cube_xy is not None and fixed_bin_xy is not None:
        cube_xy = torch.tensor(fixed_cube_xy, device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1).clone()
        bin_xy = torch.tensor(fixed_bin_xy, device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1).clone()
    else:
        cube_xy = _sample_xy_in_box(n, SPAWN_BOX_CENTER, SPAWN_BOX_HALF, device)
        bin_xy = _sample_xy_in_box(n, SPAWN_BOX_CENTER, SPAWN_BOX_HALF, device)
        # Resample bin where it overlaps with cube (rejection sampling).
        for _ in range(100):
            dist = torch.linalg.norm(cube_xy - bin_xy, dim=-1)
            too_close = dist < ITEM_BIN_MIN_DIST
            if not torch.any(too_close):
                break
            n_bad = int(too_close.sum().item())
            if n_bad == 0:
                break
            new_bin_xy = _sample_xy_in_box(n_bad, SPAWN_BOX_CENTER, SPAWN_BOX_HALF, device)
            bin_xy[too_close] = new_bin_xy

    # ---- Build cube root pose (world frame) ----
    cube_pos_w = torch.zeros(n, 3, device=device)
    cube_pos_w[:, :2] = cube_xy + env_origins[:, :2]
    cube_pos_w[:, 2] = env_origins[:, 2] + CUBE_HALF_SIZE
    if fixed_cube_yaw is not None:
        half = fixed_cube_yaw / 2
        cube_quat = torch.tensor(
            [math.cos(half), 0.0, 0.0, math.sin(half)],
            device=device, dtype=torch.float32,
        ).unsqueeze(0).expand(n, -1).clone()
    else:
        cube_quat = _random_z_quat(n, device)
    cube_pose = torch.cat([cube_pos_w, cube_quat], dim=-1)
    cube_vel = torch.zeros(n, 6, device=device)

    cube: RigidObject = env.scene[cube_cfg.name]
    cube.write_root_pose_to_sim(cube_pose, env_ids=env_ids)
    cube.write_root_velocity_to_sim(cube_vel, env_ids=env_ids)

    # ---- Distractor cube placement (face-to-face with target) ----
    # Squint's new ckpt training: distractor placed at
    # ``2 * half_size + uniform(0, 0.005)`` m from the target along a
    # uniformly random direction in [0, 2pi). Same color as a non-goal
    # palette idx; same physics + size as the target.
    if distractor_cfg is not None and distractor_cfg.name in env.scene.rigid_objects:
        theta = torch.rand(n, device=device) * (2 * math.pi)
        gap = 2 * CUBE_HALF_SIZE + torch.rand(n, device=device) * 0.005
        dx = gap * torch.cos(theta)
        dy = gap * torch.sin(theta)
        distractor_pos_w = cube_pos_w.clone()
        distractor_pos_w[:, 0] = distractor_pos_w[:, 0] + dx
        distractor_pos_w[:, 1] = distractor_pos_w[:, 1] + dy
        distractor_quat = _random_z_quat(n, device)
        distractor_pose = torch.cat([distractor_pos_w, distractor_quat], dim=-1)
        distractor_vel = torch.zeros(n, 6, device=device)
        distractor: RigidObject = env.scene[distractor_cfg.name]
        distractor.write_root_pose_to_sim(distractor_pose, env_ids=env_ids)
        distractor.write_root_velocity_to_sim(distractor_vel, env_ids=env_ids)

    # ---- Build bin part poses (world frame) ----
    bin_center_w = torch.zeros(n, 3, device=device)
    bin_center_w[:, :2] = bin_xy + env_origins[:, :2]
    bin_center_w[:, 2] = env_origins[:, 2]  # floor of bin sits on table z=0

    if fixed_bin_yaw is not None:
        half = fixed_bin_yaw / 2
        bin_quat = torch.tensor(
            [math.cos(half), 0.0, 0.0, math.sin(half)],
            device=device, dtype=torch.float32,
        ).unsqueeze(0).expand(n, -1).clone()
    else:
        bin_quat = _random_z_quat(n, device)  # rotate bin as a rigid block
    # Pre-compute rotation matrix around z for offsetting the parts.
    yaw = 2 * torch.atan2(bin_quat[:, 3], bin_quat[:, 0])  # extract z-rotation
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)

    for part_name, (lx, ly, lz) in BIN_PART_LOCAL_OFFSETS.items():
        # Rotate local xy offset into world frame.
        wx = cos_y * lx - sin_y * ly
        wy = sin_y * lx + cos_y * ly
        part_pos_w = bin_center_w.clone()
        part_pos_w[:, 0] += wx
        part_pos_w[:, 1] += wy
        part_pos_w[:, 2] += lz
        part_pose = torch.cat([part_pos_w, bin_quat], dim=-1)
        part_vel = torch.zeros(n, 6, device=device)
        part: RigidObject = env.scene[part_name]
        part.write_root_pose_to_sim(part_pose, env_ids=env_ids)
        part.write_root_velocity_to_sim(part_vel, env_ids=env_ids)
