"""Event terms — reset robot to start keyframe + spawn cube/bowl.

Replicates Squint's ``Place._initialize_episode`` (place.py:347):

1. Robot qpos = start keyframe + N(0, 0.02 rad) noise (per-joint independent)
2. Sample item xy in spawn_box around (0.3, 0) (size 20×20 cm)
3. Sample bowl xy in same box, non-overlapping with item
4. Item z = item_half_size, bowl floor z = 0
5. Random z-rotation for both item and bowl

The bowl is the SAM-3D reconstruction of the real bowl, loaded as a single
dynamic RigidObject from ``squint/meshes/bowl.usd``. Its origin is the
bowl floor's bottom-center so placing root pose at z=0 puts the floor
flush with the table surface.

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

# Legacy single-box constants — preserved for any caller still using them.
SPAWN_BOX_CENTER = (0.3, 0.0)        # robot frame xy
SPAWN_BOX_HALF = 0.125               # Squint spawn_box_half_size = 0.25/2 — place.py:189

# Cube spawn box — shrunk so the cube is ALWAYS in the wrist-cam FOV at
# home pose (audit_spawn_visibility found the wrist cone covers roughly
# x∈[0.15, 0.31], y∈[-0.09, +0.07] at home; we keep the cube box inside
# that with margin). The policy therefore never has to "search" for the
# cube — it sees it from step 0.
CUBE_SPAWN_BOX_CENTER = (0.21, -0.01)
CUBE_SPAWN_BOX_HALF = 0.04           # 8×8 cm square

# Bowl spawn box — left at the original Squint range; the bowl does NOT
# need to be visible at home pose because its xyz is added directly to
# the state vector (see ``bowl_xyz_world`` observation).
BOWL_SPAWN_BOX_CENTER = (0.30, 0.0)
BOWL_SPAWN_BOX_HALF = 0.125

CUBE_HALF_SIZE = 0.01                # 2 cm side (user override; Squint mid was 0.0125)

# Bowl AABB (post-reconstruction, from ``mesh_bowl_from_ply.py`` output):
# x ∈ [-0.0745, 0.0759], y ∈ [-0.0726, 0.0774], z ∈ [0, 0.053].
# Used here only for the cube/bowl non-overlap rejection sampler.
BOWL_HALF_X = 0.075
BOWL_HALF_Y = 0.075

# Min separation between cube and bowl centers to avoid overlap.
ITEM_BOWL_MIN_DIST = math.hypot(BOWL_HALF_X, BOWL_HALF_Y) + CUBE_HALF_SIZE + 0.01


# ---------------------------------------------------------------------------
# Event term functions
# ---------------------------------------------------------------------------


def fix_link_coms(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
) -> None:
    """Override per-link COM offsets to match Squint's URDF spec.

    Isaac's URDF converter applies the ``<inertial><origin xyz="..."/>``
    translation but is unreliable on the matching ``rpy="..."`` rotation,
    so the COM ends up in the wrong place in the link frame even though
    the diagonal inertia magnitudes are preserved correctly. We patch the
    seven links that have a nonzero spec COM with their canonical values
    from the teammate handoff (Squint URDF inertial tags, link frame).
    """
    SQUINT_COMS = {
        "base":      (-0.00636,  0.0,      -0.00240),
        "shoulder":  (-0.03040,  0.000422, -0.04170),
        "upper_arm": (-0.11257, -0.01550,   0.01870),
        "lower_arm": (-0.06485, -0.03200,   0.01820),
        "wrist":     ( 0.0,     -0.04240,   0.03060),
        "gripper":   ( 0.00770,  0.00010,  -0.02340),
        "jaw":       ( 0.0,      0.0,       0.01890),
    }
    robot = env.scene["robot"]
    body_names = list(robot.body_names)
    coms = robot.root_physx_view.get_coms().clone()  # (n_envs, n_bodies, 7)
    applied: list[str] = []
    for name, (x, y, z) in SQUINT_COMS.items():
        if name not in body_names:
            continue
        bi = body_names.index(name)
        coms[:, bi, 0] = x
        coms[:, bi, 1] = y
        coms[:, bi, 2] = z
        # Identity quat — Squint's inertia tensor is already aligned with
        # the link frame in the URDF spec we received.
        coms[:, bi, 3] = 1.0
        coms[:, bi, 4:7] = 0.0
        applied.append(f"{name}: ({x:+.5f}, {y:+.5f}, {z:+.5f})")
    robot.root_physx_view.set_coms(coms.cpu(), torch.arange(env.num_envs))

    try:
        with open(r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/_link_coms.log", "w") as f:
            f.write(f"Applied Squint URDF COMs on {len(applied)} bodies:\n")
            for line in applied:
                f.write(f"  {line}\n")
    except Exception:
        pass


def zero_finger_tip_masses(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    finger_link_names: tuple[str, ...] = ("finger1_tip", "finger2_tip"),
    epsilon_mass: float = 1e-6,
    epsilon_inertia: float = 1e-9,
) -> None:
    """Squint's ``finger1_tip`` and ``finger2_tip`` are EMPTY URDF links
    (TCP reference frames only — no inertia, no visual, no collision).
    Isaac's URDF converter falls back to ``mass=1.0`` + diagonal inertia
    1e-4 when no ``<inertial>`` tag is present, which silently adds ~2 kg
    of phantom mass fixed-jointed to the gripper/jaw bodies → the
    effective jaw inertia balloons to ~85× Squint's value and grasp
    dynamics are completely OOD for the policy.

    PhysX requires articulation rigid bodies to have STRICTLY positive
    mass-space inertia (``setMassSpaceInertiaTensor() components must be
    > 0``), so we set tiny positive values (mass 1e-6 kg, inertia 1e-9)
    instead of zero. The fixed-joint composition into gripper/jaw still
    contributes a negligible fraction of the parent's real mass.
    """
    robot = env.scene["robot"]
    body_names = list(robot.body_names)

    body_ids = [body_names.index(n) for n in finger_link_names if n in body_names]
    if not body_ids:
        return

    masses = robot.root_physx_view.get_masses().clone()
    inertias = robot.root_physx_view.get_inertias().clone()
    for bi in body_ids:
        masses[:, bi] = epsilon_mass
        # Diagonal inertia = epsilon, off-diagonals = 0. Layout is (ixx,
        # ixy, ixz, iyx, iyy, iyz, izx, izy, izz) flattened.
        inertias[:, bi] = 0.0
        inertias[:, bi, 0] = epsilon_inertia  # ixx
        inertias[:, bi, 4] = epsilon_inertia  # iyy
        inertias[:, bi, 8] = epsilon_inertia  # izz
    robot.root_physx_view.set_masses(masses.cpu(), torch.arange(env.num_envs))
    robot.root_physx_view.set_inertias(inertias.cpu(), torch.arange(env.num_envs))

    try:
        with open(r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/_finger_tip_zero.log", "w") as f:
            f.write(f"Set mass={epsilon_mass}, inertia=0 on bodies {body_ids} "
                    f"(names: {[body_names[bi] for bi in body_ids]})\n")
            new_total = float(masses[0].sum().item())
            f.write(f"Robot total mass after fix (env 0): {new_total:.6f} kg "
                    f"(target Squint: 0.632 kg)\n")
    except Exception:
        pass


def align_squint_materials(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
) -> None:
    """Apply Squint's contact-material spec across the stage at startup.

    Replicates the friction map from ``envs/robot/so101.py:_materials`` and
    the ManiSkill default ``DefaultMaterialsConfig`` (static=dynamic=0.3,
    restitution=0). Per-body overrides:

    | Body                                   | static | dynamic | torsional_patch_radius |
    | gripper / jaw / finger1_tip / finger2_tip |  2.0  |   2.0   |     0.1 m              |
    | bowl                                   |  0.5   |   0.5   |     —                   |
    | <everything else>                      |  0.3   |   0.3   |     —                   |

    Combine mode is forced to ``min`` everywhere — matches SAPIEN's
    contact-pair friction rule (PxMaterial::CombineMode = MIN).

    Cube / distractor / table set their own RigidBodyMaterialCfg in
    ``squint_scene.py`` (mid values from the per-env DR distribution) so
    we don't touch them here.
    """
    try:
        from pxr import Sdf, UsdPhysics, UsdShade, PhysxSchema
    except Exception:
        return
    stage = env.sim.stage
    if stage is None:
        return

    affected: list[str] = []

    def _set_material(mat_prim, static_f, dynamic_f, restitution=0.0, combine="min"):
        mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
        mat.CreateStaticFrictionAttr().Set(static_f)
        mat.CreateDynamicFrictionAttr().Set(dynamic_f)
        mat.CreateRestitutionAttr().Set(restitution)
        physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
        physx_mat.CreateFrictionCombineModeAttr().Set(combine)
        physx_mat.CreateRestitutionCombineModeAttr().Set(combine)

    def _set_torsional_patch(prim, radius: float):
        """``torsional_patch_radius`` lives on PhysxCollisionAPI (per-collider),
        NOT on the material — see ``isaaclab/sim/schemas/schemas.py:339``."""
        coll = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        coll.CreateTorsionalPatchRadiusAttr().Set(radius)
        coll.CreateMinTorsionalPatchRadiusAttr().Set(radius)

    def _bind(prim, mat_prim):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(mat_prim), bindingStrength=UsdShade.Tokens.weakerThanDescendants,
            materialPurpose="physics",
        )

    # 1) Global default (used by every robot link without an explicit material).
    default_mat = stage.GetPrimAtPath("/physicsScene/defaultMaterial")
    if default_mat.IsValid():
        _set_material(default_mat, 0.3, 0.3, 0.0, "min")
        affected.append("/physicsScene/defaultMaterial -> (0.3, 0.3, min)")

    # 2) Force combine_mode="min" on every other PhysicsMaterial already in the
    #    stage (cube / distractor / table — created by RigidBodyMaterialCfg).
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        if not prim.HasAPI(UsdPhysics.MaterialAPI):
            continue
        path = prim.GetPath().pathString
        if path == "/physicsScene/defaultMaterial":
            continue
        physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(prim)
        physx_mat.CreateFrictionCombineModeAttr().Set("min")
        physx_mat.CreateRestitutionCombineModeAttr().Set("min")

    # 3) Gripper-body override (2.0/2.0 + torsional_patch_radius=0.1).
    # We match link names against the converted USD (Squint's renamed URDF).
    gripper_link_names = {"gripper", "jaw", "finger1_tip", "finger2_tip"}
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        # Skip non-body prims early.
        if "/Robot/" not in path:
            continue
        leaf = path.rsplit("/", 1)[-1]
        if leaf not in gripper_link_names:
            continue
        # The body prim itself is an Xform; we want to bind the material on
        # its collision children. We create a per-body Material prim under
        # the body's Looks namespace and bind it to every collision shape.
        mat_path = f"{path}/PhysicsMaterial_gripper"
        mat_prim = stage.GetPrimAtPath(mat_path)
        if not mat_prim.IsValid():
            mat_prim = UsdShade.Material.Define(stage, mat_path).GetPrim()
        _set_material(mat_prim, 2.0, 2.0, 0.0, "min")
        for desc in prim.GetAllChildren():
            if desc.HasAPI(UsdPhysics.CollisionAPI):
                _bind(desc, mat_prim)
                _set_torsional_patch(desc, 0.1)
            for inner in desc.GetAllChildren():
                if inner.HasAPI(UsdPhysics.CollisionAPI):
                    _bind(inner, mat_prim)
                    _set_torsional_patch(inner, 0.1)
        affected.append(f"{leaf}: (2.0, 2.0, min) + tpr=0.1 on colliders under {path}")

    # 4) Bowl override (0.5/0.5, no torsional patch radius).
    bowl_prim = stage.GetPrimAtPath("/World/envs/env_0/Bowl")
    if bowl_prim.IsValid():
        mat_path = "/World/envs/env_0/Bowl/PhysicsMaterial_bowl"
        mat_prim = stage.GetPrimAtPath(mat_path)
        if not mat_prim.IsValid():
            mat_prim = UsdShade.Material.Define(stage, mat_path).GetPrim()
        _set_material(mat_prim, 0.5, 0.5, 0.0, "min")
        # Walk descendants and bind on any collision-bearing prim.
        for d in stage.Traverse():
            if not d.GetPath().pathString.startswith("/World/envs/env_0/Bowl"):
                continue
            if d.HasAPI(UsdPhysics.CollisionAPI):
                _bind(d, mat_prim)
        affected.append("bowl: (0.5, 0.5, min) bound on /World/envs/env_0/Bowl/**")

    try:
        with open(r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/_squint_materials.log", "w") as f:
            for a in affected:
                f.write(f"  {a}\n")
    except Exception:
        pass


def make_scene_emissive(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
) -> None:
    """Squint's ``_make_scene_emissive``: copy every shader's diffuse/base
    color into its emission so the rendered pixel equals the base color
    regardless of lighting.

    With all scene lights at intensity 0 and ambient = 0 (set in
    ``__post_init__``), the PBR diffuse/specular contribution is zero
    everywhere; emission is the only term that contributes. The result is
    flat unshaded silhouettes against a black void — identical to Squint's
    SAPIEN viewport at training time.

    Handles both OmniPBR (``diffuse_color_constant`` /
    ``emissive_color`` / ``emissive_intensity`` / ``enable_emission``) and
    UsdPreviewSurface (``diffuseColor`` / ``emissiveColor``). The OmniPBR
    intensity multiplier is set to 1.0 so emission == base color exactly.

    Idempotent — safe to call as a startup event AND as a reset event after
    per-episode recolor (so emissive tracks new diffuse).
    """
    try:
        from pxr import Sdf, UsdShade
    except Exception:
        return
    stage = env.sim.stage
    if stage is None:
        return

    n_omni = 0
    n_preview = 0

    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        shader = UsdShade.Shader(prim)
        if not shader:
            continue

        # ---- OmniPBR / MDL path (preferred for the converted USDs) ----
        diff_in = shader.GetInput("diffuse_color_constant")
        if diff_in:
            try:
                rgb = diff_in.Get()
            except Exception:
                rgb = None
            if rgb is not None:
                emi = shader.GetInput("emissive_color")
                if not emi:
                    emi = shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f)
                emi.Set(rgb)
                en = shader.GetInput("enable_emission")
                if not en:
                    en = shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool)
                en.Set(True)
                ei = shader.GetInput("emissive_intensity")
                if not ei:
                    ei = shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float)
                ei.Set(1.0)
                n_omni += 1
                continue

        # ---- UsdPreviewSurface fallback ----
        diff_in = shader.GetInput("diffuseColor")
        if diff_in:
            try:
                rgb = diff_in.Get()
            except Exception:
                rgb = None
            if rgb is not None:
                emi = shader.GetInput("emissiveColor")
                if not emi:
                    emi = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
                emi.Set(rgb)
                n_preview += 1

    try:
        with open(r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/_emissive_sync.log", "w") as f:
            f.write(f"OmniPBR shaders synced: {n_omni}\n")
            f.write(f"UsdPreviewSurface shaders synced: {n_preview}\n")
    except Exception:
        pass


def make_robot_matte_black(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    rgb: tuple[float, float, float] = (0.04, 0.04, 0.04),
    robot_prim_substr: str = "/Robot/",
    roughness: float = 0.6,
    specular_level: float = 0.2,
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
        # Lower roughness + slight specular shows tooth edges in the cam.
        "reflection_roughness_constant": (Sdf.ValueTypeNames.Float, float(roughness)),
        "specular_level": (Sdf.ValueTypeNames.Float, float(specular_level)),
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


COLOR_PALETTE = (
    (1.0, 0.0, 0.0),  # 0 red
    (0.0, 0.0, 1.0),  # 1 blue
    (0.0, 1.0, 0.0),  # 2 green
    (1.0, 1.0, 0.0),  # 3 yellow
    (0.6, 0.0, 0.8),  # 4 purple
    (1.0, 0.5, 0.0),  # 5 orange
)
NUM_COLORS = len(COLOR_PALETTE)


def _recolor_prim_diffuse(env, prim_subpath_substr: str, rgb: tuple[float, float, float]) -> None:
    """Set diffuse AND emission color on every Shader prim whose path
    contains ``prim_subpath_substr``. Handles OmniPBR
    (``diffuse_color_constant`` + ``emissive_color`` + ``emissive_intensity``
    + ``enable_emission``) and UsdPreviewSurface (``diffuseColor`` +
    ``emissiveColor``).

    Diffuse and emission are kept in lockstep so the emissive-only scene
    (zero ambient + zero lights) renders the new color flat & unshaded
    every episode after color randomization.
    """
    try:
        from pxr import Sdf, UsdShade
    except Exception:
        return
    stage = env.sim.stage
    if stage is None:
        return
    rgb_tuple = tuple(float(c) for c in rgb)
    # (input_name, sdf_type, kind) — kind = "omni" | "preview"
    diffuse_candidates = (
        ("diffuseColor", Sdf.ValueTypeNames.Color3f, "preview"),
        ("diffuse_color_constant", Sdf.ValueTypeNames.Color3f, "omni"),
    )
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        path = prim.GetPath().pathString
        if prim_subpath_substr not in path:
            continue
        shader = UsdShade.Shader(prim)
        if not shader:
            continue
        try:
            sid = shader.GetIdAttr().Get() if shader.GetIdAttr() else ""
        except Exception:
            sid = ""
        for attr_name, sdf_type, kind in diffuse_candidates:
            inp = shader.GetInput(attr_name)
            if inp is None or not inp:
                if kind == "preview" and "PreviewSurface" not in sid:
                    continue
                if kind == "omni" and "OmniPBR" not in sid and "PBR" not in sid:
                    continue
                inp = shader.CreateInput(attr_name, sdf_type)
            try:
                inp.Set(rgb_tuple)
            except Exception:
                pass

            # Mirror emission so the emissive-only scene tracks the new
            # base color this episode.
            if kind == "preview":
                emi = shader.GetInput("emissiveColor")
                if not emi:
                    emi = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
                try:
                    emi.Set(rgb_tuple)
                except Exception:
                    pass
            else:  # omni
                emi = shader.GetInput("emissive_color")
                if not emi:
                    emi = shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f)
                try:
                    emi.Set(rgb_tuple)
                except Exception:
                    pass
                en = shader.GetInput("enable_emission")
                if not en:
                    en = shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool)
                try:
                    en.Set(True)
                except Exception:
                    pass
                ei = shader.GetInput("emissive_intensity")
                if not ei:
                    ei = shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float)
                try:
                    ei.Set(1.0)
                except Exception:
                    pass


# Per-env shader cache for fast multi-env recolor.
# Built lazily on first call to ``reset_goal_and_distractor_colors``.
# Schema:
#   { (env_id, asset_subname): [(UsdShade.Shader, "preview" | "omni"), ...] }
# Plus an entry "__num_envs__" -> int to detect cache staleness if the env
# is rebuilt with a different num_envs.
_SHADER_CACHE: dict | None = None


def _build_shader_cache(env, asset_subnames=("Cube", "CubeDistractor")):
    """Walk the stage once and bucket every diffuse-bearing Shader by
    (env_id, asset_subname). After this runs, per-reset recolor is O(1) in
    stage traversal (just dict lookup + .Set() per shader).
    """
    try:
        from pxr import UsdShade
    except Exception:
        return {"__num_envs__": env.num_envs}
    stage = env.sim.stage
    cache: dict = {"__num_envs__": env.num_envs}
    if stage is None:
        return cache

    # Pre-compute prefix → (env_id, asset) lookup. Avoids quadratic cost
    # when many envs share a single Stage.Traverse() pass.
    prefix_map: dict[str, tuple[int, str]] = {}
    for env_id in range(env.num_envs):
        for asset in asset_subnames:
            prefix_map[f"/World/envs/env_{env_id}/{asset}/"] = (env_id, asset)
            cache[(env_id, asset)] = []

    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        path = prim.GetPath().pathString
        matched: tuple[int, str] | None = None
        for pref, key in prefix_map.items():
            if path.startswith(pref):
                matched = key
                break
        if matched is None:
            continue
        shader = UsdShade.Shader(prim)
        if not shader:
            continue
        try:
            sid = shader.GetIdAttr().Get() if shader.GetIdAttr() else ""
        except Exception:
            sid = ""
        if "PreviewSurface" in sid:
            kind = "preview"
        elif "OmniPBR" in sid or "PBR" in sid:
            kind = "omni"
        else:
            continue
        cache[matched].append((shader, kind))
    return cache


def _apply_color_to_cached(shaders, rgb: tuple[float, float, float]) -> None:
    """Set diffuse on a list of (shader, kind) tuples — no Stage traversal."""
    try:
        from pxr import Sdf
    except Exception:
        return
    rgb_t = tuple(float(c) for c in rgb)
    for shader, kind in shaders:
        if kind == "preview":
            inp = shader.GetInput("diffuseColor")
            if not inp:
                inp = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            try: inp.Set(rgb_t)
            except Exception: pass
        else:  # omni
            inp = shader.GetInput("diffuse_color_constant")
            if not inp:
                inp = shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f)
            try: inp.Set(rgb_t)
            except Exception: pass


def reset_goal_and_distractor_colors(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
) -> None:
    """Per-episode goal color + distractor color, applied PER-ENV.

    Sample goal_idx ~ Uniform{0..5}, distractor_idx ~ Uniform over the 5
    remaining colors. Update ``env._goal_color_idx`` and
    ``env._distractor_color_idx`` per-env so the ``goal_color_one_hot`` obs
    term reads the correct value for each env.

    Then repaint the cube/distractor diffuse PER-ENV using a cached
    Shader-prim lookup so each env's cube shows its own goal color
    (critical for multi-env goal-conditioned training — the visual signal
    must match the goal_color obs vector).

    Cache build is O(num_prims) once at first reset; subsequent resets are
    O(|env_ids| × shaders_per_cube).
    """
    global _SHADER_CACHE

    device = env.device
    n = env_ids.shape[0]
    goal_idx = torch.randint(NUM_COLORS, (n,), device=device, dtype=torch.long)
    offset = torch.randint(NUM_COLORS - 1, (n,), device=device, dtype=torch.long)
    distractor_idx = offset + (offset >= goal_idx).long()

    if not hasattr(env, "_goal_color_idx") or env._goal_color_idx is None:
        env._goal_color_idx = torch.zeros(env.num_envs, device=device, dtype=torch.long)
        env._distractor_color_idx = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    env._goal_color_idx[env_ids] = goal_idx
    env._distractor_color_idx[env_ids] = distractor_idx

    # Build / refresh shader cache if needed.
    if _SHADER_CACHE is None or _SHADER_CACHE.get("__num_envs__") != env.num_envs:
        _SHADER_CACHE = _build_shader_cache(env)
        try:
            with open(r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/_shader_cache.log", "w") as f:
                f.write(f"num_envs = {env.num_envs}\n")
                for k, v in _SHADER_CACHE.items():
                    if k == "__num_envs__":
                        continue
                    f.write(f"  {k}: {len(v)} shaders\n")
        except Exception:
            pass

    env_ids_l = env_ids.cpu().tolist()
    goal_l = goal_idx.cpu().tolist()
    distr_l = distractor_idx.cpu().tolist()
    for i, eid in enumerate(env_ids_l):
        goal_rgb = COLOR_PALETTE[goal_l[i]]
        distr_rgb = COLOR_PALETTE[distr_l[i]]
        _apply_color_to_cached(_SHADER_CACHE.get((eid, "Cube"), []), goal_rgb)
        _apply_color_to_cached(_SHADER_CACHE.get((eid, "CubeDistractor"), []), distr_rgb)


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
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    fixed_cube_xy: tuple[float, float] | None = None,
    fixed_bowl_xy: tuple[float, float] | None = None,
    fixed_cube_yaw: float | None = None,
    fixed_bowl_yaw: float | None = None,
) -> None:
    """Spawn the cube AND the bowl coherently.

    Cube and bowl xy sampled in the spawn box around (0.3, 0). Resampled
    until they're at least ``ITEM_BOWL_MIN_DIST`` apart (max 100 attempts).
    Both get a random z-rotation.

    For deploy / canonical audit replay we accept ``fixed_*`` overrides
    to pin the spawn locations and yaws.
    """
    device = env.device
    n = env_ids.shape[0]
    env_origins = env.scene.env_origins[env_ids]

    # ---- Sample non-overlapping cube + bowl xy ----
    # Cube uses a SMALL box guaranteed to be inside the wrist-cam FOV at
    # home pose (see audit_spawn_visibility.py). Bowl uses the wider
    # original Squint box; its xyz is exposed via the obs state vector so
    # it doesn't need to be visible at home pose.
    if fixed_cube_xy is not None and fixed_bowl_xy is not None:
        cube_xy = torch.tensor(fixed_cube_xy, device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1).clone()
        bowl_xy = torch.tensor(fixed_bowl_xy, device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1).clone()
    else:
        cube_xy = _sample_xy_in_box(n, CUBE_SPAWN_BOX_CENTER, CUBE_SPAWN_BOX_HALF, device)
        bowl_xy = _sample_xy_in_box(n, BOWL_SPAWN_BOX_CENTER, BOWL_SPAWN_BOX_HALF, device)
        for _ in range(100):
            dist = torch.linalg.norm(cube_xy - bowl_xy, dim=-1)
            too_close = dist < ITEM_BOWL_MIN_DIST
            if not torch.any(too_close):
                break
            n_bad = int(too_close.sum().item())
            if n_bad == 0:
                break
            new_bowl_xy = _sample_xy_in_box(n_bad, BOWL_SPAWN_BOX_CENTER, BOWL_SPAWN_BOX_HALF, device)
            bowl_xy[too_close] = new_bowl_xy

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

    # ---- Bowl root pose (world frame) ----
    # Origin is the bowl's floor bottom-center, so z=table_top=0.
    bowl_pos_w = torch.zeros(n, 3, device=device)
    bowl_pos_w[:, :2] = bowl_xy + env_origins[:, :2]
    bowl_pos_w[:, 2] = env_origins[:, 2]

    if fixed_bowl_yaw is not None:
        half = fixed_bowl_yaw / 2
        bowl_quat = torch.tensor(
            [math.cos(half), 0.0, 0.0, math.sin(half)],
            device=device, dtype=torch.float32,
        ).unsqueeze(0).expand(n, -1).clone()
    else:
        bowl_quat = _random_z_quat(n, device)
    bowl_pose = torch.cat([bowl_pos_w, bowl_quat], dim=-1)
    bowl_vel = torch.zeros(n, 6, device=device)
    bowl: RigidObject = env.scene[bowl_cfg.name]
    bowl.write_root_pose_to_sim(bowl_pose, env_ids=env_ids)
    bowl.write_root_velocity_to_sim(bowl_vel, env_ids=env_ids)
