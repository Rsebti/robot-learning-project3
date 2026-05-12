"""Override the SO-101 robot color to matte black (#0D0D0D).

LeIsaac's `so101_follower.usd` ships with a yellow-toned plastic look that
makes the robot blend with the lab walls in some renders. This script
overrides every Mesh prim in the robot USD with a dark UsdPreviewSurface
material so the arm shows up as matte black against the gray table.

Idempotent — running twice is a no-op (same material redefined and
re-bound). A `.bak` backup of the original USD is written on the first
run for easy revert.

Run via Isaac Sim venv:

    python sim/eval2/scripts/recolor_robot_usd.py --headless

To revert: copy `so101_follower.usd.bak` back over `so101_follower.usd`.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_parser)
_args = _parser.parse_args()
if not getattr(_args, "headless", False):
    _args.headless = True
_app_launcher = AppLauncher(_args)
_simulation_app = _app_launcher.app

ROBOT_USD = Path(
    r"C:\Users\user\Desktop\MA2\isaac\leisaac\assets\robots\so101_follower.usd"
)
BACKUP = ROBOT_USD.with_suffix(".usd.bak")

# Matte dark gray so it reads as "black" in the viewer without going pure
# RGB(0,0,0) (which can look flat under some lighting setups).
ROBOT_COLOR = (0.05, 0.05, 0.05)
ROUGHNESS = 0.6
METALLIC = 0.1

# Material lives at the stage root under /Looks (which the SO-101 USD
# already provides). If it doesn't exist we just create it.
NEW_MATERIAL_PATH = "/Root/Looks/RobotColorOverride"
NEW_SHADER_PATH = NEW_MATERIAL_PATH + "/PreviewSurface"


def main() -> int:
    if not ROBOT_USD.exists():
        print(f"[ERR] robot USD not found at {ROBOT_USD}")
        return 1

    if not BACKUP.exists():
        shutil.copyfile(ROBOT_USD, BACKUP)
        print(f"[INFO] Backup written: {BACKUP}")
    else:
        print(f"[INFO] Backup already exists: {BACKUP}  (kept)")

    from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf  # noqa: F401

    stage = Usd.Stage.Open(str(ROBOT_USD))
    if stage is None:
        print(f"[ERR] Could not open USD stage: {ROBOT_USD}")
        return 2

    # Find all Mesh prims (these are the visible geometry of the robot).
    # We deliberately do NOT filter by name because the SO-101 mesh names
    # are scattered (gripper has many sub-meshes, etc.).
    mesh_prims = [p for p in stage.Traverse() if p.GetTypeName() == "Mesh"]
    if not mesh_prims:
        print("[ERR] No Mesh prims found in robot USD. Stage prim list:")
        for prim in stage.Traverse():
            print(f"    {prim.GetPath().pathString}  ({prim.GetTypeName()})")
        return 3

    print(f"[INFO] Found {len(mesh_prims)} mesh prim(s) on the robot:")
    # Print first 10 to keep output manageable.
    for p in mesh_prims[:10]:
        print(f"    {p.GetPath().pathString}")
    if len(mesh_prims) > 10:
        print(f"    ... and {len(mesh_prims) - 10} more")

    # Decide where to put the material — under whatever root the stage uses.
    # Default Isaac Sim USDs use /Root or /World as the default prim. We use
    # the default prim's path if available, otherwise fall back.
    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        material_parent = default_prim.GetPath().pathString + "/Looks"
    else:
        material_parent = "/Looks"
    material_path = f"{material_parent}/RobotColorOverride"
    shader_path = f"{material_path}/PreviewSurface"
    print(f"[INFO] Defining material at {material_path}")

    # Ensure the parent /Looks exists (creating xform if needed).
    if not stage.GetPrimAtPath(material_parent).IsValid():
        UsdGeom.Scope.Define(stage, material_parent)

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*ROBOT_COLOR)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(ROUGHNESS)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(METALLIC)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )

    # Bind to every Mesh prim. `strongerThanDescendants` ensures this
    # binding wins over any nested per-mesh material declared in the USD.
    n_bindings = 0
    for prim in mesh_prims:
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        binding_api.Bind(
            material,
            UsdShade.Tokens.strongerThanDescendants,
        )
        n_bindings += 1

    print(f"[INFO] Bound material on {n_bindings} mesh prim(s).")

    stage.GetRootLayer().Save()
    print(f"[INFO] Saved {ROBOT_USD}")
    print()
    print("=== DONE ===")
    print(
        f"Color applied: RGB({ROBOT_COLOR[0]:.3f}, {ROBOT_COLOR[1]:.3f}, "
        f"{ROBOT_COLOR[2]:.3f}) = matte dark gray ('black')"
    )
    print(f"To revert: copy '{BACKUP}' back over '{ROBOT_USD}'")
    return 0


if __name__ == "__main__":
    print(">>> entering main()", flush=True)
    try:
        rc = main()
    except Exception as e:
        import traceback
        print(f">>> EXCEPTION: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        rc = 99
    print(f">>> main() returned rc={rc}", flush=True)
    _simulation_app.close()
    sys.exit(rc)
