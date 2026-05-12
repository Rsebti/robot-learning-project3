"""Override the LeIsaac table color to match the project spec (#B8ADA9).

The LeIsaac scene loads a textured wood-colored table from the USD asset
``scenes/table_with_cube/scene.usd``. The CLAUDE.md project spec calls for
a light gray table (`~#B8ADA9`) to match the real-world setup. We override
this by:

  1. Locating the table prim(s) in the scene USD.
  2. Defining a new ``UsdPreviewSurface`` material with diffuse color
     #B8ADA9 (RGB 184, 173, 169 → normalized 0.722, 0.678, 0.663).
  3. Binding this material to the table prims, overriding the textured
     material previously in effect.

The script is idempotent — running it twice has the same effect as once
(the binding is replaced each time). Backup is written on first run.

Run once via Isaac Sim venv (pxr requires AppLauncher to be live):

    python -m sim.eval2.scripts.recolor_table_usd --headless

To revert: copy `scene.usd.bak` -> `scene.usd`.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from isaaclab.app import AppLauncher
import argparse

_parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_parser)
_args = _parser.parse_args()
if not getattr(_args, "headless", False):
    _args.headless = True
_app_launcher = AppLauncher(_args)
_simulation_app = _app_launcher.app

SCENE_USD = Path(
    r"C:\Users\user\Desktop\MA2\isaac\leisaac\assets\scenes\table_with_cube\scene.usd"
)
BACKUP = SCENE_USD.with_suffix(".usd.bak")

# RGB normalized from #B8ADA9 (CLAUDE.md project spec)
TABLE_COLOR = (184.0 / 255.0, 173.0 / 255.0, 169.0 / 255.0)
ROUGHNESS = 0.8

# Material lives under /world/Looks (matching the existing structure of the
# LeIsaac scene USD).
NEW_MATERIAL_PATH = "/world/Looks/TableColorOverride"
NEW_SHADER_PATH = NEW_MATERIAL_PATH + "/PreviewSurface"


def _find_table_prims(stage):
    """Return list of Xform/Mesh prims that look like the table.

    LeIsaac names the table prim ``counter_right_main_group`` (with a
    Mesh child ``geometry_0``). Heuristic: any prim path containing
    'counter', 'table', 'tabletop', or 'geometry' (skipping Material
    and Shader prims which live under /world/Looks).
    """
    candidates = []
    skip_types = {"Material", "Shader", "NodeGraph"}
    for prim in stage.Traverse():
        if prim.GetTypeName() in skip_types:
            continue
        path_l = prim.GetPath().pathString.lower()
        if "/looks/" in path_l:
            continue
        name_l = prim.GetName().lower()
        if any(k in name_l for k in ("table", "tabletop", "counter", "geometry")):
            candidates.append(prim)
    return candidates


def main() -> int:
    if not SCENE_USD.exists():
        print(f"[ERR] scene.usd not found at {SCENE_USD}")
        return 1

    if not BACKUP.exists():
        shutil.copyfile(SCENE_USD, BACKUP)
        print(f"[INFO] Backup written: {BACKUP}")
    else:
        print(f"[INFO] Backup already exists: {BACKUP}  (kept)")

    from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf

    stage = Usd.Stage.Open(str(SCENE_USD))
    if stage is None:
        print(f"[ERR] Could not open USD stage: {SCENE_USD}")
        return 2

    # Find the table prim(s)
    table_prims = _find_table_prims(stage)
    if not table_prims:
        print("[ERR] No prim with 'table' in name found. Stage prim list:")
        for prim in stage.Traverse():
            print(f"    {prim.GetPath().pathString}  ({prim.GetTypeName()})")
        return 3

    print(f"[INFO] Found {len(table_prims)} table candidate prim(s):")
    for p in table_prims:
        print(f"    {p.GetPath().pathString}  type={p.GetTypeName()}")

    # Create the override material
    print(f"[INFO] Defining material at {NEW_MATERIAL_PATH}")
    material = UsdShade.Material.Define(stage, NEW_MATERIAL_PATH)
    shader = UsdShade.Shader.Define(stage, NEW_SHADER_PATH)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*TABLE_COLOR)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(ROUGHNESS)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    # Connect shader output to material's surface
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Bind material to each table prim — and to all Mesh descendants too,
    # so it overrides any nested textured mesh.
    n_bindings = 0
    for prim in table_prims:
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        binding_api.Bind(
            material,
            UsdShade.Tokens.strongerThanDescendants,  # binding wins over child overrides
        )
        n_bindings += 1
        # Also iterate child Mesh prims and bind them
        for child in Usd.PrimRange(prim):
            if child.GetTypeName() == "Mesh" and child != prim:
                child_api = UsdShade.MaterialBindingAPI.Apply(child)
                child_api.Bind(material, UsdShade.Tokens.strongerThanDescendants)
                n_bindings += 1

    print(f"[INFO] Bound material on {n_bindings} prim(s).")

    stage.GetRootLayer().Save()
    print(f"[INFO] Saved {SCENE_USD}")
    print()
    print("=== DONE ===")
    print(f"Color applied: RGB({TABLE_COLOR[0]:.3f}, {TABLE_COLOR[1]:.3f}, {TABLE_COLOR[2]:.3f}) = #B8ADA9")
    print(f"To revert: copy '{BACKUP}' to '{SCENE_USD}'")
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
