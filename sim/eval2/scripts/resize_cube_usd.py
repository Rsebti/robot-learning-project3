"""Resize the LeIsaac cube prim from 3cm to 2cm by adding a scale xformOp.

The LeIsaac scene loads the cube via a USD reference to
``scenes/table_with_cube/cube/cube.usd``, which contains a textured mesh
(BuildingBlock003) with half-extents 0.015077 m → 3 cm full size. To
match the project spec (2 cm cube), we apply a uniform scale of 2/3 on
the cube xform prim in scene.usd. This preserves the mesh (visuals +
collision + physics properties) and only scales it geometrically.

Why edit scene.usd and NOT cube.usd:
  - scene.usd holds the *reference* to cube.usd. Adding a scale on the
    referencing prim leaves cube.usd untouched.
  - Reverting is easier (clear the xformOp).
  - Other tasks/scenes that use cube.usd directly are not affected.

The Pixar USD library (`pxr`) is only importable after Isaac Sim's
AppLauncher initializes — so this script launches a minimal headless
Isaac Sim session, applies the modification, then exits.

Run once:
    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    .\\.venv\\Scripts\\Activate.ps1
    python -m sim.eval2.scripts.resize_cube_usd

To revert: copy scene.usd.bak → scene.usd.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Launch Isaac Sim FIRST so `pxr` is importable.
from isaaclab.app import AppLauncher

import argparse
_parser = argparse.ArgumentParser(description="Resize LeIsaac cube to 2cm.")
AppLauncher.add_app_launcher_args(_parser)
_args = _parser.parse_args()
# Force headless to keep this fast and uninstrusive.
if not getattr(_args, "headless", False):
    _args.headless = True
_app_launcher = AppLauncher(_args)
_simulation_app = _app_launcher.app

SCENE_USD = Path(
    r"C:\Users\user\Desktop\MA2\isaac\leisaac\assets\scenes\table_with_cube\scene.usd"
)
BACKUP = SCENE_USD.with_suffix(".usd.bak")
TARGET_SCALE = 2.0 / 3.0  # 3 cm -> 2 cm


def main() -> int:
    if not SCENE_USD.exists():
        print(f"[ERR] scene.usd not found at {SCENE_USD}")
        return 1

    # Backup if not already done
    if not BACKUP.exists():
        shutil.copyfile(SCENE_USD, BACKUP)
        print(f"[INFO] Backup written: {BACKUP}")
    else:
        print(f"[INFO] Backup already exists: {BACKUP}  (not overwritten)")

    # pxr is now available via Isaac Sim's runtime
    from pxr import Usd, UsdGeom, Gf

    stage = Usd.Stage.Open(str(SCENE_USD))
    if stage is None:
        print(f"[ERR] Could not open USD stage: {SCENE_USD}")
        return 3

    # Walk the stage and find a prim whose name or path contains "cube"
    candidates = []
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        name = prim.GetName().lower()
        if "cube" in name or "/cube" in path.lower():
            candidates.append(prim)

    if not candidates:
        print("[ERR] No prim with 'cube' in name found. Stage prim list:")
        for prim in stage.Traverse():
            print(f"    {prim.GetPath().pathString}  ({prim.GetTypeName()})")
        return 4

    print(f"[INFO] Cube candidate prims found:")
    for p in candidates:
        print(f"    {p.GetPath().pathString}  type={p.GetTypeName()}")

    # Heuristic: prefer the topmost cube xform prim (shortest path that's an Xformable)
    target = None
    for p in candidates:
        if UsdGeom.Xformable(p):
            if target is None or len(p.GetPath().pathString) < len(target.GetPath().pathString):
                target = p
    if target is None:
        # Fallback: take the first candidate's parent if it's Xformable
        target = candidates[0]
    print(f"[INFO] Applying scale ({TARGET_SCALE:.4f}, {TARGET_SCALE:.4f}, {TARGET_SCALE:.4f}) to: "
          f"{target.GetPath().pathString}")

    # Apply uniform scale via xformOp:scale
    xformable = UsdGeom.Xformable(target)
    if not xformable:
        print(f"[ERR] Target prim is not Xformable.")
        return 5

    # Check if scale op already exists
    existing_ops = xformable.GetOrderedXformOps()
    scale_op = None
    for op in existing_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break

    if scale_op is None:
        scale_op = xformable.AddXformOp(UsdGeom.XformOp.TypeScale,
                                        UsdGeom.XformOp.PrecisionFloat)
        print(f"[INFO] Added new xformOp:scale to {target.GetPath().pathString}")
    else:
        print(f"[INFO] Existing xformOp:scale found, overriding value.")

    scale_op.Set(Gf.Vec3f(TARGET_SCALE, TARGET_SCALE, TARGET_SCALE))

    # Save
    stage.GetRootLayer().Save()
    print(f"[INFO] Saved {SCENE_USD}")
    print()
    print("=== DONE ===")
    print(f"Cube scale applied: {TARGET_SCALE:.4f} (3 cm -> 2 cm)")
    print(f"To revert: copy '{BACKUP}' → '{SCENE_USD}'")
    return 0


if __name__ == "__main__":
    rc = main()
    _simulation_app.close()
    sys.exit(rc)
