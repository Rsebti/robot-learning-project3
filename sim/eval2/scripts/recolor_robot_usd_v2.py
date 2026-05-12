"""Recolor SO-101 robot to matte dark gray by patching the original OmniPBR materials.

Why v2 (and not v1):
    The first attempt (recolor_robot_usd.py) added a NEW UsdPreviewSurface
    material under /so101_new_calib/Looks and rebound all 47 mesh prims to it
    with `strongerThanDescendants`. The bindings WERE applied, but the
    rendered scene still shows the original yellow color.

    Reason: the original materials (`material_a_3d_printed`,
    `material_sts3215`) use NVIDIA's OmniPBR (MDL) which is the renderer's
    preferred material type. Our UsdPreviewSurface override doesn't win
    against an OmniPBR material in the RTX path tracer / hydra pipeline.
    Plus, USD reference scoping might prevent the override binding from
    propagating to the instance prims at /World/envs/env_0/Robot/...

    v2 takes the DIRECT route: open the USD, find every OmniPBR shader,
    set its `diffuse_color_constant` to dark gray, and disable diffuse
    textures so the constant color shows through.

    This idempotently recolors without adding any new material/binding —
    we just edit the existing ones in place.

Run:
    python sim/eval2/scripts/recolor_robot_usd_v2.py --headless

Reverting:
    Same as before — copy `so101_follower.usd.bak` over `so101_follower.usd`.
    The .bak from v1 was preserved (and v2 doesn't overwrite it on second run).
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

# Matte dark gray — reads as "black" without going pure RGB(0,0,0).
TARGET_RGB = (0.05, 0.05, 0.05)


def main() -> int:
    if not ROBOT_USD.exists():
        print(f"[ERR] robot USD not found at {ROBOT_USD}")
        return 1

    if not BACKUP.exists():
        shutil.copyfile(ROBOT_USD, BACKUP)
        print(f"[INFO] Backup written: {BACKUP}")
    else:
        print(f"[INFO] Backup already exists: {BACKUP}  (kept)")

    from pxr import Usd, UsdShade, Sdf, Gf  # noqa: E402

    stage = Usd.Stage.Open(str(ROBOT_USD))
    if stage is None:
        print(f"[ERR] Could not open USD stage")
        return 2

    # 1. Locate every Material prim and its child Shader(s).
    materials = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Material":
            materials.append(prim)

    if not materials:
        print("[ERR] No Material prims found.")
        return 3

    print(f"[INFO] Found {len(materials)} material prim(s):")
    for m in materials:
        print(f"    {m.GetPath().pathString}")

    # 2. For each material, locate Shader children and patch OmniPBR inputs.
    n_patched = 0
    n_skipped = 0
    for material_prim in materials:
        for child in material_prim.GetChildren():
            if child.GetTypeName() != "Shader":
                continue
            shader = UsdShade.Shader(child)
            shader_id_attr = child.GetAttribute("info:mdl:sourceAsset:subIdentifier")
            shader_id = shader_id_attr.Get() if shader_id_attr.IsValid() else None
            shader_path = child.GetPath().pathString
            mdl_asset = child.GetAttribute("info:mdl:sourceAsset").Get()
            mdl_str = str(mdl_asset) if mdl_asset else "(none)"

            is_omnipbr = (
                shader_id == "OmniPBR"
                or "OmniPBR" in mdl_str
            )
            if not is_omnipbr:
                print(
                    f"  [SKIP] {shader_path} — not OmniPBR "
                    f"(id={shader_id}, mdl={mdl_str})"
                )
                n_skipped += 1
                continue

            # Patch diffuse_color_constant.
            diffuse_input = shader.GetInput("diffuse_color_constant")
            if not diffuse_input:
                diffuse_input = shader.CreateInput(
                    "diffuse_color_constant", Sdf.ValueTypeNames.Color3f
                )
            diffuse_input.Set(Gf.Vec3f(*TARGET_RGB))

            # Disable diffuse texture so the constant color is the source.
            enable_tex = shader.GetInput("enable_diffuse_texture")
            if not enable_tex:
                enable_tex = shader.CreateInput(
                    "enable_diffuse_texture", Sdf.ValueTypeNames.Bool
                )
            enable_tex.Set(False)

            # Slight roughness/metallic for matte black look.
            for name, value, dtype in [
                ("reflection_roughness_constant", 0.6, Sdf.ValueTypeNames.Float),
                ("metallic_constant", 0.0, Sdf.ValueTypeNames.Float),
            ]:
                inp = shader.GetInput(name)
                if not inp:
                    inp = shader.CreateInput(name, dtype)
                inp.Set(value)

            print(f"  [OK]   {shader_path} — diffuse_color_constant set to {TARGET_RGB}")
            n_patched += 1

    print(f"[INFO] Patched {n_patched} OmniPBR shader(s), skipped {n_skipped}.")

    if n_patched == 0:
        print("[WARN] No OmniPBR shaders patched. Robot will keep its original colors.")
        return 4

    stage.GetRootLayer().Save()
    print(f"[INFO] Saved {ROBOT_USD}")
    print()
    print("=== DONE ===")
    print(
        f"OmniPBR diffuse_color_constant set to RGB"
        f"({TARGET_RGB[0]:.3f}, {TARGET_RGB[1]:.3f}, {TARGET_RGB[2]:.3f})"
        f" + diffuse texture disabled."
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
