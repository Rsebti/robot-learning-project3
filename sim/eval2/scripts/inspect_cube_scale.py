"""Inspect the LeIsaac scene.usd to verify cube scale modification.

Prints all xformOps on cube-related prims so we can verify whether the
resize_cube_usd.py modification actually took effect.
"""
from __future__ import annotations

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


def main():
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(SCENE_USD))
    if stage is None:
        print(f"[ERR] could not open {SCENE_USD}")
        return 1
    print(f"=== USD prims in {SCENE_USD} ===")
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if "cube" in path.lower():
            print(f"\nPrim: {path}  type={prim.GetTypeName()}")
            xf = UsdGeom.Xformable(prim)
            if xf:
                ops = xf.GetOrderedXformOps()
                if not ops:
                    print(f"  (no xformOps)")
                for op in ops:
                    name = op.GetOpName()
                    val = op.Get()
                    print(f"  {name} = {val}")
            else:
                print(f"  (not Xformable)")
    return 0


if __name__ == "__main__":
    rc = main()
    _simulation_app.close()
    sys.exit(rc)
