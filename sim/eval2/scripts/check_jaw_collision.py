"""Inspect the converted USD to confirm the lower jaw (gripper_link)
has TWO separate collision meshes (Fixed_part_1 + Fixed_part_2)."""
from __future__ import annotations
import argparse
import torch  # noqa

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-SquintNative-Place-Play-v0")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
args.num_envs = 1
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import sim.eval2  # noqa: F401,E402


def main():
    env_cfg = parse_env_cfg(args.task, num_envs=1)
    try:
        env_cfg.recorders = None
    except Exception:
        pass
    env = gym.make(args.task, cfg=env_cfg)
    base_env = env.unwrapped
    env.reset(seed=0)

    from pxr import Usd, Sdf, UsdPhysics, UsdGeom
    stage = base_env.sim.stage

    # Open the physics USD layer directly to see the master collision prims
    phys = r"C:/Users/user/Desktop/MA2/robot-learning-project3/squint/converted_usd/configuration/so101_squint_renamed_physics.usd"
    out_phys = r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/jaw_collision_phys.txt"
    with open(out_phys, "w") as f:
        try:
            phys_stage = Usd.Stage.Open(phys)
            f.write(f"opened: {phys}\n\n")
            for body in ("gripper", "jaw", "moving_jaw_so101_v1_link"):
                path_root = f"/colliders/{body}"
                f.write(f"\n=== {body} @ {path_root} ===\n")
                root = phys_stage.GetPrimAtPath(path_root)
                if not root.IsValid():
                    f.write(f"  (no prim)\n")
                    continue
                n_mesh = 0
                for prim in Usd.PrimRange(root, Usd.PrimAllPrimsPredicate):
                    tname = prim.GetTypeName()
                    p = prim.GetPath().pathString
                    f.write(f"  [{tname:20s}] {p}\n")
                    if tname == "Mesh":
                        n_mesh += 1
                f.write(f"  -> Mesh count = {n_mesh}\n")
        except Exception as e:
            f.write(f"failed: {e}\n")
    print(f"[jaw-coll] wrote {out_phys}")

    out_dir = r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/gripper_collision_full.txt"
    with open(out_dir, "w") as fout:
        # 1. Layer stack
        fout.write("=== Stage layer stack ===\n")
        for layer in stage.GetLayerStack():
            fout.write(f"  {layer.identifier}\n")
        fout.write("\n=== Stage sublayers ===\n")
        for layer in stage.GetUsedLayers():
            fout.write(f"  {layer.identifier}\n")

        # 2. Use predicate to get ALL prims (including under references)
        fout.write("\n=== ALL prims under /World/envs/env_0/Robot/gripper (instance+abstract) ===\n")
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper")
        if root.IsValid():
            for prim in Usd.PrimRange(root, Usd.PrimAllPrimsPredicate):
                tname = prim.GetTypeName()
                path = prim.GetPath().pathString
                is_inst = prim.IsInstanceable()
                has_coll = prim.HasAPI(UsdPhysics.CollisionAPI)
                tag = "[COLL]" if has_coll else ""
                fout.write(f"  [{tname:25s}{' inst' if is_inst else '     '}] {tag} {path}\n")
        else:
            fout.write("  (no prim)\n")

        fout.write("\n=== ALL prims under /World/envs/env_0/Robot/jaw ===\n")
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot/jaw")
        if root.IsValid():
            for prim in Usd.PrimRange(root, Usd.PrimAllPrimsPredicate):
                tname = prim.GetTypeName()
                path = prim.GetPath().pathString
                has_coll = prim.HasAPI(UsdPhysics.CollisionAPI)
                tag = "[COLL]" if has_coll else ""
                fout.write(f"  [{tname:25s}] {tag} {path}\n")
        print(f"[jaw-coll] wrote {out_dir}")
    out_path = r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/gripper_collision.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        # ALSO open the source physics USD directly to count mesh prims
        phys_path = r"C:/Users/user/Desktop/MA2/robot-learning-project3/squint/converted_usd/so101_squint_renamed_physics.usd"
        f.write(f"=== Direct read of {phys_path} ===\n")
        try:
            phys_stage = Usd.Stage.Open(phys_path)
            for body in ("gripper", "jaw", "moving_jaw_so101_v1_link"):
                path_root = f"/colliders/{body}"
                f.write(f"\n  -- {body} @ {path_root} --\n")
                root = phys_stage.GetPrimAtPath(path_root)
                if not root.IsValid():
                    f.write(f"    (no prim)\n")
                    continue
                n_mesh = 0
                for prim in Usd.PrimRange(root):
                    tn = prim.GetTypeName()
                    if tn in ("Mesh", "Xform", "Scope"):
                        f.write(f"    [{tn:18s}] {prim.GetPath().pathString}\n")
                    if tn == "Mesh":
                        n_mesh += 1
                f.write(f"    -> Mesh count = {n_mesh}\n")
        except Exception as e:
            f.write(f"   failed: {e}\n")
        f.write("\n" + "=" * 70 + "\n")
        for body_name in ("gripper", "jaw"):
            path_root = f"/World/envs/env_0/Robot/{body_name}"
            f.write(f"\n=== {body_name} → FULL prim tree under {path_root} ===\n")
            root = stage.GetPrimAtPath(path_root)
            if not root.IsValid():
                f.write(f"  (no prim)\n")
                continue
            n_meshes = 0
            for prim in Usd.PrimRange(root):
                tname = prim.GetTypeName()
                path = prim.GetPath().pathString
                # Show all prims with type/references info
                refs_str = ""
                try:
                    refs = prim.GetReferences()
                    if refs:
                        # GetReferences returns a Reference object; let's check meta
                        meta = prim.GetMetadata("references")
                        if meta:
                            refs_str = f"  [refs={meta}]"
                except Exception:
                    pass
                f.write(f"  [{tname:25s}] {path}{refs_str}\n")
                if tname == "Mesh":
                    n_meshes += 1
            f.write(f"  -> total Mesh prims = {n_meshes}\n")
    print(f"[jaw-coll] wrote {out_path}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
