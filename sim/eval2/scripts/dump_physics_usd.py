"""Open the converted physics USD directly and dump the /colliders/gripper
subtree to see how many collision meshes the gripper has."""
from __future__ import annotations
from pxr import Usd

USD_PATH = r"C:/Users/user/Desktop/MA2/robot-learning-project3/squint/converted_usd/so101_squint_renamed_physics.usd"

stage = Usd.Stage.Open(USD_PATH)
out_path = r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/physics_usd_dump.txt"
with open(out_path, "w") as f:
    f.write(f"opened: {USD_PATH}\n\n")
    for body in ("gripper", "jaw", "moving_jaw_so101_v1_link"):
        path_root = f"/colliders/{body}"
        f.write(f"\n=== {body} @ {path_root} ===\n")
        root = stage.GetPrimAtPath(path_root)
        if not root.IsValid():
            f.write(f"  (no prim)\n")
            continue
        n_meshes = 0
        for prim in Usd.PrimRange(root):
            tname = prim.GetTypeName()
            path = prim.GetPath().pathString
            f.write(f"  [{tname:25s}] {path}\n")
            if tname == "Mesh":
                n_meshes += 1
        f.write(f"  -> Mesh count = {n_meshes}\n")
print(f"wrote {out_path}")
