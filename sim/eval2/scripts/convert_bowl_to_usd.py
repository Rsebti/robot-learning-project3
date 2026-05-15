"""Convert bowl.ply (visual w/ vertex colors) and bowl.obj (collision) to USD.

Produces ``squint/meshes/bowl.usd`` with a triangle-mesh collider (we use
a kinematic bowl, so triangle-mesh collision is accurate and fast).
"""
from __future__ import annotations
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--in_mesh",
                    default=r"C:/Users/user/Desktop/MA2/robot-learning-project3/squint/meshes/bowl.obj")
parser.add_argument("--out_dir",
                    default=r"C:/Users/user/Desktop/MA2/robot-learning-project3/squint/meshes")
parser.add_argument("--mesh_collision", default="convexDecomposition",
                    choices=["triangleMesh", "convexDecomposition", "convexHull"])
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg  # noqa: E402
from isaaclab.sim.schemas import schemas_cfg  # noqa: E402


def main():
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mesh_collision == "triangleMesh":
        mesh_props = schemas_cfg.TriangleMeshPropertiesCfg()
    elif args.mesh_collision == "convexDecomposition":
        # NB: we pass an EMPTY cfg here so the converter picks the USD path
        # and sets approximation="convexDecomposition". The PhysX-specific
        # attrs (max_convex_hulls=8, error_percentage=30) are written below
        # via direct USD patching — passing them here triggers an Isaac Lab
        # gotcha where the USD approximation field is skipped.
        mesh_props = schemas_cfg.ConvexDecompositionPropertiesCfg()
    else:
        mesh_props = schemas_cfg.ConvexHullPropertiesCfg()

    cfg = MeshConverterCfg(
        asset_path=str(Path(args.in_mesh)),
        usd_dir=str(out_dir),
        usd_file_name="bowl.usd",
        force_usd_conversion=True,
        make_instanceable=False,
        mass_props=schemas_cfg.MassPropertiesCfg(density=500.0),
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            solver_position_iteration_count=15,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=5.0,
        ),
        collision_props=schemas_cfg.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
        mesh_collision_props=mesh_props,
        translation=(0.0, 0.0, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )
    converter = MeshConverter(cfg)
    print(f"[convert] {args.in_mesh} -> {converter.usd_path}")
    print(f"[convert] mesh collision approximation: {args.mesh_collision}")

    # Isaac Lab's MeshConverter skips the USD approximation field when any
    # custom PhysX attrs are passed (extract_mesh_collision_api_and_attrs in
    # isaaclab/sim/schemas/schemas.py:960), so PhysX falls back to convexHull
    # at runtime. Patch the USD here to set BOTH the approximation field AND
    # the convex-decomposition limits.
    if args.mesh_collision == "convexDecomposition":
        from pxr import Usd, UsdPhysics, PhysxSchema
        stage = Usd.Stage.Open(converter.usd_path)
        n_patched = 0
        for prim in stage.Traverse():
            if prim.GetTypeName() != "Mesh":
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr().Set("convexDecomposition")
            cd_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            cd_api.CreateMaxConvexHullsAttr().Set(8)
            cd_api.CreateErrorPercentageAttr().Set(30.0)
            cd_api.CreateShrinkWrapAttr().Set(False)
            n_patched += 1
        stage.GetRootLayer().Save()
        print(f"[convert] patched {n_patched} mesh prim(s) with approximation"
              f"='convexDecomposition' + maxHulls=8 + errorPct=30")
    app.close()


if __name__ == "__main__":
    main()
