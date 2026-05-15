"""Replicate Squint's ``scripts/mesh_bowl_from_ply.py`` — turn the SAM 3D
Gaussian-splat .ply into a watertight mesh (bowl.obj + bowl.ply) and a
CoACD convex decomposition (bowl.obj.coacd.ply).

Run with the conda env that has open3d / plyfile / coacd:

    & C:/Users/user/anaconda3/envs/squint/python.exe \
        C:/Users/user/Desktop/MA2/robot-learning-project3/sim/eval2/scripts/mesh_bowl_from_ply.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData


def load_splat_xyz_rgb(in_ply: Path):
    """Read SAM-3D Gaussian splat .ply -> (xyz, rgb [0,1]) numpy arrays."""
    ply = PlyData.read(str(in_ply))
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float64)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float64)
    rgb = np.clip(0.5 + 0.282 * f_dc, 0.0, 1.0)
    return xyz, rgb


def reconstruct_mesh(xyz: np.ndarray, rgb: np.ndarray, depth: int = 10,
                     density_prune: float = 0.02, aabb_margin: float = 0.005,
                     target_n_faces: int = 12_000):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(rgb)
    print(f"[mesh] points: {len(xyz):,}")

    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pc.orient_normals_consistent_tangent_plane(k=30)
    print(f"[mesh] normals estimated + oriented")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth)
    print(f"[mesh] Poisson @ depth={depth}: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    densities = np.asarray(densities)
    keep = densities >= np.quantile(densities, density_prune)
    mesh.remove_vertices_by_mask(~keep)
    print(f"[mesh] density-pruned (bottom {density_prune*100:.0f}%): "
          f"{len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    aabb = pc.get_axis_aligned_bounding_box()
    aabb = o3d.geometry.AxisAlignedBoundingBox(
        aabb.get_min_bound() - aabb_margin,
        aabb.get_max_bound() + aabb_margin,
    )
    mesh = mesh.crop(aabb)
    print(f"[mesh] cropped to AABB + {aabb_margin*1e3:.0f} mm: "
          f"{len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if len(cluster_n_triangles) > 1:
        biggest = int(np.argmax(cluster_n_triangles))
        mesh.remove_triangles_by_mask(triangle_clusters != biggest)
        mesh.remove_unreferenced_vertices()
        print(f"[mesh] kept largest CC ({cluster_n_triangles[biggest]:,} tris) "
              f"of {len(cluster_n_triangles)} components")

    if len(mesh.triangles) > target_n_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_n_faces)
        print(f"[mesh] decimated to ~{target_n_faces:,} tris: "
              f"{len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    print(f"[mesh] cleaned: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    # Re-origin: xy-center at 0, z_min = 0
    verts = np.asarray(mesh.vertices)
    cx, cy = verts[:, 0].mean(), verts[:, 1].mean()
    zmin = verts[:, 2].min()
    verts[:, 0] -= cx
    verts[:, 1] -= cy
    verts[:, 2] -= zmin
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    print(f"[mesh] re-origined: bowl floor at z=0, xy centered")

    # Bake vertex colors via nearest-neighbour from the source point cloud.
    pc_tree = o3d.geometry.KDTreeFlann(pc)
    verts_np = np.asarray(mesh.vertices) + np.array([cx, cy, zmin])  # back to source frame
    pc_rgb = np.asarray(pc.colors)
    new_colors = np.zeros_like(verts_np)
    for i, v in enumerate(verts_np):
        _, idx, _ = pc_tree.search_knn_vector_3d(v, 1)
        new_colors[i] = pc_rgb[idx[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    print(f"[mesh] baked vertex colors. mean rgb = "
          f"({new_colors[:,0].mean():.3f}, {new_colors[:,1].mean():.3f}, {new_colors[:,2].mean():.3f})")
    return mesh


def coacd_decompose(in_obj: Path, out_ply: Path,
                    threshold: float = 0.30, max_convex_hull: int = 8):
    """Coarse decomposition (target ≤8 hulls) — matches the Squint-side fix
    after the broad-phase contact buffer overflow at 2048 parallel envs.
    Default threshold=0.30 was the tuning point Squint converged on.
    """
    import coacd, trimesh

    tm = trimesh.load(str(in_obj), force="mesh")
    coacd_mesh = coacd.Mesh(np.asarray(tm.vertices), np.asarray(tm.faces))
    parts = coacd.run_coacd(
        coacd_mesh, threshold=threshold, max_convex_hull=max_convex_hull
    )
    print(f"[coacd] decomposed into {len(parts)} convex hulls")

    merged = o3d.geometry.TriangleMesh()
    for verts, faces in parts:
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(np.asarray(verts))
        m.triangles = o3d.utility.Vector3iVector(np.asarray(faces))
        merged += m
    o3d.io.write_triangle_mesh(str(out_ply), merged)
    print(f"[coacd] wrote {out_ply} ({len(merged.vertices):,} verts, {len(merged.triangles):,} tris)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ply", default="squint/meshes/bowl_raw.ply")
    ap.add_argument("--out_dir", default="squint/meshes")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--target_n_faces", type=int, default=12_000)
    ap.add_argument("--coacd_threshold", type=float, default=0.30)
    ap.add_argument("--max_convex_hull", type=int, default=8)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    in_ply = (project_root / args.in_ply).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz, rgb = load_splat_xyz_rgb(in_ply)
    print(f"[mesh] AABB = {xyz.min(0).round(3).tolist()} -> {xyz.max(0).round(3).tolist()}")

    mesh = reconstruct_mesh(xyz, rgb, depth=args.depth, target_n_faces=args.target_n_faces)
    aabb = mesh.get_axis_aligned_bounding_box()
    print(f"[mesh] post-AABB = {np.asarray(aabb.min_bound).round(4).tolist()} "
          f"-> {np.asarray(aabb.max_bound).round(4).tolist()}")

    obj_path = out_dir / "bowl.obj"
    ply_path = out_dir / "bowl.ply"
    o3d.io.write_triangle_mesh(str(obj_path), mesh, write_vertex_colors=False)
    o3d.io.write_triangle_mesh(str(ply_path), mesh, write_vertex_colors=True)
    print(f"[mesh] wrote {obj_path}")
    print(f"[mesh] wrote {ply_path}")

    coacd_decompose(obj_path, out_dir / "bowl.obj.coacd.ply",
                    threshold=args.coacd_threshold,
                    max_convex_hull=args.max_convex_hull)


if __name__ == "__main__":
    main()
