"""Cranial vault surface reconstruction from clipped point cloud.

Uses Poisson surface reconstruction (Kazhdan & Hoppe, 2013) via Open3D.
No temporary files — geometry stays in memory as Open3D/NumPy objects.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from src.siv.utils.logger import logger, LogLevel


def clip_cloud_above_plane(
    pcd: o3d.geometry.PointCloud,
    normal: np.ndarray,
    plane_center: np.ndarray,
) -> o3d.geometry.PointCloud:
    """Keep only points strictly above the cutting plane.

    Args:
        pcd:          Full point cloud.
        normal:       Unit normal of the cutting plane (pointing upward).
        plane_center: Any point on the cutting plane.

    Returns:
        New PointCloud with only the points above the plane.
    """
    points = np.asarray(pcd.points)
    d      = -np.dot(normal, plane_center)
    signed_dists = points @ normal + d
    mask   = signed_dists > 0

    clipped = o3d.geometry.PointCloud()
    clipped.points = o3d.utility.Vector3dVector(points[mask])

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        clipped.colors = o3d.utility.Vector3dVector(colors[mask])

    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        clipped.normals = o3d.utility.Vector3dVector(normals[mask])

    logger.log(
        f"Clipped cloud: {mask.sum()} / {len(points)} points above reconstruction plane",
        LogLevel.INFO
    )
    return clipped


def reconstruct_cranial_vault(
    pcd: o3d.geometry.PointCloud,
    poisson_depth: int = 9,
    density_quantile: float = 0.05,
) -> o3d.geometry.TriangleMesh:
    """Reconstruct the cranial vault surface using Poisson reconstruction.

    Pipeline:
        1. Estimate normals if not present (oriented toward +Z)
        2. Run Poisson reconstruction
        3. Remove low-density vertices (boundary artefacts)

    Args:
        pcd:              Clipped point cloud (above cutting plane).
        poisson_depth:    Octree depth for Poisson — higher = more detail.
                          9 is standard for head scans at ~1-2mm resolution.
        density_quantile: Vertices below this density quantile are removed.
                          0.05 removes the bottom 5% (boundary artefacts).

    Returns:
        Reconstructed TriangleMesh, cleaned of low-density artefacts.
    """
    # ── Normales ────────────────────────────────────────────────────────────
    if not pcd.has_normals():
        logger.log("Estimating normals...", LogLevel.INFO)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=6.0,
                max_nn=50
            )
        )

    # Orientar alejándose del centro de masa — robusto para superficies convexas
    # No depende de propagación por vecinos, no genera islas
    points  = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    center  = points.mean(axis=0)

    # Vector desde el centro hasta cada punto
    outward = points - center  # (N, 3)

    # Si la normal apunta hacia el centro (dot < 0), invertirla
    dots = np.einsum('ij,ij->i', normals, outward)  # dot product por fila
    normals[dots < 0] *= -1

    pcd.normals = o3d.utility.Vector3dVector(normals)

    logger.log(f"Running Poisson reconstruction (depth={poisson_depth})...", LogLevel.INFO)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=False
    )
    densities = np.asarray(densities)
    logger.log(
        f"Raw mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles",
        LogLevel.INFO
    )

    threshold = np.quantile(densities, density_quantile)
    keep_mask = densities > threshold
    mesh.remove_vertices_by_mask(~keep_mask)
    logger.log(
        f"After density filter: "
        f"{len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles",
        LogLevel.INFO
    )

    mesh.compute_vertex_normals()
    return mesh

def clip_mesh_by_plane(
    mesh: o3d.geometry.TriangleMesh,
    normal: np.ndarray,
    plane_center: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """Clip a triangle mesh with a plane, computing exact intersections.

    Uses PyVista's clip_surface which computes the exact intersection
    of each triangle with the plane, generating new vertices at the
    boundary. Result has a clean straight edge at the cutting plane.

    Args:
        mesh:         Input TriangleMesh (Open3D).
        normal:       Unit normal of the cutting plane (pointing upward).
        plane_center: Any point on the cutting plane.

    Returns:
        Clipped TriangleMesh (Open3D) with a clean boundary at the plane.
    """
    import pyvista as pv

    # Convertir O3D → PyVista
    vertices  = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    faces_pv  = np.hstack([np.full((len(triangles), 1), 3), triangles])
    mesh_pv   = pv.PolyData(vertices, faces_pv)

    # Clip con PyVista — calcula intersecciones exactas
    clipped_pv = mesh_pv.clip(
        normal   = normal.tolist(),   # sin invertir la normal
        origin   = plane_center.tolist(),
        invert   = False,
    )

    # Convertir PyVista → O3D
    pts   = np.array(clipped_pv.points)
    faces = clipped_pv.faces.reshape(-1, 4)[:, 1:]  # quitar el prefijo 3

    clipped_o3d = o3d.geometry.TriangleMesh()
    clipped_o3d.vertices  = o3d.utility.Vector3dVector(pts)
    clipped_o3d.triangles = o3d.utility.Vector3iVector(faces)
    clipped_o3d.remove_unreferenced_vertices()
    clipped_o3d.compute_vertex_normals()

    return clipped_o3d
