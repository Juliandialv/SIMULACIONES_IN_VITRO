"""Projection of 2D landmarks onto 3D geometry"""

import numpy as np
import pyvista as pv
import open3d as o3d


def render_depth_map(
    mesh: o3d.geometry.TriangleMesh,
    width: int = 1024,
    height: int = 768,
) -> np.ndarray:
    """Render a depth map of the mesh from the same frontal viewpoint.

    Args:
        mesh: Open3D TriangleMesh.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Depth map as float32 array of shape (height, width),
        values in scene units (same as mesh coordinates).
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    faces = np.hstack([np.full((len(triangles), 1), 3), triangles])
    pv_mesh = pv.PolyData(vertices, faces)

    pl = pv.Plotter(off_screen=True, window_size=[width, height])
    pl.add_mesh(pv_mesh, color="white")
    pl.view_yz()
    pl.camera.zoom(1.2)
    pl.show(auto_close=False)

    depth = pl.get_image_depth()  # (H, W) float32
    pl.close()

    return depth

def project_landmarks_to_3d(
        landmarks_2d: np.ndarray,
        depth_map: np.ndarray,
        intrinsics: o3d.camera.PinholeCameraIntrinsic,
) -> np.ndarray:
    """Project 2D pixel landmarks to 3D points usindg a depth map.
    
    Args:
        landmarks_2d: Array of shape (N, 2) with (x, y) pxel coordinates.
        depth_map: Depth image as float32 array, values in metres or mm
                   consistent with intrinsics.
        intrinsics: Open3D pinhole camera intrinsics (fx, fy, cx, cy).
        
    Returns:
        Array of shape (N, 3) with (X, Y, Z) 3D coordinates.
        Points where depth is zeero or invalid are set to NaN.
    """
    fx, fy = intrinsics.get_focal_length()
    cx, cy = intrinsics.get_principal_point()

    points_3d = []
    for (px, py) in landmarks_2d:
        x, y = int(round(px)), int(round(py))

        if y < 0 or y >= depth_map.shape[0] or x < 0 or x >= depth_map.shape[1]:
            points_3d.append([np.nan, np.nan, np.nan])
            continue

        z = float(depth_map[y, x])
        if z == 0:
            points_3d.append([np.nan, np.nan, np.nan])
            continue

        X = (px - cx) * z / fx
        Y = (py - cy) * z / fy
        points_3d.append([X, Y, z])

    return np.array(points_3d)

def landmarks_to_pointcloud(
        points_3d: np.ndarray,
        color: list[float] = [1.0, 0.0, 0.0],
) -> o3d.geometry.PointCloud:
    """Convert a 3D landmark array to a colored Open3D PointCloud
    
    Filters out NaN points (invalid depth).
    
    Args:
        points_3d: Array of shape (N, 3)
        color: RGB color for all landmark points, range [0, 1]
        
    Returns:
        Open3D PointCloud with uniform color.
    """
    valid = points_3d[~np.isnan(points_3d).any(axis=1)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid)
    pcd.paint_uniform_color(color)

    print(f"[landmarks_to_pointcloud] {len(valid)} valid landmarks out of "
          f"{len(points_3d)}.")
    return pcd
