"""Here we are going to load images/pointclouds"""

from pathlib import Path

import open3d as o3d

def load_pointcloud(path: str | Path) -> o3d.geometry.PointCloud:
    """Load a point cloud from a file.

    Args:
        path: Path to the point cloud file (.ply, .pcd, .xyz, ...).

    Returns:
        Open3D PointCloud object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file could not be read as a point cloud.
    """
    pcd = o3d.io.read_point_cloud(path)

    if pcd.is_empty():
        raise ValueError(f"Could not read point cloud from '{path}'. "
                         "Check the file format.")

    return pcd

def load_mesh(path:str | Path) -> o3d.geometry.TriangleMesh:
    """Load a triangle mesh from a file (.obj, .ply, .stl, ...).
    
    Also computes vertex normal automatically if the mesh does not
    have them, which is required for correct surface shading.
    
    Args:
        path: Path to mesh file.
        
    Returns:
        Open3D TriangleMesh object.
        
    Raises:
        ValueError: If the file could not be read or is empty.
    """
    mesh = o3d.io.read_triangle_mesh(str(path))

    if mesh.is_empty():
        raise ValueError(f"Could not read mesh from '{path}.'"
                         "Check the file format.")

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    print(f"[load_mesh] {len(mesh.vertices)} vertices, "
          f"{len(mesh.triangles)} triangles. "
          f"Normals: {mesh.has_vertex_normals()}, " 
          f"Colors: {mesh.has_vertex_colors()}")

    return mesh
