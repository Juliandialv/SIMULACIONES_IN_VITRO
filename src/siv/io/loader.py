"""Here we are going to load images/pointclouds"""

import open3d as o3d

def load_pointcloud(path):
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
