"""Saving of point clouds and meshes."""

from pathlib import Path

import open3d as o3d

def save_pointcloud(
        pcd: o3d.geometry.PointCloud,
        path: str | Path,
        overwrite: bool = False,
) -> None:
    """Save a poiint cloud to a file.
    
    Supported formats are determined by the file extension:
    .ply, .pcd, .xyz, .xyzn, .xyzrgb, .pts.
    The .ply format is recommended as it preserves all attributes
    (colors, normals) and is widely supported.
    
    Args:
        pcd: Open3D PointCloud object to save.
        path: Destination file path, including extension.
        overwrite: If False, raises FileExistsError of the file
                   already exist. Defaults to False.
                   
    Raises:
        ValueError: If the point cluod is empty.
        FileExistsError: If the file exists and overwrite is False.
        IOError: If the file could not be written.
    """
    path = Path(path)

    if pcd.is_empty():
        raise ValueError("Cannot save an empty point cloud.")

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File '{path}' already exists. Use overwrite=True to replace it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    success = o3d.io.write_point_cloud(str(path), pcd)
    if not success:
        raise IOError(f"Open3D could not write the point cloud to '{path}'.")

    print(f"[save_pointcloud] Saved {len(pcd.points)} points -> {path}")
