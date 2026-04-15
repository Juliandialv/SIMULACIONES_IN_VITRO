"""Here we are goint to define all pointcloud related functions"""

import open3d as o3d

def random_downsample(
        pcd: o3d.geometry.PointCloud,
        ratio: float,
) -> o3d.geometry.PointCloud:
    """Randomly downsample a point cloud by a given rateio.
    
    Preserves the original spatial density distribution, so dense
    regions remain denser than sparse one. Fast and simple, but not
    recommended before normal estimation or mesh reconstruction
    
    Args:
        pcd: Input point cloud
        ratio: Fraction of points to keep, in range (0, 1].
               e.g. 0.1 keeps 10% of the points.
               
    Returns:
        Downsampled point cloud
        
    Raises:
        ValueError: if ratio is not in (0, 1].
    """
    if not 0 < ratio <= 1:
        raise ValueError(f"ratio must be in (0,1], got {ratio}.")

    downsampled = pcd.random_down_sample(ratio)
    print(f"[random_downsample] {len(pcd.points)} -> "
          f"{len(downsampled.points)} "
          f"points ({ratio*100:.0f}% kept)")

    return downsampled

def voxel_downsample(
        pcd: o3d.geometry.PointCloud,
        voxel_size: float,
) -> o3d.geometry.PointCloud:
    """Downsample a point cloud using a voxel grid.
    
    Divides the space into regular grid of cubic voxels and replaces
    all points within each voxel with their centroid. The result has a
    spatially uniform distribution regardless of the original density,
    making it suitable for normal estimation and mesh reconstruction.
    
    A smaller voxel_size preserves more detail but reduces fewer points.
    A larger voxel_Size produces a coarser but more uniform cloud.
    
    Args:
        pcd: Input point cloud.
        voxel_size: Side lentgh of each voxe cube, in the same units
                    as the point cloud coordinates.
                    
    Returns:
        Downsampled point cloud.
    
    Raises:
        ValueError: if voxel_size is not positive.
    """
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}.")

    downsampled = pcd.voxel_down_sample(voxel_size)
    print(f"[voxel_downsample] {len(pcd.points)} -> "
          f"{len(downsampled.points)} "
          f"points (voxel_size: {voxel_size}).")

    return downsampled
