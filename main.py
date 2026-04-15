"""Main program of the application"""

from siv.io.loader import load_pointcloud
from siv.io.writer import save_pointcloud
from siv.visualization.viewer import show_pointcloud
from siv.processing.pointcloud import voxel_downsample
from siv.config import RAW_DIR, PROCESSED_DIR

def main():
    """Main execution code"""
    pcd = load_pointcloud(RAW_DIR / "Symmetric_Head.ply")
    print(f"[main] Point cloud loaded: {len(pcd.points)} points.")
    pcd_down = voxel_downsample(pcd, voxel_size=1.5)

    save_pointcloud(pcd_down,
                    PROCESSED_DIR / "Symmetric_Head_voxeldownsample_1.5.ply")
    show_pointcloud(pcd_down)


if __name__ == "__main__":
    main()
