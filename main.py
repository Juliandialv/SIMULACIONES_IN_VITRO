"""Main program of the application"""

from siv.io.loader import load_pointcloud
from siv.visualization.viewer import show_pointcloud

def main():
    """Main execution code"""
    path = "data/raw/Symmetric_Head.ply"
    pcd = load_pointcloud(path)
    print(f"Point cloud loaded: {len(pcd.points)} points.")

    show_pointcloud(pcd)

if __name__ == "__main__":
    main()
