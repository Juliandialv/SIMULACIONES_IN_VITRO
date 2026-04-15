"""Main program of the application"""

from siv.config import MESH_DIR
from siv.io.loader import load_mesh
from siv.landmarks.renderer import render_front_view
from siv.landmarks.detector import detect_landmarks_2d
from siv.landmarks.projector import (
    render_depth_map,
    project_landmarks_to_3d,
    landmarks_to_pointcloud,
)
from siv.visualization.viewer import show_pointcloud_with_landmarks

def main():
    """Main execution code"""
    mesh = load_mesh(MESH_DIR / "P" / "P_1_6.obj")
    print(f"[main] Mesh loaded with {len(mesh.vertices)} vertices.")

    # 2. Render frontal view + get intrinsics
    image, intrinsics = render_front_view(mesh)

    # 3. Detect 2D landmarks on the rendered image
    landmarks_2d = detect_landmarks_2d(image)
    if landmarks_2d is None:
        print("No landmarks detected. Check mesh orientation.")
        return

    # 4. Render depth map and project landmarks to 3D
    depth_map = render_depth_map(mesh)
    landmarks_3d = project_landmarks_to_3d(landmarks_2d, depth_map, intrinsics)

    # 5. Build landmark point cloud
    landmark_pcd = landmarks_to_pointcloud(landmarks_3d, color=[1.0, 0.0, 0.0])

    # 6. Convert mesh to point cloud for visualization and show
    mesh_pcd = mesh.sample_points_uniformly(number_of_points=500_000)
    show_pointcloud_with_landmarks(mesh_pcd, landmark_pcd)


if __name__ == "__main__":
    main()
