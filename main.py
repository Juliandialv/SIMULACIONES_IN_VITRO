"""Main program of the application."""

from siv.config import MESH_DIR
from siv.io.loader import load_mesh
from siv.io.picker import pick_landmarks
from siv.visualization.viewer import show_mesh_with_landmarks


def main():
    """Main application's code block"""
    mesh = load_mesh(MESH_DIR / "P" / "P_1_6.obj")
    print(f"[main] Mesh loaded with {len(mesh.vertices)} vertices.")

    landmarks = pick_landmarks(mesh)

    if landmarks:
        show_mesh_with_landmarks(mesh, landmarks, show_wireframe=True,
                                 sphere_radius=2.0)


if __name__ == "__main__":
    main()
