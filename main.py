"""Main program of the application"""

from siv.io.loader import load_mesh
from siv.io.writer import save_triangle_mesh
from siv.config import MESH_DIR
from siv.visualization.viewer import show_mesh

def main():
    """Main execution code"""
    mesh = load_mesh(MESH_DIR / "P" / "P_1_6.obj")
    print(f"[main] Mesh loaded: {len(mesh.vertices)} vertices.")
    save_triangle_mesh(mesh, MESH_DIR / "Malla_1.obj")
    show_mesh(mesh, show_wireframe=True)


if __name__ == "__main__":
    main()
