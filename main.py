"""Main program of the application"""

"""Main program of the application."""

import numpy as np
from siv.config import RAW_DIR, MESH_DIR
from siv.io.loader import load_mesh
from siv.processing.landmarks import (
    load_landmarks_pp,
    detect_landmarks_auto,
    landmarks_to_plane,
)
from siv.visualization.viewer import show_mesh_with_landmarks


def compare_landmarks(
    manual: dict[str, np.ndarray],
    auto: dict[str, np.ndarray],
) -> None:
    """Print Euclidean distance between manual and automatic landmarks."""
    print("\n--- Manual vs Automatic comparison ---")
    for name in manual:
        if name in auto:
            dist = np.linalg.norm(manual[name] - auto[name])
            print(f"  {name}: {dist:.2f} mm")
    print("--------------------------------------\n")


def main():
    """Main application's code blcock"""
    mesh = load_mesh(MESH_DIR / "P" / "P_1_6.obj")

    # Manual landmarks (gold standard)
    landmarks_manual = load_landmarks_pp(RAW_DIR / "P_1_6_picked_points.pp")

    # Automatic detection
    landmarks_auto = detect_landmarks_auto(mesh)

    # Comparison
    compare_landmarks(landmarks_manual, landmarks_auto)

    # Visual inspection — change to landmarks_auto to inspect automatic
    show_mesh_with_landmarks(mesh, landmarks_manual, show_wireframe=True, sphere_radius=2.0)


if __name__ == "__main__":
    main()
