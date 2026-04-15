"""Point cloud and mesh viewer using Open3D."""

import numpy as np
import open3d as o3d
from siv.visualization.config import (
    BACKGROUND_COLOR,
    POINT_SIZE,
    POINT_COLOR,
    WINDOW_TITLE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
)


def _build_visualizer(title: str) -> o3d.visualization.Visualizer:
    """Create and configure a base Open3D Visualizer window.
    
    Args:
        title: Window title.
        
    Returns:
        Configured Visualizer instance (window already created).
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=title,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )
    vis.get_render_option().background_color = BACKGROUND_COLOR
    return vis

def show_pointcloud(
    pcd: o3d.geometry.PointCloud,
    title: str = WINDOW_TITLE,
    point_size: float = POINT_SIZE,
) -> None:
    """Display a point cloud in an interactive Open3D window.

    Controls:
        Mouse left    : rotate
        Mouse middle  : pan
        Scroll        : zoom
        Q / Esc       : close window

    Args:
        pcd: Open3D PointCloud object.
        title: Window title.
        point_size: Rendered point size in pixels.
    """
    # Paint uniform color if the cloud has no color data
    if not pcd.has_colors():
        pcd.paint_uniform_color(POINT_COLOR)

    vis = _build_visualizer(title)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = point_size

    vis.run()
    vis.destroy_window()

def show_mesh(
        mesh: o3d.geometry.TriangleMesh,
        title: str = WINDOW_TITLE,
        show_wireframe: bool = False,
) -> None:
    """Display a triangle mesh in an interactive Open3D window.
    
    Controls:
        Mouse left    : rotate
        Mouse middle  : pan
        Scroll        : zoom
        Q / Esc       : close window

    Args:
        mesh: Open3D TriangleMesh object.
        title: Window Title.
        show_wireframe: If True, renders the mesh edges over the surface.
    """
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color(POINT_COLOR)

    vis = _build_visualizer(title)
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.mesh_show_wireframe = show_wireframe
    opt.mesh_show_back_face = True

    vis.run()
    vis.destroy_window()

def show_mesh_with_landmarks(
    mesh: o3d.geometry.TriangleMesh,
    landmarks: dict[str, np.ndarray],
    title: str = WINDOW_TITLE,
    show_wireframe: bool = False,
    sphere_radius: float = 1.0,
) -> None:
    """Display a mesh with landmark positions marked as colored spheres.

    Each landmark is rendered as a red sphere. The landmark name is
    printed to console with its coordinates for reference.

    Args:
        mesh: Open3D TriangleMesh to display.
        landmarks: Dictionary mapping name to (3,) coordinate array,
                   as returned by load_landmarks_pp or transfer_landmarks.
        title: Window title.
        show_wireframe: If True, renders mesh edges over the surface.
        sphere_radius: Radius of the landmark spheres, in the same
                       units as the mesh coordinates.
    """
    geometries = []

    # Base mesh
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color(POINT_COLOR)
    geometries.append(mesh)

    # One sphere per landmark
    colors = {
        "nose":      [1.0, 0.5, 0.0],   # orange
        "right_ear": [1.0, 0.0, 0.0],   # red
        "left_ear":  [0.0, 0.5, 1.0],   # blue
    }
    default_color = [0.0, 1.0, 0.0]     # green for any other landmark

    for name, coords in landmarks.items():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(coords)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(colors.get(name, default_color))
        geometries.append(sphere)
        print(f"[show_mesh_with_landmarks] {name}: {coords}")

    vis = _build_visualizer(title)
    for geom in geometries:
        vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.mesh_show_wireframe = show_wireframe
    opt.mesh_show_back_face = True

    vis.run()
    vis.destroy_window()
