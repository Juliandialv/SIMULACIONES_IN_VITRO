"""Point cloud and mesh viewer using Open3D."""

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

def show_pointcloud_with_landmarks(
    pcd: o3d.geometry.PointCloud,
    landmarks: o3d.geometry.PointCloud,
    title: str = WINDOW_TITLE,
    point_size: float = POINT_SIZE,
    landmark_size: float = 6.0,
) -> None:
    """Display a point cloud with landmark points overlaid.

    Landmarks are rendered larger and in a distinct color so they
    are clearly visible over the base geometry.

    Args:
        pcd: Base point cloud.
        landmarks: Landmark points as a separate colored PointCloud.
        title: Window title.
        point_size: Size for the base cloud points.
        landmark_size: Size for the landmark points.
    """
    if not pcd.has_colors():
        pcd.paint_uniform_color(POINT_COLOR)

    vis = _build_visualizer(title)
    vis.add_geometry(pcd)
    vis.add_geometry(landmarks)

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
