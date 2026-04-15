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

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=title,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )

    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = BACKGROUND_COLOR
    opt.point_size = point_size

    vis.run()
    vis.destroy_window()