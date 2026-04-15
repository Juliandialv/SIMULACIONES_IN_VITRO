"""Offscreen rendering of 3D geometry for landmark detection."""

import numpy as np
import open3d as o3d
import pyvista as pv


def _o3d_mesh_to_pyvista(mesh: o3d.geometry.TriangleMesh) -> pv.PolyData:
    """Convert an Open3D TriangleMesh to a PyVista PolyData.

    Args:
        mesh: Open3D TriangleMesh.

    Returns:
        Equivalent PyVista PolyData.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    faces = np.hstack([np.full((len(triangles), 1), 3), triangles])
    return pv.PolyData(vertices, faces)


def render_front_view(
    mesh: o3d.geometry.TriangleMesh,
    width: int = 1024,
    height: int = 768,
) -> tuple[np.ndarray, o3d.camera.PinholeCameraIntrinsic]:
    """Render a front-facing view of a mesh using PyVista offscreen.

    The camera is placed along the -Z axis looking toward the origin,
    which corresponds to a frontal view for standard facial meshes
    aligned with the XY plane.

    Args:
        mesh: Open3D TriangleMesh to render.
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        Tuple of:
            - RGB image as NumPy array of shape (height, width, 3).
            - Open3D PinholeCameraIntrinsic built from the render parameters.
    """
    pv_mesh = _o3d_mesh_to_pyvista(mesh)

    pl = pv.Plotter(off_screen=True, window_size=[width, height])
    pl.add_mesh(pv_mesh, color="lightgray", smooth_shading=True)
    pl.view_yz()  # frontal view: camera along -X, looking toward +X
    pl.camera.zoom(1.2)

    image = pl.screenshot(return_img=True)  # (H, W, 3) RGB uint8
    pl.close()

    # Build approximate pinhole intrinsics from the render dimensions.
    # fov ~60 degrees is PyVista's default.
    fov_y = 60.0
    fy = (height / 2.0) / np.tan(np.radians(fov_y / 2.0))
    fx = fy
    cx, cy = width / 2.0, height / 2.0

    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    print(f"[render_front_view] Image: {image.shape}, "
          f"fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    return image, intrinsics