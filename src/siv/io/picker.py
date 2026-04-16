"""Interactive 3D landmark picking using Open3D."""

import numpy as np
import open3d as o3d
from siv.processing.pointcloud import voxel_downsample

# Anatomical region filters (based on normalize bbox coordinates)
_REGIONS = {
    "sellion": {
        "description": "the sellion (nasal bridge)",
        "filter": lambda norm: (
            (norm[:,0] > 0.80) & # anterior X
            (np.abs(norm[:,1] - 0.5) < 0.2) # central Y
        ),
        "view": "three_quarters",
    },
    "right tragion": {
        "description": "the right tragion (right rear)",
        "filter": lambda norm: norm[:,1] < 0.20, # lateral right Y 
        "view": "right_lateral",
    },
    "left tragion": {
        "description": "the left tragion (left rear)",
        "filter": lambda norm: norm[:,1] > 0.80, # lateral left Y
        "view": "left_lateral",
    },
}

def _normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    """Normalize vertex coordinates to [0, 1] per axis."""
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    return (vertices - bbox_min) / (bbox_max - bbox_min)

def _set_camera(vis: o3d.visualization.VisualizerWithEditing,
                view: str,
                center: np.ndarray) -> None:
    """Set camera to a prefifined anatomical viewpoint
    
    Args:
        vis: Open3D visualizer instance.
        view: One of 'three_quarters', 'right_lateral', 'left_latearl'.
        center: Centroid of the geometry, used as look-at target.
    """
    ctr = vis.get_view_control()

    # Reset to a known state first
    ctr.set_lookat(center.tolist())
    ctr.set_up([0, 0, 1])

    if view == "three_quarters":
        ctr.set_front([1.0, 1.0, 1.0])
    elif view == "right_lateral":
        ctr.set_front([0.0, -1.0, -0.1]) # looking from right (+Y toward -Y)
    elif view == "left_lateral":
        ctr.set_front([0.0, 1.0, -0.1]) # looking from left (-Y toward +Y)

    ctr.set_zoom(0.6)

def _pick_single_landmark(
        pcd_region: o3d.geometry.PointCloud,
        center: np.ndarray,
        name: str,
        description: str,
        view: str,
        window_size: tuple[int, int] = (1280, 720),
) -> np.ndarray | None:
    """Open a picking window for a single landmark.
    
    Args:
        pcd_region: Point cloud filtered to the anatomical region.
        center: Geometry centroid for camera targeting.
        name: Landmark key name.
        description: Human-readable description show in window title.
        view: Camera viewpoint preset.
        window_size: Window dimensions (width, length).
        
    Returns:
        (3,) coordinate array of the selected point, or None if skipped."""
    title = f"Select {description} | Shift+click to pick | Q to confirm"
    print(f"\n[picker] Please select {description}.")
    print(f"    Shift+click to pic, Shift+right click to undo, Q to confirm.")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=window_size[0],
                      height=window_size[1])
    vis.add_geometry(pcd_region)
    _set_camera(vis, view, center)
    vis.run()

    picked_indices = vis.get_picked_points()
    vis.destroy_window()

    if not picked_indices:
        print(f"[picker] WARNING: No point selected for {name}.")
        return None

    if len(picked_indices) > 1:
        print(f"[picker] WARNING: {len(picked_indices)} points selected for "
              f"{name}, using the last one.")

    pts = np.asarray(pcd_region.points)
    coords = pts[picked_indices[-1]]
    print(f"[picker] {name}: {coords}")
    return coords

def pick_landmarks(
        mesh: o3d.geometry.TriangleMesh,
        n_samples: int = 1000000,
        window_size: tuple[int, int] = (1280, 270),
) -> dict[str, np.ndarray]:
    """Interactively pick anatomical landmarks on a head mesh.
    
    Opens one window per landmark, each showing only the relevant
    anatomical region to prevent accidental selecion of the wrong side.
    
    Landmarks picked:
        - sellion: nasal bridge (most concave point of the nose)ç
        - right_tragion: right ear tragus
        - left_tragion: left ear tragus
        
    Args:
        mesh: Input TriangleMesh of the head.
        n_samples: Number of points sampled from the mesh surface for 
                   picking. Higher values give more precision.
        window_size: Pickig window dimensions (width, height).
        
    Returns:
        Dictionary mapping landmark name to (3,) coordinate array.
        Missing landmarks (skipped by user) are absemt from the dict
    """
    pcd_full = mesh.sample_points_uniformly(number_of_points=n_samples)
    pcd_down = voxel_downsample(pcd_full, voxel_size=0.75)
    vertices = np.asarray(pcd_down.points)
    norm = _normalize_vertices(vertices)
    center = vertices.mean(axis=0)

    landmarks = {}

    for name, cfg in _REGIONS.items():
        mask = cfg["filter"](norm)
        n_filtered = mask.sum()

        if n_filtered == 0:
            print(f"[picker] WARNING: Mo points in region for {name}."
                  f"Check anatomical thresholds.")
            continue

        print(f"[picker] Region '{name}': {n_filtered} points available.")

        pcd_region = o3d.geometry.PointCloud()
        pcd_region.points = o3d.utility.Vector3dVector(vertices[mask])
        pcd_region.paint_uniform_color([0.7, 0.7, 0.7])

        coords = _pick_single_landmark(
            pcd_region=pcd_region,
            center=center,
            name=name,
            description=cfg["description"],
            view=cfg["view"],
            window_size=window_size,
        )

        if coords is not None:
            landmarks[name] = coords

    print(f"\n[picker] Picking complete. {len(landmarks)}/3 landmarks collected.")
    return landmarks
