"""Landmark loading, transfer and geometric operations."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import open3d as o3d


# --- Loading ---

def load_landmarks_pp(path: str | Path) -> dict[str, np.ndarray]:
    """Load 3D landmarks from a MeshLab PickedPoints (.pp) file.

    Args:
        path: Path to the .pp XML file.

    Returns:
        Dictionary mapping landmark name to (3,) float64 array [x, y, z].

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no active landmarks are found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Landmark file not found: '{path}'.")

    tree = ET.parse(path)
    root = tree.getroot()

    landmarks = {}
    for point in root.findall("point"):
        if point.get("active") != "1":
            continue
        name = point.get("name")
        x = float(point.get("x"))
        y = float(point.get("y"))
        z = float(point.get("z"))
        landmarks[name] = np.array([x, y, z], dtype=np.float64)

    if not landmarks:
        raise ValueError(f"No active landmarks found in '{path}'.")

    print(f"[load_landmarks_pp] Loaded {len(landmarks)} landmarks: "
          f"{list(landmarks.keys())}")
    return landmarks


# --- Transfer via ICP ---

def transfer_landmarks(
    landmarks_ref: dict[str, np.ndarray],
    mesh_ref: o3d.geometry.TriangleMesh,
    mesh_target: o3d.geometry.TriangleMesh,
    max_iterations: int = 100,
    threshold: float = 5.0,
) -> dict[str, np.ndarray]:
    """Transfer landmarks from a reference mesh to a target mesh using ICP.

    Computes the rigid transformation that aligns mesh_ref to mesh_target
    and applies it to the landmark coordinates.

    Args:
        landmarks_ref: Landmarks on the reference mesh, as returned by
                       load_landmarks_pp.
        mesh_ref: Reference mesh (the one where landmarks were picked).
        mesh_target: Target mesh to transfer landmarks to.
        max_iterations: Maximum ICP iterations.
        threshold: Maximum correspondence distance for ICP, in the same
                   units as the mesh coordinates.

    Returns:
        Dictionary mapping landmark name to transformed (3,) array.
    """
    pcd_ref = mesh_ref.sample_points_uniformly(number_of_points=50_000)
    pcd_target = mesh_target.sample_points_uniformly(number_of_points=50_000)

    result = o3d.pipelines.registration.registration_icp(
        source=pcd_ref,
        target=pcd_target,
        max_correspondence_distance=threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations,
        ),
    )

    T = result.transformation  # (4, 4) homogeneous matrix
    print(f"[transfer_landmarks] ICP fitness={result.fitness:.4f}, "
          f"inlier_rmse={result.inlier_rmse:.4f}")

    transferred = {}
    for name, point in landmarks_ref.items():
        point_h = np.append(point, 1.0)          # homogeneous
        transformed = (T @ point_h)[:3]
        transferred[name] = transformed

    return transferred


# --- Geometric operations ---

def landmarks_to_plane(
    landmarks: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the plane defined by exactly 3 landmark points.

    Args:
        landmarks: Dictionary with exactly 3 landmark entries.

    Returns:
        Tuple of:
            - normal: Unit normal vector of the plane (3,).
            - centroid: Centroid of the 3 points (3,), a point on the plane.

    Raises:
        ValueError: If landmarks does not contain exactly 3 points, or if
                    the points are collinear.
    """
    if len(landmarks) != 3:
        raise ValueError(f"Expected exactly 3 landmarks, got {len(landmarks)}.")

    pts = list(landmarks.values())
    p0, p1, p2 = pts[0], pts[1], pts[2]

    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)

    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        raise ValueError("The 3 landmark points are collinear — cannot define a plane.")

    normal = normal / norm
    centroid = (p0 + p1 + p2) / 3.0

    print(f"[landmarks_to_plane] Normal: {normal}, Centroid: {centroid}")
    return normal, centroid

# --- Automatic detection ---

def _compute_mean_curvature(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Estimate mean curvature per vertex using the umbrella operator.

    Args:
        mesh: Input mesh with vertex normals already computed.

    Returns:
        Array of shape (N,) with curvature magnitude per vertex.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Build adjacency list
    adjacency = [[] for _ in range(len(vertices))]
    for tri in triangles:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[tri[i]].append(tri[j])

    curvature = np.zeros(len(vertices))
    for i, neighbors in enumerate(adjacency):
        if not neighbors:
            continue
        neighbor_verts = vertices[list(set(neighbors))]
        laplacian = neighbor_verts.mean(axis=0) - vertices[i]
        curvature[i] = np.linalg.norm(laplacian)

    return curvature

def detect_landmarks_auto(
    mesh: o3d.geometry.TriangleMesh,
    axis_anterior: int = 0,
    axis_lateral: int = 1,
    nose_lateral_tol: float = 0.15,
    ear_anterior_tol: float = 0.45,
) -> dict[str, np.ndarray]:
    """Automatically detect nose and ear landmarks from mesh geometry.

    Uses anatomical axis extremes with a small midline/lateral filter
    to robustly find the most prominent point in each region.

    Strategy:
        - Nose      : most anterior vertex (max X) within the central
                      lateral band (Y near midline).
        - Right ear : most lateral-right vertex (min Y) within the
                      posterior region (low X).
        - Left ear  : most lateral-left vertex (max Y) within the
                      posterior region (low X).

    Args:
        mesh: Input TriangleMesh.
        axis_anterior: Index of the anteroposterior axis (default 0 = X).
        axis_lateral: Index of the lateral axis (default 1 = Y).
        nose_lateral_tol: Fraction of lateral extent around the midline
                          to consider as the nose region. Default 0.15.
        ear_anterior_tol: Fraction of anterior extent below which
                          vertices are considered in the ear region.
                          Default 0.45.

    Returns:
        Dictionary with keys 'nose', 'right_ear', 'left_ear', each
        mapping to a (3,) coordinate array on the mesh surface.

    Raises:
        ValueError: If any anatomical region contains no vertices.
    """
    vertices = np.asarray(mesh.vertices)

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_range = bbox_max - bbox_min

    ant = axis_anterior
    lat = axis_lateral

    norm_ant = (vertices[:, ant] - bbox_min[ant]) / bbox_range[ant]
    norm_lat = (vertices[:, lat] - bbox_min[lat]) / bbox_range[lat]

    # --- Nose: max X within central Y band ---
    lat_mid = 0.5
    nose_mask = np.abs(norm_lat - lat_mid) < nose_lateral_tol
    if nose_mask.sum() == 0:
        raise ValueError("No vertices in nose region. Adjust nose_lateral_tol.")
    nose_idx = np.where(nose_mask)[0][vertices[nose_mask, ant].argmax()]

    # --- Ears: min/max Y within posterior X region ---
    ear_mask = norm_ant < ear_anterior_tol
    if ear_mask.sum() == 0:
        raise ValueError("No vertices in ear region. Adjust ear_anterior_tol.")

    right_mask = ear_mask & (norm_lat < 0.5)
    left_mask  = ear_mask & (norm_lat > 0.5)

    if right_mask.sum() == 0 or left_mask.sum() == 0:
        raise ValueError("No vertices in one of the ear regions.")

    right_idx = np.where(right_mask)[0][vertices[right_mask, lat].argmin()]
    left_idx  = np.where(left_mask)[0][vertices[left_mask, lat].argmax()]

    landmarks = {
        "nose":      vertices[nose_idx],
        "right_ear": vertices[right_idx],
        "left_ear":  vertices[left_idx],
    }

    for name, coords in landmarks.items():
        print(f"[detect_landmarks_auto] {name}: {coords}")

    return landmarks
