"""Geometric primitives derived from anatomical landmarks.

All functions in this module operate on raw NumPy arrays and return either
plain NumPy structures or Open3D geometry objects.  No visualisation or UI
code lives here, keeping geometry logic fully decoupled from rendering.

Coordinate convention (same as the rest of the pipeline):
    X  →  anterior  (positive = towards face front)
    Y  →  lateral   (positive = patient's left)
    Z  →  superior  (positive = up / vertex)
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from src.siv.utils.logger import logger, LogLevel

# ── Reference plane ───────────────────────────────────────────────────────────

def compute_reference_plane(
    sellion: np.ndarray,
    right_tragion: np.ndarray,
    left_tragion: np.ndarray,
    half_size: float = 80.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the Frankfurt-like reference plane from the three cranial landmarks.

    The plane is spanned by the two inter-tragion and tragion-sellion vectors.
    Its normal is oriented consistently: always pointing anteriorly (+X),
    so that the plane faces the front of the head regardless of mesh
    orientation variability.

    The returned quad is a rectangle centred at the plane centroid, with
    side length ``2 * half_size`` along the two in-plane axes.  This gives a
    visually stable, symmetrically sized plane independent of the landmark
    spread.

    Args:
        sellion:       (3,) coordinate of the nasal bridge landmark.
        right_tragion: (3,) coordinate of the right ear tragus.
        left_tragion:  (3,) coordinate of the left ear tragus.
        half_size:     Half-edge length of the rendered quad in the same
                       units as the mesh (typically mm).  Default 80 mm gives
                       a plane slightly wider than a neonatal head.

    Returns:
        normal:   (3,) unit normal of the plane, pointing anteriorly.
        center:   (3,) centroid of the three landmarks.
        vertices: (4, 3) float32 corners of the display quad, ordered for
                  two CCW triangles: [0,1,2] and [0,2,3].
        faces:    (2, 3) int32 triangle index array [[0,1,2],[0,2,3]].

    Raises:
        ValueError: If the three landmarks are collinear (degenerate plane).
    """
    p1 = np.asarray(sellion,       dtype=float)
    p2 = np.asarray(right_tragion, dtype=float)
    p3 = np.asarray(left_tragion,  dtype=float)

    # Two edge vectors spanning the plane
    v1 = p2 - p1  # sellion → right_tragion
    v2 = p3 - p1  # sellion → left_tragion

    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-10:
        raise ValueError(
            "The three landmarks are collinear or coincident — "
            "cannot define a unique plane."
        )
    normal /= norm_len

    # Ensure the normal always points anteriorly (positive X in our convention)
    if normal[0] < 0:
        normal = -normal

    center = (p1 + p2 + p3) / 3.0

    # Build two orthogonal in-plane axes for the display quad
    # Primary in-plane axis: inter-tragion direction (roughly Y)
    u = p3 - p2  # right_tragion → left_tragion
    u_len = np.linalg.norm(u)
    if u_len < 1e-10:
        u = np.array([0.0, 1.0, 0.0])  # fallback if tragions coincide
    else:
        u /= u_len

    # Secondary in-plane axis: perpendicular to both normal and u (roughly Z)
    w = np.cross(normal, u)
    w /= np.linalg.norm(w)

    # Four corners of the display rectangle
    c0 = center - half_size * u - half_size * w  # bottom-left
    c1 = center + half_size * u - half_size * w  # bottom-right
    c2 = center + half_size * u + half_size * w  # top-right
    c3 = center - half_size * u + half_size * w  # top-left

    vertices = np.array([c0, c1, c2, c3], dtype=np.float32)
    faces    = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    return normal, center, vertices, faces


# ── Plane–mesh intersection ───────────────────────────────────────────────────

def compute_plane_mesh_intersection(
    mesh: o3d.geometry.TriangleMesh,
    normal: np.ndarray,
    d: float,
    color: list[float] | None = None,
    eps: float = 1e-8,
) -> o3d.geometry.LineSet:
    """Compute the polyline where a plane cuts a triangle mesh.

    For each mesh triangle the function evaluates the signed distance of its
    three vertices to the plane (ax + by + cz + d = 0).  A triangle is
    intersected if and only if at least one vertex lies strictly on each side.
    The two edge-crossing points are stored as a line segment.

    The result is an Open3D LineSet ready to be added to any visualizer
    alongside the original mesh.

    Robustness notes
    ────────────────
    - Vertices whose distance to the plane is below ``eps`` are snapped to the
      plane (signed distance set to 0.0).  This avoids spurious duplicate
      segments at near-tangent triangles while keeping exact intersections.
    - Triangles fully on the plane (all three distances ≈ 0) are skipped;
      they do not contribute intersection segments.
    - Triangles tangent at exactly one edge (two vertices on the plane) yield
      that edge as the intersection segment — geometrically correct.

    Args:
        mesh:   Input TriangleMesh.  Must not be empty.
        normal: (3,) unit normal of the cutting plane.  Will be normalised
                internally if not already unit length.
        d:      Plane offset such that  normal · x + d = 0  defines the plane.
                Use  d = -normal · point_on_plane  to pass through a known point.
        color:  RGB list in [0, 1] for the LineSet.  Defaults to red [1, 0, 0].
        eps:    Snap-to-plane threshold in the same units as the mesh (mm).

    Returns:
        Open3D LineSet of the intersection segments, painted in ``color``.
        The LineSet may be empty if the plane does not intersect the mesh.

    Raises:
        ValueError: If the mesh is empty.
    """
    if mesh.is_empty():
        raise ValueError("Cannot intersect an empty mesh.")

    if color is None:
        color = [1.0, 0.0, 0.0]  # red

    n = np.asarray(normal, dtype=float)
    n_len = np.linalg.norm(n)
    if n_len < 1e-12:
        raise ValueError("Plane normal must be a non-zero vector.")
    n /= n_len

    vertices  = np.asarray(mesh.vertices,  dtype=float)   # (V, 3)
    triangles = np.asarray(mesh.triangles, dtype=np.int32) # (T, 3)

    # Signed distances of all vertices to the plane — vectorised
    dist = vertices @ n + d                                # (V,)
    dist[np.abs(dist) < eps] = 0.0                        # snap near-zero

    seg_points: list[np.ndarray] = []
    seg_lines:  list[list[int]]  = []

    _EDGES = ((0, 1), (1, 2), (2, 0))

    for tri in triangles:
        f = dist[tri]          # signed distances for the 3 vertices
        f0, f1, f2 = f

        # Skip triangles fully on one side or fully on the plane
        if f0 >= 0 and f1 >= 0 and f2 >= 0:
            continue
        if f0 <= 0 and f1 <= 0 and f2 <= 0:
            continue

        pts = vertices[tri]    # (3, 3) coordinates of the triangle vertices
        inter_pts: list[np.ndarray] = []

        for i, j in _EDGES:
            fi, fj = f[i], f[j]

            if fi == 0.0 and fj == 0.0:
                # Edge lies on the plane: add both endpoints once
                inter_pts.extend([pts[i], pts[j]])
                break          # whole edge is the intersection; stop here

            if fi * fj < 0:
                # Edge crosses the plane: linear interpolation
                t = fi / (fi - fj)
                inter_pts.append(pts[i] + t * (pts[j] - pts[i]))

            elif fi == 0.0:
                # Vertex i exactly on the plane
                inter_pts.append(pts[i])

            # fj == 0 is handled when that vertex appears as fi in another edge

        # Deduplicate (can happen when a vertex is exactly on the plane)
        unique: list[np.ndarray] = []
        for p in inter_pts:
            if not any(np.allclose(p, q, atol=eps) for q in unique):
                unique.append(p)

        if len(unique) == 2:
            idx = len(seg_points)
            seg_points.extend(unique)
            seg_lines.append([idx, idx + 1])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(seg_points),
        lines=o3d.utility.Vector2iVector(seg_lines),
    )
    line_set.paint_uniform_color(color)

    n_segs = len(seg_lines)
    print(f"[geometry] Plane–mesh intersection: {n_segs} segment(s) found.")
    return line_set

def compute_cranial_height(
    points: np.ndarray,
    normal: np.ndarray,
    plane_center: np.ndarray,
) -> float:
    """Compute the total cranial height above the Frankfurt plane.

    Height is measured as the maximum signed distance from the reference
    plane to any point in the cloud, along the plane normal direction.

    Args:
        points:       (N, 3) point cloud array.
        normal:       (3,) unit normal of the Frankfurt plane.
        plane_center: (3,) any point on the Frankfurt plane (e.g. centroid
                      of the three landmarks).

    Returns:
        Scalar height in the same units as the point cloud (mm).
    """
    d = -np.dot(normal, plane_center)
    signed_dists = points @ normal + d
    return float(signed_dists.max())


def compute_cranial_levels(
    normal: np.ndarray,
    plane_center: np.ndarray,
    cranial_height: float,
    n_levels: int = 10,
) -> list[dict]:
    """Compute N equally spaced cross-sectional planes above Frankfurt.

    Follows the standard methodology (Miyabayashi et al., Kwon et al.)
    dividing the cranium superior to the reference plane into N equal
    sections, where level 0 is Frankfurt and level N is the vertex.

    Args:
        normal:         (3,) unit normal of the Frankfurt plane.
        plane_center:   (3,) centroid of the Frankfurt plane.
        cranial_height: Total height of the cranium above Frankfurt (mm).
        n_levels:       Number of sections (default 10, standard in literature).

    Returns:
        List of dicts, one per level, each containing:
            level   : int   — level index (1..N)
            offset  : float — distance above Frankfurt in mm
            ratio   : float — proportion of total height (0..1)
            center  : (3,)  — point on the plane along the normal
            d       : float — plane equation offset (normal · x + d = 0)
    """
    levels = []
    for i in range(1, n_levels + 1):
        ratio  = i / n_levels
        offset = cranial_height * ratio
        center = plane_center + normal * offset
        d      = -np.dot(normal, center)
        levels.append({
            "level":  i,
            "offset": offset,
            "ratio":  ratio,
            "center": center,
            "d":      d,
        })
    return levels


def compute_contour_band(
    points: np.ndarray,
    normal: np.ndarray,
    plane_center: np.ndarray,
    band_half_width_mm: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    d            = -np.dot(normal, plane_center)
    signed_dists = points @ normal + d
    mask         = np.abs(signed_dists) <= band_half_width_mm
    return points[mask], mask

def compute_measurement_band(
    points: np.ndarray,
    normal: np.ndarray,
    plane_center: np.ndarray,
    band_half_width_mm: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    d            = -np.dot(normal, plane_center)
    signed_dists = points @ normal + d
    mask         = np.abs(signed_dists) <= band_half_width_mm
    return points[mask], mask

def compute_mesh_plane_intersection(
    mesh: o3d.geometry.TriangleMesh,
    normal: np.ndarray,
    plane_center: np.ndarray,
) -> np.ndarray:
    """Compute the contour where a plane intersects a triangle mesh.

    For each triangle, checks if it crosses the plane and computes the
    exact intersection points. Returns a polyline representing the contour.

    Args:
        mesh:         Input TriangleMesh.
        normal:       (3,) unit normal of the cutting plane.
        plane_center: (3,) any point on the cutting plane.

    Returns:
        (N, 3) array of contour points in sequence. Empty if no intersection.
    """
    vertices  = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    d            = -np.dot(normal, plane_center)
    signed_dists = vertices @ normal + d

    contour_pts = []
    _EDGES = ((0, 1), (1, 2), (2, 0))

    for tri in triangles:
        f = signed_dists[tri]
        f0, f1, f2 = f

        # Skip triangles on one side
        if (f0 >= 0 and f1 >= 0 and f2 >= 0) or (f0 <= 0 and f1 <= 0 and f2 <= 0):
            continue

        pts = vertices[tri]
        inter_pts = []

        for i, j in _EDGES:
            fi, fj = f[i], f[j]

            if fi == 0.0 and fj == 0.0:
                inter_pts.extend([pts[i], pts[j]])
                break

            if fi * fj < 0:
                t = fi / (fi - fj)
                inter_pts.append(pts[i] + t * (pts[j] - pts[i]))

            elif fi == 0.0:
                inter_pts.append(pts[i])

        # Deduplicate
        unique = []
        for p in inter_pts:
            if not any(np.allclose(p, q, atol=1e-8) for q in unique):
                unique.append(p)

        if len(unique) == 2:
            contour_pts.extend(unique)

    return np.array(contour_pts) if contour_pts else np.empty((0, 3))


def compute_local_coordinate_system(
    sellion: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute local X, Y, Z axes for a measurement plane.

    Z: plane normal (already provided)
    Y: direction from center to sellion, projected onto the plane
    X: perpendicular to Y and Z, completing right-handed system

    Args:
        sellion:      (3,) sellion landmark position.
        center:       (3,) center of the measurement plane.
        normal:       (3,) unit normal of the plane (Z axis).

    Returns:
        x_axis: (3,) unit X axis
        y_axis: (3,) unit Y axis
        z_axis: (3,) unit Z axis (same as normal)
    """
    # Z es la normal (ya normalizada)
    z_axis = normal / np.linalg.norm(normal)

    # Vector desde center hasta sellion
    to_sellion = sellion - center

    # Proyectar onto el plano (remover componente along normal)
    y_unnorm = to_sellion - np.dot(to_sellion, z_axis) * z_axis
    y_norm   = np.linalg.norm(y_unnorm)

    if y_norm < 1e-8:
        logger.log("Warning: sellion projects to plane center", LogLevel.WARNING)
        y_axis = np.array([1.0, 0.0, 0.0])
    else:
        y_axis = y_unnorm / y_norm

    # X perpendicular a Y y Z
    # Usar el orden correcto: X = Z × Y (no Y × Z)
    x_axis = np.cross(z_axis, y_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Verificar que es un sistema ortonormal correcto
    dot_xy = np.dot(x_axis, y_axis)
    dot_xz = np.dot(x_axis, z_axis)
    dot_yz = np.dot(y_axis, z_axis)

    if abs(dot_xy) > 1e-6 or abs(dot_xz) > 1e-6 or abs(dot_yz) > 1e-6:
        logger.log(
            f"Warning: non-orthogonal axes! dot(X,Y)={dot_xy:.6f}, dot(X,Z)={dot_xz:.6f}, dot(Y,Z)={dot_yz:.6f}",
            LogLevel.WARNING
        )

    return x_axis, y_axis, z_axis

def compute_cvai_intersection(
    contour: np.ndarray,
    plane_center: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    angle_deg: float = 30.0,
) -> dict:
    """Compute CVAI from diagonal intersections with cranial contour.
    
    Find exact intersections between diagonal lines (through plane_center at ±angle)
    and the contour polygon, using linear interpolation between contour points.
    """
    import math

    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    diagonals_data = []

    for sign in [1, -1]:
        # Dirección diagonal ideal
        diagonal_dir = cos_a * y_axis + sign * sin_a * x_axis
        diagonal_dir = diagonal_dir / np.linalg.norm(diagonal_dir)

        intersections = []
        
        # Buscar intersecciones con cada segmento del contorno
        for i in range(len(contour)):
            p1 = contour[i]
            p2 = contour[(i + 1) % len(contour)]
            
            # Línea diagonal: L(s) = plane_center + s * diagonal_dir
            # Segmento: P(u) = p1 + u * (p2 - p1), u ∈ [0, 1]
            # Encontrar intersección resolviendo: L(s) = P(u)
            
            edge = p2 - p1
            rhs = p1 - plane_center
            
            # Sistema: s * diagonal_dir - u * edge = -rhs
            # Forma matricial: [diagonal_dir | -edge] * [s; u] = rhs
            
            A = np.column_stack([diagonal_dir, -edge])
            
            try:
                # Resolver por mínimos cuadrados
                sol, residuals, rank, s_vals = np.linalg.lstsq(A, rhs, rcond=None)
                s, u = sol[0], sol[1]
                
                # Verificar si u está en [0, 1] (dentro del segmento)
                if -0.01 <= u <= 1.01:
                    u = np.clip(u, 0, 1)
                    intersection = plane_center + s * diagonal_dir
                    
                    # PROYECTAR al punto del contorno más cercano
                    distances_to_contour = np.linalg.norm(contour - intersection, axis=1)
                    nearest_idx = np.argmin(distances_to_contour)
                    intersection_snapped = contour[nearest_idx]
                    
                    # Recalcular s con el punto snapped
                    s_snapped = np.dot(intersection_snapped - plane_center, diagonal_dir)
                    
                    intersections.append({
                        "point": intersection_snapped,
                        "s": s_snapped,
                        "u": u,
                        "segment_idx": i,
                    })

            except np.linalg.LinAlgError:
                continue
            
        if len(intersections) < 2:
            logger.log(f"Warning: diagonal {sign:+d}30° found {len(intersections)} intersections", LogLevel.WARNING)
            continue
        
        # Ordenar por s (componente along diagonal)
        intersections.sort(key=lambda x: x["s"])
        
        pt_neg = intersections[0]["point"]
        pt_pos = intersections[-1]["point"]
        
        proj_neg = intersections[0]["s"]
        proj_pos = intersections[-1]["s"]
        
        diagonals_data.append({
            "pt_neg": pt_neg,
            "pt_pos": pt_pos,
            "dist_neg": abs(proj_neg),
            "dist_pos": proj_pos,
            "total_dist": abs(proj_neg) + proj_pos,
        })

    if len(diagonals_data) != 2:
        return None

    d1 = diagonals_data[0]["total_dist"]
    d2 = diagonals_data[1]["total_dist"]
    ratio = min(d1, d2) / max(d1, d2)  # ← AÑADE ESTA LÍNEA

    return {
        "dist1": d1,
        "dist2": d2,
        "ratio": ratio,  # ← AÑADE ESTO
        "diag1_pts": (diagonals_data[0]["pt_neg"], diagonals_data[0]["pt_pos"]),
        "diag2_pts": (diagonals_data[1]["pt_neg"], diagonals_data[1]["pt_pos"]),
    }
