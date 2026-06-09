"""Landmark uncertainty modelling — Class 1 (explicit anatomical landmarks).

For each landmark, the position error is modelled as a 3-D anisotropic
Gaussian:
  - σ_normal    : small, along the surface normal  (probe/cursor stays on skin)
  - σ_tangential: larger, along the surface tangent (sliding to find the point)

The local surface normal at each landmark is estimated from the nearest
vertices of the cranial mesh, so the perturbation is always geometrically
consistent with the surface.

Provisional σ values (to be replaced once the systematic review is complete):
  σ_normal     = 0.5 mm
  σ_tangential = 1.5 mm
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


# ── Provisional σ values (Class 1) ───────────────────────────────────────────
# Normal is related to the perpendicular to surface error while tangential is 
# the one along surface in the landmark search
SIGMA_BY_LANDMARK = {
    "sellion":       {"sigma_normal": 0.5, "sigma_tangential": 1.0},
    "right tragion": {"sigma_normal": 0.5, "sigma_tangential": 2.0},
    "left tragion":  {"sigma_normal": 0.5, "sigma_tangential": 2.0},
}

# Number of nearest vertices used to estimate the local surface normal
NORMAL_ESTIMATION_K = 10

# ── Core: local normal estimation ────────────────────────────────────────────

def estimate_surface_normal_at_point(
    point: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    k: int = NORMAL_ESTIMATION_K,
) -> np.ndarray:
    """Estimate the outward surface normal at a point by averaging the normals
    of its k nearest mesh vertices.

    Args:
        point : (3,) query position (the picked landmark).
        mesh  : cranial vault TriangleMesh (must have vertex normals).
        k     : number of nearest vertices to average.

    Returns:
        (3,) unit normal vector, oriented outward (away from mesh centroid).
    """
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)   # (V, 3)
    normals  = np.asarray(mesh.vertex_normals)  # (V, 3)

    # k-nearest vertices by Euclidean distance
    dists = np.linalg.norm(vertices - point, axis=1)
    idx   = np.argpartition(dists, min(k, len(dists) - 1))[:k]

    avg_normal = normals[idx].mean(axis=0)
    norm_len   = np.linalg.norm(avg_normal)

    if norm_len < 1e-10:
        # Fallback: vector from mesh centroid to point
        centroid   = vertices.mean(axis=0)
        avg_normal = point - centroid
        norm_len   = np.linalg.norm(avg_normal)

    avg_normal /= norm_len

    # Ensure outward orientation
    centroid = vertices.mean(axis=0)
    if np.dot(avg_normal, point - centroid) < 0:
        avg_normal = -avg_normal

    return avg_normal


def _build_tangent_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build two unit tangent vectors orthogonal to *normal*.

    Uses the Gram-Schmidt approach with a stable fallback axis.

    Returns:
        t1, t2 : two orthogonal unit vectors spanning the tangent plane.
    """
    # Choose a reference vector not parallel to normal
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    t1 = ref - np.dot(ref, normal) * normal
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    t2 /= np.linalg.norm(t2)

    return t1, t2


# ── Core: single-landmark perturbation ───────────────────────────────────────

def perturb_landmark(
    point: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    n_samples: int,
    sigma_normal: float,
    sigma_tangential: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate *n_samples* perturbed positions for one landmark.

    The perturbation is decomposed in the local frame of the surface:
        Δp = ε_n * normal + ε_t1 * t1 + ε_t2 * t2
    where ε_n ~ N(0, σ_normal²) and ε_t1, ε_t2 ~ N(0, σ_tangential²).

    Args:
        point            : (3,) nominal landmark position.
        mesh             : cranial vault mesh (for local normal estimation).
        n_samples        : number of Monte-Carlo samples to generate.
        sigma_normal     : std-dev of error along the surface normal (mm).
        sigma_tangential : std-dev of error along the surface tangent (mm).
        rng              : numpy random Generator; created fresh if None.

    Returns:
        (n_samples, 3) array of perturbed landmark positions.
    """
    if rng is None:
        rng = np.random.default_rng()

    normal  = estimate_surface_normal_at_point(point, mesh)
    t1, t2  = _build_tangent_frame(normal)

    # Sample scalar errors in local frame
    eps_n  = rng.normal(0.0, sigma_normal,     size=n_samples)  # (N,)
    eps_t1 = rng.normal(0.0, sigma_tangential, size=n_samples)  # (N,)
    eps_t2 = rng.normal(0.0, sigma_tangential, size=n_samples)  # (N,)

    # Reconstruct 3-D displacements: (N,3)
    deltas = (
        eps_n [:, None] * normal +
        eps_t1[:, None] * t1     +
        eps_t2[:, None] * t2
    )

    return point + deltas  # (n_samples, 3)


# ── Public API: perturb all landmarks at once ─────────────────────────────────

def perturb_landmarks(
    landmarks: dict[str, np.ndarray],
    mesh: o3d.geometry.TriangleMesh,
    n_samples: int,
    sigma_errors: SIGMA_BY_LANDMARK,
    seed: int | None = None,
) -> list[dict[str, np.ndarray]]:
    """Generate *n_samples* jointly-perturbed landmark sets.

    Each sample is a dict with the same keys as *landmarks* but with
    all positions independently displaced according to the Class-1 model.
    Independence between landmarks is intentional: the operator error at
    the sellion is uncorrelated with the error at the tragion.

    Args:
        landmarks        : {'sellion': (3,), 'right tragion': (3,), ...}
        mesh             : cranial vault mesh.
        n_samples        : number of Monte-Carlo realizations.
        sigma_errors     : {"sellion": "sigma normal": 0.5, "sigma tangencial": 1.0}, ...}
        seed             : optional RNG seed for reproducibility.

    Returns:
        List of *n_samples* dicts, each shaped like *landmarks*.
    """
    rng = np.random.default_rng(seed)

    # Pre-compute perturbed positions for every landmark: {name: (N,3)}
    perturbed_arrays: dict[str, np.ndarray] = {}
    for name, point in landmarks.items():
        perturbed_arrays[name] = perturb_landmark(
            point, mesh, n_samples, sigma_errors[name]["sigma_normal"], sigma_errors[name]["sigma_tangential"], rng
        )

    # Repack into a list of dicts — one dict per realisation
    samples: list[dict[str, np.ndarray]] = []
    for i in range(n_samples):
        samples.append({name: perturbed_arrays[name][i] for name in landmarks})

    return samples


# ── Convenience: summarise a CVAI distribution ───────────────────────────────

def summarise_cvai_distribution(cvai_values: np.ndarray) -> dict:
    """Compute descriptive statistics for a vector of CVAI realisations.

    Args:
        cvai_values : (N,) array of CVAI percentages.

    Returns:
        Dict with keys: mean, std, median, p5, p25, p75, p95, iqr, cv.
        cv (coefficient of variation) = std / mean * 100 if mean != 0.
    """
    v = np.asarray(cvai_values, dtype=float)
    v = v[np.isfinite(v)]   # drop any NaN/Inf from failed intersections

    if len(v) == 0:
        return {k: float("nan") for k in
                ("mean", "std", "median", "p5", "p25", "p75", "p95", "iqr", "cv", "n_valid")}

    p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
    mean = float(v.mean())
    std  = float(v.std())

    return {
        "n_valid": int(len(v)),
        "mean":    mean,
        "std":     std,
        "median":  float(p50),
        "p5":      float(p5),
        "p25":     float(p25),
        "p75":     float(p75),
        "p95":     float(p95),
        "iqr":     float(p75 - p25),
        "cv":      float(std / mean * 100) if mean != 0 else float("nan"),
    }

def compute_landmark_ellipse(
    name: str,
    point: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    sigma_errors: dict[str, dict[str, float]],
    n_std: float = 2.0,
    n_points: int = 64,
) -> np.ndarray:
    """Genera los vértices de la elipse de incertidumbre Clase 1 en el plano
    tangente al landmark.

    La elipse tiene semiejes:
        - n_std * sigma_tangential en las dos direcciones tangenciales (t1, t2)
        - n_std * sigma_normal     en la dirección normal (visualmente plana)

    Args:
        name             : landmark name ("sellion", ...)
        point            : (3,) posición nominal del landmark.
        mesh             : malla craneal (para estimar la normal local).
        sigma_errors     : {"sellion": "sigma normal": 0.5, "sigma tangencial": 1.0}, ...}
        n_std            : número de desviaciones típicas que define el radio.
        n_points         : número de vértices del polígono de la elipse.

    Returns:
        (n_points, 3) array con los vértices de la elipse en coordenadas 3D.
    """
    normal = estimate_surface_normal_at_point(point, mesh)
    t1, t2 = _build_tangent_frame(normal)

    angles   = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r_t1     = n_std * sigma_errors[name]["sigma_tangential"] * np.cos(angles)  # (n_points,)
    r_t2     = n_std * sigma_errors[name]["sigma_tangential"] * np.sin(angles)  # (n_points,)

    # Vértices en 3D: punto + desplazamiento en el plano tangente
    vertices = (
        point
        + r_t1[:, None] * t1
        + r_t2[:, None] * t2
    )  # (n_points, 3)

    return vertices
