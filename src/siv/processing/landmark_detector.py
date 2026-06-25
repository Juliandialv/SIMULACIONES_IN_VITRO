"Automatic Landmark Detection Algorithm"
from __future__ import annotations

import numpy as np
import open3d as o3d


# ==========================================================
# Coordenadas anatómicas medias obtenidas del dataset
# ==========================================================

SELLION_TARGET = np.array([
    0.90,
    0.50,
    0.48,
])

RIGHT_TRAGION_TARGET = np.array([
    0.40,
    0.06,
    0.40,
])

LEFT_TRAGION_TARGET = np.array([
    0.40,
    0.94,
    0.40,
])


# ==========================================================
# Helpers
# ==========================================================

def _normalize_vertices(vertices: np.ndarray):
    """
    Convierte coordenadas a [0,1] dentro del bounding box.
    """

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)

    span = maxs - mins
    span[span == 0.0] = 1.0

    normalized = (vertices - mins) / span

    return normalized, mins, maxs


def _nearest_vertex(
    normalized_vertices: np.ndarray,
    target: np.ndarray,
) -> int:
    """
    Devuelve el índice del vértice más cercano
    a una coordenada normalizada objetivo.
    """

    distances = np.linalg.norm(
        normalized_vertices - target,
        axis=1,
    )

    return int(np.argmin(distances))


# ==========================================================
# Detectores
# ==========================================================

def detect_sellion(
    vertices: np.ndarray,
) -> np.ndarray:

    normalized, _, _ = _normalize_vertices(vertices)

    idx = _nearest_vertex(
        normalized,
        SELLION_TARGET,
    )

    return vertices[idx]


def detect_right_tragion(
    vertices: np.ndarray,
) -> np.ndarray:

    normalized, _, _ = _normalize_vertices(vertices)

    idx = _nearest_vertex(
        normalized,
        RIGHT_TRAGION_TARGET,
    )

    return vertices[idx]


def detect_left_tragion(
    vertices: np.ndarray,
) -> np.ndarray:

    normalized, _, _ = _normalize_vertices(vertices)

    idx = _nearest_vertex(
        normalized,
        LEFT_TRAGION_TARGET,
    )

    return vertices[idx]


# ==========================================================
# API pública
# ==========================================================

def detect_landmarks(
    mesh: o3d.geometry.TriangleMesh,
) -> dict[str, np.ndarray]:

    vertices = np.asarray(mesh.vertices)

    sellion = detect_sellion(vertices)

    right_tragion = detect_right_tragion(vertices)

    left_tragion = detect_left_tragion(vertices)

    return {
        "sellion": sellion,
        "right tragion": right_tragion,
        "left tragion": left_tragion,
    }
