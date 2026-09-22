"""
calibrate_ear_axes.py
----------------------
Herramienta de calibración (uso puntual, sobre UN modelo).

Recorta el parche local de cada oreja (tal y como lo hace ahora
landmark_detector.py) y lo guarda junto con un sistema de ejes de color
centrado en la localización gruesa, para poder identificar visualmente:

  - qué eje local corresponde a "arriba/abajo" de la oreja (vertical), y
  - qué eje corresponde a "delante/detrás" (profundidad: hacia la muesca
    del trago, frente al borde posterior del pabellón).

Convención de color del sistema de ejes de Open3D: rojo = X, verde = Y,
azul = Z.

Con esa información puedes fijar con confianza, en landmark_detector.py:
  RIGHT_EAR_LOCAL_TARGET = np.array([eje_x, eje_y, eje_z])
  LEFT_EAR_LOCAL_TARGET  = np.array([eje_x, eje_y, eje_z])
(cada componente en [0,1], relativa al bbox del propio parche: 0 = mínimo
de ese eje dentro del parche, 1 = máximo, 0.5 = mitad).

Uso:
  python calibrate_ear_axes.py <ruta_a_un_modelo.obj/.ply/.stl>

Salida (en la misma carpeta que el modelo):
  <model>_ear_patch_right.ply
  <model>_ear_patch_left.ply
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d


from src.siv.processing.landmark_detector import (
    EAR_PATCH_RADIUS_RATIO,
    LEFT_TRAGION_TARGET,
    RIGHT_TRAGION_TARGET,
    _crop_local_patch,
    _nearest_vertex,
    _normalize_vertices,
)


def save_ear_patch(
    mesh: o3d.geometry.TriangleMesh,
    target: np.ndarray,
    out_path: Path,
) -> None:

    vertices = np.asarray(mesh.vertices)

    bbox = mesh.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())

    normalized, _, _ = _normalize_vertices(vertices)
    coarse_idx = _nearest_vertex(normalized, target)
    coarse_position = vertices[coarse_idx]

    radius = diag * EAR_PATCH_RADIUS_RATIO
    patch_indices = _crop_local_patch(vertices, coarse_position, radius)

    if len(patch_indices) == 0:
        print(f"  Aviso: parche vacío para target {target}, se omite.")
        return

    patch_cloud = o3d.geometry.PointCloud()
    patch_cloud.points = o3d.utility.Vector3dVector(vertices[patch_indices])
    patch_cloud.paint_uniform_color([0.6, 0.6, 0.6])

    # Sistema de ejes de color, centrado en la localización gruesa,
    # con un tamaño proporcional al parche para que sea visible junto a él.
    axis_size = radius * 0.8
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size,
        origin=coarse_position,
    )
    axes_cloud = axes.sample_points_uniformly(3000)

    combined = patch_cloud + axes_cloud
    o3d.io.write_point_cloud(str(out_path), combined)

    print(f"  Guardado: {out_path}  ({len(patch_indices)} vértices en el parche)")


def main() -> None:

    if len(sys.argv) != 2:
        print("Uso: python calibrate_ear_axes.py <ruta_a_un_modelo>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    mesh = o3d.io.read_triangle_mesh(str(model_path))

    print(f"Modelo: {model_path.name}")

    save_ear_patch(
        mesh,
        RIGHT_TRAGION_TARGET,
        model_path.with_name(model_path.stem + "_ear_patch_right.ply"),
    )
    save_ear_patch(
        mesh,
        LEFT_TRAGION_TARGET,
        model_path.with_name(model_path.stem + "_ear_patch_left.ply"),
    )

    print("\nAbre los .ply generados en un visor 3D (MeshLab, CloudCompare...).")
    print("Sistema de ejes: rojo = X, verde = Y, azul = Z.")
    print("Identifica qué eje es vertical (arriba/abajo de la oreja) y cuál")
    print("es de profundidad (muesca del trago vs. borde posterior del pabellón),")
    print("y fija RIGHT_EAR_LOCAL_TARGET / LEFT_EAR_LOCAL_TARGET en consecuencia.")


if __name__ == "__main__":
    main()
