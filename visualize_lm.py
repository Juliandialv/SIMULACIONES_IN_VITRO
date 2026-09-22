"""
visualize_landmarks.py
-----------------------
Genera una inspección visual rápida de la detección automática de landmarks
(sellion, right tragion, left tragion) sobre cada modelo craneal del dataset.

Para cada malla en `data_path`:
  1. Carga la malla y calcula los landmarks con `detect_landmarks`.
  2. Construye pequeñas esferas de color en la posición de cada landmark.
  3. Guarda una malla combinada (.ply) con el cráneo + landmarks, para
     inspección interactiva en un visor 3D (MeshLab, Open3D, CloudCompare...).
  4. Renderiza capturas desde tres ángulos (frontal, lateral, superior) como
     PNG, para poder repasar los 74 modelos rápidamente sin abrir cada
     archivo uno a uno.
  5. Genera una hoja de contacto HTML con las tres vistas de cada modelo.

Salidas (en results/landmark_preview/):
  <model>_landmarks.ply
  <model>_view_front.png
  <model>_view_side.png
  <model>_view_top.png
  landmark_preview_index.html
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
from colorama import Fore, Style

from src.siv.processing.landmark_detector import detect_landmarks

# ── Configuración ────────────────────────────────────────────────────────
project_path = Path(__file__).resolve().parent
data_path = project_path / "data" / "raw" / "modelsSynth" / "P_sm"

results_path = project_path / "results" / "landmark_preview"
results_path.mkdir(parents=True, exist_ok=True)

# Un color distinto por landmark, consistente en todas las figuras del estudio
LANDMARK_COLORS = {
    "sellion":       (0.90, 0.10, 0.10),   # rojo
    "right tragion": (0.10, 0.45, 0.90),   # azul
    "left tragion":  (0.10, 0.75, 0.30),   # verde
}

SPHERE_RADIUS_RATIO = 0.010   # radio de la esfera, relativo a la diagonal del bbox
MESH_COLOR = (0.75, 0.75, 0.75)

IMG_WIDTH = 900
IMG_HEIGHT = 700


# ==========================================================
# Construcción de la malla anotada
# ==========================================================

def _make_landmark_sphere(
    position: np.ndarray,
    radius: float,
    color: tuple[float, float, float],
) -> o3d.geometry.TriangleMesh:

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    sphere.translate(position)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()

    return sphere


def build_annotated_mesh(
    mesh: o3d.geometry.TriangleMesh,
    landmarks: dict[str, np.ndarray],
) -> o3d.geometry.TriangleMesh:
    """Combina la malla original (gris) con esferas de color en cada landmark."""

    mesh_vis = o3d.geometry.TriangleMesh(mesh)
    mesh_vis.paint_uniform_color(MESH_COLOR)
    mesh_vis.compute_vertex_normals()

    bbox = mesh_vis.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    radius = max(diag * SPHERE_RADIUS_RATIO, 1e-6)

    combined = mesh_vis
    for name, position in landmarks.items():
        color = LANDMARK_COLORS.get(name, (1.0, 1.0, 0.0))
        combined += _make_landmark_sphere(position, radius, color)

    return combined


# ==========================================================
# Renderizado offscreen
# ==========================================================

def render_views(mesh: o3d.geometry.TriangleMesh, out_prefix: Path) -> None:
    """
    Renderiza 3 vistas (frontal, lateral, superior) a PNG.

    Usa el visualizador "legacy" (ventana GLFW oculta) en lugar de
    o3d.visualization.rendering.OffscreenRenderer: este último exige un
    backend EGL headless que en Windows no está soportado por Open3D
    (ver error "EGL Headless is not supported on this platform"). La
    ventana oculta funciona igual sobre cualquier escritorio con
    GPU/driver de pantalla normal.
    """

    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    # front = dirección desde el punto observado hacia la cámara (normalizada)
    # Las tres vistas cubren donde están los landmarks: sellion se ve de frente,
    # right/left tragion se ven cada uno desde su lateral correspondiente.
    views = {
        "side_right":      ([0, -1, 0], [0, 0, 1]),
        "front": ([1, 0, 0],  [0, 0, 1]),
        "side_left":  ([0, 1, 0], [0, 0, 1]),
    }

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=IMG_WIDTH, height=IMG_HEIGHT)
    vis.add_geometry(mesh)

    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([1.0, 1.0, 1.0])
    render_opt.light_on = True

    ctr = vis.get_view_control()

    for view_name, (front, up) in views.items():
        ctr.set_lookat(center)
        ctr.set_front(front)
        ctr.set_up(up)
        ctr.set_zoom(0.7)

        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(
            str(out_prefix.parent / f"{out_prefix.name}_view_{view_name}.png"),
            do_render=True,
        )

    vis.destroy_window()


# ==========================================================
# Hoja de contacto HTML
# ==========================================================

def build_html_index(model_names: list[str], out_dir: Path) -> None:
    """Genera una hoja de contacto HTML para revisar los modelos de un vistazo."""

    rows = []
    for name in model_names:
        rows.append(
            f"<tr><td>{name}</td>"
            f"<td><img src='{name}_view_front.png' width='260'></td>"
            f"<td><img src='{name}_view_side_right.png' width='260'></td>"
            f"<td><img src='{name}_view_side_left.png' width='260'></td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Landmark preview</title>
<style>
  body {{ font-family: sans-serif; margin: 20px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ border: 1px solid #ccc; padding: 6px; text-align: center; }}
</style></head>
<body>
<h1>Inspección visual de landmarks automáticos</h1>
<p>Rojo = sellion &middot; Azul = right tragion &middot; Verde = left tragion</p>
<table>
<tr><th>Modelo</th><th>Frontal</th><th>Lateral derecha</th><th>Lateral izquierda</th></tr>
{''.join(rows)}
</table>
</body></html>"""

    (out_dir / "landmark_preview_index.html").write_text(html, encoding="utf-8")


# ==========================================================
# Main
# ==========================================================

def main() -> None:
    files = sorted(data_path.iterdir())
    num_files = len(files)

    print(f"\n{Fore.CYAN}{'=' * 68}{Style.RESET_ALL}")
    print(f"{Fore.CYAN} Landmark visual inspection{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 68}{Style.RESET_ALL}")
    print(f"Data path : {data_path}")
    print(f"Results   : {results_path}")
    print(f"Files     : {num_files}\n")

    model_names: list[str] = []

    for i, file in enumerate(files, start=1):
        print(f"{Fore.CYAN}[{i}/{num_files}]{Style.RESET_ALL} {file.stem}", end=" ... ", flush=True)

        try:
            mesh = o3d.io.read_triangle_mesh(str(file))
            landmarks = detect_landmarks(mesh)
            annotated = build_annotated_mesh(mesh, landmarks)

            ply_path = results_path / f"{file.stem}_landmarks.ply"
            o3d.io.write_triangle_mesh(str(ply_path), annotated)

            render_views(annotated, results_path / file.stem)

            model_names.append(file.stem)
            print(f"{Fore.GREEN}OK{Style.RESET_ALL}")

        except Exception as exc:  # queremos seguir con el resto del dataset aunque uno falle
            print(f"{Fore.RED}FALLO: {exc}{Style.RESET_ALL}")
            continue

    build_html_index(model_names, results_path)

    print(f"\n{Fore.GREEN}Listo.{Style.RESET_ALL} {len(model_names)}/{num_files} modelos procesados.")
    print(f"Abre {results_path / 'landmark_preview_index.html'} para revisar todos los modelos.")


if __name__ == "__main__":
    main()
