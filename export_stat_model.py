"""
export_landmarks_txt.py
-------------------------
Para cada modelo .obj de una carpeta:
  1. Detecta sus landmarks (sellion, right tragion, left tragion) con
     landmark_detector.detect_landmarks.
  2. Guarda un .txt con esas coordenadas, una línea por landmark en ese
     mismo orden ("X Y Z"), con el mismo nombre base que el .obj.
  3. Copia el .obj original sin modificar a la misma carpeta de salida,
     también con el mismo nombre.

Uso:
  python export_landmarks_txt.py <carpeta_modelos> <carpeta_salida>

Salida (en carpeta_salida):
  <model>.obj   (copia sin modificar del original)
  <model>.txt   (3 líneas: sellion / right tragion / left tragion)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from colorama import Fore, Style

from src.siv.processing.landmark_detector  import detect_landmarks


def _format_point(point: np.ndarray) -> str:
    return f"{point[0]} {point[1]} {point[2]}"


def export_model(obj_path: Path, out_dir: Path) -> None:

    mesh = o3d.io.read_triangle_mesh(str(obj_path))
    landmarks = detect_landmarks(mesh)

    # Orden fijo: sellion, right tragion, left tragion.
    lines = [
        _format_point(landmarks["sellion"]),
        _format_point(landmarks["right tragion"]),
        _format_point(landmarks["left tragion"]),
    ]

    txt_path = out_dir / (obj_path.stem + ".txt")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    mesh.compute_vertex_normals()
    
    stl_path = out_dir / (obj_path.stem + ".stl")
    success = o3d.io.write_triangle_mesh(
        str(stl_path), 
        mesh, 
        write_ascii=False,
    )

    if not success:
        raise RuntimeError(f"No se pudo escribir el STL: {stl_path}")


def main() -> None:

    if len(sys.argv) != 3:
        print("Uso: python export_landmarks_txt.py <carpeta_modelos> <carpeta_salida>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    obj_files = sorted(input_dir.glob("*.obj"))
    num_files = len(obj_files)

    print(f"\n{Fore.CYAN}{'=' * 68}{Style.RESET_ALL}")
    print(f"{Fore.CYAN} Exportación de landmarks (sellion / right tragion / left tragion){Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 68}{Style.RESET_ALL}")
    print(f"Entrada : {input_dir}")
    print(f"Salida  : {output_dir}")
    print(f"Modelos : {num_files}\n")

    ok = 0

    for i, obj_path in enumerate(obj_files, start=1):
        print(f"{Fore.CYAN}[{i}/{num_files}]{Style.RESET_ALL} {obj_path.name}", end=" ... ", flush=True)

        try:
            export_model(obj_path, output_dir)
            ok += 1
            print(f"{Fore.GREEN}OK{Style.RESET_ALL}")

        except Exception as exc:
            print(f"{Fore.RED}FALLO: {exc}{Style.RESET_ALL}")
            continue

    print(f"\n{Fore.GREEN}Listo.{Style.RESET_ALL} {ok}/{num_files} modelos exportados a {output_dir}")


if __name__ == "__main__":
    main()
