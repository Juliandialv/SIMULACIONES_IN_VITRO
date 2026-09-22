"""Script for executing SIV in bash"""

from pathlib import Path
import csv
import time

from colorama import Fore, Style
import open3d as o3d
import numpy as np

from src.siv.processing.landmark_detector import detect_landmarks
from src.siv.processing.geometry import (compute_cranial_height,
                                         compute_cranial_levels,
                                         compute_reference_plane,
                                         compute_mesh_plane_intersection,
                                         compute_local_coordinate_system,
                                         compute_cvai_intersection)

from src.siv.processing.config import (EAR_CLEARANCE_RATIO,
                                       N_CRANIAL_LEVELS,
                                       POISSON_DEPTH,
                                       DENSITY_QUANTILE)

from src.siv.processing.reconstruction import (clip_cloud_above_plane,
                                               reconstruct_cranial_vault,
                                               clip_mesh_by_plane)

from src.siv.processing.uncertainty import (perturb_landmarks,
                                            summarise_cvai_distribution,
                                            SIGMA_BY_LANDMARK)

project_path = Path(__file__).resolve().parent
data_path = project_path / "data" / "raw" / "modelsSynth" / "P_sm"

# ── Output files ──────────────────────────────────────────────────────────────
results_path = project_path / "results"
results_path.mkdir(parents=True, exist_ok=True)

CVAI_SUMMARY_CSV = results_path / "cvai_uncertainty_summary.csv"
CVAI_SAMPLES_CSV = results_path / "cvai_uncertainty_samples.csv"

def compute_cranial_planes(mesh, landamrks):
    if not all(k in landamrks for k in ("sellion", "right tragion", "left tragion")):
        print(f"{Fore.YELLOW}Insufficient landmarks to define the reference plane.{Style.RESET_ALL}")
        return

    # Plano de Frankfurt
    normal, center, _, _ = compute_reference_plane(
        sellion       = landamrks["sellion"],
        right_tragion = landamrks["right tragion"],
        left_tragion  = landamrks["left tragion"],
    )
    frankfurt_normal = normal
    frankfurt_center = center

    if normal[2] < 0:
        normal = -normal
        frankfurt_normal = normal

    points = np.asarray(mesh.vertices)

    # ── Altura craneal total ─────────────────────────────────────────────
    cranial_height = compute_cranial_height(points, normal, center)

    # ── 10 niveles equiespaciados ────────────────────────────────────────
    cranial_levels = compute_cranial_levels(
        normal, center, cranial_height, n_levels=N_CRANIAL_LEVELS
    )

    offset_m_plane_mm = next(item['offset'] for item in cranial_levels if 
                            item['level'] == 3)   

    # ── Plano de corte inicial ───────────────────────────────────────────
    offset_mm  = cranial_height * EAR_CLEARANCE_RATIO
    cut_center = center + normal * offset_mm

    # ── Reconstrucción de la bóveda craneal ──────────────────────────────
    points_down = np.asarray(mesh.vertices)
    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(points_down)

    pcd_clipped = clip_cloud_above_plane(
        pcd          = pcd_down,
        normal       = normal,
        plane_center = cut_center,
    )

    mesh = reconstruct_cranial_vault(
        pcd              = pcd_clipped,
        poisson_depth    = POISSON_DEPTH,
        density_quantile = DENSITY_QUANTILE,
    )

    mesh = clip_mesh_by_plane(
        mesh         = mesh,
        normal       = normal,
        plane_center = cut_center,
    )

    cranial_mesh = mesh

    return frankfurt_normal, frankfurt_center, cranial_levels, cranial_mesh

def calculate_measurement_data(mesh, landmarks, frankfurt_normal, frankfurt_center, cranial_levels, cranial_mesh):
    """Calcula plane_center, contour y ejes locales para el nivel 3."""
    
    tragion_mid = (landmarks["right tragion"] + landmarks["left tragion"]) / 2
    normal = frankfurt_normal
    level_3 = cranial_levels[2]
    level_3_plane_point = level_3["center"]
    
    numerator = np.dot(tragion_mid - level_3_plane_point, normal)
    denominator = np.dot(normal, normal)
    t = -numerator / denominator
    
    plane_center = tragion_mid + t * normal
    
    contour = compute_mesh_plane_intersection(
        mesh         = cranial_mesh,
        normal       = normal,
        plane_center = plane_center,
    )

    x_axis, y_axis, z_axis = compute_local_coordinate_system(
        sellion = landmarks["sellion"],
        center  = plane_center,
        normal  = normal,
    )

    return plane_center, contour, x_axis, y_axis, z_axis

def main():
    """Main function"""
    num_files = sum(1 for _ in data_path.iterdir())

    print(f"\n{Fore.CYAN}{'=' * 68}{Style.RESET_ALL}")
    print(f"{Fore.CYAN} SIV - Cranial Vault Analysis{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 68}{Style.RESET_ALL}")
    print(f"Data path : {data_path}")
    print(f"Results   : {results_path}")
    print(f"Files     : {num_files}\n")

    # CSV de resumen: una fila por modelo y nivel
    summary_fields = [
        "model", "level", "offset_mm", "nominal_cvai",
        "mean", "std", "p5", "p95", "cv",
        "n_valid", "n_invalid", "n_samples"
    ]

    # CSV de muestras: una fila por realización y nivel
    sample_fields = ["model", "level", "sample", "cvai"]

    # Se escriben las cabeceras al comenzar una ejecución nueva.
    with open(CVAI_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f_summary, \
         open(CVAI_SAMPLES_CSV, "w", newline="", encoding="utf-8") as f_samples:

        summary_writer = csv.DictWriter(f_summary, fieldnames=summary_fields)
        sample_writer = csv.DictWriter(f_samples, fieldnames=sample_fields)

        summary_writer.writeheader()
        sample_writer.writeheader()

        for i, file in enumerate(data_path.iterdir(), start=1):
            t_start = time.perf_counter()

            print(f"{Fore.CYAN}{'─' * 68}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}[{i}/{num_files}] Processing: {file.stem}{Style.RESET_ALL}")

            # Loading mesh
            print("  ├─ Loading mesh...", end=" ", flush=True)
            mesh = o3d.io.read_triangle_mesh(file)
            print(f"{Fore.GREEN}OK{Style.RESET_ALL}")

            # Automatic landmark detection
            print("  ├─ Detecting landmarks...", end=" ", flush=True)
            landmarks = detect_landmarks(mesh)
            print(f"{Fore.GREEN}OK{Style.RESET_ALL}")

            # Compute cranial indices
            print("  ├─ Computing cranial planes...", end=" ", flush=True)
            frankfurt_normal, frankfurt_center, cranial_levels, cranial_mesh = compute_cranial_planes(mesh, landmarks)
            print(f"{Fore.GREEN}OK{Style.RESET_ALL}")

            print("  ├─ Computing nominal CVAI...", end=" ", flush=True)
            plane_center, contour, x_axis, y_axis, z_axis = calculate_measurement_data(
                mesh, landmarks, frankfurt_normal, frankfurt_center,
                cranial_levels, cranial_mesh
            )

            normal = frankfurt_normal
            cvai_results = []

            for level_idx in range(2, 9):  # índices 2-9 = niveles 3-10
                lvl = cranial_levels[level_idx]

                contour_lvl = compute_mesh_plane_intersection(
                    mesh=cranial_mesh,
                    normal=normal,
                    plane_center=lvl["center"],
                )

                if len(contour_lvl) < 10:
                    print(
                        f"\n{Fore.YELLOW}      Level {lvl['level']}: "
                        f"insufficient contour points.{Style.RESET_ALL}"
                    )
                    continue

                x_axis_lvl, y_axis_lvl, z_axis_lvl = compute_local_coordinate_system(
                    sellion=landmarks["sellion"],
                    center=lvl["center"],
                    normal=normal,
                )

                cvai_data = compute_cvai_intersection(
                    contour=contour_lvl,
                    plane_center=lvl["center"],
                    x_axis=x_axis_lvl,
                    y_axis=y_axis_lvl,
                )

                if cvai_data:
                    if cvai_data['dist1'] >= cvai_data['dist2']:
                        cvai_s = np.abs(cvai_data['dist1'] - cvai_data['dist2']) / (cvai_data['dist2']) * 100
                        cvai_l = np.abs(cvai_data['dist1'] - cvai_data['dist2']) / (cvai_data['dist1']) * 100
                    else:
                        cvai_s = np.abs(cvai_data['dist1'] - cvai_data['dist2']) / (cvai_data['dist1']) * 100
                        cvai_l = np.abs(cvai_data['dist1'] - cvai_data['dist2']) / (cvai_data['dist2']) * 100

                    cvai_results.append({
                        "level": lvl['level'],
                        "offset_mm": lvl['offset'],
                        "ratio": cvai_data['ratio'],
                        "cvai_shorter": cvai_s,
                        "cvai_longer": cvai_l,
                        "dist1": cvai_data['dist1'],
                        "dist2": cvai_data['dist2'],
                    })

            print(f"{Fore.GREEN}OK{Style.RESET_ALL}")

            cvai_results_nominal = cvai_results

            # Pre-calcular contornos nominales una sola vez por nivel
            nominal_contours = {}
            n_samples = 500

            for level_idx in range(2, 10):
                lvl = cranial_levels[level_idx]
                contour = compute_mesh_plane_intersection(
                    mesh=cranial_mesh,
                    normal=frankfurt_normal,
                    plane_center=lvl["center"],
                )
                nominal_contours[level_idx] = contour

            # Generar N realizaciones perturbadas de los landmarks
            print(
                f"  ├─ Running uncertainty analysis "
                f"({n_samples} samples/level)..."
            )

            samples = perturb_landmarks(
                landmarks=landmarks,
                mesh=cranial_mesh,
                n_samples=n_samples,
                sigma_errors=SIGMA_BY_LANDMARK,
                seed=42,
            )

            results_by_level: dict[int, dict] = {}

            for level_idx in range(2, 10):
                lvl = cranial_levels[level_idx]
                contour = nominal_contours[level_idx]

                if len(contour) < 10:
                    print(
                        f"  │  {Fore.YELLOW}Level {lvl['level']}: "
                        f"insufficient contour points.{Style.RESET_ALL}"
                    )
                    continue

                cvai_list = []
                n_invalid = 0

                for sample_idx, sample_lm in enumerate(samples):
                    try:
                        s_normal, s_center, _, _ = compute_reference_plane(
                            sellion=sample_lm["sellion"],
                            right_tragion=sample_lm["right tragion"],
                            left_tragion=sample_lm["left tragion"],
                        )

                        if s_normal[2] < 0:
                            s_normal = -s_normal

                    except ValueError:
                        n_invalid += 1
                        continue

                    x_ax, y_ax, _ = compute_local_coordinate_system(
                        sellion=sample_lm["sellion"],
                        center=lvl["center"],
                        normal=s_normal,
                    )

                    # Ordenar contorno nominal con los ejes perturbados
                    angles = np.arctan2(
                        np.dot(contour - lvl["center"], x_ax),
                        np.dot(contour - lvl["center"], y_ax),
                    )

                    contour_sorted = contour[np.argsort(angles)]

                    cvai_data = compute_cvai_intersection(
                        contour=contour_sorted,
                        plane_center=lvl["center"],
                        x_axis=x_ax,
                        y_axis=y_ax,
                    )

                    if cvai_data:
                        d1, d2 = cvai_data["dist1"], cvai_data["dist2"]

                        if min(d1, d2) < 0.1:
                            n_invalid += 1
                            continue

                        cvai_s = abs(d1 - d2) / min(d1, d2) * 100
                        cvai_list.append(cvai_s)

                        # Guardar cada realización individual
                        sample_writer.writerow({
                            "model": file.stem,
                            "level": lvl["level"],
                            "sample": sample_idx,
                            "cvai": cvai_s,
                        })

                if cvai_list:
                    stats = summarise_cvai_distribution(np.array(cvai_list))
                    results_by_level[lvl["level"]] = stats

                    # Guardar resumen del nivel
                    nominal_map = {
                        r["level"]: r["cvai_shorter"]
                        for r in cvai_results_nominal
                    }

                    summary_writer.writerow({
                        "model": file.stem,
                        "level": lvl["level"],
                        "offset_mm": lvl["offset"],
                        "nominal_cvai": nominal_map.get(lvl["level"], float("nan")),
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "p5": stats["p5"],
                        "p95": stats["p95"],
                        "cv": stats["cv"],
                        "n_valid": len(cvai_list),
                        "n_invalid": n_invalid,
                        "n_samples": n_samples,
                    })

                print(
                    f"  │  Level {lvl['level']:02d}: "
                    f"{len(cvai_list)}/{n_samples} valid"
                )

            f_summary.flush()
            f_samples.flush()

            # Resumen en terminal
            print(f"\n  {Fore.GREEN}CVAI Analysis Summary (Levels 3-10):{Style.RESET_ALL}")

            nominal_map = {
                r["level"]: r["cvai_shorter"]
                for r in cvai_results_nominal
            }

            print(
                f"  {'Lvl':>4} {'Nominal':>10} {'Mean':>10} "
                f"{'Std':>10} {'P5':>10} {'P95':>10} {'CV%':>10}"
            )

            for level_num, stats in sorted(results_by_level.items()):
                nom = nominal_map.get(level_num, float("nan"))
                print(
                    f"  {level_num:>4} {nom:>10.2f} "
                    f"{stats['mean']:>10.2f} {stats['std']:>10.2f} "
                    f"{stats['p5']:>10.2f} {stats['p95']:>10.2f} "
                    f"{stats['cv']:>10.1f}"
                )

            elapsed = time.perf_counter() - t_start

            print(f"\n  {Fore.GREEN}Results saved:{Style.RESET_ALL}")
            print(f"  ├─ {CVAI_SUMMARY_CSV}")
            print(f"  └─ {CVAI_SAMPLES_CSV}")
            print(f"  Processing time: {elapsed:.2f} s")

    print(f"\n{Fore.CYAN}{'=' * 68}{Style.RESET_ALL}")
    print(f"{Fore.GREEN} Analysis completed successfully.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 68}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
