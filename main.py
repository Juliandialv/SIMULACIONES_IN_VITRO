import sys
import ctypes

import inspect
import numpy as np
import open3d as o3d
from enum import Enum
from datetime import datetime

from PySide6.QtCore import QSize, QObject, Signal
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QFileDialog
from PySide6.QtGui import QIcon, QAction

# Generado por pyside6-uic
from src.siv.ui.ventana_ui import Ui_MainWindow
from src.siv.utils.logger import logger, LogLevel
from src.siv.visualization.viewer import PointCloudViewer
from src.siv.processing.reconstruction import (
    clip_cloud_above_plane,
    reconstruct_cranial_vault,
    clip_mesh_by_plane,
)
from src.siv.processing.config import (
    EAR_CLEARANCE_RATIO,
    CONTOUR_BAND_MM,
    N_CRANIAL_LEVELS,
    POISSON_DEPTH,
    DENSITY_QUANTILE
)
from src.siv.processing.geometry import (
    compute_reference_plane,
    compute_cranial_height,
    compute_cranial_levels,
    compute_contour_band,
    compute_measurement_band
)

class LogLevel(Enum):
    INFO    = "#dcdcdc"
    SUCCESS = "#5CB85C"
    WARNING = "#f0ad4e"
    ERROR   = "#d9534f"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self._pcd_o3d      = None
        self._landmarks    = {}
        self._cranial_mesh = None

        # Viewport
        self.viewer = PointCloudViewer()
        layout = QVBoxLayout(self.ui.viewport_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.viewer)

        # Consola
        logger.message.connect(self._append_log)

        # Botones
        self.ui.qpushButton_landmarks.clicked.connect(self._start_landmark_picking)
        self.ui.qpushButton_compute.clicked.connect(self._compute_indices)
        self.ui.qpushButton_uncertainty.clicked.connect(lambda: self._run_uncertainty())
        self.ui.qpushButton_automatic.clicked.connect(self._run_automatic_landmarks)

        # Texto hint picking
        self.viewer.picking_hint.connect(self._update_picking_hint)

        # Toolbarc
        self._init_toolbar()

        # Stylesheet
        self._load_stylesheet("src/siv/resources/style.qss")

        logger.log("Application loaded successfully", LogLevel.SUCCESS)

    def _init_toolbar(self):
        from PySide6.QtWidgets import QToolBar
        from PySide6.QtGui import QAction
        from PySide6.QtCore import QSize

        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(16, 16))

        action_open = QAction("Open", self)
        action_open.triggered.connect(self._open_file)
        toolbar.addAction(action_open)

        toolbar.addSeparator()

        action_clear = QAction("Clean Window", self)
        action_clear.triggered.connect(self._clean_window)
        toolbar.addAction(action_clear)

        toolbar.addSeparator()

        action_reset = QAction("Reset Application", self)
        action_reset.triggered.connect(self._reset_application)
        toolbar.addAction(action_reset)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir modelo 3D", "", "Modelos 3D (*.ply *.obj)"
        )
        if not path:
            return

        if path.endswith(".obj"):
            self._pcd_o3d = o3d.io.read_triangle_mesh(path)
        else:
            self._pcd_o3d = o3d.io.read_point_cloud(path)  # ← guardar referencia O3D

        if path.endswith(".obj"):
            self.viewer.load_mesh(path)
        else:
            self.viewer.load_pointcloud(path)
        self.ui.textEdit_log.setVisible(False)
        self.ui.action_bar.setVisible(True)
        self.ui.textEdit_log.setVisible(True)

    def _append_log(self, timestamp: str, tag: str, text: str, color: str):
        tag_fixed = f"[{tag}]".ljust(100)  # ajusta el número según tus tags más largos
        self.ui.textEdit_log.append(
            f'<span style="font-family:Consolas; font-size:13px;">'
            f'<span style="color:#ffffff;">[{timestamp}]&nbsp;</span>'
            f'<span style="color:#ffffff;">{tag_fixed}&nbsp;</span>'
            f'<span style="color:{color};">{text}</span>'
            f'</span>'
        )
        self.ui.textEdit_log.verticalScrollBar().setValue(
            self.ui.textEdit_log.verticalScrollBar().maximum()
        )

    def _load_stylesheet(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print(f"Stylesheet not found: {path}")

    def _start_landmark_picking(self):
        # Ocultar botón — no tiene sentido volver a pulsarlo
        self.ui.qpushButton_landmarks.setVisible(False)
        self.ui.qpushButton_automatic.setVisible(False)
        self.viewer.start_landmark_picking(self._on_landmarks_complete)

    def _on_landmarks_complete(self, landmarks: dict):
        self._landmarks = landmarks
        logger.log(
            f"Landmarks received: {list(landmarks.keys())}",
            LogLevel.SUCCESS
        )
        # Mostrar botón Compute Indices
        self.ui.qpushButton_compute.setVisible(True)
        self.ui.qpushButton_uncertainty.setVisible(False)
        self._compute_cranial_planes()

    def _calculate_measurement_data(self):
        """Calcula plane_center, contour y ejes locales para el nivel 3."""
        from src.siv.processing.geometry import (
            compute_mesh_plane_intersection,
            compute_local_coordinate_system,
        )
        
        tragion_mid = (self._landmarks["right tragion"] + self._landmarks["left tragion"]) / 2
        normal = self._frankfurt_normal
        level_3 = self._cranial_levels[2]
        level_3_plane_point = level_3["center"]
        
        numerator = np.dot(tragion_mid - level_3_plane_point, normal)
        denominator = np.dot(normal, normal)
        t = -numerator / denominator
        
        plane_center = tragion_mid + t * normal
        
        contour = compute_mesh_plane_intersection(
            mesh         = self._cranial_mesh,
            normal       = normal,
            plane_center = plane_center,
        )
        
        x_axis, y_axis, z_axis = compute_local_coordinate_system(
            sellion = self._landmarks["sellion"],
            center  = plane_center,
            normal  = normal,
        )
        
        return plane_center, contour, x_axis, y_axis, z_axis

    def _compute_cranial_planes(self):
        lm = self._landmarks

        if not all(k in lm for k in ("sellion", "right tragion", "left tragion")):
            logger.log("Insufficient landmarks to define the reference plane", LogLevel.WARNING)
            return

        # ── Plano de Frankfurt ───────────────────────────────────────────────
        normal, center, _, _ = compute_reference_plane(
            sellion       = lm["sellion"],
            right_tragion = lm["right tragion"],
            left_tragion  = lm["left tragion"],
        )
        self._frankfurt_normal = normal
        self._frankfurt_center = center

        if normal[2] < 0:
            normal = -normal
            self._frankfurt_normal = normal

        points = np.asarray(self.viewer._current_cloud.points)

        # ── Altura craneal total ─────────────────────────────────────────────
        cranial_height = compute_cranial_height(points, normal, center)

        # ── 10 niveles equiespaciados ────────────────────────────────────────
        self._cranial_levels = compute_cranial_levels(
            normal, center, cranial_height, n_levels=N_CRANIAL_LEVELS
        )

        offset_m_plane_mm = next(item['offset'] for item in self._cranial_levels if 
                                item['level'] == 3)   

        logger.log(
            f"Measurement plane: level 03, {offset_m_plane_mm:.1f} mm to reference plane",
            LogLevel.INFO
        )

        # ── Plano de corte inicial ───────────────────────────────────────────
        offset_mm  = cranial_height * EAR_CLEARANCE_RATIO
        cut_center = center + normal * offset_mm
        logger.log(
            f"Mesh reconstructed {offset_mm:.1f} mm from reference plane "
            f"({EAR_CLEARANCE_RATIO*100:.0f}% of total cranial height)",
            LogLevel.INFO
        )

        # ── Reconstrucción de la bóveda craneal ──────────────────────────────
        points_down = np.asarray(self.viewer._current_cloud.points)
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

        self._cranial_mesh = mesh
        self.viewer.show_measurement_preview(
            plane_normal = normal,
            all_levels   = self._cranial_levels,
            mesh         = self._cranial_mesh,
            landmarks    = self._landmarks,
        )

    def _compute_indices(self):
        if self._cranial_mesh is None:
            logger.log("No mesh reconstructed", LogLevel.WARNING)
            return

        # ── Mostrar visualización igual que en cranial_planes ────────────────
        plane_center, contour, x_axis, y_axis, z_axis = self._calculate_measurement_data()
        
        self.viewer.show_measurement_view(
            contour      = contour,
            plane_center = plane_center,
            plane_normal = self._frankfurt_normal,
            x_axis       = x_axis,
            y_axis       = y_axis,
            z_axis       = z_axis,
            all_levels   = self._cranial_levels,
            mesh         = self._cranial_mesh,
            landmarks    = self._landmarks,
        )
        
        # ── Calcular CVAI para todos los planos 3-10 ──────────────────────────
        from src.siv.processing.geometry import (
            compute_cvai_intersection,
            compute_mesh_plane_intersection,
            compute_local_coordinate_system,
        )
        
        normal = self._frankfurt_normal
        cvai_results = []

        for level_idx in range(2, 10):  # índices 2-9 = niveles 3-10
            lvl = self._cranial_levels[level_idx]
            
            contour_lvl = compute_mesh_plane_intersection(
                mesh         = self._cranial_mesh,
                normal       = normal,
                plane_center = lvl["center"],
            )
            
            if len(contour_lvl) < 10:
                logger.log(f"Level {lvl['level']}: insufficient contour points", LogLevel.WARNING)
                continue
            
            x_axis_lvl, y_axis_lvl, z_axis_lvl = compute_local_coordinate_system(
                sellion = self._landmarks["sellion"],
                center  = lvl["center"],
                normal  = normal,
            )
            
            cvai_data = compute_cvai_intersection(
                contour      = contour_lvl,
                plane_center = lvl["center"],
                x_axis       = x_axis_lvl,
                y_axis       = y_axis_lvl,
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

        # ── Resumen final ────────────────────────────────────────────────────
        if cvai_results:
            logger.log("CVAI Analysis Summary (Levels 3-10):", LogLevel.INFO)
            for result in cvai_results:
                logger.log(
                    f"Level {result['level']:02d} @ {result['offset_mm']:.1f}mm: "
                    f"ratio={result['ratio']:.4f}, shorter={result['cvai_shorter']:.2f}%, longer={result['cvai_longer']:.2f}%",
                    LogLevel.SUCCESS
                )
        
        self.ui.qpushButton_uncertainty.setVisible(True)
        self._cvai_results_nominal = cvai_results
        self.viewer.lock_camera_rotation()

    def _clean_window(self):
        self.viewer.clear()

    def _reset_application(self):
        """Resetea la aplicación al estado inicial."""
        # Datos
        self._pcd_o3d      = None
        self._landmarks    = {}
        self._cranial_mesh = None

        # UI — ocultar action_bar y resetear botones
        self.ui.action_bar.setVisible(False)
        self.ui.qpushButton_landmarks.setVisible(True)
        self.ui.qpushButton_compute.setVisible(False)
        self.ui.qpushButton_uncertainty.setVisible(False)
        self.ui.qpushButton_automatic.setVisible(False)

        # UI - ocultar texto hint picker
        self.ui.label_picking_hint.setVisible(False)
        self.ui.label_picking_hint.setText("")

        # Log
        self.ui.textEdit_log.clear()

        # Viewer — volver a estado inicial
        self.viewer.reset()

        logger.log("Application reset", LogLevel.INFO)

    def _update_picking_hint(self, text: str) -> None:
        if text:
            self.ui.label_picking_hint.setText(text)
            self.ui.label_picking_hint.setVisible(True)
        else:
            self.ui.label_picking_hint.setVisible(False)
            self.ui.label_picking_hint.setText("")
    
    def _run_uncertainty(self, n_samples: int = 500):
        """Run Class-1 Monte Carlo perturbation and report CVAI distribution."""
        from src.siv.processing.uncertainty import (
            perturb_landmarks,
            summarise_cvai_distribution,
            SIGMA_BY_LANDMARK,
        )
        from src.siv.processing.geometry import (
            compute_cvai_intersection,
            compute_mesh_plane_intersection,
            compute_local_coordinate_system,
            compute_reference_plane,
        )

        logger.log("Running uncertainty analysis...", LogLevel.INFO)
        QApplication.processEvents()

        if not self._landmarks or self._cranial_mesh is None:
            logger.log("No landmarks or mesh available", LogLevel.WARNING)
            return

        logger.log(f"Uncertainty analysis — Class 1 — {n_samples} samples", LogLevel.INFO)

        # ── Pre-calcular contornos nominales una sola vez por nivel ──────────────
        nominal_contours = {}
        for level_idx in range(2, 10):
            lvl     = self._cranial_levels[level_idx]
            contour = compute_mesh_plane_intersection(
                mesh         = self._cranial_mesh,
                normal       = self._frankfurt_normal,
                plane_center = lvl["center"],
            )
            nominal_contours[level_idx] = contour

        # ── Generar N realizaciones perturbadas de los landmarks ─────────────────
        samples = perturb_landmarks(
            landmarks    = self._landmarks,
            mesh         = self._cranial_mesh,
            n_samples    = n_samples,
            sigma_errors = SIGMA_BY_LANDMARK,
            seed         = 42,
        )

        # ── Para cada nivel, acumular CVAI de todas las realizaciones ────────────
        results_by_level: dict[int, dict] = {}

        for level_idx in range(2, 10):
            lvl     = self._cranial_levels[level_idx]
            contour = nominal_contours[level_idx]

            if len(contour) < 10:
                continue

            cvai_list = []

            for sample_lm in samples:
                try:
                    s_normal, s_center, _, _ = compute_reference_plane(
                        sellion       = sample_lm["sellion"],
                        right_tragion = sample_lm["right tragion"],
                        left_tragion  = sample_lm["left tragion"],
                    )
                    if s_normal[2] < 0:
                        s_normal = -s_normal
                except ValueError:
                    continue

                x_ax, y_ax, _ = compute_local_coordinate_system(
                    sellion = sample_lm["sellion"],
                    center  = lvl["center"],
                    normal  = s_normal,
                )

                # Ordenar contorno nominal con los ejes perturbados
                angles         = np.arctan2(
                    np.dot(contour - lvl["center"], x_ax),
                    np.dot(contour - lvl["center"], y_ax),
                )
                contour_sorted = contour[np.argsort(angles)]

                cvai_data = compute_cvai_intersection(
                    contour      = contour_sorted,
                    plane_center = lvl["center"],
                    x_axis       = x_ax,
                    y_axis       = y_ax,
                )

                if cvai_data:
                    d1, d2 = cvai_data["dist1"], cvai_data["dist2"]
                    if min(d1, d2) < 0.1:   # realización inválida → descartar
                        continue
                    cvai_s = abs(d1 - d2) / min(d1, d2) * 100
                    cvai_list.append(cvai_s)

            if cvai_list:
                stats = summarise_cvai_distribution(np.array(cvai_list))
                results_by_level[lvl["level"]] = stats

        # ── Log de resultados ────────────────────────────────────────────────────
        logger.log(
            f"{'Lvl':>4}  {'Nominal':>8}  {'Mean':>8}  {'Std':>6}  "
            f"{'P5':>7}  {'P95':>7}  {'CV%':>6}",
            LogLevel.INFO
        )

        nominal_map = {r["level"]: r["cvai_shorter"] for r in self._cvai_results_nominal}

        for level_num, stats in sorted(results_by_level.items()):
            nom = nominal_map.get(level_num, float("nan"))
            logger.log(
                f"{level_num:>4}  {nom:>8.2f}  {stats['mean']:>8.2f}  "
                f"{stats['std']:>6.2f}  {stats['p5']:>7.2f}  "
                f"{stats['p95']:>7.2f}  {stats['cv']:>6.1f}",
                LogLevel.SUCCESS
            )


        # Guardar para batch/export posterior
        self._uncertainty_results = results_by_level

        # ── Visualizar elipses de incertidumbre sobre la nube de puntos ──────────────
        from src.siv.processing.uncertainty import compute_landmark_ellipse
        import pyvista as pv

        ELLIPSE_COLORS = {
            "sellion":       "yellow",
            "right tragion": "red",
            "left tragion":  "green",
        }

        for name, point in self._landmarks.items():
            verts = compute_landmark_ellipse(
                name = name,
                point = point,
                mesh  = self._cranial_mesh,
                sigma_errors = SIGMA_BY_LANDMARK
            )

            verts_closed = np.vstack([verts, verts[0]])
            spline = pv.Spline(verts_closed, n_points=200)

            self.viewer.plotter.add_mesh(
                spline,
                color      = ELLIPSE_COLORS.get(name, "white"),
                line_width = 3,
                label      = f"{name} ±2σ",
            )

        self.viewer.plotter.render()
    
    def _run_automatic_landmarks(self):
        """Run Automatic Landmark Recognition algorithm."""
        from src.siv.processing.landmark_detector import(
            detect_landmarks,
        )

        if self._pcd_o3d is None:
            return

        logger.log("Automatic landmark detection algorithm running...", LogLevel.INFO)
        QApplication.processEvents()
        landmarks = detect_landmarks(self._pcd_o3d)
        self._on_landmarks_complete(landmarks)
        self.ui.qpushButton_automatic.setVisible(False)
        self.ui.qpushButton_landmarks.setVisible(False)
        

    def closeEvent(self, event):
        self.viewer.closeEvent(event)
        super().closeEvent(event)

if __name__ == "__main__":
    myappid = "siv.1.0"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("./src/siv/resources/icons/PC_ICON.ico"))
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())
