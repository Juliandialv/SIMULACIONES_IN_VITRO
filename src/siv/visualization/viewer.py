import numpy as np
import open3d as o3d
import pyvista as pv

from pyvistaqt import QtInteractor
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter

from src.siv.utils.logger import logger, LogLevel
from src.siv.io.picker import LandmarkPicker, LANDMARK_COLORS
from src.siv.visualization.config import BACKGROUND_COLOR, POINT_SIZE


class PointCloudViewer(QWidget):

    picking_hint = Signal(str)  # ← re-emite el hint del picker hacia main

    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_cloud = None
        self._current_actor = None

        self._splitter = QSplitter(Qt.Horizontal, self)

        self.plotter = QtInteractor(self)
        self.plotter.set_background(BACKGROUND_COLOR)

        self.plotter_pick = QtInteractor(self)
        self.plotter_pick.set_background(BACKGROUND_COLOR)

        self._splitter.addWidget(self.plotter)
        self._splitter.addWidget(self.plotter_pick)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._splitter)

        # Arranca con el plotter_pick oculto, pero lo dejamos inicializar
        # un instante antes de ocultarlo para que el contexto OpenGL exista
        self._picking_mode = False
        self._set_single_viewport()

    # ── API pública ─────────────────────────────────────────────────────────

    def load_pointcloud(self, path: str) -> None:
        from src.siv.processing.pointcloud import voxel_downsample
        from src.siv.processing.config import DEFAULT_VOXEL_SIZE

        pcd_o3d = o3d.io.read_point_cloud(path)
        if not pcd_o3d.has_points():
            logger.log(f"No se pudieron cargar puntos desde: {path}", LogLevel.ERROR)
            return

        if path.endswith(".ply"):
            pcd_o3d = voxel_downsample(pcd_o3d, voxel_size=DEFAULT_VOXEL_SIZE)

        points = np.asarray(pcd_o3d.points)
        cloud  = pv.PolyData(points)
        if pcd_o3d.has_colors():
            colors = (np.asarray(pcd_o3d.colors) * 255).astype(np.uint8)
            cloud["RGB"] = colors

        self._render_cloud(cloud)

    def load_mesh(self, path: str) -> None:
        mesh_o3d = o3d.io.read_triangle_mesh(path)
        mesh_o3d.compute_vertex_normals()

        vertices  = np.asarray(mesh_o3d.vertices)
        faces_o3d = np.asarray(mesh_o3d.triangles)
        faces_pv  = np.hstack([np.full((len(faces_o3d), 1), 3), faces_o3d])

        self._render_cloud(pv.PolyData(vertices, faces_pv), as_mesh=True)

    def clear(self) -> None:
        self.plotter.clear()
        self._current_cloud = None
        self._current_actor = None

    def _on_hint_changed(self, text: str) -> None:
        self.picking_hint.emit(text)

    def start_landmark_picking(self, on_complete) -> None:
        if self._current_cloud is None:
            logger.log("No hay modelo cargado", LogLevel.WARNING)
            return

        self._set_split_viewport()

        picker = LandmarkPicker(
            plotter_global=self.plotter,
            plotter_pick=self.plotter_pick,
            cloud=self._current_cloud,
            points=np.asarray(self._current_cloud.points),
        )
        picker.hint_changed.connect(self._on_hint_changed)
        picker.start(lambda lm: self._on_picking_done(lm, on_complete))

    def show_landmarks(self, landmarks: dict) -> None:
        bounds = self._current_cloud.bounds
        span   = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        radius = span * 0.01

        for name, coords in landmarks.items():
            sphere = pv.Sphere(radius=radius, center=coords.tolist())
            self.plotter.add_mesh(
                sphere,
                color=LANDMARK_COLORS.get(name, "#ffffff"),
                smooth_shading=True,
            )
        self.plotter.render()

    # ── Viewport helpers ─────────────────────────────────────────────────────

    def _set_single_viewport(self) -> None:
        self._splitter.setSizes([self.width(), 0])
        self.plotter_pick.hide()

    def _set_split_viewport(self) -> None:
        self.plotter_pick.show()
        w = self.width()
        self._splitter.setSizes([w // 2, w // 2])

    # ── Internos ─────────────────────────────────────────────────────────────

    def _on_picking_done(self, landmarks, on_complete) -> None:
        self.plotter_pick.disable_parallel_projection()
        self._splitter.setSizes([0, self.width()])
        on_complete(landmarks)

    def _render_cloud(self, geometry: pv.PolyData, as_mesh: bool = False) -> None:
        self._current_cloud = geometry

        # Restaurar plotter principal y ocultar plotter_pick
        self.plotter.clear()
        self.plotter.show_axes()
        self.plotter.show()
        self._set_single_viewport()
        
        scalar_bar = "RGB" in geometry.point_data
        if scalar_bar:
            self._current_actor = self.plotter.add_points(
                geometry,
                scalars="RGB",
                rgb=True,
                point_size=POINT_SIZE,
                render_points_as_spheres=False,
            )
        else:
            self._current_actor = self.plotter.add_points(
                geometry,
                color="#4488ff",
                point_size=POINT_SIZE,
                render_points_as_spheres=False,
            )

        self.plotter.add_axes(
            line_width=3, labels_off=False,
            x_color="#ff4444", y_color="#44ff44", z_color="#4488ff",
        )
        self.plotter.reset_camera()
        self.plotter.render()

    def show_contour(self, contour_pts: np.ndarray) -> None:
        """Renderiza la banda de contorno en rojo sobre el modelo actual."""
        if contour_pts is None or len(contour_pts) == 0:
            return

        contour_cloud = pv.PolyData(np.array(contour_pts, dtype=np.float32))

        # Añadir sin tocar nada más del estado del plotter
        actor = self.plotter.add_points(
            contour_cloud,
            color="#ff2222",
            point_size=10.0,
            render_points_as_spheres=True,
            reset_camera=False,  # ← crítico: no tocar la cámara
        )

        logger.log(f"Contorno actor: {actor}", LogLevel.INFO)
        self.plotter.render()

    def refresh_scene(
        self,
        landmarks: dict = None,
        contour_pts: np.ndarray = None,
        contour_mask: np.ndarray = None,
        measurement_pts: np.ndarray = None,
        measurement_mask: np.ndarray = None,
    ) -> None:
        # Ocultar mientras se prepara la escena
        self.plotter_pick.hide()

        self.plotter_pick.clear()

        points = np.asarray(self._current_cloud.points)
        n      = len(points)
        colors = np.full((n, 3), [68, 136, 255], dtype=np.uint8)

        if contour_mask is not None:
            colors[contour_mask] = [255, 34, 34]

        if measurement_mask is not None:
            colors[measurement_mask] = [34, 255, 34]

        cloud_colored = pv.PolyData(points)
        cloud_colored["RGB"] = colors

        self.plotter_pick.add_points(
            cloud_colored,
            scalars="RGB",
            rgb=True,
            point_size=POINT_SIZE,
            render_points_as_spheres=False,
        )

        if landmarks:
            bounds = self._current_cloud.bounds
            span   = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            radius = span * 0.01
            for name, coords in landmarks.items():
                sphere = pv.Sphere(radius=radius, center=coords.tolist())
                self.plotter_pick.add_mesh(
                    sphere,
                    color=LANDMARK_COLORS.get(name, "#ffffff"),
                    smooth_shading=True,
                )

        self.plotter_pick.add_axes(
            line_width=3, labels_off=False,
            x_color="#ff4444", y_color="#44ff44", z_color="#4488ff",
        )

        # Orientar antes de mostrar
        self.plotter_pick.reset_camera()
        self.plotter_pick.view_isometric()

        # Mostrar solo cuando la escena está lista
        self.plotter_pick.show()
        self.plotter_pick.render()

    def show_mesh(self, mesh_o3d: "o3d.geometry.TriangleMesh") -> None:
        """Renderiza la malla reconstruida sobre la escena actual."""
        import open3d as o3d

        vertices  = np.asarray(mesh_o3d.vertices)
        faces_o3d = np.asarray(mesh_o3d.triangles)

        if len(vertices) == 0 or len(faces_o3d) == 0:
            logger.log("Malla vacía, nada que renderizar", LogLevel.WARNING)
            return

        faces_pv = np.hstack([
            np.full((len(faces_o3d), 1), 3),
            faces_o3d
        ])
        mesh_pv = pv.PolyData(vertices, faces_pv)

        self.plotter_pick.add_mesh(
            mesh_pv,
            color="#e8d5b0",       # color hueso/carne neutro
            opacity=0.85,
            smooth_shading=True,
            show_edges=False,
        )
        self.plotter_pick.render()
        logger.log(
            f"Malla renderizada: {len(vertices)} vértices, {len(faces_o3d)} triángulos",
            LogLevel.SUCCESS
        )

    def show_measurement_view(
        self,
        contour: np.ndarray,
        plane_center: np.ndarray,
        plane_normal: np.ndarray,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        z_axis: np.ndarray,
        all_levels: list,
        mesh: "o3d.geometry.TriangleMesh",
        landmarks: dict = None,
    ) -> None:
        """Setup measurement view: left=lateral profile, right=level 3 contour."""
        
        self._set_split_viewport()
        
        # ── Viewport IZQUIERDO: vista lateral con nube + landmarks + malla + contornos ────
        self.plotter.clear()
        
        # Nube base azul
        points = np.asarray(self._current_cloud.points)
        cloud_colored = pv.PolyData(points)
        cloud_colored["RGB"] = np.full((len(points), 3), [68, 136, 255], dtype=np.uint8)
        
        self.plotter.add_points(
            cloud_colored,
            scalars="RGB",
            rgb=True,
            point_size=POINT_SIZE,
            render_points_as_spheres=False,
        )
        
        # Landmarks
        if landmarks:
            bounds = self._current_cloud.bounds
            span   = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            radius = span * 0.01
            for name, coords in landmarks.items():
                sphere = pv.Sphere(radius=radius, center=coords.tolist())
                self.plotter.add_mesh(
                    sphere,
                    color=LANDMARK_COLORS.get(name, "#ffffff"),
                    smooth_shading=True,
                )
        
        # Malla reconstruida
        vertices_o3d = np.asarray(mesh.vertices)
        faces_o3d    = np.asarray(mesh.triangles)
        faces_pv     = np.hstack([np.full((len(faces_o3d), 1), 3), faces_o3d])
        mesh_pv      = pv.PolyData(vertices_o3d, faces_pv)
        
        self.plotter.add_mesh(
            mesh_pv,
            color="#e8d5b0",
            opacity=0.7,
            smooth_shading=True,
        )
        
        from src.siv.processing.geometry import compute_mesh_plane_intersection

        for i, lvl in enumerate(all_levels):
            if i < 2:
                continue

            lvl_contour = compute_mesh_plane_intersection(
                mesh         = mesh,
                normal       = plane_normal,
                plane_center = lvl["center"],
            )
            if len(lvl_contour) < 3:
                continue

            # Ordenar contorno
            cc = lvl_contour - lvl["center"]
            ang = np.arctan2(np.dot(cc, x_axis), np.dot(cc, y_axis))
            lvl_contour = lvl_contour[np.argsort(ang)]

            n_pts = len(lvl_contour)
            lines_arr = np.hstack([[2, k, (k+1) % n_pts] for k in range(n_pts)]).astype(np.int32)
            poly = pv.PolyData(lvl_contour)
            poly.lines = lines_arr

            color = "#b8860b" if i == 2 else "#8b0000"   # nivel 3: amarillo oscuro, resto: rojo oscuro
            self.plotter.add_mesh(poly, color=color, line_width=2, render_lines_as_tubes=False)

        # Vista lateral sin perspectiva
        self.plotter.enable_parallel_projection()
        self.plotter.view_xz()
        self.plotter.add_axes(
            line_width=3, labels_off=False,
            x_color="#ff4444", y_color="#44ff44", z_color="#4488ff",
        )
        self.plotter.render()
        
        # ── Viewport DERECHO: contorno level 3 + ejes locales ─────────────────
        self.plotter_pick.clear()
        
        if len(contour) > 0:
            # Ordenar contorno para formar polígono cerrado
            contour_centered = contour - plane_center
            angles = np.arctan2(
                np.dot(contour_centered, x_axis),
                np.dot(contour_centered, y_axis),
            )
            sorted_idx = np.argsort(angles)
            contour_sorted = contour[sorted_idx]
            
            # Crear líneas conectando puntos
            n_pts = len(contour_sorted)
            lines = [[i, (i + 1) % n_pts] for i in range(n_pts)]
            
            contour_poly = pv.PolyData(contour_sorted)
            contour_poly.lines = np.hstack([[2, i, j] for i, j in lines]).astype(np.int32)
            
            self.plotter_pick.add_mesh(
                contour_poly,
                color="#b8860b",  # amarillo oscuro (mismo que izquierda)
                line_width=3,
                render_lines_as_tubes=False,
            )
                    
        # Ejes locales X (rojo) e Y (verde)
        axis_length = 30.0
        for direction, color in [(x_axis, "#ff0000"), (y_axis, "#00ff00")]:
            start = plane_center
            end   = plane_center + direction * axis_length
            line  = pv.Line(start, end)
            self.plotter_pick.add_mesh(line, color=color, line_width=3)
                
        # ── Diagonales a ±30° del eje Y local ─────────────────────────────────────
        from src.siv.processing.geometry import compute_cvai_intersection
        # Calcular intersecciones
        cvai_data = compute_cvai_intersection(
            contour      = contour_sorted,
            plane_center = plane_center,
            x_axis       = x_axis,
            y_axis       = y_axis,
        )

        if cvai_data:
            for i, diag_pts in enumerate([cvai_data["diag1_pts"], cvai_data["diag2_pts"]]):
                pt_neg, pt_pos = diag_pts
                
                dist_to_neg = np.linalg.norm(pt_neg - plane_center)
                dist_to_pos = np.linalg.norm(pt_pos - plane_center)
                
                # Línea diagonal
                line = pv.Line(pt_neg, pt_pos)
                self.plotter_pick.add_mesh(line, color="#5865F2", line_width=1)
                
                # Pequeñas esferas en los puntos de intersección
                radius = 1.5
                sphere_neg = pv.Sphere(radius=radius, center=pt_neg.tolist())
                sphere_pos = pv.Sphere(radius=radius, center=pt_pos.tolist())
                self.plotter_pick.add_mesh(sphere_neg, color="#5865F2", smooth_shading=True)
                self.plotter_pick.add_mesh(sphere_pos, color="#5865F2", smooth_shading=True)
        else:
            logger.log("Could not compute CVAI intersections", LogLevel.WARNING)
        
        # Cámara: vista desde arriba perpendicular al plano
        # Mirando "hacia abajo" (hacia Frankfurt), con Y apuntando hacia arriba en pantalla
        span = 65.0

        # Fijar la cámara ANTES de reset_camera para que no se recalcule
        contour_center = np.mean(contour, axis=0)
        self.plotter_pick.camera.position    = (contour_center + plane_normal * span).tolist()
        self.plotter_pick.camera.focal_point = contour_center.tolist()
        self.plotter_pick.camera.up          = y_axis.tolist()
        self.plotter_pick.enable_parallel_projection()
        self.plotter_pick.camera.zoom(0.25)

        self.plotter.camera.position    = (contour_center + plane_normal * span).tolist()
        self.plotter.camera.focal_point = contour_center.tolist()
        self.plotter.camera.up          = y_axis.tolist()
        self.plotter.enable_parallel_projection()
        self.plotter.camera.zoom(0.25)
        self.plotter.render()
        self.plotter_pick.render()

    def show_measurement_preview(
        self,
        plane_normal: np.ndarray,
        all_levels: list,
        mesh: "o3d.geometry.TriangleMesh",
        landmarks: dict = None,
    ) -> None:
        """Vista previa de medición: nube + malla + contornos 3-10 en único viewport."""
        
        # Asegurar que estamos en viewport único
        self._set_single_viewport()
        
        self.plotter.clear()
        
        # Nube base azul
        points = np.asarray(self._current_cloud.points)
        cloud_colored = pv.PolyData(points)
        cloud_colored["RGB"] = np.full((len(points), 3), [68, 136, 255], dtype=np.uint8)
        
        self.plotter.add_points(
            cloud_colored,
            scalars="RGB",
            rgb=True,
            point_size=POINT_SIZE,
            render_points_as_spheres=False,
        )
        
        # Landmarks
        if landmarks:
            bounds = self._current_cloud.bounds
            span   = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            radius = span * 0.01
            for name, coords in landmarks.items():
                sphere = pv.Sphere(radius=radius, center=coords.tolist())
                self.plotter.add_mesh(
                    sphere,
                    color=LANDMARK_COLORS.get(name, "#ffffff"),
                    smooth_shading=True,
                )
        
        # Malla reconstruida
        vertices_o3d = np.asarray(mesh.vertices)
        faces_o3d    = np.asarray(mesh.triangles)
        faces_pv     = np.hstack([np.full((len(faces_o3d), 1), 3), faces_o3d])
        mesh_pv      = pv.PolyData(vertices_o3d, faces_pv)
        
        self.plotter.add_mesh(
            mesh_pv,
            color="#e8d5b0",
            opacity=0.7,
            smooth_shading=True,
        )
        
        from src.siv.processing.geometry import compute_mesh_plane_intersection

        for i, lvl in enumerate(all_levels):
            if i < 2:
                continue

            lvl_contour = compute_mesh_plane_intersection(
                mesh         = mesh,
                normal       = plane_normal,
                plane_center = lvl["center"],
            )
            if len(lvl_contour) < 3:
                continue

            # Ordenar contorno
            # Usamos un eje de referencia genérico para ordenar en preview
            ref_ax = np.array([1.0, 0.0, 0.0])
            ref_ax2 = np.array([0.0, 1.0, 0.0])
            cc = lvl_contour - lvl["center"]
            ang = np.arctan2(np.dot(cc, ref_ax), np.dot(cc, ref_ax2))
            lvl_contour = lvl_contour[np.argsort(ang)]

            n_pts = len(lvl_contour)
            lines_arr = np.hstack([[2, k, (k+1) % n_pts] for k in range(n_pts)]).astype(np.int32)
            poly = pv.PolyData(lvl_contour)
            poly.lines = lines_arr

            color = "#00ff00" if i == 2 else "#8b0000"
            self.plotter.add_mesh(poly, color=color, line_width=2, render_lines_as_tubes=False)
        
        # Configurar vista
        self.plotter.enable_parallel_projection()
        self.plotter.view_isometric()
        self.plotter.add_axes(
            line_width=3, labels_off=False,
            x_color="#ff4444", y_color="#44ff44", z_color="#4488ff",
        )
        self.plotter.render()

    def reset(self) -> None:
        """Vuelve al estado inicial: un único viewport vacío."""
        self._current_cloud = None
        self._current_actor = None

        self.plotter.show()
        self._set_single_viewport()
        self.plotter.set_background(BACKGROUND_COLOR)

        # clear() + remove_all_lights garantiza limpieza total incluyendo ejes
        self.plotter.clear()
        self.plotter.remove_all_lights()
        # Eliminar el widget de ejes específicamente
        try:
            self.plotter.hide_axes()
        except Exception:
            pass
        self.plotter.render()

        # Limpiar plotter_pick también
        self.plotter_pick.clear()
        try:
            self.plotter_pick.hide_axes()
        except Exception:
            pass
        self.plotter_pick.disable_picking()
        self.plotter_pick.enable_trackball_style()
        self.plotter_pick.reset_key_events()
        self.plotter_pick.render()
    
    def lock_camera_rotation(self):
        """Deshabilita la rotación libre manteniendo zoom y paneo."""
        interactor = self.plotter_pick.interactor
        interactor.RemoveObservers("LeftButtonPressEvent")
        interactor.RemoveObservers("MouseMoveEvent")
        interactor.AddObserver("LeftButtonPressEvent", lambda o, e: None)

    def closeEvent(self, event):
        # Suprimir warnings de VTK durante el cierre
        import vtkmodules.vtkRenderingCore as vtk_rc
        vtk_rc.vtkObject.GlobalWarningDisplayOff()

        try:
            self.plotter.close()
        except Exception:
            pass
        try:
            self.plotter_pick.close()
        except Exception:
            pass
        super().closeEvent(event)
