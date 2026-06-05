"""Interactive landmark picking integrated in the PyVista/Qt viewport."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from src.siv.utils.logger import logger, LogLevel
from src.siv.visualization.config import BACKGROUND_COLOR

from PySide6.QtCore import QObject, Signal

CLR_CONTEXT  = "#4488ff"
CLR_SELLION  = "#f5a623"
CLR_RIGHT    = "#e74c3c"
CLR_LEFT     = "#2ecc71"

LANDMARK_COLORS = {
    "sellion":       CLR_SELLION,
    "right tragion": CLR_RIGHT,
    "left tragion":  CLR_LEFT,
}

CAMERA_PRESETS = {
    "anterior": {
        "position": (0.0, -1.0, 0.0),
        "up":       (0.0,  0.0, 1.0),
    },
    "right_lateral": {
        "position": (0.0,  1.0, 0.0),
        "up":       (0.0,  0.0, 1.0),
    },
    "left_lateral": {
        "position": (0.0, -1.0, 0.0),
        "up":       (0.0,  0.0, 1.0),
    },
}

LANDMARK_SEQUENCE = [
    {
        "name":   "sellion",
        "label":  "Sellion (nasal bridge)",
        "camera": "anterior",
        "clip_mask": None,  # sin recorte, se ve todo
    },
    {
        "name":   "right tragion",
        "label":  "Right tragion (right ear tragus)",
        "camera": "right_lateral",
        "clip_mask": lambda n: n[:, 1] < 0.55,  # solo hemisferio derecho (Y bajo)
    },
    {
        "name":   "left tragion",
        "label":  "Left tragion (left ear tragus)",
        "camera": "left_lateral",
        "clip_mask": lambda n: n[:, 1] > 0.45,  # solo hemisferio izquierdo (Y alto)
    },
]


def _normalize(points: np.ndarray) -> np.ndarray:
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent   = bbox_max - bbox_min
    extent[extent == 0] = 1.0
    return (points - bbox_min) / extent


class LandmarkPicker(QObject):

    hint_changed = Signal(str)   # ← señal que lleva el texto del hint

    def __init__(
        self,
        plotter_global,
        plotter_pick,
        cloud: pv.PolyData,
        points: np.ndarray,
    ):
        super().__init__()  # ← necesario para QObject
        self._plotter_global = plotter_global
        self._plotter_pick   = plotter_pick
        self._cloud          = cloud
        self._points         = points
        bounds               = cloud.bounds
        self._center         = np.array([
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ])
        self._landmarks: dict[str, np.ndarray] = {}
        self._step     = 0
        self._callback = None

    def start(self, on_complete) -> None:
        self._callback = on_complete
        self._step     = 0
        self._landmarks = {}
        logger.log("Landmark picking session started", LogLevel.INFO)
        self._setup_global_panel()
        self._pick_current_step()

    def _setup_global_panel(self) -> None:
        self._plotter_global.clear()
        self._plotter_global.add_points(
            self._cloud,
            color=CLR_CONTEXT,
            point_size=1.5,
            render_points_as_spheres=False,
        )
        self._plotter_global.add_axes(
            line_width=3, labels_off=False,
            x_color="#ff4444", y_color="#44ff44", z_color="#4488ff",
        )
        self._plotter_global.reset_camera()
        self._plotter_global.render()

    def _pick_current_step(self) -> None:
        if self._step >= len(LANDMARK_SEQUENCE):
            self._finish()
            return

        cfg  = LANDMARK_SEQUENCE[self._step]
        name = cfg["name"]
        clip_mask = cfg["clip_mask"]

        # Aplicar recorte si existe
        if clip_mask is not None:
            norm        = _normalize(self._points)
            visible_pts = self._points[clip_mask(norm)]
        else:
            visible_pts = self._points

        logger.log(
            f"Step {self._step + 1}/3 — {cfg['label']} "
            f"({len(visible_pts)} pts visibles)",
            LogLevel.INFO
        )

        # Panel derecho: nube completa en blanco
        self._plotter_pick.clear()
        self._plotter_pick.set_background(BACKGROUND_COLOR)
        self._plotter_pick.add_points(
            self._cloud,
            color="#ffffff",
            point_size=2.0,
            render_points_as_spheres=False,
        )
        self.hint_changed.emit(
            f"Step {self._step + 1} / 3  —  Select {cfg['label']}"
        )
        self._plotter_pick.add_axes(
            line_width=3, labels_off=False,
            x_color="#ff4444", y_color="#44ff44", z_color="#4488ff",
        )

        self._set_camera(cfg["camera"])

        self._plotter_pick.enable_point_picking(
            callback=self._on_point_picked,
            show_message=False,
            color=LANDMARK_COLORS[name],
            point_size=12,
            use_picker=True,
            pickable_window=False,
        )
        self._plotter_pick.render()

    def _set_camera(self, preset_key: str) -> None:
        preset = CAMERA_PRESETS[preset_key]
        c      = self._center
        bounds = self._cloud.bounds
        span   = max(
            bounds[1]-bounds[0],
            bounds[3]-bounds[2],
            bounds[5]-bounds[4],
        )
        offset = np.array(preset["position"]) * span * 1.8

        self._plotter_pick.camera.position    = (c + offset).tolist()
        self._plotter_pick.camera.focal_point = c.tolist()
        self._plotter_pick.camera.up          = preset["up"]
        self._plotter_pick.enable_parallel_projection()
        self._plotter_pick.reset_camera()

    def _on_point_picked(self, point: np.ndarray, picker) -> None:
        if point is None:
            return

        cfg       = LANDMARK_SEQUENCE[self._step]
        name      = cfg["name"]
        clip_mask = cfg["clip_mask"]

        if clip_mask is not None:
            norm        = _normalize(self._points)
            visible_pts = self._points[clip_mask(norm)]
        else:
            visible_pts = self._points

        renderer = self._plotter_pick.renderer

        # Proyectar el punto clickado a pantalla
        renderer.SetWorldPoint(point[0], point[1], point[2], 1.0)
        renderer.WorldToDisplay()
        cx, cy, _ = renderer.GetDisplayPoint()
        click_2d  = np.array([cx, cy])

        # Proyectar TODOS los puntos visibles a pantalla
        pts_2d = []
        for p in visible_pts:
            renderer.SetWorldPoint(p[0], p[1], p[2], 1.0)
            renderer.WorldToDisplay()
            dx, dy, _ = renderer.GetDisplayPoint()
            pts_2d.append([dx, dy])
        pts_2d = np.array(pts_2d)

        # Buscar el más cercano en 2D
        dists_2d = np.linalg.norm(pts_2d - click_2d, axis=1)
        nearest  = visible_pts[dists_2d.argmin()]

        self._landmarks[name] = nearest
        logger.log(f"{name}: {np.round(nearest, 2)}", LogLevel.SUCCESS)

        self._add_landmark_marker(name, nearest)
        self._plotter_pick.disable_picking()

        self._step += 1
        self._pick_current_step()

    def _add_landmark_marker(self, name: str, coords: np.ndarray) -> None:
        bounds = self._cloud.bounds
        span   = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        radius = span * 0.015

        sphere = pv.Sphere(radius=radius, center=coords.tolist())
        self._plotter_global.add_mesh(
            sphere,
            color=LANDMARK_COLORS[name],
            smooth_shading=True,
        )
        self._plotter_global.render()

    def _finish(self) -> None:
        n = len(self._landmarks)
        logger.log(
            f"Picking complete — {n}/3 landmarks collected",
            LogLevel.SUCCESS if n == 3 else LogLevel.WARNING
        )
        self.hint_changed.emit("")  # limpiar texto
        self._plotter_pick.disable_picking()
        self._plotter_pick.enable_trackball_style()
        self._plotter_pick.reset_key_events()

        if self._callback:
            self._callback(self._landmarks)
