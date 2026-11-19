import sys
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Final

import colorcet as cc
import numpy as np
from PySide6.QtCore import (
    QObject,
    QPoint,
    QRectF,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QCloseEvent,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QMainWindow,
    QWidget,
)
from pyvista import PolyData
from pyvistaqt import QtInteractor
from rich.live import Live
from rich.text import Text

from flowtier.common import UserError, console
from flowtier.data_handling import AlgConfig, DomainData

BACKGROUND_COLOR = QColor("#383838")
VIEW_COLOR = QColor("#2b2b2b")
BORDER_COLOR = QColor("#2b2b2b")
FALSE_COLOR = QColor("#4a6fa5")
TRUE_COLOR = QColor("#a3eaff")


STYLE_SHEET = f"""
    QGraphicsView {{
        border: 2px solid {BORDER_COLOR.name()};
        border-radius: 3px;
        background-color: {VIEW_COLOR.name()};
    }}

    QMainWindow {{
        background-color: {BACKGROUND_COLOR.name()};
    }}
"""


class AppState(QObject):
    cross_section_changed = Signal()
    sweep_changed = Signal(object, object)
    computing_changed = Signal(bool)

    def __init__(self, domain_data: DomainData) -> None:
        super().__init__()

        self._domain_data = domain_data

        self._worker = None
        self._pending = False
        self._revision = 0

    @property
    def is_computing(self) -> bool:
        return self._worker is not None or self._pending

    @property
    def domain_data(self) -> DomainData:
        return self._domain_data

    def paint_cross_section(self, i: np.ndarray, j: np.ndarray, value: bool) -> None:
        if self._domain_data.paint_cross_section(i, j, value):
            self._revision += 1
            self.cross_section_changed.emit()

    def save_cross_section(self, path: Path) -> None:
        np.savez_compressed(path, cross_section=self.domain_data.cropped_cross_section)

    def update_segmentation(self) -> None:
        self.domain_data.update_segmentation()

    def save_segmentation(self, path: Path) -> None:
        self.domain_data.save_segmentation(path)

    def update_sweep(self) -> None:
        self._pending = True

        if self._worker is None:
            self._start_sweep()

    def _start_sweep(self) -> None:
        self._pending = False

        self._worker = SweepWorker(self._domain_data, self._domain_data.cross_section.copy(), self._revision)

        self._worker.result_ready.connect(self._on_sweep_ready)
        self._worker.failed.connect(self._on_sweep_failed)
        self._worker.finished.connect(self._on_sweep_finished)

        self.computing_changed.emit(True)
        self._worker.start()

    def _on_sweep_ready(
        self,
        revision: int,
        confidence: np.ndarray,
        indices: np.ndarray,
        labels: np.ndarray,
        surfaces: list[PolyData],
        sweep_surface: PolyData,
    ) -> None:
        if revision != self._revision:
            return

        self._domain_data.apply_sweep_dependencies(confidence, indices, labels)

        self.sweep_changed.emit(surfaces, sweep_surface)

    def _on_sweep_failed(self, message: str) -> None:
        console.print(f"[red]Sweep computation failed: {message}[/red]")

    def _on_sweep_finished(self) -> None:
        self._worker.deleteLater()
        self._worker = None

        if self._pending:
            self._start_sweep()
        else:
            self.computing_changed.emit(False)

    @classmethod
    def build(cls, domain_data: DomainData, app_config: "AppConfig") -> tuple["AppState", "Mode"]:
        app_state = cls(domain_data)
        mode = app_config.mode

        if mode != Mode.RESET and app_config.cross_section_state_path.is_file():
            with np.load(app_config.cross_section_state_path) as data:
                if len(data.files) != 1:
                    raise UserError("The .npz file must contain exactly one array.")

                app_state.domain_data.set_cross_section(data[data.files[0]].astype(np.bool))

        else:
            mode = Mode.RESET

        return app_state, mode


class Mode(StrEnum):
    RESET = "reset"
    RESUME = "resume"
    SILENT = "silent"


@dataclass(frozen=True)
class AppConfig:
    surface_path: Path
    center_lines_path: Path
    cross_section_state_path: Path
    output_path: Path
    mode: Mode
    autosave_interval_ms: int = 500


class FlowtierStudio(QMainWindow):
    def __init__(self, app_state: AppState, app_config: AppConfig) -> None:
        super().__init__()
        self._closing = False

        self.app_state = app_state
        self._app_config = app_config

        self._build_save_connections()

        self._build_ui()
        self._build_connections()

        self.app_state.update_sweep()

    def _build_save_connections(self) -> None:
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(self._app_config.autosave_interval_ms)
        self._save_timer.timeout.connect(
            partial(self.app_state.save_cross_section, self._app_config.cross_section_state_path)
        )

        self.app_state.cross_section_changed.connect(self._save_timer.start)

    def _build_ui(self) -> None:
        self.setWindowTitle("Flowtier")
        self.resize(1000, 700)

        main_container = QWidget()
        self.setCentralWidget(main_container)
        main_layout = QHBoxLayout(main_container)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        left_container = self._build_3d_view()
        right_container = CrossSectionSketcher(self.app_state)

        main_layout.addWidget(left_container, 1)
        main_layout.addWidget(right_container, 1)

    def _build_connections(self) -> None:
        self.app_state.sweep_changed.connect(self._update_3d_view)
        self.app_state.computing_changed.connect(self._on_computing_changed)

    def _on_computing_changed(self, computing: bool) -> None:
        self._computing_indicator.setVisible(computing)

        if self._closing and not computing:
            self.close()

    def _build_3d_view(self) -> QWidget:
        self._3d_plotter = QtInteractor()
        self._3d_plotter.setMinimumSize(300, 300)

        self._3d_plotter.set_background(BACKGROUND_COLOR.name())

        bounds = self.app_state.domain_data.boundaries
        self._3d_plotter.show_grid(
            bounds=(0, bounds[1, 0], 0, bounds[1, 1], 0, bounds[1, 2]),
            xtitle="Axis 0",
            ytitle="Axis 1",
            ztitle="Axis 2",
            color="white",
        )

        self._3d_plotter.reset_camera()
        self._3d_plotter.camera.zoom(1 / 1.5)

        self._computing_indicator = LoadingSpinner(self._3d_plotter)
        self._computing_indicator.move(
            (self._3d_plotter.width() - self._computing_indicator.width()) // 2,
            (self._3d_plotter.height() - self._computing_indicator.height()) // 2,
        )
        self._computing_indicator.hide()

        return self._3d_plotter

    def showEvent(self, event) -> None:
        super().showEvent(event)

        self._computing_indicator.move(
            (self._3d_plotter.width() - self._computing_indicator.width()) // 2,
            (self._3d_plotter.height() - self._computing_indicator.height()) // 2,
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

        if hasattr(self, "_computing_indicator"):
            self._computing_indicator.move(
                (self._3d_plotter.width() - self._computing_indicator.width()) // 2,
                (self._3d_plotter.height() - self._computing_indicator.height()) // 2,
            )

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._save_timer.isActive():
            self._save_timer.stop()
            self.app_state.save_cross_section(self._app_config.cross_section_state_path)

        if self.app_state.is_computing:
            self._closing = True
            event.ignore()
            return

        super().closeEvent(event)

    def _update_3d_view(self, surfaces: list[PolyData], sweep_surface: PolyData) -> None:
        for actor in getattr(self, "_surface_actors", []):
            self._3d_plotter.remove_actor(actor, render=False)

        self._3d_plotter.remove_actor("sweep_surface", render=False)

        self._surface_actors = []

        for i, surface in enumerate(surfaces):
            actor = self._3d_plotter.add_mesh(
                surface, color=cc.glasbey[i % len(cc.glasbey)], show_edges=False, smooth_shading=False, render=False
            )
            self._surface_actors.append(actor)

        if sweep_surface.n_cells:
            self._3d_plotter.add_mesh(
                sweep_surface,
                name="sweep_surface",
                color="white",
                style="surface",
                opacity=0.2,
                show_edges=False,
                smooth_shading=False,
                lighting=False,
                render=False,
            )

        self._3d_plotter.render()


class CrossSectionSketcher(QGraphicsView):
    IMAGE_MARGIN: Final = 10
    PIXEL_RESOLUTION: Final = 32
    MIN_VISIBLE_PIXELS: Final = 11

    ZOOM_FACTOR: Final = 1.15

    def __init__(self, app_state: AppState) -> None:
        super().__init__()

        self.app_state = app_state

        self._build_view()
        self._build_dependencies()
        self._build_connections()

        self.setMinimumSize(300, 300)

        self._update_view()

    def _build_view(self) -> None:
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._panning = False
        self._pan_start = QPoint()

        self.setDragMode(QGraphicsView.NoDrag)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self._painting = False
        self._paint_value = True
        self._last_voxel = None

    def _build_dependencies(self) -> None:
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.scene.setSceneRect(self._image_rect)

        self._pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pixmap_item)

        self._build_center_marker()

    def _build_connections(self) -> None:
        self.app_state.cross_section_changed.connect(self._update_view)

    def showEvent(self, event) -> None:
        super().showEvent(event)

        if not getattr(self, "_initial_fit_done", False):
            self._fit_content()
            self._initial_fit_done = True

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

        if not hasattr(self, "_pixmap_item"):
            return

        rect = self._navigation_rect
        viewport = self.viewport().rect()

        min_scale = min(viewport.width() / rect.width(), viewport.height() / rect.height())

        current_scale = self.transform().m11()

        if current_scale < min_scale:
            self.scale(min_scale / current_scale, min_scale / current_scale)

        self._constrain_pan()
        self._update_view()

    @staticmethod
    def _build_rounded_image(binary: np.ndarray, resolution: int) -> np.ndarray:
        y, x = np.mgrid[:resolution, :resolution]

        center = resolution / 2
        half_size = resolution * 0.44
        radius = resolution * 0.12

        dx = np.maximum(np.abs(x + 0.5 - center) - (half_size - radius), 0)
        dy = np.maximum(np.abs(y + 0.5 - center) - (half_size - radius), 0)

        distance = np.sqrt(dx**2 + dy**2)
        alpha = np.clip(radius + 0.5 - distance, 0, 1)

        # Use configured colours for False and True
        false_color = np.array(
            [
                FALSE_COLOR.red(),
                FALSE_COLOR.green(),
                FALSE_COLOR.blue(),
            ],
            dtype=np.uint8,
        )

        true_color = np.array(
            [
                TRUE_COLOR.red(),
                TRUE_COLOR.green(),
                TRUE_COLOR.blue(),
            ],
            dtype=np.uint8,
        )

        colors = np.where(binary[..., None], true_color, false_color)

        # Combine pixels and rounded-square template
        rgba = np.empty(
            (
                binary.shape[0],
                binary.shape[1],
                resolution,
                resolution,
                4,
            ),
            dtype=np.uint8,
        )

        rgba[..., :3] = colors[:, :, None, None, :]
        rgba[..., 3] = (alpha * 255).astype(np.uint8)

        # Reorder into a standard image
        return rgba.transpose(0, 2, 1, 3, 4).reshape(
            binary.shape[0] * resolution,
            binary.shape[1] * resolution,
            4,
        )

    @staticmethod
    def _get_render_resolution(zoom: float) -> int:
        return min(256, 2.0 ** np.ceil(np.log2(max(zoom, 1e-6))))

    def _get_visible_binary(self) -> tuple[np.ndarray, int, int]:
        rect = self.mapToScene(self.viewport().rect()).boundingRect()
        nx, ny = self.app_state.domain_data.cross_section.shape

        x0 = max(0, int(np.floor(rect.left())))
        x1 = min(nx, int(np.ceil(rect.right())))

        y0 = max(0, int(np.floor(rect.top())))
        y1 = min(ny, int(np.ceil(rect.bottom())))

        cross_section = self.app_state.domain_data.cross_section
        ny = cross_section.shape[1]

        binary = cross_section[x0:x1, ny - y1 : ny - y0]
        binary = np.ascontiguousarray(np.flipud(binary.T))

        return binary, x0, y0

    def _update_view(self) -> None:
        binary, x0, y0 = self._get_visible_binary()

        if binary.size == 0:
            self._pixmap_item.setPixmap(QPixmap())
            return

        zoom = self.transform().m11() * self.devicePixelRatioF()
        resolution = self._get_render_resolution(zoom)

        if resolution < 1:
            factor = int(round(1 / resolution))

            # Align sampling to the global voxel grid
            offset_x = (-x0) % factor
            offset_y = (-y0) % factor

            binary = binary[offset_y::factor, offset_x::factor]

            x0 += offset_x
            y0 += offset_y

        if resolution <= 16:
            rgba = np.empty((*binary.shape, 4), dtype=np.uint8)
            rgba[binary] = (*TRUE_COLOR.getRgb()[:3], 255)
            rgba[~binary] = (*FALSE_COLOR.getRgb()[:3], 255)

            resolution = min(resolution, 1)
        else:
            rgba = self._build_rounded_image(binary, int(resolution))

        height, width = rgba.shape[:2]

        qimage = QImage(
            rgba.data,
            width,
            height,
            4 * width,
            QImage.Format.Format_RGBA8888,
        ).copy()

        self._pixmap_item.setPixmap(QPixmap.fromImage(qimage))
        self._pixmap_item.setScale(1 / resolution)
        self._pixmap_item.setPos(x0, y0)

    def _build_center_marker(self) -> None:
        nx, ny = self.app_state.domain_data.cross_section.shape

        size = 0.3
        center_i = nx / 2
        center_j = ny / 2

        path = QPainterPath()
        path.addRoundedRect(QRectF(-size / 2, -size / 2, size, size), 0.08, 0.08)

        self._center_marker = QGraphicsPathItem(path)
        self._center_marker.setBrush(QBrush(QColor("#ff5555")))
        self._center_marker.setPen(QPen(Qt.NoPen))
        self._center_marker.setPos(center_i, ny - center_j)
        self._center_marker.setZValue(10)

        self.scene.addItem(self._center_marker)

    @property
    def _image_rect(self) -> QRectF:
        nx, ny = self.app_state.domain_data.cross_section.shape
        return QRectF(0, 0, nx, ny)

    @property
    def _content_rect(self) -> QRectF:
        cross_section = self.app_state.domain_data.cross_section.copy()

        shape = cross_section.shape
        cross_section[shape[0] // 2, shape[1] // 2] = True

        i, j = np.nonzero(cross_section)

        x_min = i.min() - self.IMAGE_MARGIN
        x_max = i.max() + self.IMAGE_MARGIN + 1

        y_min = shape[1] - 1 - j.max() - self.IMAGE_MARGIN
        y_max = shape[1] - 1 - j.min() + self.IMAGE_MARGIN + 1

        return QRectF(float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)).intersected(
            self._image_rect
        )

    def _fit_content(self) -> None:
        rect = self._content_rect
        viewport = self.viewport().rect()

        scale = min(viewport.width() / rect.width(), viewport.height() / rect.height())

        self.resetTransform()
        self.scale(scale, scale)
        self.centerOn(rect.center())
        self._update_view()

    @property
    def _navigation_rect(self) -> QRectF:
        rect = self._content_rect

        return QRectF(
            rect.center().x() - rect.width(), rect.center().y() - rect.height(), rect.width() * 2, rect.height() * 2
        ).intersected(self._image_rect)

    def _constrain_pan(self) -> None:
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        bounds = self._navigation_rect

        center = visible.center()

        x = (
            bounds.center().x()
            if visible.width() >= bounds.width()
            else np.clip(
                center.x(),
                bounds.left() + visible.width() / 2,
                bounds.right() - visible.width() / 2,
            )
        )

        y = (
            bounds.center().y()
            if visible.height() >= bounds.height()
            else np.clip(
                center.y(),
                bounds.top() + visible.height() / 2,
                bounds.bottom() - visible.height() / 2,
            )
        )

        self.centerOn(float(x), float(y))

    def _mouse_to_voxel(self, pos: QPoint) -> tuple[int, int] | None:
        point = self.mapToScene(pos)
        nx, ny = self.app_state.domain_data.cross_section.shape

        i = int(np.floor(point.x()))
        j = ny - 1 - int(np.floor(point.y()))

        if 0 <= i < nx and 0 <= j < ny:
            return i, j

        return None

    def _paint_to(self, pos: QPoint) -> None:
        voxel = self._mouse_to_voxel(pos)

        if voxel is None:
            return

        start = self._last_voxel or voxel
        end = voxel

        steps = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1

        i = np.rint(np.linspace(start[0], end[0], steps)).astype(int)
        j = np.rint(np.linspace(start[1], end[1], steps)).astype(int)

        self.app_state.paint_cross_section(i, j, self._paint_value)
        self._last_voxel = end

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            voxel = self._mouse_to_voxel(event.pos())

            if voxel is not None:
                self._painting = True
                self._paint_value = not bool(self.app_state.domain_data.cross_section[voxel])
                self._last_voxel = None
                self._paint_to(event.pos())

            event.accept()
            return

        if event.button() == Qt.RightButton:
            self._panning = True
            self._pan_start = event.pos()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._painting:
            self._paint_to(event.pos())
            event.accept()
            return

        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()

            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._constrain_pan()
            self._update_view()

            event.accept()
            self.viewport().update()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._painting = False
            self._last_voxel = None

            self.app_state.update_sweep()

            event.accept()
            return

        if event.button() == Qt.RightButton:
            self._panning = False
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QMouseEvent) -> None:
        factor = self.ZOOM_FACTOR if event.angleDelta().y() > 0 else 1 / self.ZOOM_FACTOR

        image_rect = self._navigation_rect
        viewport_rect = self.viewport().rect()

        min_scale = min(
            viewport_rect.width() / image_rect.width(),
            viewport_rect.height() / image_rect.height(),
        )

        max_scale = (
            min(
                viewport_rect.width(),
                viewport_rect.height(),
            )
            / self.MIN_VISIBLE_PIXELS
        )

        current_scale = self.transform().m11()
        new_scale = np.clip(current_scale * factor, min_scale, max_scale)

        self.scale(new_scale / current_scale, new_scale / current_scale)
        self._constrain_pan()
        self._update_view()
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            self._fit_content()
            event.accept()
            return

        super().mouseDoubleClickEvent(event)


class SweepWorker(QThread):
    result_ready = Signal(int, object, object, object, object, object)
    failed = Signal(str)

    def __init__(self, domain_data: DomainData, cross_section: np.ndarray, revision: int) -> None:
        super().__init__()

        self._domain_data = domain_data
        self._cross_section = cross_section
        self._revision = revision

    def run(self) -> None:
        try:
            confidence, indices, labels, surfaces = self._domain_data.compute_segmentation(self._cross_section)
            sweep_surface = self._domain_data.compute_sweep_surface(self._cross_section)

            self.result_ready.emit(self._revision, confidence, indices, labels, surfaces, sweep_surface)

        except Exception as exc:
            self.failed.emit(str(exc))


class LoadingSpinner(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.setFixedSize(48, 48)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        self._angle = 0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)

    def _rotate(self) -> None:
        self._angle = (self._angle + 10) % 360
        self.update()

    def setVisible(self, visible: bool) -> None:
        super().setVisible(visible)

        if visible:
            self._timer.start(16)
        else:
            self._timer.stop()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor("#a3eaff"), 4)
        pen.setCapStyle(Qt.RoundCap)

        painter.setPen(pen)
        painter.drawArc(
            self.rect().adjusted(6, 6, -6, -6),
            self._angle * 16,
            270 * 16,
        )


def run_flowtier_studio(app_config: AppConfig, alg_config: AlgConfig) -> int:
    with console.status("Building domain data..."):
        domain_data = DomainData(app_config.surface_path, app_config.center_lines_path, alg_config)

    for warning in domain_data.warnings:
        console.print(warning)

    console.print("[green]✔ Domain data built[/green]")

    with console.status("Building AppState..."):
        app_state, mode = AppState.build(domain_data, app_config)

    console.print("[green]✔ AppState built[/green]")

    if app_config.mode != mode:
        console.print(
            f"[bold yellow]⚠ Warning:[/bold yellow] "
            f"Cross-section file not found: {app_config.cross_section_state_path}.\n"
            f"Starting in {mode} mode instead of {app_config.mode} mode"
        )

    if mode == Mode.SILENT:
        with console.status("Computing segmentation..."):
            app_state.update_segmentation()

    else:
        app = QApplication.instance() or QApplication(sys.argv)
        app.setStyleSheet(STYLE_SHEET)

        with Live(Text.from_markup("Opening Flowtier Studio..."), console=console, transient=True):
            window = FlowtierStudio(app_state, app_config)
            window.show()
            exit_code = app.exec()

        if exit_code != 0:
            return exit_code

        console.print("[green]✔ Segmentation handling complete[/green]")

    with console.status("Writing outputs..."):
        app_state.save_segmentation(app_config.output_path)

    try:
        rel = app_config.output_path.relative_to(Path.cwd())
    except ValueError:
        rel = app_config.output_path  # fallback to absolute path
    console.print(f"💾 Wrote labelled voxelisation to {rel}", highlight=False)

    return 0
