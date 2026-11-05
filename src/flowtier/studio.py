import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Final

import numpy as np
from PySide6.QtCore import (
    QObject,
    QPoint,
    QRectF,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QMainWindow,
    QWidget,
)
from pyvistaqt import QtInteractor
from rich.live import Live
from rich.text import Text

from flowtier.common import console
from flowtier.data_handling import DomainData

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
    domain_data_changed = Signal()

    def __init__(self, domain_data: DomainData) -> None:
        super().__init__()
        self._domain_data = domain_data

    @property
    def domain_data(self):
        return self._domain_data


class Mode(StrEnum):
    RESET = "reset"
    RESUME = "resume"
    SILENT = "silent"


@dataclass(frozen=True)
class AlgConfig:
    precision: float


@dataclass(frozen=True)
class AppConfig:
    surface_path: Path
    center_lines_path: Path
    output_path: Path
    mode: Mode
    autosave_interval_ms: int = 500


class FlowtierStudio(QMainWindow):
    def __init__(self, app_state: AppState, app_config: AppConfig) -> None:
        super().__init__()

        self.app_state = app_state
        self._app_config = app_config

        self._build_ui()
        self._update_3d_view()

        self.app_state.domain_data_changed.emit()

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

        return self._3d_plotter

    def _update_3d_view(self) -> None:
        start = perf_counter()

        for surface in self.app_state.domain_data.plotting_surfaces:
            self._3d_plotter.add_mesh(
                surface,
                name="voxel_surface",
                color="cornflowerblue",
                style="surface",
                show_edges=False,
                smooth_shading=False,
            )

        print(f"Surface plotting: {perf_counter() - start:.3f} s")


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
        self._build_conections()

        self.setMinimumSize(300, 300)

    def _build_view(self) -> None:
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._panning = False
        self._pan_start = QPoint()

        self.setDragMode(QGraphicsView.NoDrag)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def _build_dependencies(self) -> None:
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self._pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pixmap_item)

    @staticmethod
    def _build_rounded_image(
        binary: np.ndarray,
        resolution: int = 8,
    ) -> np.ndarray:
        # Create a small rounded-square opacity mask
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

    def _build_conections(self) -> None:
        self.app_state.domain_data_changed.connect(self._update_view)
        pass

    def showEvent(self, event) -> None:
        super().showEvent(event)

        self._fit_content()

    def _update_view(self) -> None:
        cross_section = self.app_state.domain_data.cross_section

        binary = np.flipud(cross_section.T)

        rgba = self._build_rounded_image(
            binary,
            resolution=self.PIXEL_RESOLUTION,
        )

        height, width = rgba.shape[:2]

        qimage = QImage(
            rgba.data,
            width,
            height,
            4 * width,
            QImage.Format.Format_RGBA8888,
        ).copy()

        self._pixmap_item.setPixmap(QPixmap.fromImage(qimage))
        self._pixmap_item.setScale(1 / self.PIXEL_RESOLUTION)

        self.scene.setSceneRect(
            0,
            0,
            cross_section.shape[0],
            cross_section.shape[1],
        )

    def _fit_content(self) -> None:
        cross_section = self.app_state.domain_data.cross_section
        i, j = np.nonzero(cross_section)

        if i.size == 0:
            return

        shape = cross_section.shape
        center = np.array(shape) // 2

        x_min = min(i.min(), center[0]) - self.IMAGE_MARGIN
        x_max = max(i.max(), center[0]) + self.IMAGE_MARGIN + 1

        y_min = shape[1] - 1 - max(j.max(), center[1]) - self.IMAGE_MARGIN
        y_max = shape[1] - 1 - min(j.min(), center[1]) + self.IMAGE_MARGIN + 1

        rect = QRectF(
            float(x_min),
            float(y_min),
            float(x_max - x_min),
            float(y_max - y_min),
        )

        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            self._panning = True
            self._pan_start = event.pos()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()

            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

            event.accept()
            self.viewport().update()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            self._panning = False
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QMouseEvent) -> None:
        factor = self.ZOOM_FACTOR if event.angleDelta().y() > 0 else 1 / self.ZOOM_FACTOR

        image_rect = self.scene.sceneRect()
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
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            self._fit_content()
            event.accept()
            return

        super().mouseDoubleClickEvent(event)


def run_flowtier_studio(app_config: AppConfig, alg_config: AlgConfig) -> int:
    with console.status("Building domain data..."):
        domain_data = DomainData(app_config.surface_path, app_config.center_lines_path, alg_config.precision)

    for warning in domain_data.warnings:
        console.print(warning)

    console.print("[green]✔ Domain data built[/green]")

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)

    with Live(Text.from_markup("Opening CTidy Studio..."), console=console, transient=True):
        window = FlowtierStudio(AppState(domain_data), app_config)
        window.show()
        exit_code = app.exec()

    if exit_code != 0:
        return exit_code

    console.print("[green]✔ Segmentation handling complete[/green]")

    return 0
