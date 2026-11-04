import math
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter

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
    QPixmap,
    QTransform,
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

BACKGROUND_COLOR = QColor("#353535")

STYTE_SHEET = """
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
        self.setStyleSheet("""
            QMainWindow {
                background-color: #353535;
            }
        """)

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

        bounds = self.app_state.domain_data.boundaries
        self._3d_plotter.show_grid(
            bounds=(0, bounds[1, 0], 0, bounds[1, 1], 0, bounds[1, 2]),
            xtitle="Axis 0",
            ytitle="Axis 1",
            ztitle="Axis 2",
        )

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


class PixelPixmapItem(QGraphicsPixmapItem):
    def paint(self, painter, option, widget=None):
        if self.scene() is None or not self.scene().views():
            return

        view = self.scene().views()[0]
        visible = view.mapToScene(view.viewport().rect()).boundingRect()

        x_min = math.ceil(visible.left())
        y_min = math.ceil(visible.top())
        x_max = math.floor(visible.right())
        y_max = math.floor(visible.bottom())

        clip = QRectF(x_min, y_min, max(0, x_max - x_min), max(0, y_max - y_min))

        painter.save()
        painter.setClipRect(self.mapRectFromScene(clip), Qt.ClipOperation.IntersectClip)
        super().paint(painter, option, widget)
        painter.restore()


class CrossSectionSketcher(QGraphicsView):
    IMAGE_MARGIN = 10
    GRID_RESOLUTION = 8

    def __init__(self, app_state: AppState) -> None:
        super().__init__()

        self.app_state = app_state

        self._build_view()
        self._build_dependencies()
        self._build_conections()

        self.setMinimumSize(300, 300)
        self.setStyleSheet("""
            QGraphicsView {
                border: 1px solid #353535;
            }
        """)

    def _build_view(self) -> None:
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._panning = False
        self._pan_start = QPoint()

        self.setDragMode(QGraphicsView.NoDrag)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self.setBackgroundBrush(BACKGROUND_COLOR)

    def _build_dependencies(self) -> None:

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self._pixmap_item = PixelPixmapItem()
        self.scene.addItem(self._pixmap_item)

        self.scene.addItem(self._grid_overlay())

    def _build_conections(self) -> None:
        self.app_state.domain_data_changed.connect(self._update_view)

    def showEvent(self, event) -> None:
        super().showEvent(event)

        self._fit_content()

    def _update_view(self) -> None:
        image = np.ascontiguousarray(self.app_state.domain_data.cross_section.astype(np.uint8) * 255)

        qimage = QImage(
            image.data, image.shape[1], image.shape[0], image.shape[1], QImage.Format.Format_Grayscale8
        ).copy()

        self._pixmap_item.setPixmap(QPixmap.fromImage(qimage))
        self.scene.setSceneRect(self._pixmap_item.boundingRect())

        self._fit_content()

    def _fit_content(self) -> None:
        cross_section = self.app_state.domain_data.cross_section
        rows, cols = np.nonzero(cross_section)

        if rows.size == 0:
            return

        center = np.array(cross_section.shape) // 2

        x_min = min(cols.min(), center[1]) - self.IMAGE_MARGIN + 1
        x_max = max(cols.max(), center[1]) + self.IMAGE_MARGIN + 1

        y_min = min(rows.min(), center[0]) - self.IMAGE_MARGIN + 1
        y_max = max(rows.max(), center[0]) + self.IMAGE_MARGIN + 1

        rect = QRectF(
            float(x_min),
            float(y_min),
            float(x_max - x_min),
            float(y_max - y_min),
        )

        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def _grid_overlay(self) -> QGraphicsPixmapItem:
        grid_overlay = QGraphicsPixmapItem()
        grid_overlay.setZValue(1)

        cross_section_shape = np.array(self.app_state.domain_data.cross_section.shape)
        overlay_shape = cross_section_shape * self.GRID_RESOLUTION + 1

        overlay = np.zeros(overlay_shape, dtype=bool)
        overlay[:: self.GRID_RESOLUTION, :] = True
        overlay[:, :: self.GRID_RESOLUTION] = True

        rgba = np.zeros((*overlay.shape, 4), dtype=np.uint8)
        rgba[overlay] = [53, 53, 53, 255]  # Opaque blue; False remains transparent

        height, width = overlay.shape
        image = QImage(rgba.data, width, height, 4 * width, QImage.Format.Format_RGBA8888)

        grid_overlay.setPixmap(QPixmap.fromImage(image.copy()))
        grid_overlay.setTransform(QTransform.fromScale(*tuple(cross_section_shape / overlay_shape)))

        return grid_overlay


def run_flowtier_studio(app_config: AppConfig, alg_config: AlgConfig) -> int:
    with console.status("Building domain data..."):
        domain_data = DomainData(app_config.surface_path, app_config.center_lines_path, alg_config.precision)

    for warning in domain_data.warnings:
        console.print(warning)

    console.print("[green]✔ Domain data built[/green]")

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYTE_SHEET)

    with Live(Text.from_markup("Opening CTidy Studio..."), console=console, transient=True):
        window = FlowtierStudio(AppState(domain_data), app_config)
        window.show()
        exit_code = app.exec()

    if exit_code != 0:
        return exit_code

    console.print("[green]✔ Segmentation handling complete[/green]")

    return 0
