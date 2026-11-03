import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter

import pyvista as pv
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QWidget,
)
from pyvistaqt import QtInteractor
from rich.live import Live
from rich.text import Text

from flowtier.common import console
from flowtier.data_handling import DomainData

STYTE_SHEET = """
"""


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
    def __init__(self, domain_data: DomainData, app_config: AppConfig) -> None:
        super().__init__()

        self._domain_data = domain_data
        self._app_config = app_config

        self._build_ui()
        self._update_3d_view()

    def _build_ui(self) -> None:
        self.setWindowTitle("Flowtier")
        self.resize(1000, 700)

        main_container = QWidget()
        self.setCentralWidget(main_container)
        main_layout = QHBoxLayout(main_container)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        left_container = self._build_3d_view()
        right_container = self._build_cross_section_sketcher()

        main_layout.addWidget(left_container, 1)
        main_layout.addWidget(right_container, 1)

    def _build_3d_view(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self._3d_plotter = QtInteractor(container)
        layout.addWidget(self._3d_plotter.interactor)

        return container

    def _update_3d_view(self) -> None:
        grid = pv.ImageData()
        grid.dimensions = self._domain_data.grid_shape + 1
        grid.spacing = (self._domain_data.precision,) * 3

        start = perf_counter()

        grid.cell_data["values"] = (self._domain_data.void_distance_field > 0).ravel(order="F")
        surface = grid.threshold(0.5).extract_surface(algorithm="dataset_surface")

        elapsed = perf_counter() - start
        print(f"Elapsed time: {elapsed:.3f} s")

        self._3d_plotter.add_mesh(
            surface,
            name="binary_mesh",
            color="cornflowerblue",
            show_edges=True,
            edge_color="black",
            line_width=1,
        )

        self._3d_plotter.show_grid(xtitle="Axis 0", ytitle="Axis 1", ztitle="Axis 2")

    def _build_cross_section_sketcher(self) -> QWidget:
        container = QWidget()

        return container


def run_flowtier_studio(app_config: AppConfig, alg_config: AlgConfig) -> int:
    with console.status("Building domain data..."):
        domain_data = DomainData(app_config.surface_path, app_config.center_lines_path, alg_config.precision)

    for warning in domain_data.warnings:
        console.print(warning)

    console.print("[green]✔ Domain data built[/green]")

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYTE_SHEET)

    with Live(Text.from_markup("Opening CTidy Studio..."), console=console, transient=True):
        window = FlowtierStudio(domain_data, app_config)
        window.show()
        exit_code = app.exec()

    if exit_code != 0:
        return exit_code

    console.print("[green]✔ Segmentation handling complete[/green]")

    return 0
