import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import (
    QObject,
    Signal,
)
from PySide6.QtWidgets import (
    QApplication,
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
        bounds = self._domain_data.boundaries
        self._3d_plotter.show_grid(
            bounds=(0, bounds[1, 0], 0, bounds[1, 1], 0, bounds[1, 2]),
            xtitle="Axis 0",
            ytitle="Axis 1",
            ztitle="Axis 2",
        )

        layout.addWidget(self._3d_plotter.interactor)

        return container

    def _update_3d_view(self) -> None:
        start = perf_counter()

        for surface in self._domain_data.plotting_surfaces:
            self._3d_plotter.add_mesh(
                surface,
                name="voxel_surface",
                color="cornflowerblue",
                style="surface",
                show_edges=False,
                smooth_shading=False,
            )

        print(f"Surface plotting: {perf_counter() - start:.3f} s")

    def _build_cross_section_sketcher(self) -> QWidget:
        container = QWidget()

        return container


class CrossSectionView(QGraphicsView):
    def __init__(self, app_state: AppState) -> None:
        super().__init__()

        self.app_state = app_state


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
