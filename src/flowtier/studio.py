import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
)
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
