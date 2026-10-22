from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from flowtier.common import console
from flowtier.data_handling import DomainData


class Mode(str, Enum):
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


def run_flowtier_studio(app_config: AppConfig, alg_config: AlgConfig) -> int:
    with console.status("Building domain data..."):
        domain_data = DomainData(
            app_config.surface_path, app_config.center_lines_path, alg_config.precision
        )

    console.print("[green]✔ Domain data build[/green]")
