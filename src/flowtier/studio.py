from pathlib import Path
from enum import Enum
from dataclasses import dataclass


class Mode(str, Enum):
    RESET = "reset"
    RESUME = "resume"
    SILENT = "silent"


@dataclass(frozen=True)
class AppConfig:
    surface_path: Path
    center_lines_path: Path
    output_path: Path
    mode: Mode
    autosave_interval_ms: int = 500


def run_flowtier_studio(app_config: AppConfig) -> int:
    pass