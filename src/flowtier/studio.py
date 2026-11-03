import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter

import numpy as np
import pyvista as pv
from numba import njit, prange
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
        start = perf_counter()

        distance = self._domain_data.void_distance_field
        precision = self._domain_data.precision

        binary = distance > 0

        candidates = binary & (distance <= precision)

        # Include the six domain boundaries
        candidates[0, :, :] |= binary[0, :, :]
        candidates[-1, :, :] |= binary[-1, :, :]
        candidates[:, 0, :] |= binary[:, 0, :]
        candidates[:, -1, :] |= binary[:, -1, :]
        candidates[:, :, 0] |= binary[:, :, 0]
        candidates[:, :, -1] |= binary[:, :, -1]

        surface = binary_to_voxel_surface(binary, candidates, precision)

        surface.translate(
            self._domain_data.boundaries[0] - precision / 2,
            inplace=True,
        )

        print(f"Surface extraction: {perf_counter() - start:.3f} s")

        self._3d_plotter.add_mesh(
            surface,
            name="voxel_surface",
            color="cornflowerblue",
            style="surface",
            show_edges=False,
            smooth_shading=False,
        )

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


_DIRECTIONS = np.array(
    [
        [-1, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1],
    ],
    dtype=np.int8,
)

_CORNERS = np.array(
    [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    ],
    dtype=np.int8,
)


@njit(inline="always")
def _exposed_faces(binary, i, j, k):
    nx, ny, nz = binary.shape

    return (
        i == 0 or not binary[i - 1, j, k],
        i == nx - 1 or not binary[i + 1, j, k],
        j == 0 or not binary[i, j - 1, k],
        j == ny - 1 or not binary[i, j + 1, k],
        k == 0 or not binary[i, j, k - 1],
        k == nz - 1 or not binary[i, j, k + 1],
    )


@njit(parallel=True, cache=True)
def _count_candidate_faces(binary, indices):
    nx, ny, nz = binary.shape
    counts = np.empty(len(indices), dtype=np.uint8)

    for p in prange(len(indices)):
        index = indices[p]

        i = index // (ny * nz)
        j = (index // nz) % ny
        k = index % nz

        if not binary[i, j, k]:
            counts[p] = 0
            continue

        exposed = _exposed_faces(binary, i, j, k)

        count = 0
        for d in range(6):
            count += exposed[d]

        counts[p] = count

    return counts


@njit(parallel=True, cache=True)
def _build_candidate_faces(binary, indices, offsets, voxel_size, corners, vertices, faces):
    ny, nz = binary.shape[1:]

    for p in prange(len(indices)):
        index = indices[p]

        i = index // (ny * nz)
        j = (index // nz) % ny
        k = index % nz

        if not binary[i, j, k]:
            continue

        exposed = _exposed_faces(binary, i, j, k)
        face_id = offsets[p]

        for d in range(6):
            if not exposed[d]:
                continue

            base = face_id * 4
            faces[face_id, 0] = 4

            for c in range(4):
                vertices[base + c, 0] = (i + corners[d, c, 0]) * voxel_size
                vertices[base + c, 1] = (j + corners[d, c, 1]) * voxel_size
                vertices[base + c, 2] = (k + corners[d, c, 2]) * voxel_size
                faces[face_id, c + 1] = base + c

            face_id += 1


def binary_to_voxel_surface(
    binary: np.ndarray,
    candidates: np.ndarray,
    voxel_size: float,
) -> pv.PolyData:
    # Candidate voxel indices
    indices = np.flatnonzero(candidates)

    # Count exposed faces
    counts = _count_candidate_faces(binary, indices)

    # Calculate each candidate's output offset
    offsets = np.empty(len(indices) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    n_faces = int(offsets[-1])

    # Allocate output
    vertices = np.empty((n_faces * 4, 3), dtype=np.float32)
    faces = np.empty((n_faces, 5), dtype=np.int64)

    # Build geometry in parallel
    _build_candidate_faces(binary, indices, offsets, voxel_size, _CORNERS, vertices, faces)

    return pv.PolyData(vertices, faces.ravel())
