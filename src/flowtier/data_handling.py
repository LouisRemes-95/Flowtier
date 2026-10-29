from pathlib import Path

import igl
import numpy as np
from rich.progress import BarColumn, Progress, TextColumn

from flowtier.common import UserError, console


class DomainData:
    def __init__(self, surface_path: Path, center_lines_path: Path, precision: float) -> None:
        self.warnings: list[str] = []

        if precision <= 0:
            raise UserError("precision must be positive.")
        self.precision = precision

        self._set_and_adjust_boundaries(surface_path)
        self._build_void_distance_field(surface_path, subdivisions=2)

    @staticmethod
    def _read_ply_bounds(surface_path: Path) -> np.ndarray | None:
        with surface_path.open("rb") as file:
            for line in file:
                if line.startswith(b"comment domain boundaries "):
                    bounds = np.fromstring(
                        line.decode().removeprefix("comment domain boundaries "),
                        sep=" ",
                    )
                    return bounds.reshape(2, 3)

                if line.startswith(b"end_header"):
                    break

        return None

    def _set_and_adjust_boundaries(self, surface_path: Path) -> None:
        boundaries = self._read_ply_bounds(surface_path)

        if boundaries is None:
            vertices, _ = igl.read_triangle_mesh(surface_path)
            boundaries = np.array([np.min(vertices, axis=0), vertices.max(axis=0)])

        self._grid_shape = np.ceil((boundaries[1, :] - boundaries[0, :]) / self.precision).astype(int)

        adjusted_boundaries = np.vstack((boundaries[0], boundaries[0] + (self._grid_shape - 1) * self.precision))

        if not np.array_equal(boundaries[1], adjusted_boundaries[1]):
            difference = np.abs(adjusted_boundaries[1] - boundaries[1])
            nonzero = difference[difference > 0]
            decimals = max(2, int(np.ceil(-np.log10(nonzero.min()))) + 1) if nonzero.size else 2

            with np.printoptions(precision=decimals, floatmode="fixed", suppress=True):
                previous = np.array2string(boundaries)
                updated = np.array2string(adjusted_boundaries)

            indent = " " * 12

            warning = (
                "[bold yellow]⚠ Warning:[/bold yellow] Boundaries adjusted to match desired precision.\n"
                f"  Previous: {previous.replace(chr(10), chr(10) + indent)}\n"
                f"  Updated:  {updated.replace(chr(10), chr(10) + indent)}"
            )

            self.warnings.append(warning)

        self.boundaries = adjusted_boundaries

    def _build_void_distance_field(self, surface_path: Path, chunk_size: int = 100_000, subdivisions: int = 0) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        vertices, faces = igl.read_triangle_mesh(surface_path)

        vertices, faces = igl.loop(vertices, faces, subdivisions)

        self.void_distance_field = np.empty(self._grid_shape, dtype=np.float32)
        flat_distance_field = self.void_distance_field.ravel()

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(complete_style="blue", finished_style="blue"),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("[white]{task.elapsed:0.1f}s"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Computing void distance field", total=flat_distance_field.size)

            for start_index in range(0, flat_distance_field.size, chunk_size):
                end_index = min(start_index + chunk_size, flat_distance_field.size)

                flat_indices = np.arange(start_index, end_index)
                grid_indices = np.column_stack(np.unravel_index(flat_indices, self._grid_shape))
                points = self.boundaries[0, :] + grid_indices * self.precision

                distances, _, _, _ = igl.signed_distance(points, vertices, faces, igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)

                flat_distance_field[flat_indices] = distances
                progress.update(task, completed=end_index)
