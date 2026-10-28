from pathlib import Path

import igl
import numpy as np
from rich.progress import BarColumn, Progress, TextColumn

from flowtier.common import UserError, console


class DomainData:
    def __init__(self, surface_path: Path, center_lines_path: Path, precision: float) -> None:

        if precision <= 0:
            raise UserError("precision must be positive.")
        self.precision = precision

        self.set_boundaries(surface_path)
        self.build_void_distance_field(surface_path, subdivisions=2)

    @staticmethod
    def read_ply_bounds(surface_path: Path) -> np.ndarray | None:
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

    def set_boundaries(self, surface_path: Path) -> None:
        boundaries = self.read_ply_bounds(surface_path)

        if boundaries is None:
            vertices, _ = igl.read_triangle_mesh(surface_path)
            boundaries = np.array([np.min(vertices, axis=0), vertices.max(axis=0)])

        self.boundaries = boundaries

    def build_void_distance_field(self, surface_path: Path, chunk_size: int = 100_000, subdivisions: int = 0) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        vertices, faces = igl.read_triangle_mesh(surface_path)

        vertices, faces = igl.loop(vertices, faces, subdivisions)

        shape = np.ceil((self.boundaries[1, :] - self.boundaries[0, :]) / self.precision).astype(int)

        self.void_distance_field = np.empty(shape, dtype=np.float32)
        flat_distance_field = self.void_distance_field.ravel()

        new_boundaries = np.vstack((self.boundaries[0], self.boundaries[0] + (shape - 1) * self.precision))

        if not np.array_equal(self.boundaries[1], new_boundaries[1]):
            difference = np.abs(new_boundaries[1] - self.boundaries[1])
            nonzero = difference[difference > 0]

            decimals = max(2, int(np.ceil(-np.log10(nonzero.min()))) + 1) if nonzero.size else 2

            with np.printoptions(precision=decimals, floatmode="fixed", suppress=True):
                previous = np.array2string(self.boundaries)
                updated = np.array2string(new_boundaries)

                console.print(
                    "\n[bold yellow]⚠ Warning:[/bold yellow] Boundaries adjusted to match desired precision.\n"
                    f"  Previous: {previous.replace(chr(10), chr(10) + ' ' * 12)}\n"
                    f"  Updated:  {updated.replace(chr(10), chr(10) + ' ' * 12)}\n"
                )

            self.boundaries[1, :] = new_boundaries[1]

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
                grid_indices = np.column_stack(np.unravel_index(flat_indices, shape))
                points = self.boundaries[0, :] + grid_indices * self.precision

                distances, _, _, _ = igl.signed_distance(points, vertices, faces, igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)

                flat_distance_field[flat_indices] = distances
                progress.update(task, completed=end_index)
