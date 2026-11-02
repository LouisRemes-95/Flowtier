from pathlib import Path

import igl
import numpy as np
import trimesh
from numba import njit, prange
from rich.progress import BarColumn, Progress, TextColumn
from scipy.ndimage import binary_dilation, binary_propagation, distance_transform_cdt

from flowtier.common import UserError, console


class DomainData:
    def __init__(self, surface_path: Path, center_lines_path: Path, precision: float) -> None:
        self.warnings: list[str] = []

        if precision <= 0:
            raise UserError("precision must be positive.")
        self._precision = precision

        self._set_and_adjust_boundaries(surface_path)
        # self._build_void_distance_field(surface_path, subdivisions=2, chunk_size=1_000_000)
        self._build_center_lines_dependencies(center_lines_path)

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

        self._grid_shape = np.ceil((boundaries[1, :] - boundaries[0, :]) / self._precision).astype(int)

        adjusted_boundaries = np.vstack((boundaries[0], boundaries[0] + (self._grid_shape - 1) * self._precision))

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
        for _ in range(subdivisions):
            smooth_vertices, smooth_faces = igl.loop(vertices, faces, 1)
            fixed_vertices, fixed_faces = igl.upsample(vertices, faces, 1)

            assert np.array_equal(smooth_faces, fixed_faces)

            tolerance = self._precision * 0.1

            boundary_mask = np.any(
                np.isclose(
                    fixed_vertices[:, :, None],
                    self.boundaries.T[None, :, :],
                    atol=tolerance,
                    rtol=0,
                ),
                axis=(1, 2),
            )

            smooth_vertices[boundary_mask] = fixed_vertices[boundary_mask]

            vertices, faces = smooth_vertices, smooth_faces

        mesh = trimesh.Trimesh(vertices, faces, process=False)

        shifted_mesh = mesh.copy()
        shifted_mesh.apply_translation(-self.boundaries[0, :])

        with console.status("Voxelising the interface..."):
            edge_lengths = shifted_mesh.edges_unique_length
            pitch = self._precision

            edge_factor = 1.1
            max_iter = max(
                0,
                int(np.ceil(np.log2(edge_lengths.max() * edge_factor / pitch))),
            )

            voxel_grid = shifted_mesh.voxelized(
                pitch=pitch,
                method="subdivide",
                edge_factor=edge_factor,
                max_iter=max_iter,
            )

            indices = np.rint(voxel_grid.points / self._precision).astype(int)

            shape = np.asarray(self._grid_shape)
            valid = np.all((indices >= 0) & (indices < shape), axis=1)

            binary_interface = np.zeros(tuple(shape), dtype=bool)
            binary_interface[tuple(indices[valid].T)] = True

            binary_dilation(binary_interface, iterations=3, output=binary_interface)

        self.void_distance_field = np.full(self._grid_shape, np.nan, dtype=np.float32)

        interface_indices = np.flatnonzero(binary_interface)
        flat_distance_field = self.void_distance_field.ravel()

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(complete_style="blue", finished_style="blue"),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("[white]{task.elapsed:0.1f}s"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "Computing void distance field at interface",
                total=len(interface_indices),
            )

            for start_index in range(0, len(interface_indices), chunk_size):
                chunk_indices = interface_indices[start_index : start_index + chunk_size]

                grid_indices = np.column_stack(np.unravel_index(chunk_indices, self._grid_shape))
                points = self.boundaries[0, :] + grid_indices * self._precision

                distances, _, _, _ = igl.signed_distance(
                    points,
                    vertices,
                    faces,
                    igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
                )

                flat_distance_field[chunk_indices] = distances
                progress.update(task, advance=len(chunk_indices))

        with console.status("Propagating the void binary..."):
            structure = np.ones((3,) * 3, dtype=bool)
            binary_voids = binary_propagation(self.void_distance_field < 0, mask=~binary_interface, structure=structure)

        with console.status("Computing the rest of the void distance field..."):
            void_sdf = (
                distance_transform_cdt(~binary_interface, metric="chessboard").astype(np.float32) * self._precision
            )

            void_sdf[binary_voids == 1] = -void_sdf[binary_voids == 1]
            self.void_distance_field[np.isnan(self.void_distance_field)] = void_sdf[np.isnan(self.void_distance_field)]

        print(self.void_distance_field.size)

    def _build_center_lines_dependencies(self, center_lines_path: Path) -> None:
        with np.load(center_lines_path) as data:
            if len(data.files) != 1:
                raise UserError("The .npz file must contain exactly one array.")

            center_lines = data[data.files[0]].astype(np.float32)

        if center_lines.ndim != 3 or center_lines.shape[1:] != (2, 3):
            raise UserError(f"Expected an (N, 2, 3) array, got {center_lines.shape}.")

        center_lines = (center_lines - self.boundaries[0]) / self._precision

        self._build_affine_projection_coefficients(center_lines)
        self._build_closest_center_lines()

    def _build_affine_projection_coefficients(self, center_lines: np.ndarray) -> None:
        origins = center_lines.mean(axis=1)

        directions = center_lines[:, 1] - center_lines[:, 0]
        norms = np.linalg.norm(directions, axis=1)

        if not np.all(np.isfinite(norms)) or np.any(norms <= np.finfo(np.float32).eps):
            raise ValueError("Center lines must have finite, distinct endpoints.")

        directions /= norms[:, None]

        vertical = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        ex = np.cross(vertical, directions)

        norms_ex = np.linalg.norm(ex, axis=1)
        is_vertical = norms_ex <= np.finfo(np.float32).eps

        ex[is_vertical] = [1.0, 0.0, 0.0]
        ex[~is_vertical] /= norms_ex[~is_vertical, None]

        ey = np.cross(directions, ex)

        # Affine coefficients: a0, ax, ay, b0, bx, by, bz
        self._affine_projection_coefficients = np.column_stack(
            (
                -np.sum(origins * ex, axis=1),
                ex[:, 0],
                ex[:, 1],
                -np.sum(origins * ey, axis=1),
                ey[:, 0],
                ey[:, 1],
                ey[:, 2],
            )
        ).astype(np.float32)

    def _build_closest_center_lines(self, chunk_size: int = 1_000_000) -> None:
        n_lines = self._affine_projection_coefficients.shape[0]

        if n_lines < 4:
            raise ValueError("At least four center lines are required.")

        shape = tuple(self._grid_shape)
        n_voxels = int(np.prod(shape))

        dtype = np.uint8 if n_lines <= 256 else np.uint16

        self._closest_center_lines = np.empty((*shape, 4), dtype=dtype)
        flat_output = self._closest_center_lines.reshape(-1, 4)

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(complete_style="blue", finished_style="blue"),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("[white]{task.elapsed:0.1f}s"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Finding closest centre lines", total=n_voxels)

            for start in range(0, n_voxels, chunk_size):
                stop = min(start + chunk_size, n_voxels)

                indices = np.arange(start, stop)
                coordinates = np.column_stack(np.unravel_index(indices, shape)).astype(np.float32)

                # Numba computes into a temporary int32 array
                closest = np.empty((stop - start, 4), dtype=np.int32)

                _find_closest_center_lines(
                    coordinates,
                    self._affine_projection_coefficients,
                    closest,
                )

                flat_output[start:stop] = closest

                progress.update(task, advance=stop - start)


@njit(inline="always")
def _bilinear_interpolate(distance_field: np.ndarray, u: float, v: float) -> float:
    nx, ny = distance_field.shape

    i = int(np.floor(u))
    j = int(np.floor(v))

    if i < 0 or i >= nx - 1 or j < 0 or j >= ny - 1:
        return np.inf

    du = u - i
    dv = v - j

    return (1 - du) * ((1 - dv) * distance_field[i, j] + dv * distance_field[i, j + 1]) + du * (
        (1 - dv) * distance_field[i + 1, j] + dv * distance_field[i + 1, j + 1]
    )


@njit(parallel=True, cache=True)
def _find_closest_center_lines(
    coordinates: np.ndarray,
    coefficients: np.ndarray,
    output: np.ndarray,
) -> None:
    n_lines = coefficients.shape[0]
    n_closest = output.shape[1]

    for p in prange(coordinates.shape[0]):
        i, j, k = coordinates[p]

        best_distances = np.full(n_closest, np.inf)
        best_indices = np.full(n_closest, -1, dtype=np.int32)

        for line in range(n_lines):
            a0, ax, ay, b0, bx, by, bz = coefficients[line]

            u = a0 + ax * i + ay * j
            v = b0 + bx * i + by * j + bz * k

            distance_squared = u * u + v * v

            worst = np.argmax(best_distances)

            if distance_squared < best_distances[worst]:
                best_distances[worst] = distance_squared
                best_indices[worst] = line

        output[p] = best_indices
