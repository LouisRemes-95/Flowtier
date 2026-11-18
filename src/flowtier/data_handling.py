from pathlib import Path

import edt
import igl
import numpy as np
import trimesh
from numba import njit, prange
from pyvista import PolyData
from rich.progress import BarColumn, Progress, TextColumn
from scipy.ndimage import binary_dilation, binary_erosion, binary_propagation
from skimage.morphology import local_minima
from skimage.segmentation import watershed

from flowtier.common import UserError, console


class DomainData:
    def __init__(
        self, surface_path: Path, center_lines_path: Path, precision: float, build_direction: np.ndarray
    ) -> None:
        self.warnings: list[str] = []

        if precision <= 0:
            raise UserError("precision must be positive.")
        self._precision = precision

        self._set_build_direction(build_direction)
        self._set_and_adjust_boundaries(surface_path)
        self._build_void_confidence_field(surface_path, subdivisions=2, chunk_size=1_000_000)
        self._build_initial_cross_section()
        self._build_centerline_dependencies(center_lines_path)

    @property
    def precision(self) -> float:
        return self._precision

    @property
    def grid_shape(self) -> float:
        return self._grid_shape

    @property
    def boundaries(self) -> np.ndarray:
        return self._boundaries

    @property
    def cross_section(self) -> np.ndarray:
        return self._cross_section

    def _set_build_direction(self, build_direction: np.ndarray):
        self._build_direction = np.asarray(build_direction, dtype=np.float64)

        norm = np.linalg.norm(self._build_direction)

        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError("Build direciton must be finite and nonzero vector.")

        self._build_direction /= norm

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

        self._boundaries = adjusted_boundaries

    def _build_void_confidence_field(
        self, surface_path: Path, chunk_size: int = 100_000, subdivisions: int = 0
    ) -> np.ndarray:

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
            max_iter = max(0, int(np.ceil(np.log2(edge_lengths.max() * edge_factor / pitch)))) + 2

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

        distance_field = np.full(self._grid_shape, np.nan, dtype=np.float32)

        interface_indices = np.flatnonzero(binary_interface)
        flat_distance_field = distance_field.ravel()

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
            binary_voids = binary_propagation(distance_field < 0, mask=~binary_interface, structure=structure)

        with console.status("Computing the rest of the void distance field..."):
            void_sdf = edt.edt(~binary_interface, anisotropy=(self._precision,) * 3, parallel=4, black_border=False)

            void_sdf[binary_voids] *= -1

            mask = np.isnan(distance_field)
            distance_field[mask] = void_sdf[mask]

        self._void_confidence_field = np.clip(1.0 - distance_field / distance_field.max(), 0, 1)

    def _build_centerline_dependencies(self, center_lines_path: Path) -> None:
        with np.load(center_lines_path) as data:
            if len(data.files) != 1:
                raise UserError("The .npz file must contain exactly one array.")

            center_lines = data[data.files[0]].astype(np.float32)

        if center_lines.ndim != 3 or center_lines.shape[1:] != (2, 3):
            raise UserError(f"Expected an (N, 2, 3) array, got {center_lines.shape}.")

        self._center_lines = center_lines.copy()
        center_lines = (center_lines - self.boundaries[0]) / self._precision

        self._build_affine_projection_coefficients(center_lines)
        self._build_nearest_centerline_indices()

    def _build_affine_projection_coefficients(self, center_lines: np.ndarray) -> None:
        origins = center_lines.mean(axis=1)

        directions = center_lines[:, 1] - center_lines[:, 0]
        norms = np.linalg.norm(directions, axis=1)

        if not np.all(np.isfinite(norms)) or np.any(norms <= np.finfo(np.float32).eps):
            raise ValueError("Centerlines must have finite, distinct endpoints.")

        directions /= norms[:, None]

        ex, ey = self._compute_local_basis(directions)

        # Affine coefficients: a0, ax, ay, az, b0, bx, by, bz
        self._affine_projection_coefficients = np.column_stack(
            (
                -np.sum(origins * ex, axis=1),
                ex[:, 0],
                ex[:, 1],
                ex[:, 2],
                -np.sum(origins * ey, axis=1),
                ey[:, 0],
                ey[:, 1],
                ey[:, 2],
            )
        ).astype(np.float32)

    def _compute_local_basis(self, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ey = self._build_direction - (np.sum(directions * self._build_direction, axis=1, keepdims=True) * directions)

        norms = np.linalg.norm(ey, axis=1)

        if np.any(norms < 1e-3):
            raise ValueError("A centerline direction is too close to the vertical vector.")

        ey /= norms[:, None]

        ex = np.cross(ey, directions)

        return ex, ey

    def _build_nearest_centerline_indices(self, chunk_size: int = 1_000_000) -> None:
        coefficients = self._affine_projection_coefficients
        n_lines = coefficients.shape[0]

        if n_lines < 4:
            raise ValueError("At least four center lines are required.")

        shape = tuple(self._grid_shape)
        n_voxels = int(np.prod(shape))

        dtype = np.uint8 if n_lines <= 256 else np.uint16

        self._nearest_centerline_indices = np.empty((*shape, 4), dtype=dtype)
        flat_output = self._nearest_centerline_indices.reshape(-1, 4)

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(complete_style="blue", finished_style="blue"),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("[white]{task.elapsed:0.1f}s"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Finding closest centrelines", total=n_voxels)

            for start in range(0, n_voxels, chunk_size):
                stop = min(start + chunk_size, n_voxels)

                closest = np.empty((stop - start, 4), dtype=np.int32)

                _find_closest_center_lines(
                    start,
                    coefficients,
                    closest,
                    shape[1],
                    shape[2],
                )

                flat_output[start:stop] = closest

                progress.update(task, advance=stop - start)

    def _build_initial_cross_section(self) -> None:
        n = int(np.max(self._grid_shape))
        shape = (n * 2 + 1,) * 2

        self._cross_section = np.zeros(shape, dtype=bool)
        self._cross_section[n, n] = True

    def update_segmentation(self):
        confidence, indices, labels, _ = self.compute_segmentation(self.cross_section.copy())
        self.apply_sweep_dependencies(confidence, indices, labels)

    def compute_segmentation(
        self, cross_section: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[PolyData]]:

        confidence, indices = self._compute_sweep_dependencies(cross_section)
        labels = self._compute_labeled_voxelisation(confidence, indices)
        surfaces = self._compute_plotting_surfaces(labels)

        return confidence, indices, labels, surfaces

    def apply_sweep_dependencies(
        self, confidence_field: np.ndarray, nearest_indices: np.ndarray, labels: np.ndarray
    ) -> None:
        self._sweep_confidence_field = confidence_field
        self._nearest_sweep_indices = nearest_indices
        self._labeled_voxelisation = labels

    def _compute_sweep_dependencies(
        self, cross_section: np.ndarray, chunk_size: int = 1_000_000
    ) -> tuple[np.ndarray, np.ndarray]:
        shape = tuple(self._grid_shape)
        n_voxels = int(np.prod(shape))

        distance_field = np.empty(shape, dtype=np.float32)
        nearest_indices = np.empty(
            shape,
            dtype=self._nearest_centerline_indices.dtype,
        )

        flat_output = distance_field.ravel()
        flat_indices = nearest_indices.ravel()
        flat_closest = self._nearest_centerline_indices.reshape(-1, 4)

        cross_section_distance_field = edt.edt(
            ~cross_section,
            parallel=4,
            black_border=False,
        )

        for start in range(0, n_voxels, chunk_size):
            stop = min(start + chunk_size, n_voxels)

            _compute_minimum_distance(
                start,
                flat_closest[start:stop],
                self._affine_projection_coefficients,
                cross_section_distance_field,
                flat_output[start:stop],
                flat_indices[start:stop],
                shape[1],
                shape[2],
            )

        confidence_field = np.clip(1.0 - distance_field / distance_field.max(), 0, 1)

        return confidence_field, nearest_indices

    def _compute_labeled_voxelisation(
        self, sweep_confidence_field: np.ndarray, nearest_sweep_indices: np.ndarray, power: int = 4
    ) -> np.ndarray:
        segmentation_field = self._void_confidence_field**power - sweep_confidence_field**power

        mask = self._void_confidence_field < 1
        minima = local_minima(segmentation_field, connectivity=3) & mask

        markers = np.zeros(minima.shape, dtype=np.uint16)

        _, inverse = np.unique(nearest_sweep_indices[minima], return_inverse=True)
        markers[minima] = inverse + 1
        print(segmentation_field.shape)

        return watershed(segmentation_field, markers=markers, mask=mask, connectivity=3)

    def _compute_plotting_surfaces(self, labeled_voxelisation: np.ndarray) -> list[PolyData]:
        surfaces = []

        labels = np.unique(labeled_voxelisation)
        labels = labels[labels != 0]

        for label in labels:
            binary = labeled_voxelisation == label

            shell = binary & ~binary_erosion(binary, border_value=0)

            surface = self.binary_to_voxel_surface(binary, shell, self.precision)

            surface.translate(self.boundaries[0] - self.precision / 2, inplace=True)

            surfaces.append(surface)

        return surfaces

    @staticmethod
    def binary_to_voxel_surface(binary: np.ndarray, candidates: np.ndarray, voxel_size: float) -> PolyData:
        indices = np.flatnonzero(candidates)
        counts = _count_candidate_faces(binary, indices)

        offsets = np.empty(len(indices) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])

        n_faces = int(offsets[-1])

        vertices = np.empty((n_faces * 4, 3), dtype=np.float32)
        faces = np.empty((n_faces, 5), dtype=np.int64)

        _build_candidate_faces(binary, indices, offsets, voxel_size, _CORNERS, vertices, faces)

        return PolyData(vertices, faces.ravel())

    def paint_cross_section(self, i: np.ndarray, j: np.ndarray, value: bool) -> bool:
        cross_section = self.cross_section

        mirrored_i = cross_section.shape[0] - 1 - i

        if np.all((cross_section[i, j] == value) & (cross_section[mirrored_i, j] == value)):
            return False

        cross_section[i, j] = value
        cross_section[mirrored_i, j] = value

        return True

    def compute_sweep_surface(self, cross_section: np.ndarray) -> PolyData:
        pixels = np.argwhere(cross_section)

        if len(pixels) == 0:
            return PolyData()

        pixels = pixels - (np.array(cross_section.shape) - 1) / 2

        box_min = self.boundaries[0] - self.precision
        box_max = self.boundaries[1] + self.precision
        box_center = (box_min + box_max) / 2

        half_width = self.precision / 2

        all_vertices = []
        all_faces = []
        offset = 0

        square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])

        # Six faces: two caps and four sides
        face_pattern = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])

        for center_line in self._center_lines:
            origin = center_line.mean(axis=0).astype(np.float64)

            direction = center_line[1].astype(np.float64) - center_line[0].astype(np.float64)
            direction /= np.linalg.norm(direction)

            ex, ey = self._compute_local_basis(direction[None, :])
            ex, ey = ex[0], ey[0]

            # One infinite line for each painted pixel
            origin += np.dot(box_center - origin, direction) * direction

            centers = origin + self.precision * (pixels[:, 0, None] * ex + pixels[:, 1, None] * ey)

            # Clip the lines to the expanded domain box
            t_min = np.full(len(centers), -np.inf)
            t_max = np.full(len(centers), np.inf)
            valid = np.ones(len(centers), dtype=bool)

            for axis in range(3):
                if abs(direction[axis]) < 1e-12:
                    valid &= (centers[:, axis] >= box_min[axis]) & (centers[:, axis] <= box_max[axis])
                    continue

                t0 = (box_min[axis] - centers[:, axis]) / direction[axis]
                t1 = (box_max[axis] - centers[:, axis]) / direction[axis]

                t_min = np.maximum(t_min, np.minimum(t0, t1))
                t_max = np.minimum(t_max, np.maximum(t0, t1))

            valid &= t_min < t_max

            if not np.any(valid):
                continue

            starts = centers[valid] + t_min[valid, None] * direction
            ends = centers[valid] + t_max[valid, None] * direction

            # Square cross-section, aligned with ex and ey
            corners = half_width * (square[:, 0, None] * ex + square[:, 1, None] * ey)

            vertices = np.concatenate((starts[:, None, :] + corners, ends[:, None, :] + corners), axis=1)

            n = len(vertices)

            faces = face_pattern[None, :, :] + 8 * np.arange(n)[:, None, None] + offset

            faces = np.concatenate((np.full((*faces.shape[:2], 1), 4), faces), axis=2)

            all_vertices.append(vertices.reshape(-1, 3))
            all_faces.append(faces.reshape(-1, 5))

            offset += 8 * n

        if not all_vertices:
            return PolyData()

        return PolyData(np.vstack(all_vertices), np.vstack(all_faces).ravel())


@njit(inline="always")
def _bilinear_interpolate(distance_field: np.ndarray, u: float, v: float) -> float:
    nx, ny = distance_field.shape

    u += nx // 2
    v += ny // 2

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
def _find_closest_center_lines(start: int, coefficients: np.ndarray, output: np.ndarray, ny: int, nz: int) -> None:
    n_lines = coefficients.shape[0]
    n_closest = output.shape[1]

    for p in prange(output.shape[0]):
        index = start + p

        i = index // (ny * nz)
        j = (index // nz) % ny
        k = index % nz

        best_distances = np.full(n_closest, np.inf)
        best_indices = np.full(n_closest, -1, dtype=np.int32)

        for line in range(n_lines):
            a0, ax, ay, az, b0, bx, by, bz = coefficients[line]

            u = a0 + ax * i + ay * j + az * k
            v = b0 + bx * i + by * j + bz * k

            distance_squared = u * u + v * v

            if distance_squared < best_distances[-1]:
                pos = n_closest - 1

                while pos > 0 and distance_squared < best_distances[pos - 1]:
                    best_distances[pos] = best_distances[pos - 1]
                    best_indices[pos] = best_indices[pos - 1]
                    pos -= 1

                best_distances[pos] = distance_squared
                best_indices[pos] = line

        output[p] = best_indices


@njit(parallel=True, cache=True)
def _compute_minimum_distance(
    start: int,
    closest_lines: np.ndarray,
    coefficients: np.ndarray,
    distance_field: np.ndarray,
    output: np.ndarray,
    indices_output: np.ndarray,
    ny: int,
    nz: int,
) -> None:
    for p in prange(output.shape[0]):
        index = start + p

        i = index // (ny * nz)
        j = (index // nz) % ny
        k = index % nz

        minimum = np.inf
        nearest_index = closest_lines[p, 0]

        for c in range(closest_lines.shape[1]):
            line = closest_lines[p, c]

            a0, ax, ay, az, b0, bx, by, bz = coefficients[line]

            u = a0 + ax * i + ay * j + az * k
            v = b0 + bx * i + by * j + bz * k

            distance = _bilinear_interpolate(distance_field, u, v)

            if distance < minimum:
                minimum = distance
                nearest_index = line

        output[p] = minimum
        indices_output[p] = nearest_index


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
