from pathlib import Path

import igl
import numpy as np
import trimesh
from rich.progress import BarColumn, Progress, TextColumn
from scipy.ndimage import binary_dilation, binary_propagation
from trimesh.ray.ray_pyembree import RayMeshIntersector

from flowtier.common import UserError, console


class DomainData:
    def __init__(self, surface_path: Path, center_lines_path: Path, precision: float) -> None:
        self.warnings: list[str] = []

        if precision <= 0:
            raise UserError("precision must be positive.")
        self._precision = precision

        self._set_and_adjust_boundaries(surface_path)
        # self._build_void_distance_field(surface_path, subdivisions=2, chunk_size=1_000_000)
        # self._build_void_binary(surface_path, subdivisions=2)
        self._build_void_distance_field_2(surface_path, subdivisions=2, chunk_size=1_000_000)

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

    def _build_void_distance_field_2(
        self, surface_path: Path, chunk_size: int = 100_000, subdivisions: int = 0
    ) -> None:
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
                "Computing void distance field",
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
                points = self.boundaries[0, :] + grid_indices * self._precision

                distances, _, _, _ = igl.signed_distance(points, vertices, faces, igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)

                flat_distance_field[flat_indices] = distances
                progress.update(task, completed=end_index)

    def _ray_classify_along_axis(
        self,
        intersector: RayMeshIntersector,
        face_normals: np.ndarray,
        classification: np.ndarray,
        axis: int,
        chunk_size: int = 100_000,
        relative_tolerance: float = 1e-5,
        normal_tolerance: float = 1e-5,
    ) -> None:
        position_tolerance = relative_tolerance * self._precision

        other_axes = [i for i in range(3) if i != axis]
        ray_shape = self._grid_shape[other_axes]
        ray_count = int(np.prod(ray_shape))

        aligned_classification = np.moveaxis(classification, axis, 0)
        grid_coordinates = self.boundaries[0, axis] + np.arange(self._grid_shape[axis]) * self._precision

        for start_index in range(0, ray_count, chunk_size):
            end_index = min(start_index + chunk_size, ray_count)

            flat_indices = np.arange(start_index, end_index)
            ray_grid_indices = np.column_stack(np.unravel_index(flat_indices, ray_shape))

            unresolved_columns = np.any(
                aligned_classification[:, ray_grid_indices[:, 0], ray_grid_indices[:, 1]] == -1,
                axis=0,
            )
            ray_grid_indices = ray_grid_indices[unresolved_columns]

            if len(ray_grid_indices) == 0:
                continue

            ray_origins = np.tile(self.boundaries[0], (len(ray_grid_indices), 1))
            ray_origins[:, axis] -= self._precision
            ray_origins[:, other_axes] += ray_grid_indices * self._precision

            ray_directions = np.zeros_like(ray_origins)
            ray_directions[:, axis] = 1.0

            hit_positions, hit_ray_ids, hit_triangle_ids = intersector.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                multiple_hits=True,
            )

            order = np.lexsort((hit_positions[:, axis], hit_ray_ids))

            hit_positions = hit_positions[order]
            hit_triangle_ids = hit_triangle_ids[order]
            hit_ray_ids = hit_ray_ids[order]

            ray_counts = np.bincount(hit_ray_ids, minlength=len(ray_origins))
            ray_splits = np.concatenate(([0], np.cumsum(ray_counts)))

            for ray_id, (i, j) in enumerate(ray_grid_indices):
                first_index = ray_splits[ray_id]
                last_index = ray_splits[ray_id + 1]

                if first_index == last_index:
                    continue

                ray_hit_triangle_ids = hit_triangle_ids[first_index:last_index]
                ray_hit_positions = hit_positions[first_index:last_index, axis]
                ray_hit_normal_components = face_normals[ray_hit_triangle_ids, axis]

                if np.any(np.abs(ray_hit_normal_components) < normal_tolerance):
                    continue

                if np.any(np.diff(ray_hit_positions) < position_tolerance):
                    continue

                if np.any(np.sign(ray_hit_normal_components[1:]) == np.sign(ray_hit_normal_components[:-1])):
                    continue

                interval_indices = np.searchsorted(ray_hit_positions, grid_coordinates, side="right")

                interval_phases = np.concatenate(
                    ([int(ray_hit_normal_components[0] < 0)], (ray_hit_normal_components > 0).astype(np.int8))
                )
                proposed_classification = interval_phases[interval_indices]

                left_indices = np.clip(interval_indices - 1, 0, len(ray_hit_positions) - 1)
                right_indices = np.clip(interval_indices, 0, len(ray_hit_positions) - 1)

                distance_to_surface = np.minimum(
                    np.abs(grid_coordinates - ray_hit_positions[left_indices]),
                    np.abs(grid_coordinates - ray_hit_positions[right_indices]),
                )

                proposed_classification[distance_to_surface <= position_tolerance] = -1

                current_classification = aligned_classification[:, i, j]

                conflicts = (
                    (proposed_classification != -1)
                    & (current_classification != -1)
                    & (current_classification != proposed_classification)
                )

                if np.any(conflicts):
                    raise UserError("Conflicting solid/void classifications detected.")

                unclassified_mask = (current_classification == -1) & (proposed_classification != -1)
                current_classification[unclassified_mask] = proposed_classification[unclassified_mask]

    def _build_void_binary(
        self,
        surface_path: Path,
        chunk_size: int = 10_000,
        subdivisions: int = 0,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        vertices, faces = igl.read_triangle_mesh(surface_path)
        vertices, faces = igl.loop(vertices, faces, subdivisions)

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )

        face_normals = mesh.face_normals

        if np.any(np.linalg.norm(face_normals, axis=1) < 1e-10):
            raise UserError("Surface mesh contains degenerate triangles.")

        intersector = RayMeshIntersector(mesh)

        # -1 = unclassified, 0 = solid, 1 = void
        classification = np.full(self._grid_shape, -1, dtype=np.int8)

        # Start with the longest grid axis to minimise the number of rays.
        axis_order = np.argsort(self._grid_shape)[::-1]

        for axis in axis_order:
            self._ray_classify_along_axis(
                intersector=intersector,
                face_normals=face_normals,
                classification=classification,
                axis=int(axis),
                chunk_size=chunk_size,
            )

            if not np.any(classification == -1):
                break

        unresolved_count = np.count_nonzero(classification == -1)

        if unresolved_count:
            raise UserError(f"Unable to classify {unresolved_count:,} grid samples using ray casting.")

        self.void_binary = classification.astype(bool)
