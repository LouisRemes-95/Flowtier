from pathlib import Path

import igl
import numpy as np

from flowtier.common import UserError


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

        shape = self.boundaries[1, :] - self.boundaries[0, :]

        # # Load triangular surface mesh

        # # Define regular grid
        # origin = vertices.min(axis=0)
        # upper = vertices.max(axis=0)

        # shape = tuple(np.ceil((upper - origin) / voxel_size).astype(int) + 1)

        # # Allocate distance field
        # distance_field = np.empty(shape, dtype=np.float32)
        # flat = distance_field.ravel()

        # # Compute signed distances in chunks
        # for start in range(0, flat.size, chunk_size):
        #     end = min(start + chunk_size, flat.size)

        #     indices = np.arange(start, end)
        #     ijk = np.column_stack(np.unravel_index(indices, shape))
        #     points = origin + ijk * voxel_size

        #     distances, _, _, _ = igl.signed_distance(
        #         points,
        #         vertices,
        #         faces,
        #         igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
        #     )

        #     flat[start:end] = distances

        # return distance_field, origin
