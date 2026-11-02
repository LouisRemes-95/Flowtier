from time import perf_counter

import numpy as np
from numba import njit, prange


class Line:
    def __init__(self, points: np.ndarray):
        points = np.asarray(points, dtype=np.float64)

        if points.shape != (2, 3):
            raise ValueError("Expected points with shape (2, 3).")

        self.origin = points.mean(axis=0)

        direction = points[1] - points[0]
        norm = np.linalg.norm(direction)

        if norm == 0:
            raise ValueError("Line endpoints must be distinct.")

        self.direction = direction / norm

        # Horizontal local X-axis
        vertical = np.array([0.0, 0.0, 1.0])
        self.ex = np.cross(vertical, self.direction)

        if np.linalg.norm(self.ex) < 1e-12:
            self.ex = np.array([1.0, 0.0, 0.0])
        else:
            self.ex /= np.linalg.norm(self.ex)

        # Local Y-axis
        self.ey = np.cross(self.direction, self.ex)

    def to_2d(self, coordinates: np.ndarray) -> np.ndarray:
        relative = coordinates - self.origin

        return np.column_stack(
            (
                relative @ self.ex,
                relative @ self.ey,
            )
        )

    def precompute_grid(
        self,
        boundaries: np.ndarray,
        voxel_size: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        shape = np.rint((boundaries[:, 1] - boundaries[:, 0]) / voxel_size).astype(int)

        nx, ny, nz = shape

        start = boundaries[:, 0] + voxel_size / 2
        origin_2d = self.to_2d(start[None, :])[0]

        increments = voxel_size * np.column_stack((self.ex, self.ey))

        i = np.arange(nx, dtype=np.float32)
        j = np.arange(ny, dtype=np.float32)
        k = np.arange(nz, dtype=np.float32)

        u_xy = (origin_2d[0] + i[:, None] * increments[0, 0] + j[None, :] * increments[1, 0]).astype(np.float32)

        v_xy = (origin_2d[1] + i[:, None] * increments[0, 1] + j[None, :] * increments[1, 1]).astype(np.float32)

        v_z = (k * increments[2, 1]).astype(np.float32)

        return u_xy, v_xy, v_z


# ============================================================
# Projection implementations
# ============================================================


def project_numpy(u_xy, v_xy, v_z, output):
    output[..., 0] = u_xy[:, :, None]

    np.add(
        v_xy[:, :, None],
        v_z[None, None, :],
        out=output[..., 1],
    )


@njit
def project_numba(u_xy, v_xy, v_z, output):
    nx, ny, nz, _ = output.shape

    for i in range(nx):
        for j in range(ny):
            u = u_xy[i, j]
            v = v_xy[i, j]

            for k in range(nz):
                output[i, j, k, 0] = u
                output[i, j, k, 1] = v + v_z[k]


@njit(parallel=True)
def project_numba_parallel(u_xy, v_xy, v_z, output):
    nx, ny, nz, _ = output.shape

    for i in prange(nx):
        for j in range(ny):
            u = u_xy[i, j]
            v = v_xy[i, j]

            for k in range(nz):
                output[i, j, k, 0] = u
                output[i, j, k, 1] = v + v_z[k]


# ============================================================
# Benchmark
# ============================================================

if __name__ == "__main__":
    line = Line(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 2.0],
            ]
        )
    )

    boundaries = np.array(
        [
            [0.0, 100.0],
            [0.0, 100.0],
            [0.0, 100.0],
        ]
    )

    voxel_size = 0.1
    batch_size = 16

    # Precomputation
    start = perf_counter()
    u_xy, v_xy, v_z = line.precompute_grid(boundaries, voxel_size)
    precompute_time = perf_counter() - start

    nx, ny = u_xy.shape
    nz = len(v_z)
    n = nx * ny * nz

    buffer = np.empty((batch_size, ny, nz, 2), dtype=np.float32)

    # Compile Numba before timing
    project_numba(u_xy[:1], v_xy[:1], v_z, buffer[:1])
    project_numba_parallel(u_xy[:1], v_xy[:1], v_z, buffer[:1])

    # Check correctness
    test_shape = (2, 3, nz, 2)
    reference = np.empty(test_shape, dtype=np.float32)
    result = np.empty_like(reference)

    project_numpy(u_xy[:2, :3], v_xy[:2, :3], v_z, reference)

    for function in (project_numba, project_numba_parallel):
        function(u_xy[:2, :3], v_xy[:2, :3], v_z, result)
        np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-6)

    def benchmark(name, function):
        start = perf_counter()

        for i in range(0, nx, batch_size):
            count = min(batch_size, nx - i)

            function(
                u_xy[i : i + count],
                v_xy[i : i + count],
                v_z,
                buffer[:count],
            )

        elapsed = perf_counter() - start

        print(f"{name:16s}: {elapsed:.3f} s | {n / elapsed / 1e6:.2f} million voxels/s")

    print(f"Total voxels: {n:,}")
    print(f"Precomputation: {precompute_time:.6f} s")
    print(f"Precomputed memory: {(u_xy.nbytes + v_xy.nbytes + v_z.nbytes) / 1e6:.2f} MB")
    print()

    benchmark("NumPy", project_numpy)
    benchmark("Numba", project_numba)
    benchmark("Numba parallel", project_numba_parallel)
