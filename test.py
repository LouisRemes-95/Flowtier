import time

import numpy as np
from numba import njit, prange

# --------------------------------------------------
# Flattened bilinear interpolation
# --------------------------------------------------


@njit(parallel=True, fastmath=True)
def bilinear_flat(matrix, points, output, ny):
    for p in prange(points.shape[0]):
        x = points[p, 0]
        y = points[p, 1]

        x0 = int(x)
        y0 = int(y)

        dx = x - x0
        dy = y - y0

        idx = x0 * ny + y0

        v00 = matrix[idx]
        v01 = matrix[idx + 1]
        v10 = matrix[idx + ny]
        v11 = matrix[idx + ny + 1]

        one_dx = 1.0 - dx
        one_dy = 1.0 - dy

        v0 = v00 * one_dy + v01 * dy
        v1 = v10 * one_dy + v11 * dy

        output[p] = v0 * one_dx + v1 * dx


# --------------------------------------------------
# Bucket sorting
# --------------------------------------------------


@njit
def scatter_into_buckets(points, x0, offsets, sorted_points):
    """
    Scatter points into buckets according to integer x0.

    offsets contains the initial write position of each bucket.
    It is modified in place.
    """

    for i in range(points.shape[0]):
        bucket = x0[i]

        pos = offsets[bucket]

        sorted_points[pos, 0] = points[i, 0]
        sorted_points[pos, 1] = points[i, 1]

        offsets[bucket] += 1


# --------------------------------------------------
# Configuration
# --------------------------------------------------

size = 1000
n = 1_000_000_000
chunk_size = 10_000_000

bins = size - 1

rng = np.random.default_rng(42)

matrix = rng.random((size, size), dtype=np.float32)
matrix_flat = matrix.ravel()

output = np.empty(chunk_size, dtype=np.float32)


# --------------------------------------------------
# Warm-up
# --------------------------------------------------

dummy_points = np.random.rand(100, 2).astype(np.float32)
dummy_x0 = dummy_points[:, 0].astype(np.int32)

dummy_sorted = np.empty_like(dummy_points)
dummy_offsets = np.zeros(bins + 1, dtype=np.int64)

scatter_into_buckets(
    dummy_points,
    dummy_x0,
    dummy_offsets,
    dummy_sorted,
)

bilinear_flat(
    matrix_flat,
    dummy_points,
    np.empty(100, dtype=np.float32),
    size,
)


# --------------------------------------------------
# Benchmark
# --------------------------------------------------

generation_time = 0.0
sorting_time = 0.0
sampling_time = 0.0
checksum = 0.0

start_total = time.perf_counter()

for i in range(0, n, chunk_size):
    current_size = min(chunk_size, n - i)

    # ----------------------------------------------
    # Generate random coordinates
    # ----------------------------------------------

    t0 = time.perf_counter()

    points = rng.random(
        (current_size, 2),
        dtype=np.float32,
    )

    points *= np.float32(size - 1)

    generation_time += time.perf_counter() - t0

    # ----------------------------------------------
    # Bucket sort by integer x coordinate
    # ----------------------------------------------

    t0 = time.perf_counter()

    x0 = points[:, 0].astype(np.int32)

    counts = np.bincount(x0, minlength=bins)

    offsets = np.empty(bins + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    sorted_points = np.empty_like(points)

    scatter_into_buckets(
        points,
        x0,
        offsets[:-1].copy(),
        sorted_points,
    )

    sorting_time += time.perf_counter() - t0

    # ----------------------------------------------
    # Bilinear interpolation
    # ----------------------------------------------

    t0 = time.perf_counter()

    bilinear_flat(
        matrix_flat,
        sorted_points,
        output[:current_size],
        size,
    )

    sampling_time += time.perf_counter() - t0

    # ----------------------------------------------
    # Checksum
    # ----------------------------------------------

    checksum += float(np.sum(output[:current_size], dtype=np.float64))

    # ----------------------------------------------
    # Progress
    # ----------------------------------------------

    print(f"Processed: {i + current_size:,} / {n:,} ({100 * (i + current_size) / n:.1f}%)")

    del points, x0, counts, offsets, sorted_points


elapsed = time.perf_counter() - start_total


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n---------- BENCHMARK RESULTS ----------")

print(f"Total points:          {n:,}")
print(f"Chunk size:            {chunk_size:,}")
print(f"Number of buckets:     {bins:,}")

print()
print(f"Coordinate generation: {generation_time:.3f} s")
print(f"Bucket sorting:        {sorting_time:.3f} s")
print(f"Numba interpolation:   {sampling_time:.3f} s")
print(f"Total elapsed time:    {elapsed:.3f} s")

print()
print(f"Generation throughput: {n / generation_time / 1e6:.2f} M points/s")
print(f"Sorting throughput:    {n / sorting_time / 1e6:.2f} M points/s")
print(f"Sampling throughput:   {n / sampling_time / 1e6:.2f} M points/s")

print()
print(f"Checksum: {checksum:.6f}")
