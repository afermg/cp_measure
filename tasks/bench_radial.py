"""A/B benchmark: numba radial_distribution vs numpy reference (1080^2, JIT warmed).

Note: numba FIXES Issue #22, so it does not match the numpy baseline on multi-object
fields — this measures speed only.
"""

import time

import numpy as np

from cp_measure.core.measureobjectintensitydistribution import (
    get_radial_distribution as ref,
)
from cp_measure.core.numba.measureobjectintensitydistribution import (
    get_radial_distribution as nb,
)


def make_image(size=1080, grid=12, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.zeros((size, size), np.int32)
    step = size // grid
    obj = step * 3 // 4
    lab = 0
    for i in range(grid):
        for j in range(grid):
            lab += 1
            r, c = i * step, j * step
            labels[r : r + obj, c : c + obj] = lab
    return labels, rng.random((size, size))


def bench(fn, *a, reps=4):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def main():
    labels, pixels = make_image()
    nb(labels, pixels)  # warm JIT
    t_np = bench(ref, labels, pixels)
    t_nb = bench(nb, labels, pixels)
    print(f"image 1080^2, {int(labels.max())} objects\n")
    print(f"{'feature':<22}{'numpy ms':>10}{'numba ms':>10}{'speedup':>9}")
    print(f"{'radial_distribution':<22}{t_np:>10.2f}{t_nb:>10.2f}{t_np / t_nb:>8.1f}x")


if __name__ == "__main__":
    main()
