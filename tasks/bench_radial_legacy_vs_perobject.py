"""PR #68: per-object (legacy=False) vs whole-image (legacy=True) radial distribution.

legacy=True does one centrosome geometry pass over the whole field; legacy=False
does one pass per object crop. Sweep object count (the driver of per-object overhead)
on a fixed field to find where the crossover is.
"""

import time

import numpy as np

from cp_measure.core.measureobjectintensitydistribution import get_radial_distribution


def grid(size, n, seed=0):
    """n x n non-touching square objects on a size x size field."""
    labels = np.zeros((size, size), np.int32)
    step = size // n
    for i in range(n):
        for j in range(n):
            labels[i * step : i * step + step * 3 // 4, j * step : j * step + step * 3 // 4] = i * n + j + 1
    return labels, np.random.default_rng(seed).random((size, size)).astype(np.float32)


def bench(labels, pixels, legacy, reps=5):
    get_radial_distribution(labels, pixels, legacy=legacy)  # warm
    return min(_timed(labels, pixels, legacy) for _ in range(reps)) * 1e3


def _timed(labels, pixels, legacy):
    t = time.perf_counter()
    get_radial_distribution(labels, pixels, legacy=legacy)
    return time.perf_counter() - t


if __name__ == "__main__":
    print(f"{'objects':>9}{'legacy ms':>12}{'perobj ms':>12}{'ratio':>8}{'ms/obj':>9}")
    for n in (1, 2, 4, 8, 12, 20, 32, 40, 64):
        labels, pixels = grid(1080, n)
        nobj = int(labels.max())
        leg, obj = bench(labels, pixels, True), bench(labels, pixels, False)
        print(f"{nobj:>9}{leg:>12.2f}{obj:>12.2f}{obj / leg:>7.2f}x{obj / nobj:>9.3f}")
