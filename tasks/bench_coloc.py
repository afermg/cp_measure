"""A/B benchmark: numba colocalization vs the numpy reference.

Realistic single image (1080^2, ~144 objects), float pixels. numba JIT is warmed
up before timing. Reports per-feature numpy ms, numba ms, and speedup.
"""

import time

import numpy as np

import cp_measure.core.measurecolocalization as ref
import cp_measure.core.numba.measurecolocalization as nb

FUNCS = ["pearson", "manders_fold", "rwc", "overlap", "costes"]


def make_image(size=1080, grid=12, seed=0):
    rng = np.random.default_rng(seed)
    masks = np.zeros((size, size), np.int32)
    step = size // grid
    obj = step * 3 // 4
    label = 0
    for i in range(grid):
        for j in range(grid):
            label += 1
            r, c = i * step, j * step
            masks[r : r + obj, c : c + obj] = label
    return masks, rng.random((size, size)), rng.random((size, size))


def bench(fn, *args, reps=5):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def main():
    masks, p1, p2 = make_image()
    n = int(masks.max())
    print(f"image 1080^2, {n} objects, float pixels\n")
    print(f"{'feature':<14}{'numpy ms':>10}{'numba ms':>10}{'speedup':>9}")
    for name in FUNCS:
        rfn = getattr(ref, f"get_correlation_{name}")
        nfn = getattr(nb, f"get_correlation_{name}")
        nfn(p1, p2, masks)  # warm up JIT
        t_np = bench(rfn, p1, p2, masks)
        t_nb = bench(nfn, p1, p2, masks)
        print(f"{name:<14}{t_np:>10.2f}{t_nb:>10.2f}{t_np / t_nb:>8.1f}x")


if __name__ == "__main__":
    main()
