"""Component profile of the numba costes path (float pixels, scale=1).

Splits the per-call cost into prep (labels_to_offsets + flatten) vs the
costes_per_object kernel, and isolates the kernel's internal passes (regression,
search, final C1/C2) to see what, if anything, is worth optimising while staying
exact + bzyx.
"""

import time

import numpy as np
from numba import njit

from cp_measure.core.numba._costes import (
    _bisection,
    _regression_ab,
    costes_per_object,
)
from cp_measure.primitives._segment_numba import flatten_pairs_grouped
from cp_measure.primitives.segment import labels_to_offsets


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


@njit(cache=True, error_model="numpy")
def _regression_only(g1, g2, offsets, n):
    acc = 0.0
    for k in range(n):
        a, b = _regression_ab(g1, g2, offsets[k], offsets[k + 1])
        acc += a + b
    return acc


@njit(cache=True, error_model="numpy")
def _search_only(g1, g2, offsets, n, scale):
    acc = 0.0
    for k in range(n):
        a, b = _regression_ab(g1, g2, offsets[k], offsets[k + 1])
        tf, ts = _bisection(g1, g2, offsets[k], offsets[k + 1], a, b, scale)
        acc += tf + ts
    return acc


def t(fn, reps=7):
    best = float("inf")
    out = None
    for _ in range(reps):
        s = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - s)
    return best * 1e3, out


def main():
    masks, p1, p2 = make_image()
    mc = np.ascontiguousarray(masks)
    p1c = np.ascontiguousarray(p1)
    p2c = np.ascontiguousarray(p2)

    ms_lut, (lut, n, offs) = t(lambda: labels_to_offsets(mc))
    ms_flat, (g1, g2) = t(lambda: flatten_pairs_grouped(mc[None], p1c[None], p2c[None], lut, offs))

    # warm up
    _regression_only(g1, g2, offs, n)
    _search_only(g1, g2, offs, n, 1.0)
    costes_per_object(g1, g2, offs, n, 1.0, 0)

    ms_reg, _ = t(lambda: _regression_only(g1, g2, offs, n))
    ms_search, _ = t(lambda: _search_only(g1, g2, offs, n, 1.0))
    ms_full, _ = t(lambda: costes_per_object(g1, g2, offs, n, 1.0, 0))

    M = int(offs[-1])
    print(f"image 1080^2, {n} objects, {M} masked px ({M / n:.0f} px/obj), scale=1\n")
    print(f"{'labels_to_offsets':<28}{ms_lut:>8.2f} ms")
    print(f"{'flatten_pairs_grouped':<28}{ms_flat:>8.2f} ms")
    print(f"{'kernel: regression only':<28}{ms_reg:>8.2f} ms")
    print(f"{'kernel: + bisection search':<28}{ms_search:>8.2f} ms  (search = {ms_search - ms_reg:.2f})")
    print(f"{'kernel: full (+final C1/C2)':<28}{ms_full:>8.2f} ms  (final = {ms_full - ms_search:.2f})")
    print(f"\nprep total                  {ms_lut + ms_flat:>8.2f} ms")
    print(f"kernel total                {ms_full:>8.2f} ms")


if __name__ == "__main__":
    main()
