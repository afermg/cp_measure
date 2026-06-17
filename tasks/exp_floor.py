"""Experiment: cut the sort-free floor shared by all 4 coloc features.

Current floor ~10ms = find_objects (3.5) + flatten 2 scans (2.6) + kernel (4.2).
Try: np.bincount(masks.ravel()) -> counts/lut/offsets in one C pass, replacing
find_objects AND the flatten count-scan (flatten then does only the scatter).
Compare timing + verify lut/offsets identical.
"""

import time

import numpy as np
from numba import njit

from cp_measure.primitives._segment_numba import flatten_pairs_grouped
from cp_measure.primitives.segment import label_to_idx_lut


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


@njit(cache=True)
def _scatter_only(masks, pix1, pix2, lut, offsets):
    """Single scatter scan; offsets precomputed from bincount."""
    Z, Y, X = masks.shape
    M = offsets[-1]
    g1 = np.empty(M, np.float64)
    g2 = np.empty(M, np.float64)
    fill = offsets[:-1].copy()
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                L = masks[z, y, x]
                if L > 0:
                    k = lut[L]
                    p = fill[k]
                    g1[p] = pix1[z, y, x]
                    g2[p] = pix2[z, y, x]
                    fill[k] = p + 1
    return g1, g2


def via_bincount(masks):
    counts = np.bincount(masks.ravel())  # index 0 = background
    present = np.nonzero(counts[1:])[0] + 1  # positive labels actually present
    n = present.size
    max_label = int(present[-1]) if n else 0
    lut = np.full(max_label + 1, -1, np.int64)
    lut[present] = np.arange(n)
    offsets = np.zeros(n + 1, np.int64)
    offsets[1:] = np.cumsum(counts[present])
    return lut, n, offsets


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

    # current path
    ms_fo, (lut, n) = t(lambda: label_to_idx_lut(mc))
    ms_flat, (g1, g2, offs) = t(
        lambda: flatten_pairs_grouped(mc[None], p1c[None], p2c[None], lut, n)
    )

    # bincount path
    ms_bc, (lut2, n2, offs2) = t(lambda: via_bincount(mc))
    _scatter_only(mc[None], p1c[None], p2c[None], lut2, offs2)  # warm up
    ms_sc, (h1, h2) = t(lambda: _scatter_only(mc[None], p1c[None], p2c[None], lut2, offs2))

    print(f"n={n} (bincount n={n2})  offsets match: {np.array_equal(offs, offs2)}")
    print(f"lut match: {np.array_equal(lut, lut2)}")
    print(f"g1 match (set per obj): {np.array_equal(np.sort(g1), np.sort(h1))}\n")
    print(f"CURRENT:  find_objects {ms_fo:5.2f} + flatten(2 scans) {ms_flat:5.2f} = {ms_fo + ms_flat:5.2f} ms")
    print(f"BINCOUNT: bincount     {ms_bc:5.2f} + scatter(1 scan)  {ms_sc:5.2f} = {ms_bc + ms_sc:5.2f} ms")


if __name__ == "__main__":
    main()
