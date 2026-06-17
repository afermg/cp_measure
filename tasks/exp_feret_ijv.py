"""Compare ijv-builders for feret: per-label scan vs nonzero+argsort vs numba counting-scatter.

masks_to_ijv groups rows by label ascending, within-label row-major (np.where order).
A single row-major numba scan writing each pixel at a per-label running offset produces
the SAME order -> bit-exact. And it avoids both nonzero and the O(n log n) argsort.
"""

import time

import numpy as np
from numba import njit

from cp_measure.utils import masks_to_ijv

rng = np.random.default_rng(0)
side, n = 1080, 144
yy, xx = np.mgrid[0:side, 0:side]
centers = rng.integers(0, side, size=(n, 2))
lab = np.zeros((side, side), np.int32)
best = np.full((side, side), np.inf)
for i, (cy, cx) in enumerate(centers, start=1):
    d = (yy - cy) ** 2 + (xx - cx) ** 2
    m = d < best
    best[m] = d[m]
    lab[m] = i
masks = lab


def ijv_nonzero(masks):
    i, j = np.nonzero(masks)
    v = masks[i, j]
    order = np.argsort(v, kind="stable")
    out = np.empty((i.size, 3), dtype=np.int64)
    out[:, 0] = i[order]
    out[:, 1] = j[order]
    out[:, 2] = v[order]
    return out


@njit(cache=True)
def _ijv_scatter(masks, max_label):
    Y, X = masks.shape
    counts = np.zeros(max_label + 1, np.int64)
    for y in range(Y):
        for x in range(X):
            v = masks[y, x]
            if v > 0:
                counts[v] += 1
    # offsets for labels 1..max in ascending order (skip background 0)
    offs = np.zeros(max_label + 2, np.int64)
    for lbl in range(1, max_label + 1):
        offs[lbl + 1] = offs[lbl] + counts[lbl]
    total = offs[max_label + 1]
    out = np.empty((total, 3), np.int64)
    cur = offs[:-1].copy()  # cur[lbl] = next write pos for label lbl
    for y in range(Y):
        for x in range(X):
            v = masks[y, x]
            if v > 0:
                p = cur[v]
                out[p, 0] = y
                out[p, 1] = x
                out[p, 2] = v
                cur[v] = p + 1
    return out


def ijv_numba(masks):
    return _ijv_scatter(masks, int(masks.max()))


def t(fn, *a, reps=5):
    fn(*a)
    ts = []
    for _ in range(reps):
        s = time.perf_counter()
        fn(*a)
        ts.append(time.perf_counter() - s)
    return min(ts) * 1e3


a = masks_to_ijv(masks)
b = ijv_nonzero(masks)
c = ijv_numba(masks)
print("nonzero bit-exact to masks_to_ijv:", np.array_equal(a, b))
print("numba   bit-exact to masks_to_ijv:", np.array_equal(a, c))
print()
print(f"masks_to_ijv (per-label scan) {t(masks_to_ijv, masks):8.1f} ms")
print(f"nonzero + argsort             {t(ijv_nonzero, masks):8.1f} ms")
print(f"numba counting-scatter        {t(ijv_numba, masks):8.1f} ms")
