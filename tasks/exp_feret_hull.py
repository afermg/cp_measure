"""Can we shrink convex_hull_ijv's input bit-exactly by passing only boundary pixels?

hull(object) == hull(boundary(object)): interior pixels are never hull vertices.
A pixel is a boundary pixel if any 8-neighbour is not the same label (or out of bounds);
that superset contains every extreme/corner pixel, so the hull (and feret_diameter) are
identical. Boundary extraction is a mechanical numba pass (not numerically-sensitive geometry).
"""

import time

import centrosome.cpmorphology
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


@njit(cache=True)
def _boundary_ijv_scatter(masks, max_label):
    Y, X = masks.shape
    counts = np.zeros(max_label + 1, np.int64)
    # pass 1: count boundary pixels per label
    for y in range(Y):
        for x in range(X):
            v = masks[y, x]
            if v <= 0:
                continue
            is_b = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    y2 = y + dy
                    x2 = x + dx
                    if y2 < 0 or y2 >= Y or x2 < 0 or x2 >= X or masks[y2, x2] != v:
                        is_b = True
                        break
                if is_b:
                    break
            if is_b:
                counts[v] += 1
    offs = np.zeros(max_label + 2, np.int64)
    for lbl in range(1, max_label + 1):
        offs[lbl + 1] = offs[lbl] + counts[lbl]
    out = np.empty((offs[max_label + 1], 3), np.int64)
    cur = offs[:-1].copy()
    for y in range(Y):
        for x in range(X):
            v = masks[y, x]
            if v <= 0:
                continue
            is_b = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    y2 = y + dy
                    x2 = x + dx
                    if y2 < 0 or y2 >= Y or x2 < 0 or x2 >= X or masks[y2, x2] != v:
                        is_b = True
                        break
                if is_b:
                    break
            if is_b:
                p = cur[v]
                out[p, 0] = y
                out[p, 1] = x
                out[p, 2] = v
                cur[v] = p + 1
    return out


def boundary_ijv(masks):
    return _boundary_ijv_scatter(masks, int(masks.max()))


# reference
ijv_full = masks_to_ijv(masks)
indices = np.unique(ijv_full[:, 2])
indices = indices[indices > 0]
ch_full, cc_full = centrosome.cpmorphology.convex_hull_ijv(ijv_full, indices)
fmin_full, fmax_full = centrosome.cpmorphology.feret_diameter(ch_full, cc_full, indices)

# boundary-only
ijv_b = boundary_ijv(masks)
ch_b, cc_b = centrosome.cpmorphology.convex_hull_ijv(ijv_b, indices)
fmin_b, fmax_b = centrosome.cpmorphology.feret_diameter(ch_b, cc_b, indices)

print(f"full ijv points:     {ijv_full.shape[0]:>9d}")
print(f"boundary ijv points: {ijv_b.shape[0]:>9d}  ({100 * ijv_b.shape[0] / ijv_full.shape[0]:.1f}%)")
print()
print("convex hull bit-exact:", np.array_equal(ch_full, ch_b) and np.array_equal(cc_full, cc_b))
print("feret_min bit-exact:  ", np.array_equal(fmin_full, fmin_b))
print("feret_max bit-exact:  ", np.array_equal(fmax_full, fmax_b))


def t(fn, *a, reps=5):
    fn(*a)
    ts = []
    for _ in range(reps):
        s = time.perf_counter()
        fn(*a)
        ts.append(time.perf_counter() - s)
    return min(ts) * 1e3


print()
print(f"boundary_ijv (numba)            {t(boundary_ijv, masks):8.1f} ms")
print(f"convex_hull_ijv on FULL ijv     {t(lambda: centrosome.cpmorphology.convex_hull_ijv(ijv_full, indices)):8.1f} ms")
print(f"convex_hull_ijv on BOUNDARY ijv {t(lambda: centrosome.cpmorphology.convex_hull_ijv(ijv_b, indices)):8.1f} ms")
print(f"feret_diameter                  {t(lambda: centrosome.cpmorphology.feret_diameter(ch_b, cc_b, indices)):8.1f} ms")
