"""Step-0 profile for the feret numba lane.

get_feret = masks_to_ijv  -> reducible (per-label scan, like zernike's; -> one nonzero)
          + convex_hull_ijv -> IMPORT (centrosome computational geometry)
          + feret_diameter  -> IMPORT (centrosome antipodal over hull points)
"""

import time

import centrosome.cpmorphology
import numpy as np

from cp_measure.core.measureobjectsizeshape import get_feret
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
pixels = rng.random((side, side)).astype(np.float32)


def t(fn, *a, reps=3):
    fn(*a)
    ts = []
    for _ in range(reps):
        s = time.perf_counter()
        fn(*a)
        ts.append(time.perf_counter() - s)
    return min(ts) * 1e3


# candidate reducible replacement for masks_to_ijv: one nonzero
def ijv_nonzero(masks):
    i, j = np.nonzero(masks)
    v = masks[i, j]
    order = np.argsort(v, kind="stable")  # group by label, like the per-label scan
    out = np.empty((i.size, 3), dtype=int)
    out[:, 0] = i[order]
    out[:, 1] = j[order]
    out[:, 2] = v[order]
    return out


t_full = t(get_feret, masks, pixels)
t_ijv = t(masks_to_ijv, masks)
t_ijv_nz = t(ijv_nonzero, masks)

ijv = masks_to_ijv(masks)
indices = np.unique(ijv[:, 2])
indices = indices[indices > 0]


def hull():
    return centrosome.cpmorphology.convex_hull_ijv(ijv, indices)


t_hull = t(hull)
chulls, chull_counts = hull()


def feret():
    return centrosome.cpmorphology.feret_diameter(chulls, chull_counts, indices)


t_feret = t(feret)

print(f"image {masks.shape}, {len(indices)} objects\n")
print(f"get_feret (full)        {t_full:8.1f} ms   100%")
print(f"  masks_to_ijv          {t_ijv:8.1f} ms   {100 * t_ijv / t_full:4.1f}%   (reducible)")
print(f"  convex_hull_ijv       {t_hull:8.1f} ms   {100 * t_hull / t_full:4.1f}%   (IMPORT)")
print(f"  feret_diameter        {t_feret:8.1f} ms   {100 * t_feret / t_full:4.1f}%   (IMPORT)")
print()
print(f"  masks_to_ijv -> nonzero replacement: {t_ijv:.1f} -> {t_ijv_nz:.1f} ms")
# verify nonzero replacement is equivalent (same rows, possibly different intra-label order)
a = masks_to_ijv(masks)
b = ijv_nonzero(masks)
same = a.shape == b.shape and np.array_equal(
    a[np.lexsort((a[:, 1], a[:, 0], a[:, 2]))],
    b[np.lexsort((b[:, 1], b[:, 0], b[:, 2]))],
)
print(f"  nonzero rows == masks_to_ijv rows (order-insensitive): {same}")
print()
reducible = t_ijv - t_ijv_nz
print(f"Amdahl ceiling replacing masks_to_ijv with nonzero (host, NOT numba):")
print(f"  {t_full:.1f} -> {t_full - reducible:.1f} ms  -> {t_full / (t_full - reducible):.2f}x")
