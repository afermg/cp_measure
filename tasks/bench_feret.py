"""Benchmark numba feret vs numpy baseline (1080^2, 144 objects)."""

import time

import numpy as np

from cp_measure.core.measureobjectsizeshape import get_feret as feret_numpy
from cp_measure.core.numba import get_feret as feret_nb

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


def t(fn, *a, reps=5):
    fn(*a)
    ts = []
    for _ in range(reps):
        s = time.perf_counter()
        fn(*a)
        ts.append(time.perf_counter() - s)
    return min(ts) * 1e3


# correctness
r_np = feret_numpy(masks, pixels)
r_nb = feret_nb(masks, pixels)
ok = all(np.array_equal(r_np[k], r_nb[k]) for k in r_np) and set(r_np) == set(r_nb)
print(f"bit-exact vs numpy: {ok}\n")

t_np = t(feret_numpy, masks, pixels)
t_nb = t(feret_nb, masks, pixels)
print(f"image {side}^2, {n} objects\n")
print(f"{'feature':12s} {'numpy ms':>10s} {'numba ms':>10s} {'speedup':>8s}")
print(f"{'feret':12s} {t_np:10.2f} {t_nb:10.2f} {t_np / t_nb:7.1f}x")
