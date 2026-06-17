"""POC: validate the granularity 'numba bilinear gather replaces map_coordinates' win.

Checks (1) accuracy of a numba order-1/mode=constant gather vs scipy.map_coordinates
and (2) per-call speed, on the ACTUAL (sy,sx)+rec the granularity loop produces.
"""

import time

import numpy as np
import scipy.ndimage
from numba import njit


@njit(cache=True)
def _bilinear_gather(img, y0, x0, fy, fx, H, W):
    """order-1, mode='constant' cval=0 bilinear sample at precomputed floor/frac."""
    out = np.empty(y0.shape[0])
    for t in range(y0.shape[0]):
        iy = y0[t]
        ix = x0[t]
        gy = fy[t]
        gx = fx[t]
        v00 = img[iy, ix] if (0 <= iy < H and 0 <= ix < W) else 0.0
        v01 = img[iy, ix + 1] if (0 <= iy < H and 0 <= ix + 1 < W) else 0.0
        v10 = img[iy + 1, ix] if (0 <= iy + 1 < H and 0 <= ix < W) else 0.0
        v11 = img[iy + 1, ix + 1] if (0 <= iy + 1 < H and 0 <= ix + 1 < W) else 0.0
        out[t] = (v00 * (1 - gx) + v01 * gx) * (1 - gy) + (v10 * (1 - gx) + v11 * gx) * gy
    return out


rng = np.random.default_rng(0)
side, n = 1080, 144
yy, xx = np.mgrid[0:side, 0:side]
c = rng.integers(0, side, size=(n, 2))
best = np.full((side, side), np.inf)
lab = np.zeros((side, side), np.int32)
for i, (cy, cx) in enumerate(c, 1):
    d = (yy - cy) ** 2 + (xx - cx) ** 2
    m = d < best
    best[m] = d[m]
    lab[m] = i

# emulate the granularity sample coords: object pixels mapped to the downsampled grid
flat_pos = np.flatnonzero(lab.ravel() > 0)
oy, ox = np.unravel_index(flat_pos, (side, side))
new = int(side * 0.25)
sy = oy * (float(new - 1) / float(side - 1))
sx = ox * (float(new - 1) / float(side - 1))
rec = rng.random((new, new))  # a reconstructed (downsampled) image
H, W = rec.shape

# scipy reference
ref = scipy.ndimage.map_coordinates(rec, (sy, sx), order=1)  # mode='constant' default

# numba gather (precompute floor/frac ONCE, as the real impl would)
y0 = np.floor(sy).astype(np.int64)
x0 = np.floor(sx).astype(np.int64)
fy = sy - y0
fx = sx - x0
got = _bilinear_gather(rec, y0, x0, fy, fx, H, W)

print(f"n sample points: {sy.size}")
print(f"max abs diff vs map_coordinates: {np.max(np.abs(got - ref)):.2e}")
print(f"all close <1e-12: {np.allclose(got, ref, rtol=0, atol=1e-12)}")


def t(fn, *a, reps=20):
    fn(*a)
    ts = []
    for _ in range(reps):
        s = time.perf_counter()
        fn(*a)
        ts.append(time.perf_counter() - s)
    return min(ts) * 1e3


t_scipy = t(lambda: scipy.ndimage.map_coordinates(rec, (sy, sx), order=1))
t_numba = t(lambda: _bilinear_gather(rec, y0, x0, fy, fx, H, W))
print(f"map_coordinates per call: {t_scipy:.3f} ms")
print(f"numba gather  per call:   {t_numba:.3f} ms  ({t_scipy / t_numba:.1f}x)")
print(f"x16 calls saved ~ {16 * (t_scipy - t_numba):.1f} ms/image (floor/frac precomputed once)")
