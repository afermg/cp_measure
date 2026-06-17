"""Granularity speedup over the numpy baseline (X-factor framing), 1080^2 / 144 obj.

Compares, at default (subsample=0.25) and fullres (subsample=1.0):
  - numpy baseline  (cp_measure.core.measuregranularity.get_granularity)
  - numba PRE-opt   (single raster pair + int64 FIFO  == commit 89dde43)
  - numba SHIPPED   (triple raster + int32-packed FIFO == commit c8b9414)
all as wall-clock + the speedup factor vs baseline."""

import time
import numpy as np
from numba import njit

import cp_measure.core.numba._granularity as G
import cp_measure.core.numba.measuregranularity as MG
from cp_measure.core.measuregranularity import get_granularity as numpy_granularity

SHIPPED = G.reconstruction_by_dilation_2d  # c8b9414 (triple raster + int32)


@njit(cache=True)
def PRE_OPT(seed, mask):
    """The previous kernel: ONE forward+backward raster pair + int64 flat FIFO (89dde43)."""
    H, W = seed.shape
    out = np.empty((H, W), np.float64)
    for i in range(H):
        for j in range(W):
            s = seed[i, j]
            m = mask[i, j]
            out[i, j] = s if s < m else m
    for i in range(H):
        for j in range(W):
            v = out[i, j]
            if i > 0 and out[i - 1, j] > v:
                v = out[i - 1, j]
            if j > 0 and out[i, j - 1] > v:
                v = out[i, j - 1]
            m = mask[i, j]
            if v > m:
                v = m
            out[i, j] = v
    cap = H * W + 1
    queue = np.empty(cap, np.int64)
    inq = np.zeros((H, W), np.uint8)
    head = 0
    tail = 0
    for i in range(H - 1, -1, -1):
        for j in range(W - 1, -1, -1):
            v = out[i, j]
            if i < H - 1 and out[i + 1, j] > v:
                v = out[i + 1, j]
            if j < W - 1 and out[i, j + 1] > v:
                v = out[i, j + 1]
            m = mask[i, j]
            if v > m:
                v = m
            out[i, j] = v
            seed_p = (
                i < H - 1 and out[i + 1, j] < v and out[i + 1, j] < mask[i + 1, j]
            ) or (j < W - 1 and out[i, j + 1] < v and out[i, j + 1] < mask[i, j + 1])
            if seed_p:
                queue[tail] = i * W + j
                inq[i, j] = 1
                tail += 1
                if tail == cap:
                    tail = 0
    while head != tail:
        code = queue[head]
        head += 1
        if head == cap:
            head = 0
        i = code // W
        j = code % W
        inq[i, j] = 0
        v = out[i, j]
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ii = i + di
            jj = j + dj
            if 0 <= ii < H and 0 <= jj < W:
                mq = mask[ii, jj]
                nv = v if v < mq else mq
                if nv > out[ii, jj]:
                    out[ii, jj] = nv
                    if inq[ii, jj] == 0:
                        inq[ii, jj] = 1
                        queue[tail] = ii * W + jj
                        tail += 1
                        if tail == cap:
                            tail = 0
    return out


# synthetic 1080^2 / 144-object textured field
rng = np.random.default_rng(0)
side, n = 1080, 144
yy, xx = np.mgrid[0:side, 0:side]
c = rng.integers(0, side, size=(n, 2))
best = np.full((side, side), np.inf)
lab = np.zeros((side, side), np.int32)
for i, (cy, cx) in enumerate(c, 1):
    d = (yy - cy) ** 2 + (xx - cx) ** 2
    msk = d < best
    best[msk] = d[msk]
    lab[msk] = i
pix = (rng.random((side, side)) * 0.5 + 0.5 * np.sin(xx / 17.0) * np.sin(yy / 13.0) ** 2).astype(np.float64)


def bench(fn, reps=5):
    fn()  # warmup (compile)
    ts = []
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - s)
    return min(ts) * 1e3


for sub, label in [(0.25, "default (subsample=0.25)"), (1.0, "fullres (subsample=1.0)")]:
    t_base = bench(lambda: numpy_granularity(lab, pix, subsample_size=sub))

    MG.reconstruction_by_dilation_2d = PRE_OPT
    t_pre = bench(lambda: MG.get_granularity(lab, pix, subsample_size=sub))

    MG.reconstruction_by_dilation_2d = SHIPPED
    t_ship = bench(lambda: MG.get_granularity(lab, pix, subsample_size=sub))

    print(f"\n== {label} ==")
    print(f"  numpy baseline : {t_base:8.1f} ms   (1.00x)")
    print(f"  numba pre-opt  : {t_pre:8.1f} ms   ({t_base / t_pre:5.2f}x vs baseline)   [89dde43]")
    print(f"  numba SHIPPED  : {t_ship:8.1f} ms   ({t_base / t_ship:5.2f}x vs baseline)   [c8b9414]")
    print(f"  -> the recon opt moved numba {t_base / t_pre:.2f}x -> {t_base / t_ship:.2f}x "
          f"(numba itself {t_pre / t_ship:.2f}x faster)")
