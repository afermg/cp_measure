"""A/B the two granularity reconstruction opts (int32-packed FIFO queue + triple
raster pass) vs the current single-pair kernel. Bit-exact check + per-call speed,
on the REAL (seed,mask) the spectrum loop produces, at default (270^2) and fullres
(1080^2). Both opts are result-preserving (exact reconstruction either way)."""

import time
import numpy as np
from numba import njit

import cp_measure.core.numba._granularity as G
import cp_measure.core.numba.measuregranularity as MG

OLD = G.reconstruction_by_dilation_2d  # current shipped kernel


@njit(cache=True)
def NEW(seed, mask):
    """Triple-raster + int32-packed-queue hybrid reconstruction (4-conn)."""
    H, W = seed.shape
    out = np.empty((H, W), np.float64)
    for i in range(H):
        for j in range(W):
            s = seed[i, j]
            m = mask[i, j]
            out[i, j] = s if s < m else m

    # 3 forward+backward raster pairs (pairs 1 and 2 do not seed)
    for _pair in range(2):
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

    # forward scan 3
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

    cap = 12 * H * W
    queue = np.empty(cap, np.int32)
    tail = 0
    mask16 = np.int32(0xFFFF)
    # backward scan 3 + seeding
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
            if (
                (i > 0 and out[i - 1, j] < v and out[i - 1, j] < mask[i - 1, j])
                or (j > 0 and out[i, j - 1] < v and out[i, j - 1] < mask[i, j - 1])
                or (i < H - 1 and out[i + 1, j] < v and out[i + 1, j] < mask[i + 1, j])
                or (j < W - 1 and out[i, j + 1] < v and out[i, j + 1] < mask[i, j + 1])
            ):
                queue[tail] = np.int32((i << 16) | j)
                tail += 1

    head = 0
    while head < tail:
        packed = queue[head]
        head += 1
        ci = packed >> np.int32(16)
        cj = packed & mask16
        v = out[ci, cj]
        if ci > 0 and out[ci - 1, cj] < v and out[ci - 1, cj] < mask[ci - 1, cj]:
            mq = mask[ci - 1, cj]
            out[ci - 1, cj] = v if v < mq else mq
            queue[tail] = np.int32(((ci - 1) << 16) | cj)
            tail += 1
        if ci < H - 1 and out[ci + 1, cj] < v and out[ci + 1, cj] < mask[ci + 1, cj]:
            mq = mask[ci + 1, cj]
            out[ci + 1, cj] = v if v < mq else mq
            queue[tail] = np.int32(((ci + 1) << 16) | cj)
            tail += 1
        if cj > 0 and out[ci, cj - 1] < v and out[ci, cj - 1] < mask[ci, cj - 1]:
            mq = mask[ci, cj - 1]
            out[ci, cj - 1] = v if v < mq else mq
            queue[tail] = np.int32((ci << 16) | (cj - 1))
            tail += 1
        if cj < W - 1 and out[ci, cj + 1] < v and out[ci, cj + 1] < mask[ci, cj + 1]:
            mq = mask[ci, cj + 1]
            out[ci, cj + 1] = v if v < mq else mq
            queue[tail] = np.int32((ci << 16) | (cj + 1))
            tail += 1
    return out


# --- synthetic field (same as profiler) ---
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

captured = {}


def capture(subsample):
    real = G.reconstruction_by_dilation_2d
    store = []

    def tap(seed, mask):
        store.append((seed.copy(), mask.copy()))
        return real(seed, mask)

    MG.reconstruction_by_dilation_2d = tap
    MG.get_granularity(lab, pix, subsample_size=subsample)
    MG.reconstruction_by_dilation_2d = real
    captured[subsample] = store


capture(0.25)
capture(1.0)


def bench(fn, seed, mask, reps=10):
    fn(seed, mask)
    ts = []
    for _ in range(reps):
        s = time.perf_counter()
        fn(seed, mask)
        ts.append(time.perf_counter() - s)
    return min(ts) * 1e3


for sub, label in [(0.25, "default 270^2"), (1.0, "fullres 1080^2")]:
    pairs = captured[sub]
    # bit-exact check across all 16 steps
    exact = all(np.array_equal(OLD(s, m), NEW(s, m)) for s, m in pairs)
    t_old = sum(bench(OLD, s, m) for s, m in pairs)
    t_new = sum(bench(NEW, s, m) for s, m in pairs)
    print(f"[{label}] bit-exact={exact}  16-step recon: OLD={t_old:7.1f}ms  NEW={t_new:7.1f}ms  "
          f"speedup={t_old/t_new:.2f}x  saved={t_old-t_new:.1f}ms")
