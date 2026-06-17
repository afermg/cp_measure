"""A/B Option A: triple-raster + int32-packed coords WHILE KEEPING the overflow-proof
ring buffer + in-queue dedup flag (our existing invariant). Compare to the current
shipped kernel (OLD) for bit-exactness + speed, on real recon inputs."""

import time
import numpy as np
from numba import njit

import cp_measure.core.numba._granularity as G
import cp_measure.core.numba.measuregranularity as MG
from ab_granularity_recon import lab, pix  # reuse synthetic field

OLD = G.reconstruction_by_dilation_2d


@njit(cache=True)
def _geodesic_fwd(out, mask):
    H, W = out.shape
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


@njit(cache=True)
def _geodesic_bwd(out, mask):
    H, W = out.shape
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


@njit(cache=True)
def NEW2(seed, mask):
    H, W = seed.shape
    out = np.empty((H, W), np.float64)
    for i in range(H):
        for j in range(W):
            s = seed[i, j]
            m = mask[i, j]
            out[i, j] = s if s < m else m

    # 3 raster pairs (Vincent/Robinson) before FIFO seeding; pairs 1-2 no seed
    _geodesic_fwd(out, mask)
    _geodesic_bwd(out, mask)
    _geodesic_fwd(out, mask)
    _geodesic_bwd(out, mask)
    _geodesic_fwd(out, mask)

    cap = H * W + 1
    queue = np.empty(cap, np.int32)  # packed (i<<16)|j; ring buffer, dedup => bounded
    inq = np.zeros((H, W), np.uint8)
    mask16 = np.int32(0xFFFF)
    head = 0
    tail = 0

    # backward raster 3 + FIFO seeding
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
                queue[tail] = np.int32((i << 16) | j)
                inq[i, j] = 1
                tail += 1
                if tail == cap:
                    tail = 0

    while head != tail:
        code = queue[head]
        head += 1
        if head == cap:
            head = 0
        i = code >> np.int32(16)
        j = code & mask16
        inq[i, j] = 0
        v = out[i, j]
        if i > 0:
            mq = mask[i - 1, j]
            nv = v if v < mq else mq
            if nv > out[i - 1, j]:
                out[i - 1, j] = nv
                if inq[i - 1, j] == 0:
                    inq[i - 1, j] = 1
                    queue[tail] = np.int32(((i - 1) << 16) | j)
                    tail += 1
                    if tail == cap:
                        tail = 0
        if i < H - 1:
            mq = mask[i + 1, j]
            nv = v if v < mq else mq
            if nv > out[i + 1, j]:
                out[i + 1, j] = nv
                if inq[i + 1, j] == 0:
                    inq[i + 1, j] = 1
                    queue[tail] = np.int32(((i + 1) << 16) | j)
                    tail += 1
                    if tail == cap:
                        tail = 0
        if j > 0:
            mq = mask[i, j - 1]
            nv = v if v < mq else mq
            if nv > out[i, j - 1]:
                out[i, j - 1] = nv
                if inq[i, j - 1] == 0:
                    inq[i, j - 1] = 1
                    queue[tail] = np.int32((i << 16) | (j - 1))
                    tail += 1
                    if tail == cap:
                        tail = 0
        if j < W - 1:
            mq = mask[i, j + 1]
            nv = v if v < mq else mq
            if nv > out[i, j + 1]:
                out[i, j + 1] = nv
                if inq[i, j + 1] == 0:
                    inq[i, j + 1] = 1
                    queue[tail] = np.int32((i << 16) | (j + 1))
                    tail += 1
                    if tail == cap:
                        tail = 0
    return out


captured = {}
for sub in (0.25, 1.0):
    real = G.reconstruction_by_dilation_2d
    store = []
    MG.reconstruction_by_dilation_2d = lambda s, m, _st=store, _r=real: (_st.append((s.copy(), m.copy())), _r(s, m))[1]
    MG.get_granularity(lab, pix, subsample_size=sub)
    MG.reconstruction_by_dilation_2d = real
    captured[sub] = store


def bench(fn, s, m, reps=10):
    fn(s, m)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn(s, m)
        ts.append(time.perf_counter() - t)
    return min(ts) * 1e3


for sub, label in [(0.25, "default 270^2"), (1.0, "fullres 1080^2")]:
    pairs = captured[sub]
    exact = all(np.array_equal(OLD(s, m), NEW2(s, m)) for s, m in pairs)
    t_old = sum(bench(OLD, s, m) for s, m in pairs)
    t_new = sum(bench(NEW2, s, m) for s, m in pairs)
    print(f"[{label}] bit-exact={exact}  OLD={t_old:7.1f}ms NEW2={t_new:7.1f}ms  "
          f"speedup={t_old / t_new:.2f}x")
