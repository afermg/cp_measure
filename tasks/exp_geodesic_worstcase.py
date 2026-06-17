"""Pick the chamfer-geodesic kernel design: raster-sweeps vs FIFO-queue.

Raster-sweep-to-convergence is cache-friendly but needs O(path-windings) passes —
pathological on a spiral. A FIFO Bellman-Ford (re-push only improved pixels, like
the granularity reconstruction) is O(N) amortised and bounded. Compare correctness
(both vs centrosome propagate) and sweep count / time on convex, concave, spiral.
"""

import time

import centrosome.cpmorphology
import centrosome.propagate
import numpy as np
import scipy.ndimage
from numba import njit

SQ2 = np.sqrt(2.0)
_DR = np.array([-1, 1, 0, 0, -1, -1, 1, 1], np.int64)
_DC = np.array([0, 0, -1, 1, -1, 1, -1, 1], np.int64)
_DW = np.array([1.0, 1.0, 1.0, 1.0, SQ2, SQ2, SQ2, SQ2], np.float64)


@njit(cache=True)
def geodesic_sweeps(mask, si, sj):
    H, W = mask.shape
    d = np.full((H, W), 1e18)
    d[si, sj] = 0.0
    npass = 0
    changed = True
    while changed:
        changed = False
        npass += 1
        for r in range(H):
            for c in range(W):
                if not mask[r, c]:
                    continue
                best = d[r, c]
                if r > 0 and mask[r-1, c] and d[r-1, c]+1.0 < best: best = d[r-1, c]+1.0
                if c > 0 and mask[r, c-1] and d[r, c-1]+1.0 < best: best = d[r, c-1]+1.0
                if r > 0 and c > 0 and mask[r-1, c-1] and d[r-1, c-1]+SQ2 < best: best = d[r-1, c-1]+SQ2
                if r > 0 and c < W-1 and mask[r-1, c+1] and d[r-1, c+1]+SQ2 < best: best = d[r-1, c+1]+SQ2
                if best < d[r, c]: d[r, c] = best; changed = True
        for r in range(H-1, -1, -1):
            for c in range(W-1, -1, -1):
                if not mask[r, c]:
                    continue
                best = d[r, c]
                if r < H-1 and mask[r+1, c] and d[r+1, c]+1.0 < best: best = d[r+1, c]+1.0
                if c < W-1 and mask[r, c+1] and d[r, c+1]+1.0 < best: best = d[r, c+1]+1.0
                if r < H-1 and c < W-1 and mask[r+1, c+1] and d[r+1, c+1]+SQ2 < best: best = d[r+1, c+1]+SQ2
                if r < H-1 and c > 0 and mask[r+1, c-1] and d[r+1, c-1]+SQ2 < best: best = d[r+1, c-1]+SQ2
                if best < d[r, c]: d[r, c] = best; changed = True
    return d, npass


@njit(cache=True)
def geodesic_fifo(mask, si, sj):
    """Bellman-Ford with a ring-buffer FIFO; each pixel re-pushed when improved."""
    H, W = mask.shape
    d = np.full((H, W), 1e18)
    d[si, sj] = 0.0
    cap = H * W + 1
    qr = np.empty(cap, np.int64)
    qc = np.empty(cap, np.int64)
    inq = np.zeros((H, W), np.bool_)
    head = 0
    tail = 0
    qr[tail] = si; qc[tail] = sj; tail = (tail + 1) % cap
    inq[si, sj] = True
    while head != tail:
        r = qr[head]; c = qc[head]; head = (head + 1) % cap
        inq[r, c] = False
        dr = d[r, c]
        for k in range(8):
            rr = r + _DR[k]; cc = c + _DC[k]; w = _DW[k]
            if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and dr + w < d[rr, cc]:
                d[rr, cc] = dr + w
                if not inq[rr, cc]:
                    inq[rr, cc] = True
                    qr[tail] = rr; qc[tail] = cc; tail = (tail + 1) % cap
    return d


def make_spiral(n=120):
    m = np.zeros((n, n), bool)
    cx = cy = n // 2
    import math
    for tt in range(0, 4000):
        ang = tt * 0.15
        rad = tt * 0.06
        if rad > n // 2 - 3:
            break
        y = int(cy + rad * math.sin(ang)); x = int(cx + rad * math.cos(ang))
        m[max(y-2, 0):y+3, max(x-2, 0):x+3] = True
    return m


def seed_of(m):
    d = scipy.ndimage.distance_transform_edt(m)
    i, j = centrosome.cpmorphology.maximum_position_of_labels(d, m.astype(np.int32), [1])
    return int(i[0]), int(j[0])


def cent(m, si, sj):
    ctr = np.zeros(m.shape, int); ctr[si, sj] = 1
    _, d = centrosome.propagate.propagate(np.zeros(m.shape), ctr, m, 1)
    return d


def t(fn, reps=5):
    best = float("inf"); out = None
    for _ in range(reps):
        s = time.perf_counter(); out = fn(); best = min(best, time.perf_counter()-s)
    return best*1e3, out


def check(name, m):
    m = m.astype(np.bool_); si, sj = seed_of(m)
    geodesic_sweeps(m, si, sj); geodesic_fifo(m, si, sj)  # warm
    (_, npass) = geodesic_sweeps(m, si, sj)[0:2] if False else (None, 0)
    dswp, npass = geodesic_sweeps(m, si, sj)
    dfifo = geodesic_fifo(m, si, sj)
    dc = cent(m, si, sj)
    mm = m
    ok_s = np.allclose(dswp[mm], dc[mm], atol=1e-9)
    ok_f = np.allclose(dfifo[mm], dc[mm], atol=1e-9)
    ms_s, _ = t(lambda: geodesic_sweeps(m, si, sj))
    ms_f, _ = t(lambda: geodesic_fifo(m, si, sj))
    ms_c, _ = t(lambda: cent(m, si, sj))
    print(f"{name:<12} px={mm.sum():6d}  sweeps={npass:3d}  "
          f"swp={ms_s:6.2f}ms(exact={ok_s})  fifo={ms_f:6.2f}ms(exact={ok_f})  cent={ms_c:6.2f}ms")


def main():
    sq = np.zeros((100, 100), bool); sq[5:95, 5:95] = True
    check("convex", sq)
    U = np.zeros((100, 100), bool); U[5:95, 5:95] = True; U[5:70, 30:70] = False
    check("concave U", U)
    check("spiral", make_spiral())


if __name__ == "__main__":
    main()
