"""Numba segment kernels (single-threaded, cached).

These are the numba implementation of the segment-reduce / segment-quantile
primitives. They loop over the flat ``(values, seg0, coords)`` arrays produced
by :mod:`cp_measure.primitives.segment` — no image shape, no batch axis — so one
kernel set covers 2D, 3D, and (future) batched inputs unchanged.

All kernels are ``@njit(cache=True)`` and serial: no ``prange``/``nogil``.
Parallelism is the job of the (future) batch layer over images, not the kernel.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def segment_moments(values, seg0, xc, yc, zc, n):
    """One pass over the flat pixels accumulating, per segment:

    count, sum, min, max, and the six centroid cross-sums
    (Sx, Sy, Sz, Sx*I, Sy*I, Sz*I). max position is intentionally NOT computed
    here (it is done on the host via scipy for exact parity).
    """
    count = np.zeros(n, np.int64)
    sumI = np.zeros(n, np.float64)
    minI = np.full(n, np.inf)
    maxI = np.full(n, -np.inf)
    sx = np.zeros(n, np.float64)
    sy = np.zeros(n, np.float64)
    sz = np.zeros(n, np.float64)
    sxI = np.zeros(n, np.float64)
    syI = np.zeros(n, np.float64)
    szI = np.zeros(n, np.float64)
    M = values.shape[0]
    for i in range(M):
        k = seg0[i]
        v = values[i]
        count[k] += 1
        sumI[k] += v
        if v < minI[k]:
            minI[k] = v
        if v > maxI[k]:
            maxI[k] = v
        x = xc[i]
        y = yc[i]
        z = zc[i]
        sx[k] += x
        sy[k] += y
        sz[k] += z
        sxI[k] += x * v
        syI[k] += y * v
        szI[k] += z * v
    return count, sumI, minI, maxI, sx, sy, sz, sxI, syI, szI


@njit(cache=True)
def segment_resid_sumsq(values, seg0, n, mean):
    """Second pass: per-segment sum of squared residuals (for population std)."""
    ss = np.zeros(n, np.float64)
    M = values.shape[0]
    for i in range(M):
        k = seg0[i]
        d = values[i] - mean[k]
        ss[k] += d * d
    return ss


@njit(cache=True)
def _interp(sorted_seg, n, frac):
    """Linear interpolation at position ``n * frac`` within a sorted segment.

    Matches the reference's ``cumsum(areas)``-offset interpolation: the local
    position is ``count * frac``; clamp the upper neighbour at the last element.
    """
    pos = n * frac
    lo = int(pos)
    if lo > n - 1:
        lo = n - 1
    hi = lo + 1
    if hi > n - 1:
        hi = n - 1
    f = pos - lo
    return sorted_seg[lo] * (1.0 - f) + sorted_seg[hi] * f


@njit(cache=True)
def segment_quantiles(values, seg0, counts, n, mad_frac):
    """Per-segment quartiles/median + MAD via scatter-into-segments + sort.

    Builds CSR offsets from ``counts``, scatters values into one flat buffer,
    sorts each segment slice in place, and interpolates. MAD reuses the sorted
    absolute deviations from the median at ``mad_frac`` (= 1 / original ndim).
    """
    starts = np.zeros(n, np.int64)
    acc = 0
    for k in range(n):
        starts[k] = acc
        acc += counts[k]
    total = acc

    buf = np.empty(total, np.float64)
    cursor = starts.copy()
    M = values.shape[0]
    for i in range(M):
        k = seg0[i]
        buf[cursor[k]] = values[i]
        cursor[k] += 1

    lq = np.zeros(n, np.float64)
    med = np.zeros(n, np.float64)
    uq = np.zeros(n, np.float64)
    mad = np.zeros(n, np.float64)
    for k in range(n):
        cnt = counts[k]
        if cnt == 0:
            continue
        s = starts[k]
        seg = buf[s : s + cnt]
        seg.sort()
        lq[k] = _interp(seg, cnt, 0.25)
        med[k] = _interp(seg, cnt, 0.5)
        uq[k] = _interp(seg, cnt, 0.75)
        ad = np.abs(seg - med[k])
        ad.sort()
        mad[k] = _interp(ad, cnt, mad_frac)
    return lq, med, uq, mad
