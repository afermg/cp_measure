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
def flatten_numba(masks, pixels, lut):
    """Flatten a labeled (Z, Y, X) image to ``(values, seg0, xc, yc, zc)``.

    Two grid scans (count, then fill) replace the numpy ``(masks>0)&isfinite``
    mask + ``numpy.nonzero`` + fancy-index gathers; coordinates are the loop
    indices. Background and non-finite pixels are dropped, in C (raster) order.
    ``masks`` and ``pixels`` must be C-contiguous; ``pixels`` may be any float
    dtype (kept values are upcast into the float64 ``values`` output).
    """
    Z, Y, X = masks.shape
    M = 0
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if masks[z, y, x] > 0 and np.isfinite(pixels[z, y, x]):
                    M += 1
    values = np.empty(M, np.float64)
    seg0 = np.empty(M, np.int64)
    xc = np.empty(M, np.float64)
    yc = np.empty(M, np.float64)
    zc = np.empty(M, np.float64)
    i = 0
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                L = masks[z, y, x]
                if L <= 0:
                    continue
                v = pixels[z, y, x]
                if not np.isfinite(v):
                    continue
                values[i] = v
                seg0[i] = lut[L]
                xc[i] = x
                yc[i] = y
                zc[i] = z
                i += 1
    return values, seg0, xc, yc, zc


@njit(cache=True)
def segment_moments(values, seg0, xc, yc, zc, n):
    """One pass over the flat pixels accumulating, per segment:

    count, sum, min, max, the max-pixel coordinates, and the six centroid
    cross-sums (Sx, Sy, Sz, Sx*I, Sy*I, Sz*I).

    The max position uses ``>=`` so the LAST pixel (in the flat raster order the
    host builds the arrays in) attaining the maximum wins. On real, continuous
    data the max is unique, so this is bit-identical to the numpy backend's
    ``scipy.ndimage.maximum_position``; on exact-value ties it is a deterministic
    rule rather than scipy's quicksort-dependent (version-unstable) pick.
    """
    count = np.zeros(n, np.int64)
    sumI = np.zeros(n, np.float64)
    minI = np.full(n, np.inf)
    maxI = np.full(n, -np.inf)
    mx = np.zeros(n, np.float64)
    my = np.zeros(n, np.float64)
    mz = np.zeros(n, np.float64)
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
        x = xc[i]
        y = yc[i]
        z = zc[i]
        count[k] += 1
        sumI[k] += v
        if v < minI[k]:
            minI[k] = v
        if v >= maxI[k]:  # >= -> keep LAST max in raster order
            maxI[k] = v
            mx[k] = x
            my[k] = y
            mz[k] = z
        sx[k] += x
        sy[k] += y
        sz[k] += z
        sxI[k] += x * v
        syI[k] += y * v
        szI[k] += z * v
    return count, sumI, minI, maxI, mx, my, mz, sx, sy, sz, sxI, syI, szI


@njit(cache=True)
def segment_stats(values, seg0, n):
    """One pass accumulating per-segment count, sum, min, max only.

    The lightweight reduce for callers (e.g. edge measurements) that need
    neither the max position nor the centroid cross-sums.
    """
    count = np.zeros(n, np.int64)
    sumI = np.zeros(n, np.float64)
    minI = np.full(n, np.inf)
    maxI = np.full(n, -np.inf)
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
    return count, sumI, minI, maxI


@njit(cache=True)
def inner_boundary(masks2d):
    """Labeled inner boundary of a 2D label plane (0 = not a boundary pixel).

    A foreground pixel is a boundary pixel iff any in-bounds 4-neighbour has a
    different label; out-of-bounds neighbours are ignored. Bit-identical to
    ``skimage.segmentation.find_boundaries(mode="inner")`` on a single 2D plane,
    ~12-27x faster (one pass, no morphology). ``masks2d`` must be C-contiguous.
    """
    H, W = masks2d.shape
    out = np.zeros((H, W), masks2d.dtype)
    for r in range(H):
        for c in range(W):
            L = masks2d[r, c]
            if L <= 0:
                continue
            if (
                (r > 0 and masks2d[r - 1, c] != L)
                or (r < H - 1 and masks2d[r + 1, c] != L)
                or (c > 0 and masks2d[r, c - 1] != L)
                or (c < W - 1 and masks2d[r, c + 1] != L)
            ):
                out[r, c] = L
    return out


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


@njit(cache=True)
def flatten_pairs_grouped(masks, pix1, pix2, lut, n):
    """Flatten two co-registered ``(Z, Y, X)`` channels to per-object blocks.

    Returns ``(g1, g2, offsets)`` where object ``k`` owns the contiguous slices
    ``g1[offsets[k] : offsets[k + 1]]`` and ``g2[...]``. Both channels are
    gathered at every ``masks > 0`` pixel in one raster scan, so ``g1`` and
    ``g2`` stay aligned. Unlike :func:`flatten_numba`, non-finite pixels are
    KEPT — this mirrors the colocalization reference's ``pixels[mask]``
    extraction, which applies no finiteness filter. Two scans (count, then
    scatter) give an O(M) counting-sort grouping; ``masks`` must be C-contiguous
    integer, ``lut`` the ``label -> 0..n-1`` map from ``label_to_idx_lut``.
    """
    Z, Y, X = masks.shape
    counts = np.zeros(n, np.int64)
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                L = masks[z, y, x]
                if L > 0:
                    counts[lut[L]] += 1
    offsets = np.zeros(n + 1, np.int64)
    for k in range(n):
        offsets[k + 1] = offsets[k] + counts[k]
    M = offsets[n]
    g1 = np.empty(M, np.float64)
    g2 = np.empty(M, np.float64)
    fill = offsets[:n].copy()
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                L = masks[z, y, x]
                if L > 0:
                    k = lut[L]
                    p = fill[k]
                    g1[p] = pix1[z, y, x]
                    g2[p] = pix2[z, y, x]
                    fill[k] = p + 1
    return g1, g2, offsets
