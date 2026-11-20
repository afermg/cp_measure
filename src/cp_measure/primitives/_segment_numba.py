"""Numba segment kernels (single-threaded, cached).

These are the numba implementation of the segment-reduce / segment-quantile
primitives. They loop over the flat ``(values, seg0, coords)`` arrays that
:func:`flatten_numba` builds — no image shape, no batch axis — so one kernel set
covers 2D, 3D, and (future) batched inputs unchanged.

All kernels are ``@njit(cache=True, nogil=True)`` and serial: no ``prange``.
Parallelism is the job of the batch layer over images (threads, which need the
GIL released), not the kernel.
"""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def flatten_numba(masks, pixels):
    """Flatten a labeled (Z, Y, X) image to ``(values, seg0, xc, yc, zc)``.

    Two grid scans (count, then fill) replace the numpy ``(masks>0)&isfinite``
    mask + ``numpy.nonzero`` + fancy-index gathers; coordinates are the loop
    indices. Background and non-finite pixels are dropped, in C (raster) order.
    Labels are the contiguous ``1..N`` cp_measure guarantees (see
    :mod:`cp_measure._sanitize`), so the segment index is ``label - 1``.
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
                seg0[i] = L - 1
                xc[i] = x
                yc[i] = y
                zc[i] = z
                i += 1
    return values, seg0, xc, yc, zc


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
def segment_resid_sumsq(values, seg0, n, mean):
    """Second pass: per-segment sum of squared residuals (for population std)."""
    ss = np.zeros(n, np.float64)
    M = values.shape[0]
    for i in range(M):
        k = seg0[i]
        d = values[i] - mean[k]
        ss[k] += d * d
    return ss


@njit(cache=True, nogil=True)
def _interp_pos(n, frac, legacy):
    """Interpolation position within a segment of ``n`` sorted values.

    ``legacy=True`` uses ``pos = n * frac`` (the CellProfiler / cumsum-offset
    convention); ``legacy=False`` uses ``pos = (n - 1) * frac``
    (``numpy.percentile`` 'linear'). Returns ``(lo, hi, f)``: the two neighbouring
    ranks (upper clamped at the last element) and the weight of ``hi``.
    """
    pos = n * frac if legacy else (n - 1) * frac
    lo = int(pos)
    if lo > n - 1:
        lo = n - 1
    hi = lo + 1
    if hi > n - 1:
        hi = n - 1
    return lo, hi, pos - lo


@njit(cache=True, nogil=True)
def segment_quantiles(values, seg0, counts, n, mad_frac, legacy):
    """Per-segment quartiles/median + MAD via scatter-into-segments + selection.

    Builds CSR offsets from ``counts``, scatters values into one flat buffer,
    places the ranks the interpolation reads with ``np.partition`` (O(n)
    selection instead of an O(n log n) sort; the values at those ranks, and so
    the results, match the sorted segment's), and interpolates with the
    convention selected by ``legacy`` (see :func:`_interp_pos`). MAD is the same interpolation over the absolute
    deviations from the median: in legacy mode at ``mad_frac`` (= 1 / original
    ndim), otherwise the textbook median.
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
        l1, h1, f1 = _interp_pos(cnt, 0.25, legacy)
        l2, h2, f2 = _interp_pos(cnt, 0.5, legacy)
        l3, h3, f3 = _interp_pos(cnt, 0.75, legacy)
        part = np.partition(seg, np.unique(np.array([l1, h1, l2, h2, l3, h3])))
        lq[k] = part[l1] * (1.0 - f1) + part[h1] * f1
        med[k] = part[l2] * (1.0 - f2) + part[h2] * f2
        uq[k] = part[l3] * (1.0 - f3) + part[h3] * f3
        ad = np.abs(seg - med[k])
        lm, hm, fm = _interp_pos(cnt, mad_frac if legacy else 0.5, legacy)
        part = np.partition(ad, np.unique(np.array([lm, hm])))
        mad[k] = part[lm] * (1.0 - fm) + part[hm] * fm
    return lq, med, uq, mad


@njit(cache=True, nogil=True)
def label_counts(masks, n):
    """Pixels of each label ``1..n`` in a C-contiguous label image; other values
    (background, negatives) are skipped. One pass, no full-image ``bincount``."""
    counts = np.zeros(n, np.int64)
    for v in masks.ravel():
        if v > 0 and v <= n:
            counts[v - 1] += 1
    return counts


@njit(cache=True, nogil=True)
def flatten_pairs_grouped(masks, pix1, pix2, offsets):
    """Flatten two co-registered ``(Z, Y, X)`` channels to per-object blocks.

    Returns ``(g1, g2)`` where object ``k`` owns the contiguous slices
    ``g1[offsets[k] : offsets[k + 1]]`` and ``g2[...]``. Both channels are
    gathered at every ``masks > 0`` pixel in one raster scan, so ``g1`` and
    ``g2`` stay aligned. Unlike :func:`flatten_numba`, non-finite pixels are
    KEPT — this mirrors the colocalization reference's ``pixels[mask]``
    extraction, which applies no finiteness filter.

    A SINGLE scatter scan: ``offsets`` (the CSR block bounds, length ``n + 1``)
    is precomputed from a ``bincount``, so the count scan this kernel used to do
    is gone. Labels are the contiguous ``1..N`` cp_measure guarantees (see
    :mod:`cp_measure._sanitize`), so the segment index is ``label - 1`` — no
    lookup table. ``masks`` must be C-contiguous integer.
    """
    Z, Y, X = masks.shape
    g1 = np.empty(offsets[-1], np.float64)
    g2 = np.empty(offsets[-1], np.float64)
    fill = offsets[:-1].copy()
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                L = masks[z, y, x]
                if L > 0:
                    k = L - 1
                    p = fill[k]
                    g1[p] = pix1[z, y, x]
                    g2[p] = pix2[z, y, x]
                    fill[k] = p + 1
    return g1, g2


@njit(cache=True, nogil=True)
def group_pairs(masks, pix1, pix2):
    """:func:`flatten_pairs_grouped` with its block bounds, in one GIL-free call.

    Returns ``(g1, g2, offsets)`` for labels ``1..masks.max()``; a missing label
    gets an empty block. Float32 pixels are widened to float64 exactly.
    """
    n = np.int64(0)
    for v in masks.ravel():
        if v > n:
            n = np.int64(v)
    offsets = np.zeros(n + 1, np.int64)
    offsets[1:] = np.cumsum(label_counts(masks, n))
    g1, g2 = flatten_pairs_grouped(masks, pix1, pix2, offsets)
    return g1, g2, offsets


# NumPy reduces a contiguous float64 array with a pairwise sum (``pairwise_sum`` in
# numpy/_core/src/umath/loops_utils.h.src). Before NumPy 2.3 the reduction ran over
# chunks of its buffer size (``np.getbufsize()``, 8192 by default), added left to
# right. Note: numba's on-disk cache checks a cached function only against its own
# file, so a kernel in another file that calls these keeps a stale copy after an edit
# here; clear the numba caches (``__pycache__/*.nb[ic]``) after editing this file.
_NUMPY_CHUNKS = tuple(int(p) for p in np.__version__.split(".")[:2]) < (2, 3)


def numpy_sum_chunk() -> int:
    """The ``chunk`` argument of :func:`numpy_sum`: NumPy's current buffer size
    before 2.3, else 0. Read at call time, as ``np.setbufsize`` changes it."""
    return np.getbufsize() if _NUMPY_CHUNKS else 0


@njit(cache=True, nogil=True)
def _block_sum(a, lo, n):
    """``pairwise_sum``'s leaf: a serial sum below 8 values, else eight interleaved
    partial sums combined as a balanced tree, then the remainder."""
    if n < 8:
        res = 0.0
        for i in range(lo, lo + n):
            res += a[i]
        return res
    r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
    r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
    i = 8
    while i < n - (n % 8):
        j = lo + i
        r0, r1, r2, r3 = r0 + a[j], r1 + a[j + 1], r2 + a[j + 2], r3 + a[j + 3]
        r4, r5, r6, r7 = r4 + a[j + 4], r5 + a[j + 5], r6 + a[j + 6], r7 + a[j + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@njit(cache=True, nogil=True)
def _pairwise_sum(a, lo, n):
    """NumPy's ``pairwise_sum`` over ``a[lo : lo + n]``: blocks of at most 128 values,
    halves split at a multiple of 8 and added as ``left + right``.

    Evaluated with an explicit stack because numba's on-disk cache does not handle
    recursive functions (a cached recursive version crashed on reload).
    """
    if n <= 128:
        return _block_sum(a, lo, n)
    # frames: (start, length, stage); stage 0 = expand left, 1 = expand right, 2 = add
    starts = np.empty(64, np.int64)
    lengths = np.empty(64, np.int64)
    stages = np.empty(64, np.int64)
    values = np.empty(64)
    top = 0
    nvals = 0
    starts[0], lengths[0], stages[0] = lo, n, 0
    while top >= 0:
        start, length, stage = starts[top], lengths[top], stages[top]
        if length <= 128:
            values[nvals] = _block_sum(a, start, length)
            nvals += 1
            top -= 1
            continue
        half = length // 2
        half -= half % 8
        if stage == 0:
            stages[top] = 1
            top += 1
            starts[top], lengths[top], stages[top] = start, half, 0
        elif stage == 1:
            stages[top] = 2
            top += 1
            starts[top], lengths[top], stages[top] = start + half, length - half, 0
        else:
            nvals -= 1
            values[nvals - 1] = values[nvals - 1] + values[nvals]
            top -= 1
    return values[0]


@njit(cache=True, nogil=True)
def numpy_sum(a, chunk):
    """``np.sum(a)`` of a contiguous 1D float64 array, bit for bit (``chunk`` is
    :func:`numpy_sum_chunk`). NumPy starts the reduction from ``0.0``, which turns
    a ``-0.0`` result into ``0.0``."""
    n = a.size
    step = chunk if chunk > 0 else max(n, 1)  # NumPy >= 2.3: one chunk
    res = 0.0
    for lo in range(0, n, step):
        res += _pairwise_sum(a, lo, min(step, n - lo))
    return res


@njit(cache=True, nogil=True)
def numpy_var(a, chunk):
    """``np.var(a, ddof=1)`` of a contiguous 1D float64 array, bit for bit."""
    n = a.size
    mean = numpy_sum(a, chunk) / n
    sq = np.empty(n)
    for i in range(n):
        d = a[i] - mean
        sq[i] = d * d
    return numpy_sum(sq, chunk) / (n - 1)
