"""Numba kernels for radial intensity distribution (single-threaded, cached).

Two kernels backing :func:`cp_measure.core.numba.measureobjectintensitydistribution.get_radial_distribution`:

- :func:`geodesic_chamfer_fifo`: geodesic distance from the object centre within
  the object mask, as a 1/sqrt(2) chamfer shortest path (FIFO Bellman-Ford with a
  ring-buffer queue). Shortest paths do not depend on the algorithm, so it equals
  centrosome's ``propagate(zeros, seed, mask, 1)`` exactly (tested on convex,
  concave, ring and single-pixel shapes) and is ~35x faster on small crops.
- :func:`radial_object`: one fused pass per crop computing the centre, the
  geodesic, the per-bin histograms and the per-(bin, wedge) sums that replace the
  reference's repeated ``scipy.sparse.coo_matrix(...).toarray()`` builds.

The exact-Euclidean distance to edge (scipy EDT) and the final ``RadialCV``
(``numpy.ma``, to keep the reference's masked-division results) stay on the host.
Serial per call; the kernels release the GIL so batches can run on threads.
"""

import numpy as np
from numba import njit

from cp_measure.primitives._edt_numba import squared_edt_2d

_SQ2 = np.sqrt(2.0)
# 8-neighbour offsets (orthogonal weight 1, diagonal weight sqrt(2)).
_DR = np.array([-1, 1, 0, 0, -1, -1, 1, 1], np.int64)
_DC = np.array([0, 0, -1, 1, -1, 1, -1, 1], np.int64)
_DW = np.array([1.0, 1.0, 1.0, 1.0, _SQ2, _SQ2, _SQ2, _SQ2], np.float64)

# Sentinel for unreached pixels (disconnected from the centre seed).
UNREACHED = 1e18


@njit(cache=True, nogil=True, error_model="numpy")
def geodesic_chamfer_fifo(mask, si, sj):
    """1/sqrt(2) chamfer geodesic distance from ``(si, sj)`` within ``mask``.

    FIFO Bellman-Ford: each pixel is (re)queued when a shorter path is found; the
    ring buffer + in-queue flag keep it O(N) amortised and bounded for any shape.
    Pixels not reachable from the seed within the mask keep ``UNREACHED``.
    Bit-exact to ``centrosome.propagate.propagate(zeros, seed, mask, 1)``.
    """
    H, W = mask.shape
    d = np.full((H, W), UNREACHED)
    d[si, sj] = 0.0
    cap = H * W + 1
    qr = np.empty(cap, np.int64)
    qc = np.empty(cap, np.int64)
    inq = np.zeros((H, W), np.bool_)
    head = 0
    tail = 0
    qr[tail] = si
    qc[tail] = sj
    tail = (tail + 1) % cap
    inq[si, sj] = True
    while head != tail:
        r = qr[head]
        c = qc[head]
        head = (head + 1) % cap
        inq[r, c] = False
        dr = d[r, c]
        for k in range(8):
            rr = r + _DR[k]
            cc = c + _DC[k]
            if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
                nd = dr + _DW[k]
                if nd < d[rr, cc]:
                    d[rr, cc] = nd
                    if not inq[rr, cc]:
                        inq[rr, cc] = True
                        qr[tail] = rr
                        qc[tail] = cc
                        tail = (tail + 1) % cap
    return d


@njit(cache=True, nogil=True, error_model="numpy")
def radial_object(m, pix, d_to_edge, scaled, bin_count, maximum_radius):
    """All per-object radial work for one cropped object, in one numba pass.

    Folds the centre (argmax of ``d_to_edge`` within ``m``), the chamfer geodesic,
    the per-bin intensity/count histograms and the per-(bin, 8-wedge) sums into a
    single kernel, so the host loop does no per-object ``mgrid``/``np.where``.
    Returns ``(frac_at_d, mean_frac, wedge_sums, wedge_counts)``: two arrays of
    length ``bin_count + 1`` (last entry = overflow bin) and two ``(bin_count + 1,
    8)`` arrays from which the host computes ``RadialCV``.

    The centre is the first maximal ``d_to_edge`` pixel in C order, the
    reference's ``_maximum_position_of_labels`` rule. ``error_model="numpy"``
    gives ``nan`` for a ``0/0`` like the reference.
    """
    if bin_count < 0:
        raise ValueError("bin_count must be >= 0")
    H, W = m.shape
    # Centre = first (C-order) pixel with the maximal distance-to-edge.
    best = -1.0
    ci = 0
    cj = 0
    for r in range(H):
        for c in range(W):
            if m[r, c] and d_to_edge[r, c] > best:
                best = d_to_edge[r, c]
                ci = r
                cj = c

    d_from = geodesic_chamfer_fifo(m, ci, cj)

    nb = bin_count + 1
    hist = np.zeros(nb)
    num = np.zeros(nb)
    wsum = np.zeros((nb, 8))
    wcnt = np.zeros((nb, 8))
    for r in range(H):
        for c in range(W):
            if not (m[r, c] and d_from[r, c] < UNREACHED):
                continue
            df = d_from[r, c]
            if scaled:
                nd = df / (df + d_to_edge[r, c] + 0.001)
            else:
                nd = df / maximum_radius
            x = nd * bin_count
            if not np.isfinite(x):
                # Never cast non-finite values into unchecked array indices.
                # The host delegates zero/NaN radii to the NumPy reference.
                raise ValueError("non-finite radial bin")
            if x >= bin_count + 1:
                b = bin_count
            elif x <= -1:  # would index out of bounds; NumPy raises here too
                raise ValueError("negative radial bin; maximum_radius must be > 0")
            else:
                b = int(x)
            w = 0
            if r > ci:
                w += 1
            if c > cj:
                w += 2
            if abs(r - ci) > abs(c - cj):
                w += 4
            v = pix[r, c]
            hist[b] += v
            num[b] += 1.0
            wsum[b, w] += v
            wcnt[b, w] += 1.0

    eps = np.finfo(np.float64).eps
    frac_at_d = np.zeros(nb)
    mean_frac = np.zeros(nb)
    tot_h = hist.sum()
    tot_n = num.sum()
    for b in range(nb):
        frac_at_d[b] = hist[b] / tot_h
        mean_frac[b] = frac_at_d[b] / (num[b] / tot_n + eps)
    return frac_at_d, mean_frac, wsum, wcnt


@njit(cache=True, nogil=True, error_model="numpy")
def radial_all(labels, pixels, n, scaled, bin_count, maximum_radius):
    """:func:`radial_object` for every label ``1..n`` of a 2D image in one call.

    Each object is cropped to its bounding box grown by 1 pixel and clipped to the
    image, as the host loop did (the margin gives the distance transform its
    background edge, the clip keeps the image border from counting as one). The
    distance to edge is the exact EDT (:func:`squared_edt_2d`, then ``sqrt``, as
    scipy forms it). A crop without background, which scipy measures differently,
    is flagged in ``fallback`` for the host to measure. Absent labels stay zero.
    Returns ``(frac_at_d, mean_frac, wedge_sums, wedge_counts, fallback)``.
    """
    H, W = labels.shape
    r0 = np.full(n, H, np.int64)
    r1 = np.full(n, -1, np.int64)
    c0 = np.full(n, W, np.int64)
    c1 = np.full(n, -1, np.int64)
    for r in range(H):
        for c in range(W):
            k = np.int64(labels[r, c]) - 1  # unsigned labels must not wrap
            if k >= 0:
                r0[k] = min(r0[k], r)
                r1[k] = max(r1[k], r)
                c0[k] = min(c0[k], c)
                c1[k] = max(c1[k], c)
    nb = bin_count + 1
    frac_at_d = np.zeros((n, nb))
    mean_frac = np.zeros((n, nb))
    wedge_sums = np.zeros((n, nb, 8))
    wedge_counts = np.zeros((n, nb, 8))
    fallback = np.zeros(n, np.bool_)
    for k in range(n):
        if r1[k] < 0:
            continue
        a, b = max(r0[k] - 1, 0), min(r1[k] + 2, H)
        c, d = max(c0[k] - 1, 0), min(c1[k] + 2, W)
        m = np.empty((b - a, d - c), np.bool_)
        pix = np.empty((b - a, d - c))
        background = False
        for r in range(a, b):
            for q in range(c, d):
                inside = np.int64(labels[r, q]) == k + 1
                m[r - a, q - c] = inside
                pix[r - a, q - c] = pixels[r, q]
                background |= not inside
        if not background:
            fallback[k] = True
            continue
        d_to_edge = np.sqrt(squared_edt_2d(m).astype(np.float64))
        frac_at_d[k], mean_frac[k], wedge_sums[k], wedge_counts[k] = radial_object(
            m, pix, d_to_edge, scaled, bin_count, maximum_radius
        )
    return frac_at_d, mean_frac, wedge_sums, wedge_counts, fallback
