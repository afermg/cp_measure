"""Fused per-object colocalization kernel (single-threaded, cached).

One pass-set over the per-object value blocks produced by
:func:`cp_measure.primitives._segment_numba.flatten_pairs_grouped` yields every
PR-A colocalization feature — Pearson correlation + slope, Manders M1/M2,
Overlap + K1/K2, and (optionally) the rank-weighted RWC1/RWC2 — for all objects.

The math is the per-object reduction of
:mod:`cp_measure.core.measurecolocalization`, where each reference ``_ind`` call
processes a single object via ``scipy.ndimage`` over an all-ones label
(``lrange=[1]``, hence the ``[0]`` indexing throughout). Here that becomes a
plain loop over each object's contiguous block.

Serial by construction: objects are iterated in a single ``for``; no
``prange``/``nogil``. Parallelism is the job of a future batch layer over images,
never the kernel.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _dense_rank(vals):
    """Per-element 0-based dense rank (ties share a rank), plus the max rank.

    Bit-reproduces the reference's ``lexsort`` + unique-diff + ``cumsum``: the
    rank of an element is the number of DISTINCT smaller values, so it is
    independent of the tie-break order ``argsort`` happens to pick.
    """
    m = vals.shape[0]
    ranks = np.empty(m, np.int64)
    order = np.argsort(vals)
    r = 0
    ranks[order[0]] = 0
    for j in range(1, m):
        if vals[order[j]] != vals[order[j - 1]]:
            r += 1
        ranks[order[j]] = r
    return ranks, r


@njit(cache=True)
def coloc_per_object(g1, g2, offsets, n, thr_frac, compute_rwc):
    """Per-object colocalization features over grouped value blocks.

    ``g1``/``g2`` hold both channels' intensities with object ``k`` in
    ``[offsets[k] : offsets[k + 1]]``. ``thr_frac`` is the threshold as a
    fraction of each object's per-channel max (reference ``thr/100``). The rank
    sort for RWC is gated by ``compute_rwc`` so the three rank-free features skip
    it. Returns nine length-``n`` arrays:
    ``(corr, slope, m1, m2, overlap, k1, k2, rwc1, rwc2)``.

    Threshold-derived features default to ``0.0`` for an object with no pixel
    above the combined threshold (matching the reference's initialised values);
    Pearson is threshold-free and is always computed (``NaN`` when undefined).
    """
    corr = np.zeros(n)
    slope = np.zeros(n)
    m1 = np.zeros(n)
    m2 = np.zeros(n)
    overlap = np.zeros(n)
    k1 = np.zeros(n)
    k2 = np.zeros(n)
    rwc1 = np.zeros(n)
    rwc2 = np.zeros(n)

    for k in range(n):
        lo = offsets[k]
        hi = offsets[k + 1]
        cnt = hi - lo
        if cnt == 0:
            corr[k] = np.nan
            slope[k] = np.nan
            continue

        # Pass 1: means + per-channel maxima.
        s1 = 0.0
        s2 = 0.0
        mx1 = -np.inf
        mx2 = -np.inf
        for i in range(lo, hi):
            v1 = g1[i]
            v2 = g2[i]
            s1 += v1
            s2 += v2
            if v1 > mx1:
                mx1 = v1
            if v2 > mx2:
                mx2 = v2
        mean1 = s1 / cnt
        mean2 = s2 / cnt

        # Pass 2: centred second moments → Pearson r + regression slope.
        c11 = 0.0
        c22 = 0.0
        c12 = 0.0
        for i in range(lo, hi):
            d1 = g1[i] - mean1
            d2 = g2[i] - mean2
            c11 += d1 * d1
            c22 += d2 * d2
            c12 += d1 * d2
        corr[k] = c12 / np.sqrt(c11 * c22)
        slope[k] = c12 / c11

        tff = thr_frac * mx1
        tss = thr_frac * mx2

        # Per-pixel RWC weights need the object's dense ranks up front.
        if compute_rwc:
            r1, r1max = _dense_rank(g1[lo:hi])
            r2, r2max = _dense_rank(g2[lo:hi])
            rmax = r1max if r1max > r2max else r2max
            R = rmax + 1

        # Pass 3: threshold-gated accumulations.
        tot_fi = 0.0
        tot_si = 0.0
        sum_fi_c = 0.0
        sum_si_c = 0.0
        sum_fi2_c = 0.0
        sum_si2_c = 0.0
        sum_fisi_c = 0.0
        sum_fiw_c = 0.0
        sum_siw_c = 0.0
        n_comb = 0
        for i in range(lo, hi):
            v1 = g1[i]
            v2 = g2[i]
            above1 = v1 >= tff
            above2 = v2 >= tss
            if above1:
                tot_fi += v1
            if above2:
                tot_si += v2
            if above1 and above2:
                n_comb += 1
                sum_fi_c += v1
                sum_si_c += v2
                sum_fi2_c += v1 * v1
                sum_si2_c += v2 * v2
                sum_fisi_c += v1 * v2
                if compute_rwc:
                    j = i - lo
                    diff = r1[j] - r2[j]
                    if diff < 0:
                        diff = -diff
                    w = (R - diff) / R
                    sum_fiw_c += v1 * w
                    sum_siw_c += v2 * w

        if n_comb > 0:
            m1[k] = sum_fi_c / tot_fi
            m2[k] = sum_si_c / tot_si
            overlap[k] = sum_fisi_c / np.sqrt(sum_fi2_c * sum_si2_c)
            k1[k] = sum_fisi_c / sum_fi2_c
            k2[k] = sum_fisi_c / sum_si2_c
            if compute_rwc:
                rwc1[k] = sum_fiw_c / tot_fi
                rwc2[k] = sum_siw_c / tot_si

    return corr, slope, m1, m2, overlap, k1, k2, rwc1, rwc2
