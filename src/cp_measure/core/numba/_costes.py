"""Costes automated-threshold colocalization kernel (single-threaded, cached).

Per-object port of ``measurecolocalization.get_correlation_costes_ind`` and its
``bisection_costes`` / ``linear_costes`` search. Each object's two channels are
the contiguous blocks ``g1[offsets[k]:offsets[k+1]]`` / ``g2[...]`` from
:func:`cp_measure.primitives._segment_numba.flatten_pairs_grouped` (the same
grouped layout the other coloc features use — no sort here).

The reference's ``calculate_threshold`` call inside ``..._ind`` is dead (its
outputs are never read), so it is not ported. ``thr`` likewise has no effect on
costes. ``scale`` (``infer_scale``, dtype-keyed: float→1) is passed in host-side.

Exactness: bit-exact vs the numpy reference on **float** pixels (the realistic
input; ``scale==1``). Integer-dtype input diverges by design — the reference
computes ``z = fi + si`` in that dtype and overflows (uint8/uint16), corrupting
the regression; the float64 kernel here does not. See ``tasks/numba_costes_plan.md``.

Serial: a plain object loop, no ``prange``/``nogil``.
"""

import numpy as np
from numba import njit


@njit(cache=True, error_model="numpy")
def _regression_ab(g1, g2, lo, hi):
    """Costes orthogonal-regression line ``si ≈ a·fi + b`` over non-zero pixels.

    ``non_zero = (fi > 0) | (si > 0)``; ``a`` from the covariance recovered via
    ``cov = 0.5(var(fi+si) - var(fi) - var(si))`` (all ``ddof=1``), matching the
    reference exactly.
    """
    cnt = 0
    s1 = 0.0
    s2 = 0.0
    for i in range(lo, hi):
        if g1[i] > 0 or g2[i] > 0:
            cnt += 1
            s1 += g1[i]
            s2 += g2[i]
    m1 = s1 / cnt
    m2 = s2 / cnt
    vx = 0.0
    vy = 0.0
    vz = 0.0
    for i in range(lo, hi):
        if g1[i] > 0 or g2[i] > 0:
            d1 = g1[i] - m1
            d2 = g2[i] - m2
            dz = (g1[i] + g2[i]) - (m1 + m2)
            vx += d1 * d1
            vy += d2 * d2
            vz += dz * dz
    denom_n = cnt - 1
    xvar = vx / denom_n
    yvar = vy / denom_n
    zvar = vz / denom_n
    covar = 0.5 * (zvar - (xvar + yvar))
    yx = yvar - xvar
    a = (yx + np.sqrt(yx * yx + 4.0 * covar * covar)) / (2.0 * covar)
    b = m2 - a * m1
    return a, b


@njit(cache=True, error_model="numpy")
def _count_combt(g1, g2, lo, hi, thr_fi, thr_si):
    """``count_nonzero((fi < thr_fi) | (si < thr_si))`` over the object's block."""
    cnt = 0
    for i in range(lo, hi):
        if g1[i] < thr_fi or g2[i] < thr_si:
            cnt += 1
    return cnt


@njit(cache=True, error_model="numpy")
def _pearson_combt(g1, g2, lo, hi, thr_fi, thr_si):
    """Pearson r over the ``(fi < thr_fi) | (si < thr_si)`` subset.

    Mirrors ``scipy.stats.pearsonr``'s operation order — centre, normalise each
    vector by its L2 norm, accumulate the normalised products, clamp to [-1, 1] —
    to stay as close as possible to the value the reference branches on. Pass
    ``thr_fi = thr_si = inf`` for the full-block correlation.
    """
    cnt = 0
    s1 = 0.0
    s2 = 0.0
    for i in range(lo, hi):
        if g1[i] < thr_fi or g2[i] < thr_si:
            cnt += 1
            s1 += g1[i]
            s2 += g2[i]
    if cnt == 0:
        return np.nan
    m1 = s1 / cnt
    m2 = s2 / cnt
    nx = 0.0
    ny = 0.0
    for i in range(lo, hi):
        if g1[i] < thr_fi or g2[i] < thr_si:
            d1 = g1[i] - m1
            d2 = g2[i] - m2
            nx += d1 * d1
            ny += d2 * d2
    normx = np.sqrt(nx)
    normy = np.sqrt(ny)
    r = 0.0
    for i in range(lo, hi):
        if g1[i] < thr_fi or g2[i] < thr_si:
            r += ((g1[i] - m1) / normx) * ((g2[i] - m2) / normy)
    if r > 1.0:
        r = 1.0
    elif r < -1.0:
        r = -1.0
    return r


@njit(cache=True, error_model="numpy")
def _bisection(g1, g2, lo, hi, a, b, scale):
    """``bisection_costes`` (M_FASTER): narrowing-window search for the threshold."""
    left = 1.0
    right = float(scale)
    mid = np.floor((right - left) / 1.2) + left
    lastmid = 0.0
    valid = 1.0
    while lastmid != mid:
        thr_fi = mid / scale
        thr_si = a * thr_fi + b
        cnt = _count_combt(g1, g2, lo, hi, thr_fi, thr_si)
        if cnt <= 2:
            left = mid - 1
        else:
            r = _pearson_combt(g1, g2, lo, hi, thr_fi, thr_si)
            if r < 0:
                left = mid - 1
            elif r >= 0:
                right = mid + 1
                valid = mid
            # NaN (constant subset): neither bound moves; loop exits next step.
        lastmid = mid
        if right - left > 6:
            mid = np.floor((right - left) / 1.2) + left
        else:
            mid = np.floor((right - left) / 2.0) + left
    thr_fi_c = (valid - 1) / scale
    thr_si_c = a * thr_fi_c + b
    return thr_fi_c, thr_si_c


@njit(cache=True, error_model="numpy")
def _linear(g1, g2, lo, hi, a, b, scale, accurate):
    """``linear_costes`` (M_FAST / M_ACCURATE): step the threshold down to R≈0."""
    i_step = 1.0 / scale
    fi_max = -np.inf
    si_max = -np.inf
    for i in range(lo, hi):
        if g1[i] > fi_max:
            fi_max = g1[i]
        if g2[i] > si_max:
            si_max = g2[i]
    img_max = fi_max if fi_max > si_max else si_max
    cur = i_step * (np.floor(img_max / i_step) + 1)
    thr_fi_c = cur
    thr_si_c = a * cur + b
    while cur > fi_max and (a * cur + b) > si_max:
        cur -= i_step
    r = 0.0
    num_true = -1  # sentinel (reference's None: forces recompute on first pass)
    while cur > i_step:
        thr_fi_c = cur
        thr_si_c = a * cur + b
        cnt = _count_combt(g1, g2, lo, hi, thr_fi_c, thr_si_c)
        if cnt < 2:  # reference: scipy.stats.pearsonr raises -> break
            break
        if cnt != num_true:
            r = _pearson_combt(g1, g2, lo, hi, thr_fi_c, thr_si_c)
            num_true = cnt
        if r <= 0:
            break
        elif accurate or cur < i_step * 10:
            cur -= i_step
        elif r > 0.45:
            cur -= i_step * 10
        elif r > 0.35:
            cur -= i_step * 5
        elif r > 0.25:
            cur -= i_step * 2
        else:
            cur -= i_step
    return thr_fi_c, thr_si_c


@njit(cache=True, error_model="numpy")
def costes_per_object(g1, g2, offsets, n, scale, mode):
    """Costes C1/C2 per object over grouped value blocks.

    ``mode``: 0 = bisection (M_FASTER), 1 = linear M_FAST, 2 = linear M_ACCURATE.
    Returns ``(C1[n], C2[n])``. An object with no pixel above the Costes threshold
    yields ``0.0`` (matching the reference's fringe-case default).
    """
    C1 = np.zeros(n)
    C2 = np.zeros(n)
    for k in range(n):
        lo = offsets[k]
        hi = offsets[k + 1]
        if hi - lo == 0:
            continue
        a, b = _regression_ab(g1, g2, lo, hi)
        if mode == 0:
            thr_fi_c, thr_si_c = _bisection(g1, g2, lo, hi, a, b, scale)
        else:
            thr_fi_c, thr_si_c = _linear(g1, g2, lo, hi, a, b, scale, mode == 2)

        any_fi = False
        any_si = False
        sum_ge_fi = 0.0
        sum_ge_si = 0.0
        sum_fi_c = 0.0
        sum_si_c = 0.0
        n_comb = 0
        for i in range(lo, hi):
            v1 = g1[i]
            v2 = g2[i]
            if v1 > thr_fi_c:
                any_fi = True
            if v2 > thr_si_c:
                any_si = True
            if v1 >= thr_fi_c:
                sum_ge_fi += v1
            if v2 >= thr_si_c:
                sum_ge_si += v2
            if v1 > thr_fi_c and v2 > thr_si_c:
                n_comb += 1
                sum_fi_c += v1
                sum_si_c += v2
        if n_comb > 0:
            tot_fi = sum_ge_fi if any_fi else 0.0
            tot_si = sum_ge_si if any_si else 0.0
            C1[k] = sum_fi_c / tot_fi
            C2[k] = sum_si_c / tot_si
    return C1, C2
