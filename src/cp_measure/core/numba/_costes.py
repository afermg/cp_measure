"""Costes threshold searches (single-threaded, cached).

The host keeps NumPy's regression reductions: tiny changes in the fitted line
can change threshold inclusion (#118). Only the repeated background searches
are compiled. Undefined searches return NaN to the wrapper, which replays the
reference for its precise RuntimeWarning. No fastmath or grid rounding is used.
"""

import numpy as np
from numba import njit


def _regression_ab(g1, g2, lo, hi):
    """Use the reference's exact NumPy reduction order, not serial JIT sums."""
    fi, si = g1[lo:hi], g2[lo:hi]
    non_zero = (fi > 0) | (si > 0)
    fz, sz = fi[non_zero], si[non_zero]
    xvar = np.var(fz, axis=0, ddof=1)
    yvar = np.var(sz, axis=0, ddof=1)
    xmean = np.mean(fz, axis=0)
    ymean = np.mean(sz, axis=0)
    zvar = np.var(fz + sz, axis=0, ddof=1)
    covar = 0.5 * (zvar - (xvar + yvar))
    if abs(covar) <= 8 * np.finfo(float).eps * max(xvar, yvar):
        return np.nan, np.nan
    num = (yvar - xvar) + np.sqrt((yvar - xvar) * (yvar - xvar) + 4 * (covar * covar))
    a = num / (2 * covar)
    return a, ymean - a * xmean


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
    """Background Pearson r; fewer than three or constant pairs are undefined."""
    return _count_pearson_combt(g1, g2, lo, hi, thr_fi, thr_si)[1]


@njit(cache=True, error_model="numpy")
def _count_pearson_combt(g1, g2, lo, hi, thr_fi, thr_si):
    """Fuse subset count and Pearson passes; mirror SciPy's scaled norms."""
    cnt = 0
    s1 = 0.0
    s2 = 0.0
    min1 = min2 = np.inf
    max1 = max2 = -np.inf
    for i in range(lo, hi):
        if g1[i] < thr_fi or g2[i] < thr_si:
            cnt += 1
            s1 += g1[i]
            s2 += g2[i]
            min1 = min(min1, g1[i])
            max1 = max(max1, g1[i])
            min2 = min(min2, g2[i])
            max2 = max(max2, g2[i])
    if cnt < 3 or min1 == max1 or min2 == max2:
        return cnt, np.nan
    m1 = s1 / cnt
    m2 = s2 / cnt
    xmax = ymax = 0.0
    for i in range(lo, hi):
        if g1[i] < thr_fi or g2[i] < thr_si:
            xmax = max(xmax, abs(g1[i] - m1))
            ymax = max(ymax, abs(g2[i] - m2))
    nx = ny = 0.0
    for i in range(lo, hi):
        if g1[i] < thr_fi or g2[i] < thr_si:
            nx += ((g1[i] - m1) / xmax) ** 2
            ny += ((g2[i] - m2) / ymax) ** 2
    normx = xmax * np.sqrt(nx)
    normy = ymax * np.sqrt(ny)
    # SciPy warns on near-constant subsets. Replay those searches on the host
    # rather than silently dropping the warning or branching on unstable r.
    if normx < np.finfo(np.float64).eps ** 0.75 * abs(m1) or normy < np.finfo(
        np.float64
    ).eps ** 0.75 * abs(m2):
        return cnt, np.inf
    r = 0.0
    for i in range(lo, hi):
        if g1[i] < thr_fi or g2[i] < thr_si:
            r += ((g1[i] - m1) / normx) * ((g2[i] - m2) / normy)
    # Search decisions are discontinuous at zero and the fast-step cutoffs.
    # Serial reductions need not share NumPy/BLAS's last bits. Use a conservative
    # forward-error bound (gamma_k = k*eps/(1-k*eps), at most 32*n scalar
    # operations per evaluation, allowing for BOTH reduction orders and
    # centering's condition number) only to request
    # reference replay. This does NOT round r, move a cutoff, or relax parity.
    operations_eps = 64 * cnt * np.finfo(np.float64).eps
    gamma = operations_eps / (1 - operations_eps)
    conditioning = 1 + np.sqrt(cnt) * (abs(m1) / normx + abs(m2) / normy)
    error_bound = gamma * conditioning
    for cutoff in (0.0, 0.25, 0.35, 0.45):
        if abs(r - cutoff) <= error_bound:
            return cnt, np.inf
    return cnt, min(1.0, max(-1.0, r))


@njit(cache=True, error_model="numpy")
def _bisection(g1, g2, lo, hi, a, b, scale):
    """``bisection_costes`` (M_FASTER): narrowing-window search for the threshold."""
    left = 1.0
    right = float(scale)
    mid = ((right - left) // (6 / 5)) + left
    lastmid = 0.0
    valid = np.nan
    iterations = 0
    while lastmid != mid:
        iterations += 1
        if iterations > 256:
            return np.nan, np.nan
        thr_fi = mid / scale
        thr_si = a * thr_fi + b
        cnt, r = _count_pearson_combt(g1, g2, lo, hi, thr_fi, thr_si)
        if np.isinf(r):
            return np.nan, np.nan
        if not np.isfinite(r) or r < 0:
            left = mid - 1
        else:
            right = mid + 1
            valid = mid
        lastmid = mid
        if right - left > 6:
            mid = ((right - left) // (6 / 5)) + left
        else:
            mid = ((right - left) // 2) + left
    thr_fi_c = (valid - 1) / scale
    thr_si_c = a * thr_fi_c + b
    if not np.isfinite(_pearson_combt(g1, g2, lo, hi, thr_fi_c, thr_si_c)):
        return np.nan, np.nan
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
    cur = i_step * ((img_max // i_step) + 1)
    # The reference calls SciPy on the full object before its first candidate;
    # preserve possible near-constant diagnostics even if later subsets differ.
    if np.isinf(_pearson_combt(g1, g2, lo, hi, np.inf, np.inf)):
        return np.nan, np.nan
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
        if cnt != num_true:
            r = _pearson_combt(g1, g2, lo, hi, thr_fi_c, thr_si_c)
            num_true = cnt
        if not np.isfinite(r):
            return np.nan, np.nan
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
    if not np.isfinite(_pearson_combt(g1, g2, lo, hi, thr_fi_c, thr_si_c)):
        return np.nan, np.nan
    return thr_fi_c, thr_si_c
