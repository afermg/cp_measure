"""Costes threshold searches (single-threaded, cached).

The host keeps NumPy's regression reductions: tiny changes in the fitted line
can change threshold inclusion (#118). Only the repeated background searches
are compiled. Undefined searches return NaN to the wrapper, which reports the
common "no valid threshold" case directly and replays the reference for the
others' precise RuntimeWarning. Long linear searches switch to sorted prefix
sums but make the exact kernel's decisions. No fastmath or grid rounding is used.
"""

import numpy as np
from numba import njit

from cp_measure.primitives._segment_numba import numpy_sum, numpy_var

# Search status, returned with the thresholds. SEARCHED: use the thresholds, or replay the
# reference when they are NaN (its warnings need SciPy's exact diagnostics). The other two
# are undefined results whose reference warning the wrapper can emit without a replay.
SEARCHED = 0
NO_VALID_THRESHOLD = 1  # bisection: no candidate had a finite non-negative correlation
UNDEFINED_IN_SEARCH = 2  # linear: a background subset had < 3 pixels or was constant
UNDEFINED_BACKGROUND = (
    3  # the chosen threshold's background subset is < 3 pixels or constant
)
# Per-object statuses of costes_per_object, beyond the search's: undefined results the
# wrapper reports with the reference's warning, an out-of-range pixel it raises for, and
# REPLAY for cases whose diagnostics need the reference itself.
EMPTY_OBJECT = 4
FEW_OR_NONFINITE = 5
FEW_NONZERO = 6
CONSTANT_CHANNEL = 7
ZERO_DENOMINATOR = 8
OUT_OF_RANGE = 9
REPLAY = 10


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
    """``bisection_costes`` (M_FASTER): narrowing-window search for the threshold.

    Returns ``(thr_fi, thr_si, status)``; see the status codes at the top of the module.
    """
    left = 1.0
    right = float(scale)
    mid = ((right - left) // (6 / 5)) + left
    lastmid = 0.0
    valid = np.nan
    iterations = 0
    totals = _moment_totals(g1, g2, lo, hi)
    while lastmid != mid:
        iterations += 1
        if iterations > 256:
            return np.nan, np.nan, SEARCHED
        thr_fi = mid / scale
        thr_si = a * thr_fi + b
        cnt, r = _direct_count_pearson(g1, g2, lo, hi, totals, thr_fi, thr_si)
        if np.isinf(r):
            return np.nan, np.nan, SEARCHED
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
    if np.isnan(valid):
        return np.nan, np.nan, NO_VALID_THRESHOLD
    thr_fi_c = (valid - 1) / scale
    thr_si_c = a * thr_fi_c + b
    return _finish(g1, g2, lo, hi, thr_fi_c, thr_si_c)


@njit(cache=True, error_model="numpy")
def _finish(g1, g2, lo, hi, thr_fi, thr_si):
    """The reference's final check: the chosen background subset must have a defined
    correlation. Too few or constant pixels (NaN) is decided exactly, so the wrapper
    warns without a replay; only SciPy's near-constant diagnostics (inf) need one."""
    r = _pearson_combt(g1, g2, lo, hi, thr_fi, thr_si)
    if np.isnan(r):
        return np.nan, np.nan, UNDEFINED_BACKGROUND
    if np.isinf(r):
        return np.nan, np.nan, SEARCHED
    return thr_fi, thr_si, SEARCHED


# The linear search tests one grid step at a time: up to ``scale`` candidates (65535 for
# 16-bit data), each an O(n) pass. After this many exact candidates the search switches to
# sorted prefix sums, which answer each further candidate in O(log n). Short searches (most
# objects stop within a few steps) never pay for the sort.
_SWEEP_AFTER = 32
_CUTOFFS = (
    0.0,
    0.25,
    0.35,
    0.45,
)  # the search's decision points (see _count_pearson_combt)


@njit(cache=True)
def _sorted_prefix(keys, g1, g2, c1, c2, use):
    """Sorted ``keys`` of the pixels with ``use``, and prefix sums of the centred
    ``(u, v, u*u, v*v, u*v)``, ``u = g1 - c1`` and ``v = g2 - c2``, in key order."""
    order = np.argsort(keys)
    m = 0
    for i in order:
        if use[i]:
            m += 1
    ks = np.empty(m)
    P = np.zeros((5, m + 1))
    j = 0
    for i in order:
        if not use[i]:
            continue
        u = g1[i] - c1
        v = g2[i] - c2
        ks[j] = keys[i]
        P[0, j + 1] = P[0, j] + u
        P[1, j + 1] = P[1, j] + v
        P[2, j + 1] = P[2, j] + u * u
        P[3, j + 1] = P[3, j] + v * v
        P[4, j + 1] = P[4, j] + u * v
        j += 1
    return ks, P


@njit(cache=True)
def _moment_totals(g1, g2, lo, hi):
    """Sums of ``u, v, uu, vv, uv, |u|, |v|`` over the object, then the centres
    ``c1, c2``, with ``u = g1 - c1`` and ``v = g2 - c2`` centred on the object means."""
    totals = np.zeros(9)
    c1 = g1[lo:hi].mean()
    c2 = g2[lo:hi].mean()
    totals[7] = c1
    totals[8] = c2
    for i in range(lo, hi):
        u = g1[i] - c1
        v = g2[i] - c2
        totals[0] += u
        totals[1] += v
        totals[2] += u * u
        totals[3] += v * v
        totals[4] += u * v
        totals[5] += abs(u)
        totals[6] += abs(v)
    return totals


@njit(cache=True, error_model="numpy")
def _sweep_prep(g1, g2, a, b):
    """Tables for :func:`_sweep_count_pearson` on one object's pixels.

    A pixel is EXCLUDED from the background at threshold ``t`` when ``g1 >= t`` and
    ``g2 >= a*t + b``. With ``q = (g2 - b) / a`` that is ``t <= min(g1, q)`` for ``a > 0``
    (one sorted key), and ``q <= t <= g1`` for ``a < 0`` (the pixels with ``q <= t`` minus
    those with ``g1 < t``, over pixels where that interval is non-empty).
    """
    n = g1.size
    totals = _moment_totals(g1, g2, 0, n)
    c1, c2 = totals[7], totals[8]
    q = (g2 - b) / a
    if a > 0:
        k1, P1 = _sorted_prefix(np.minimum(g1, q), g1, g2, c1, c2, np.ones(n, np.bool_))
        k2, P2 = k1[:0], P1[:, :1]
    else:
        nonempty = q <= g1
        k1, P1 = _sorted_prefix(q, g1, g2, c1, c2, nonempty)
        k2, P2 = _sorted_prefix(g1, g1, g2, c1, c2, nonempty)
    return totals, k1, P1, k2, P2, np.sort(q)


@njit(cache=True, error_model="numpy")
def _sweep_count_pearson(g1, g2, lo, hi, a, b, tab, thr_fi, thr_si):
    """Same ``(count, r)`` as :func:`_count_pearson_combt`, from the prefix tables.

    Exactness: ``g1 < t`` is the same comparison either way, but ``q`` vs ``t`` can
    disagree with ``g2 < a*t + b`` by rounding when they are within ``tol``; any such pixel
    sends the candidate to the exact pass. Likewise an ``r`` within its forward-error bound
    of a decision cutoff, or a near-constant subset, uses the exact pass, so every search
    decision equals the exact kernel's.
    """
    totals, k1, P1, k2, P2, qs = tab
    n = hi - lo
    eps = np.finfo(np.float64).eps
    t = thr_fi
    tol = 64 * eps * (1.0 + abs(t) + (1.0 + abs(b)) / abs(a))
    j = np.searchsorted(qs, t - tol)
    if j < qs.size and qs[j] <= t + tol:
        return _count_pearson_combt(g1, g2, lo, hi, thr_fi, thr_si)
    if a > 0:
        cut = np.searchsorted(k1, t)  # keys < t stay in the background
        excluded = P1[:, -1] - P1[:, cut]
        n_excluded = k1.size - cut
    else:
        hq = np.searchsorted(k1, t, side="right")  # q <= t
        hf = np.searchsorted(k2, t)  # g1 < t
        excluded = P1[:, hq] - P2[:, hf]
        n_excluded = hq - hf
    m = n - n_excluded
    return _r_or_exact(
        g1,
        g2,
        lo,
        hi,
        thr_fi,
        thr_si,
        totals,
        m,
        totals[0] - excluded[0],
        totals[1] - excluded[1],
        totals[2] - excluded[2],
        totals[3] - excluded[3],
        totals[4] - excluded[4],
    )


@njit(cache=True, error_model="numpy")
def _r_or_exact(g1, g2, lo, hi, thr_fi, thr_si, totals, m, su, sv, suu, svv, suv):
    """``(count, r)`` of the background subset from its centred moments, or from the
    exact kernel when the moments cannot decide the search exactly.

    ``m`` and the sums ``su .. suv`` are over the subset; ``totals`` from
    :func:`_moment_totals` bound their rounding error.
    """
    n = hi - lo
    eps = np.finfo(np.float64).eps
    if m < 3:
        return m, np.nan
    varu = suu - su * su / m
    varv = svv - sv * sv / m
    cov = suv - su * sv / m
    # Forward-error bound of the prefix-sum moments (each a sum of at most 2n terms,
    # then a subtraction), propagated to r; conservative by a factor of two.
    e = 8 * n * eps
    err_u = e * (totals[2] + 2 * abs(su) * totals[5] / m)
    err_v = e * (totals[3] + 2 * abs(sv) * totals[6] / m)
    # Near-constant subsets (ill-conditioned here, or flagged by the exact kernel's
    # norm < eps**0.75 * |mean| test, checked with a 4x margin): the exact pass decides.
    flag = 4 * eps**1.5
    mean1 = totals[7] + su / m
    mean2 = totals[8] + sv / m
    if (
        varu <= 1e3 * err_u
        or varv <= 1e3 * err_v
        or varu <= flag * mean1 * mean1
        or varv <= flag * mean2 * mean2
    ):
        return _count_pearson_combt(g1, g2, lo, hi, thr_fi, thr_si)
    err_c = e * (
        np.sqrt(totals[2] * totals[3]) + (abs(su) * totals[6] + abs(sv) * totals[5]) / m
    )
    r = cov / np.sqrt(varu * varv)
    bound = 2 * (err_c / np.sqrt(varu * varv) + abs(r) * (err_u / varu + err_v / varv))
    # The exact kernel replays the reference when ITS r is within its own error
    # bound of a cutoff (gamma * conditioning, see _count_pearson_combt). Its
    # conditioning grows with |mean| / norm, so near-constant objects need that
    # margin too, or this path would decide a step the exact kernel replays.
    ops = 64 * m * eps
    conditioning = 1 + np.sqrt(m) * (
        abs(mean1) / np.sqrt(varu) + abs(mean2) / np.sqrt(varv)
    )
    bound += 2 * ops / (1 - ops) * conditioning
    for cutoff in _CUTOFFS:
        if abs(r - cutoff) <= bound:
            return _count_pearson_combt(g1, g2, lo, hi, thr_fi, thr_si)
    return m, min(1.0, max(-1.0, r))


@njit(cache=True, error_model="numpy")
def _direct_count_pearson(g1, g2, lo, hi, totals, thr_fi, thr_si):
    """Same ``(count, r)`` decisions as :func:`_count_pearson_combt` in one pass over
    the object (the exact kernel takes four), via :func:`_r_or_exact`."""
    c1, c2 = totals[7], totals[8]
    m = 0
    su = sv = suu = svv = suv = 0.0
    for i in range(lo, hi):
        if g1[i] < thr_fi or g2[i] < thr_si:
            u = g1[i] - c1
            v = g2[i] - c2
            m += 1
            su += u
            sv += v
            suu += u * u
            svv += v * v
            suv += u * v
    return _r_or_exact(g1, g2, lo, hi, thr_fi, thr_si, totals, m, su, sv, suu, svv, suv)


@njit(cache=True)
def _next_candidate(cur, r, accurate, i_step):
    """The reference's step rule: one grid step, or 2/5/10 steps in Fast mode while r is
    well above zero (never below ten grid steps)."""
    if accurate or cur < i_step * 10:
        return cur - i_step
    if r > 0.45:
        return cur - i_step * 10
    if r > 0.35:
        return cur - i_step * 5
    if r > 0.25:
        return cur - i_step * 2
    return cur - i_step


@njit(cache=True, error_model="numpy")
def _linear(g1, g2, lo, hi, a, b, scale, accurate):
    """``linear_costes`` (M_FAST / M_ACCURATE): step the threshold down to R≈0.

    Returns ``(thr_fi, thr_si, status)``; see the status codes at the top of the module.
    """
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
        return np.nan, np.nan, SEARCHED
    thr_fi_c = cur
    thr_si_c = a * cur + b
    while cur > fi_max and (a * cur + b) > si_max:
        cur -= i_step
    r = 0.0
    num_true = -1  # sentinel (reference's None: forces recompute on first pass)
    evaluated = 0
    found = False
    while cur > i_step and evaluated < _SWEEP_AFTER:
        thr_fi_c = cur
        thr_si_c = a * cur + b
        cnt = _count_combt(g1, g2, lo, hi, thr_fi_c, thr_si_c)
        evaluated += 1
        if cnt != num_true:
            r = _pearson_combt(g1, g2, lo, hi, thr_fi_c, thr_si_c)
            num_true = cnt
        if not np.isfinite(r):
            return np.nan, np.nan, UNDEFINED_IN_SEARCH if np.isnan(r) else SEARCHED
        if r <= 0:
            found = True
            break
        cur = _next_candidate(cur, r, accurate, i_step)
    if not found and cur > i_step:
        # Long walk: continue the identical search on sorted prefix sums.
        tab = _sweep_prep(g1[lo:hi], g2[lo:hi], a, b)
        while cur > i_step:
            thr_fi_c = cur
            thr_si_c = a * cur + b
            cnt, r_new = _sweep_count_pearson(g1, g2, lo, hi, a, b, tab, cur, thr_si_c)
            # the reference recomputes r only when the count changes
            if cnt != num_true:
                r = r_new
                num_true = cnt
            if not np.isfinite(r):
                return np.nan, np.nan, UNDEFINED_IN_SEARCH if np.isnan(r) else SEARCHED
            if r <= 0:
                break
            cur = _next_candidate(cur, r, accurate, i_step)
    return _finish(g1, g2, lo, hi, thr_fi_c, thr_si_c)


@njit(cache=True)
def _canonical(v, scale):
    """The reference's ``_canonical_costes_pixels`` for one value: snap a value encoded
    as ``level * (1 / scale)`` to ``level / scale``; leave every other value alone."""
    if v >= 1.0 / scale and v <= 1:
        level = np.rint(v * scale)
        if v == level * (1.0 / scale):
            return level / scale
    return v


@njit(cache=True)
def _gather_sum(values, lo, hi, keep, buf, chunk):
    """``values[lo:hi][keep].sum()`` with NumPy's summation order."""
    m = 0
    for i in range(lo, hi):
        if keep[i - lo]:
            buf[m] = values[i]
            m += 1
    return numpy_sum(buf[:m], chunk)


@njit(cache=True, nogil=True, error_model="numpy")
def costes_per_object(g1, g2, offsets, scale, mode, chunk):
    """Costes coefficients of every object in one pass, as the reference computes them.

    ``g1``/``g2`` are the grouped pixels (object ``k`` owns ``offsets[k]:offsets[k+1]``),
    already divided by the acquisition scale. ``mode`` is 0 (Faster, bisection),
    1 (Fast) or 2 (Accurate); ``chunk`` is ``NUMPY_SUM_CHUNK``. Canonicalization,
    validity checks, the regression line (NumPy's pairwise sums, so the same pixels
    count), the search and the coefficients follow ``reference._costes_pair``.
    Returns ``(c1, c2, status)``; a non-zero status leaves NaN for the wrapper to
    report (see the status codes at the top of the module).
    """
    n = offsets.size - 1
    c1 = np.full(n, np.nan)
    c2 = np.full(n, np.nan)
    status = np.zeros(n, np.int64)
    f = np.empty(g1.size)
    s = np.empty(g2.size)
    for i in range(g1.size):
        f[i] = _canonical(g1[i], scale)
        s[i] = _canonical(g2[i], scale)
    longest = 0
    for k in range(n):
        longest = max(longest, offsets[k + 1] - offsets[k])
    bx = np.empty(longest)
    by = np.empty(longest)
    bz = np.empty(longest)
    keep = np.empty(longest, np.bool_)
    for k in range(n):
        lo, hi = offsets[k], offsets[k + 1]
        if hi == lo:
            status[k] = EMPTY_OBJECT
            continue
        # _valid_costes_pixels
        finite = True
        for i in range(lo, hi):
            if not (np.isfinite(f[i]) and np.isfinite(s[i])):
                finite = False
                break
        if hi - lo < 3 or not finite:
            status[k] = FEW_OR_NONFINITE
            continue
        outside = False
        for i in range(lo, hi):
            if f[i] < 0 or s[i] < 0 or f[i] > 1 or s[i] > 1:
                outside = True
                break
        if outside:
            status[k] = OUT_OF_RANGE
            continue
        m = 0
        for i in range(lo, hi):
            if f[i] > 0 or s[i] > 0:
                bx[m] = f[i]
                by[m] = s[i]
                bz[m] = f[i] + s[i]
                m += 1
        if m < 3:
            status[k] = FEW_NONZERO
            continue
        if bx[:m].min() == bx[:m].max() or by[:m].min() == by[:m].max():
            status[k] = CONSTANT_CHANNEL
            continue
        # The reference's regression (bisection_costes / linear_costes), same sums.
        xvar = numpy_var(bx[:m], chunk)
        yvar = numpy_var(by[:m], chunk)
        xmean = numpy_sum(bx[:m], chunk) / m
        ymean = numpy_sum(by[:m], chunk) / m
        zvar = numpy_var(bz[:m], chunk)
        covar = 0.5 * (zvar - (xvar + yvar))
        num = (yvar - xvar) + np.sqrt(
            (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar)
        )
        if abs(covar) <= 8 * np.finfo(np.float64).eps * max(xvar, yvar):
            status[k] = REPLAY  # rare; the reference also reports NumPy's own warnings
            continue
        a = num / (2 * covar)
        b = ymean - a * xmean
        if not (np.isfinite(a) and np.isfinite(b)):
            status[k] = REPLAY
            continue
        if mode == 0:
            thr1, thr2, st = _bisection(f, s, lo, hi, a, b, scale)
        else:
            thr1, thr2, st = _linear(f, s, lo, hi, a, b, scale, mode == 2)
        if st != SEARCHED:
            status[k] = st
            continue
        if not (np.isfinite(thr1) and np.isfinite(thr2)):
            status[k] = REPLAY
            continue
        for i in range(lo, hi):
            keep[i - lo] = f[i] >= thr1
        tot1 = _gather_sum(f, lo, hi, keep, bx, chunk)
        for i in range(lo, hi):
            keep[i - lo] = s[i] >= thr2
        tot2 = _gather_sum(s, lo, hi, keep, bx, chunk)
        if tot1 <= 0 or tot2 <= 0:
            status[k] = ZERO_DENOMINATOR
            continue
        for i in range(lo, hi):
            keep[i - lo] = f[i] > thr1 and s[i] > thr2
        c1[k] = _gather_sum(f, lo, hi, keep, bx, chunk) / tot1
        c2[k] = _gather_sum(s, lo, hi, keep, bx, chunk) / tot2
    return c1, c2, status
