"""Costes for all objects in one compiled pass (cached; the entry kernels
release the GIL).

The regression line and coefficient sums use NumPy's pairwise summation
(``numpy_sum``/``numpy_var``): tiny changes in the fitted line can change
threshold inclusion (#118). Undefined results return a status per object; the
wrapper emits the reference's warning, or replays the reference where its
diagnostics need SciPy. Bisection steps and long linear searches answer from
moments or sorted prefix sums but make the exact kernel's decisions. No
fastmath or grid rounding is used. Positive slopes use exact float-entry keys
and bounded grid buckets from #167; the same rounding guard and SciPy-resume
protocol serve every path. Long linear searches keep their delayed sort,
including the guarded negative-slope sweep.
"""

import numpy as np
from numba import njit

from cp_measure.primitives._segment_numba import group_pairs, numpy_sum, numpy_var

# The values of r each check branches on. A check must defer to the reference when
# r is within rounding of one of ITS decision points, and only those: bisection
# branches on r < 0; the Fast linear search also steps by r > 0.25/0.35/0.45; the
# final check and the linear search's initial check only need r to be defined.
_FAST_CUTOFFS = np.array([0.0, 0.25, 0.35, 0.45])
_SIGN_CUTOFF = np.array([0.0])
_NO_CUTOFFS = np.zeros(0)

# Search status, returned with the thresholds. SEARCHED: use the thresholds, or replay the
# reference when they are NaN (its warnings need SciPy's exact diagnostics). As in
# CellProfiler, small or constant search backgrounds end or steer the search; they never
# make the result undefined.
SEARCHED = 0
# Per-object statuses of costes_faster / costes_linear, beyond the search's: undefined results the
# wrapper reports with the reference's warning, an out-of-range pixel it raises for, and
# REPLAY for cases whose diagnostics need the reference itself.
EMPTY_OBJECT = 4
FEW_OR_NONFINITE = 5
FEW_NONZERO = 6
CONSTANT_CHANNEL = 7
ZERO_DENOMINATOR = 8
OUT_OF_RANGE = 9
REPLAY = 10
RESUME = 11  # a bisection step was ambiguous; the wrapper asks the reference for it

# _bisection_steps outcomes, and its state layout (one float64 row per object).
STEPS_DONE, STEPS_AMBIGUOUS, STEPS_NOT_CONVERGED = 0, 1, 2
_LEFT, _RIGHT, _MID, _LASTMID, _VALID, _ITERATIONS, _A, _B = range(8)


@njit(cache=True, error_model="numpy")
def _regression_ab(g1, g2, lo, hi, chunk):
    """The reference's regression line over the object's nonzero pixels.

    Same expressions and NumPy's pairwise sums (``numpy_sum``/``numpy_var`` with
    ``chunk`` from ``numpy_sum_chunk()``), so ``a`` and ``b`` are bit-identical and
    the same pixels count against the line (#118). NaN when ill-conditioned.
    """
    m = 0
    for i in range(lo, hi):
        if g1[i] > 0 or g2[i] > 0:
            m += 1
    fz = np.empty(m)
    sz = np.empty(m)
    zz = np.empty(m)
    m = 0
    for i in range(lo, hi):
        if g1[i] > 0 or g2[i] > 0:
            fz[m] = g1[i]
            sz[m] = g2[i]
            zz[m] = g1[i] + g2[i]
            m += 1
    xvar = numpy_var(fz, chunk)
    yvar = numpy_var(sz, chunk)
    xmean = numpy_sum(fz, chunk) / m
    ymean = numpy_sum(sz, chunk) / m
    zvar = numpy_var(zz, chunk)
    covar = 0.5 * (zvar - (xvar + yvar))
    if abs(covar) <= 8 * np.finfo(np.float64).eps * max(xvar, yvar):
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
def _pearson_combt(g1, g2, lo, hi, thr_fi, thr_si, cutoffs, defer):
    """Background Pearson r; fewer than two or constant pairs are undefined."""
    return _count_pearson_combt(g1, g2, lo, hi, thr_fi, thr_si, cutoffs, defer)[1]


@njit(cache=True, error_model="numpy")
def _count_pearson_combt(g1, g2, lo, hi, thr_fi, thr_si, cutoffs, defer):
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
    if cnt < 2 or min1 == max1 or min2 == max2:
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
    if defer and (
        normx < np.finfo(np.float64).eps ** 0.75 * abs(m1)
        or normy < np.finfo(np.float64).eps ** 0.75 * abs(m2)
    ):
        return cnt, np.inf
    if cnt == 2:
        # SciPy's exact +/-1 for two pairs (linear search permits them), after its
        # near-constant check above so scipy_ties keeps SciPy's warning.
        previous = -1
        for i in range(lo, hi):
            if g1[i] < thr_fi or g2[i] < thr_si:
                if previous >= 0:
                    return cnt, np.sign(g1[i] - g1[previous]) * np.sign(
                        g2[i] - g2[previous]
                    )
                previous = i
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
    for cutoff in cutoffs:
        if defer and abs(r - cutoff) <= error_bound:
            return cnt, np.inf
    return cnt, min(1.0, max(-1.0, r))


@njit(cache=True)
def _bisection_state(a, b, scale):
    """The reference's initial bisection window. ``valid`` starts at 1: CellProfiler's
    fallback to thresholds ``(0, b)`` when no candidate is nonnegative."""
    state = np.empty(8)
    state[_LEFT] = 1.0
    state[_RIGHT] = float(scale)
    state[_MID] = ((state[_RIGHT] - state[_LEFT]) // (6 / 5)) + state[_LEFT]
    state[_LASTMID] = 0.0
    state[_VALID] = 1.0
    state[_ITERATIONS] = 0
    state[_A] = a
    state[_B] = b
    return state


@njit(cache=True, error_model="numpy")
def _bisection_steps(g1, g2, lo, hi, scale, state, totals, defer):
    """Run ``bisection_costes``' loop from ``state`` (updated in place).

    Returns ``STEPS_DONE``, ``STEPS_NOT_CONVERGED``, or ``STEPS_AMBIGUOUS`` when a
    step's r is too close to 0 (or the subset too near-constant) to decide like
    SciPy; ``state[_MID]`` is then that step's candidate, already counted, for the
    caller to decide with the reference and resume.
    """
    left, right, mid = state[_LEFT], state[_RIGHT], state[_MID]
    lastmid, valid, iterations = state[_LASTMID], state[_VALID], state[_ITERATIONS]
    a, b = state[_A], state[_B]
    grid = None
    if _use_grid(a, hi - lo, scale):
        grid = _grid_background(g1, g2, lo, hi, a, b, scale, totals)
    code = STEPS_DONE
    while lastmid != mid:
        iterations += 1
        if iterations > 256:
            code = STEPS_NOT_CONVERGED
            break
        thr_fi = mid / scale
        thr_si = a * thr_fi + b
        if grid is not None:
            cnt, r = _grid_count_pearson(
                g1, g2, lo, hi, a, b, scale, mid, grid, totals, defer
            )
        else:
            cnt, r = _direct_count_pearson(
                g1, g2, lo, hi, totals, thr_fi, thr_si, _SIGN_CUTOFF, defer
            )
        if cnt <= 2:  # skipped without a correlation, as in the reference
            left = mid - 1
        elif np.isinf(r):
            code = STEPS_AMBIGUOUS
            break
        elif r < 0:
            left = mid - 1
        elif r >= 0:
            right = mid + 1
            valid = mid
        # A constant subset (NaN) leaves the bounds unchanged, as in the reference.
        lastmid = mid
        if right - left > 6:
            mid = ((right - left) // (6 / 5)) + left
        else:
            mid = ((right - left) // 2) + left
    state[_LEFT], state[_RIGHT], state[_MID] = left, right, mid
    state[_LASTMID], state[_VALID], state[_ITERATIONS] = lastmid, valid, iterations
    return code


@njit(cache=True)
def _bisection_end(scale, state):
    """The thresholds of a finished bisection: ``(thr_fi, thr_si)``."""
    thr_fi_c = (state[_VALID] - 1) / scale
    return thr_fi_c, state[_A] * thr_fi_c + state[_B]


@njit(cache=True, error_model="numpy")
def _bisection(g1, g2, lo, hi, a, b, scale):
    """``bisection_costes`` (M_FASTER): narrowing-window search for the threshold.

    Returns ``(thr_fi, thr_si, status)``; NaN thresholds with ``SEARCHED`` where
    only the reference can decide (see the status codes at the top of the module).
    """
    state = _bisection_state(a, b, scale)
    totals = _moment_totals(g1, g2, lo, hi)
    if _bisection_steps(g1, g2, lo, hi, scale, state, totals, True):
        return np.nan, np.nan, SEARCHED
    thr_fi, thr_si = _bisection_end(scale, state)
    return thr_fi, thr_si, SEARCHED


# The linear search tests one grid step at a time: up to ``scale`` candidates (65535 for
# 16-bit data), each an O(n) pass. After this many exact candidates the search switches to
# sorted prefix sums, which answer each further candidate in O(log n). Short searches (most
# objects stop within a few steps) never pay for the sort.
_SWEEP_AFTER = 32


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
    """Exact positive-slope entry keys; guarded excluded intervals for negative slopes."""
    n = g1.size
    totals = _moment_totals(g1, g2, 0, n)
    c1, c2 = totals[7], totals[8]
    if a > 0:
        scratch = np.empty(1)
        entries = np.empty(n)
        for index in range(n):
            entries[index] = min(
                np.nextafter(g1[index], np.inf),
                _second_entry(g2[index], a, b, scratch),
            )
        k1, P1 = _sorted_prefix(entries, g1, g2, c1, c2, np.ones(n, np.bool_))
        q = np.empty(0)
        k2, P2 = k1[:0], P1[:, :1]
    else:
        q = (g2 - b) / a
        nonempty = q <= g1
        k1, P1 = _sorted_prefix(q, g1, g2, c1, c2, nonempty)
        k2, P2 = _sorted_prefix(g1, g1, g2, c1, c2, nonempty)
    return totals, k1, P1, k2, P2, np.sort(q)


@njit(cache=True, error_model="numpy")
def _sweep_count_pearson(g1, g2, lo, hi, a, b, tab, thr_fi, thr_si, cutoffs, defer):
    """Same ``(count, r)`` as :func:`_count_pearson_combt`, from the prefix tables.

    Positive slopes use exact entry thresholds. For negative slopes, ``q`` vs ``t`` can
    disagree with ``g2 < a*t + b`` by rounding when they are within ``tol``; any such pixel
    sends the candidate to the exact pass. Likewise an ``r`` within its forward-error bound
    of a decision cutoff, or a near-constant subset, uses the exact pass, so every search
    decision equals the exact kernel's.
    """
    totals, k1, P1, k2, P2, qs = tab

    n = hi - lo
    if a > 0:
        count = np.searchsorted(k1, thr_fi, side="right")
        sums = P1[:, count]
    else:
        eps = np.finfo(np.float64).eps
        tol = 64 * eps * (1.0 + abs(thr_fi) + (1.0 + abs(b)) / abs(a))
        index = np.searchsorted(qs, thr_fi - tol)
        if index < qs.size and qs[index] <= thr_fi + tol:
            return _count_pearson_combt(g1, g2, lo, hi, thr_fi, thr_si, cutoffs, defer)
        upper = np.searchsorted(k1, thr_fi, side="right")
        lower = np.searchsorted(k2, thr_fi)
        count = n - upper + lower
        sums = totals[:5] - (P1[:, upper] - P2[:, lower])
    return _r_or_exact(
        g1,
        g2,
        lo,
        hi,
        thr_fi,
        thr_si,
        totals,
        count,
        sums[0],
        sums[1],
        sums[2],
        sums[3],
        sums[4],
        cutoffs,
        defer,
    )


@njit(cache=True, error_model="numpy")
def _r_or_exact(
    g1,
    g2,
    lo,
    hi,
    thr_fi,
    thr_si,
    totals,
    m,
    su,
    sv,
    suu,
    svv,
    suv,
    cutoffs,
    defer,
    nops=0,
):
    """``(count, r)`` of the background subset from its centred moments, or from the
    exact kernel when the moments cannot decide the search exactly.

    ``m`` and the sums ``su .. suv`` are over the subset; ``totals`` from
    :func:`_moment_totals` bound their rounding error.
    """
    n = hi - lo
    eps = np.finfo(np.float64).eps
    if m < 2:
        return m, np.nan
    if m == 2:  # SciPy's exact two-pair result: the exact pass computes it
        return _count_pearson_combt(g1, g2, lo, hi, thr_fi, thr_si, cutoffs, defer)
    varu = suu - su * su / m
    varv = svv - sv * sv / m
    cov = suv - su * sv / m
    # Forward-error bound of the prefix-sum moments (each a sum of at most 2n terms,
    # then a subtraction), propagated to r; conservative by a factor of two.
    e = 8 * max(n, nops) * eps
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
        return _count_pearson_combt(g1, g2, lo, hi, thr_fi, thr_si, cutoffs, defer)
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
    for cutoff in cutoffs:
        if abs(r - cutoff) <= bound:
            return _count_pearson_combt(g1, g2, lo, hi, thr_fi, thr_si, cutoffs, defer)
    return m, min(1.0, max(-1.0, r))


@njit(cache=True, error_model="numpy")
def _direct_count_pearson(g1, g2, lo, hi, totals, thr_fi, thr_si, cutoffs, defer):
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
    return _r_or_exact(
        g1,
        g2,
        lo,
        hi,
        thr_fi,
        thr_si,
        totals,
        m,
        su,
        sv,
        suu,
        svv,
        suv,
        cutoffs,
        defer,
        nops=0,
    )


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
def _linear(g1, g2, lo, hi, a, b, scale, accurate, defer):
    """``linear_costes`` (M_FAST / M_ACCURATE): step the threshold down to R≈0.

    Returns ``(thr_fi, thr_si, status)``; see the status codes at the top of the module.
    """
    i_step = 1.0 / scale
    cutoffs = _SIGN_CUTOFF if accurate else _FAST_CUTOFFS
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
    if np.isinf(_pearson_combt(g1, g2, lo, hi, np.inf, np.inf, _NO_CUTOFFS, defer)):
        return np.nan, np.nan, SEARCHED
    thr_fi_c = cur
    thr_si_c = a * cur + b
    while cur > fi_max and (a * cur + b) > si_max:
        cur -= i_step
    r = 0.0
    num_true = -1  # sentinel (reference's None: forces recompute on first pass)
    evaluated = 0
    found = False
    use_sweep = _use_sorted(a, hi - lo) or (
        a < 0 and _SORTED_MIN_PIXELS <= hi - lo <= _SORTED_MAX_PIXELS
    )
    while cur > i_step and (evaluated < _SWEEP_AFTER or not use_sweep):
        thr_fi_c = cur
        thr_si_c = a * cur + b
        cnt = _count_combt(g1, g2, lo, hi, thr_fi_c, thr_si_c)
        evaluated += 1
        if cnt < 2:  # too few pairs: the current thresholds stand
            found = True
            break
        if cnt != num_true:
            r = _pearson_combt(g1, g2, lo, hi, thr_fi_c, thr_si_c, cutoffs, defer)
            num_true = cnt
        if np.isinf(r):
            return np.nan, np.nan, SEARCHED
        # A constant subset (NaN) takes the ordinary one-step descent.
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
            cnt, r_new = _sweep_count_pearson(
                g1, g2, lo, hi, a, b, tab, cur, thr_si_c, cutoffs, defer
            )
            if cnt < 2:
                break
            # the reference recomputes r only when the count changes
            if cnt != num_true:
                r = r_new
                num_true = cnt
            if np.isinf(r):
                return np.nan, np.nan, SEARCHED
            if r <= 0:
                break
            cur = _next_candidate(cur, r, accurate, i_step)
    return thr_fi_c, thr_si_c, SEARCHED


@njit(cache=True)
def _canonicalize(g, scale):
    """The reference's ``_canonical_costes_pixels``, in place on private buffers: snap
    a value encoded as ``level * (1 / scale)`` to ``level / scale``; leave every
    other value alone. Idempotent, so replayed objects see the same values."""
    for i in range(g.size):
        v = g[i]
        if v >= 1.0 / scale and v <= 1:
            level = np.rint(v * scale)
            if v == level * (1.0 / scale):
                g[i] = level / scale


@njit(cache=True)
def _validity(g1, g2, lo, hi):
    """The reference's ``_invalid_costes_reason`` checks, in its order, as a status."""
    if hi - lo < 3:
        return FEW_OR_NONFINITE
    for i in range(lo, hi):
        if not (np.isfinite(g1[i]) and np.isfinite(g2[i])):
            return FEW_OR_NONFINITE
    for i in range(lo, hi):
        if g1[i] < 0 or g2[i] < 0 or g1[i] > 1 or g2[i] > 1:
            return OUT_OF_RANGE
    m = 0
    min1 = min2 = np.inf
    max1 = max2 = -np.inf
    for i in range(lo, hi):
        if g1[i] > 0 or g2[i] > 0:
            m += 1
            min1, max1 = min(min1, g1[i]), max(max1, g1[i])
            min2, max2 = min(min2, g2[i]), max(max2, g2[i])
    if m < 3:
        return FEW_NONZERO
    if min1 == max1 or min2 == max2:
        return CONSTANT_CHANNEL
    return SEARCHED


@njit(cache=True)
def _gather_sum(values, lo, hi, keep, buf, chunk):
    """``values[lo:hi][keep].sum()`` with NumPy's summation order."""
    m = 0
    for i in range(lo, hi):
        if keep[i - lo]:
            buf[m] = values[i]
            m += 1
    return numpy_sum(buf[:m], chunk)


@njit(cache=True)
def _coefficients(g1, g2, lo, hi, thr1, thr2, keep, buf, chunk):
    """The reference's coefficients for the chosen thresholds: ``(status, c1, c2)``."""
    for i in range(lo, hi):
        keep[i - lo] = g1[i] >= thr1
    tot1 = _gather_sum(g1, lo, hi, keep, buf, chunk)
    for i in range(lo, hi):
        keep[i - lo] = g2[i] >= thr2
    tot2 = _gather_sum(g2, lo, hi, keep, buf, chunk)
    if tot1 <= 0 or tot2 <= 0:
        return ZERO_DENOMINATOR, np.nan, np.nan
    for i in range(lo, hi):
        keep[i - lo] = g1[i] > thr1 and g2[i] > thr2
    c1 = _gather_sum(g1, lo, hi, keep, buf, chunk) / tot1
    return SEARCHED, c1, _gather_sum(g2, lo, hi, keep, buf, chunk) / tot2


@njit(cache=True)
def _prepare(g1, g2, offsets, scale):
    """Canonicalize in place; allocate the per-object outputs and scratch buffers."""
    _canonicalize(g1, scale)
    _canonicalize(g2, scale)
    n = offsets.size - 1
    longest = 0
    for k in range(n):
        longest = max(longest, offsets[k + 1] - offsets[k])
    status = np.zeros(n, np.int64)
    return (
        np.full(n, np.nan),
        np.full(n, np.nan),
        status,
        np.empty(longest),
        np.empty(longest, np.bool_),
    )


# Two entry kernels share the per-object helpers, so a call compiles only its own
# search family (one runtime-dispatched kernel compiled both, ~8x the first call).


@njit(cache=True, nogil=True, error_model="numpy")
def costes_faster(g1, g2, offsets, scale, defer, chunk):
    """Costes coefficients of every object, Faster mode (bisection).

    ``g1``/``g2`` are private grouped pixels (object ``k`` owns
    ``offsets[k]:offsets[k + 1]``), already divided by the acquisition scale; they
    are canonicalized in place. ``chunk`` is ``numpy_sum_chunk()``. Validity, the
    regression line, the search and the coefficients follow
    ``reference._costes_pair``. Returns ``(c1, c2, status, states)``; a non-zero
    status leaves NaN for the wrapper to report (see the status codes above), and
    a ``RESUME`` object's row of ``states`` holds its paused bisection. The pass
    stops at the first ``OUT_OF_RANGE`` object, where the reference raises.
    """
    c1, c2, status, buf, keep = _prepare(g1, g2, offsets, scale)
    states = np.empty((offsets.size - 1, 8))
    for k in range(offsets.size - 1):
        lo, hi = offsets[k], offsets[k + 1]
        st = _validity(g1, g2, lo, hi) if hi > lo else EMPTY_OBJECT
        if st == SEARCHED:
            a, b = _regression_ab(g1, g2, lo, hi, chunk)
            if not (np.isfinite(a) and np.isfinite(b)):
                st = REPLAY  # rare; the reference's NumPy diagnostics decide
            else:
                state = _bisection_state(a, b, scale)
                totals = _moment_totals(g1, g2, lo, hi)
                code = _bisection_steps(g1, g2, lo, hi, scale, state, totals, defer)
                if code == STEPS_AMBIGUOUS:
                    st = RESUME
                    states[k] = state
                elif code == STEPS_NOT_CONVERGED:
                    st = REPLAY
                else:
                    thr1, thr2 = _bisection_end(scale, state)
                    st, c1[k], c2[k] = _coefficients(
                        g1, g2, lo, hi, thr1, thr2, keep, buf, chunk
                    )
        status[k] = st
        if st == OUT_OF_RANGE:
            break
    return c1, c2, status, states


@njit(cache=True, nogil=True, error_model="numpy")
def costes_linear(g1, g2, offsets, scale, accurate, defer, chunk):
    """:func:`costes_faster` for the Fast and Accurate modes (linear search)."""
    c1, c2, status, buf, keep = _prepare(g1, g2, offsets, scale)
    for k in range(offsets.size - 1):
        lo, hi = offsets[k], offsets[k + 1]
        st = _validity(g1, g2, lo, hi) if hi > lo else EMPTY_OBJECT
        if st == SEARCHED:
            a, b = _regression_ab(g1, g2, lo, hi, chunk)
            if not (np.isfinite(a) and np.isfinite(b)):
                st = REPLAY
            else:
                thr1, thr2, st = _linear(g1, g2, lo, hi, a, b, scale, accurate, defer)
                if st == SEARCHED and not (np.isfinite(thr1) and np.isfinite(thr2)):
                    st = REPLAY
                elif st == SEARCHED:
                    st, c1[k], c2[k] = _coefficients(
                        g1, g2, lo, hi, thr1, thr2, keep, buf, chunk
                    )
        status[k] = st
        if st == OUT_OF_RANGE:
            break
    return c1, c2, status


# One call per image: grouping and the search in the same GIL-free call, so the
# grouped pixels never reach Python unless the reference must decide an object.


@njit(cache=True)
def _for_replay(g1, g2, status):
    """The grouped pixels if an object needs the reference, else empty arrays."""
    for st in status:
        if st == REPLAY or st == RESUME:
            return g1, g2
    return np.empty(0), np.empty(0)


@njit(cache=True, nogil=True, error_model="numpy")
def costes_faster_image(masks, pix1, pix2, scale, defer, chunk):
    """:func:`costes_faster` of one ``(Z, Y, X)`` image triple, grouped by
    ``group_pairs``. Returns ``(c1, c2, status, states, g1, g2, offsets)``, with
    ``g1``/``g2`` empty unless an object needs the reference (see
    :func:`_for_replay`)."""
    g1, g2, offsets = group_pairs(masks, pix1, pix2)
    c1, c2, status, states = costes_faster(g1, g2, offsets, scale, defer, chunk)
    g1, g2 = _for_replay(g1, g2, status)
    return c1, c2, status, states, g1, g2, offsets


@njit(cache=True, nogil=True, error_model="numpy")
def costes_linear_image(masks, pix1, pix2, scale, accurate, defer, chunk):
    """:func:`costes_faster_image` for the Fast and Accurate modes; returns
    ``(c1, c2, status, g1, g2, offsets)``."""
    g1, g2, offsets = group_pairs(masks, pix1, pix2)
    c1, c2, status = costes_linear(g1, g2, offsets, scale, accurate, defer, chunk)
    g1, g2 = _for_replay(g1, g2, status)
    return c1, c2, status, g1, g2, offsets


_SORTED_MAX_PIXELS = 1 << 22
_SORTED_MIN_PIXELS = 32
_GRID_PAD = 258
_GRID_MAX_SCALE = 1 << 20
_GRID_STEPS_PER_PIXEL = 2
_GRID_MIN_PIXELS = 1 << 15
_MAGNITUDE = np.int64(0x7FFFFFFFFFFFFFFF)
_SIGN = np.int64(-0x8000000000000000)


@njit(cache=True)
def _float_key(t, scratch):
    """Map a non-NaN float to an int64 with the same ordering."""
    scratch[0] = t
    bits = scratch.view(np.int64)[0]
    return bits if bits >= 0 else -(bits & _MAGNITUDE)


@njit(cache=True)
def _key_float(key, scratch):
    ints = scratch.view(np.int64)
    ints[0] = key if key >= 0 else (-key) | _SIGN
    return scratch[0]


@njit(cache=True, error_model="numpy")
def _second_entry(y, a, b, scratch):
    """Smallest float t with ``a*t + b > y`` (a > 0, all finite)."""
    t = (y - b) / a
    if np.isfinite(t):
        # The quotient is normally within an ulp or two of the boundary.
        if a * t + b > y:
            for _ in range(8):
                below = np.nextafter(t, -np.inf)
                if not (a * below + b > y):
                    return t
                t = below
        else:
            for _ in range(8):
                t = np.nextafter(t, np.inf)
                if a * t + b > y:
                    return t
    # Exact bisection over the float ordering: false at -inf, true at +inf.
    lo = _float_key(-np.inf, scratch)
    hi = _float_key(np.inf, scratch)
    while True:
        mid = (lo >> 1) + (hi >> 1) + (lo & hi & 1)
        if mid == lo or mid == hi:
            break
        if a * _key_float(mid, scratch) + b > y:
            hi = mid
        else:
            lo = mid
    return _key_float(hi, scratch)


@njit(cache=True, error_model="numpy")
def _grid_background_at(x, y, k, a, b, scale):
    """The scan's background test at candidate ``mid = k`` (k integer-valued)."""
    t = k / scale
    return x < t or y < a * t + b


@njit(cache=True, error_model="numpy")
def _grid_entry(x, y, a, b, scale, k_lo, k_hi):
    """Smallest integer k in [k_lo, k_hi] with the pixel in background at k.

    Returns ``k_lo`` if it is already background there and ``k_hi + 1`` if it
    never is. The guess is normally exact or off by one; far guesses (e.g.
    cancellation in ``y - b``) fall back to an exact integer bisection.
    """
    g = min(np.floor(x * scale), np.floor((y - b) / a * scale)) + 1.0
    if not (g >= k_lo):  # also catches NaN
        g = k_lo
    if not (g <= k_hi + 1):
        g = k_hi + 1
    if g <= k_hi and _grid_background_at(x, y, g, a, b, scale):
        for _ in range(4):
            if g == k_lo or not _grid_background_at(x, y, g - 1, a, b, scale):
                return g
            g -= 1
        lo, hi = k_lo, g
    else:
        if g > k_hi:
            return k_hi + 1
        for _ in range(4):
            g += 1
            if g > k_hi or _grid_background_at(x, y, g, a, b, scale):
                return g
        lo, hi = g + 1, k_hi + 1
    # Invariant: background at hi (k_hi + 1 counts as background), not below lo.
    while lo < hi:
        m = np.floor((lo + hi) / 2)
        if _grid_background_at(x, y, m, a, b, scale):
            hi = m
        else:
            lo = m + 1
    return lo


@njit(cache=True, error_model="numpy")
def _grid_background(g1, g2, lo, hi, a, b, scale, totals):
    """Cumulative per-grid-step background statistics.

    Returns ``(acc, ext, mx, my, k_lo, k_hi)``. Row ``j`` describes the
    background at ``mid = k_lo + j``: ``acc[j]`` holds count and sums of
    dx, dy, dx², dy², dx·dy (shifted by object means ``mx``/``my``);
    ``ext[j]`` holds min/max of x and y.
    """
    k_lo = 1.0 - _GRID_PAD
    k_hi = float(scale) + _GRID_PAD
    nb = int(k_hi - k_lo) + 2  # last row: never background in range
    mx, my = totals[7], totals[8]
    acc = np.zeros((nb, 6), dtype=np.float64)
    ext = np.empty((nb, 4), dtype=np.float64)
    ext[:, 0] = np.inf
    ext[:, 1] = -np.inf
    ext[:, 2] = np.inf
    ext[:, 3] = -np.inf
    for i in range(lo, hi):
        x = g1[i]
        y = g2[i]
        j = int(_grid_entry(x, y, a, b, scale, k_lo, k_hi) - k_lo)
        dx = x - mx
        dy = y - my
        acc[j, 0] += 1.0
        acc[j, 1] += dx
        acc[j, 2] += dy
        acc[j, 3] += dx * dx
        acc[j, 4] += dy * dy
        acc[j, 5] += dx * dy
        ext[j, 0] = min(ext[j, 0], x)
        ext[j, 1] = max(ext[j, 1], x)
        ext[j, 2] = min(ext[j, 2], y)
        ext[j, 3] = max(ext[j, 3], y)
    for j in range(1, nb):
        for c in range(6):
            acc[j, c] += acc[j - 1, c]
        ext[j, 0] = min(ext[j, 0], ext[j - 1, 0])
        ext[j, 1] = max(ext[j, 1], ext[j - 1, 1])
        ext[j, 2] = min(ext[j, 2], ext[j - 1, 2])
        ext[j, 3] = max(ext[j, 3], ext[j - 1, 3])
    return acc, ext, mx, my, k_lo, k_hi


@njit(cache=True)
def _use_sorted(a, n):
    """Sorted searches need a > 0, enough pixels to amortize the sort, and
    bounded scratch memory."""
    return a > 0 and _SORTED_MIN_PIXELS <= n <= _SORTED_MAX_PIXELS


@njit(cache=True)
def _use_grid(a, n, scale):
    """Grid buckets need a > 0, bounded scale, and pixels to amortize O(scale)."""
    return (
        a > 0
        and n >= _GRID_MIN_PIXELS
        and scale <= _GRID_MAX_SCALE
        and scale + 2 * _GRID_PAD <= _GRID_STEPS_PER_PIXEL * n
    )


@njit(cache=True, error_model="numpy")
def _grid_count_pearson(g1, g2, lo, hi, a, b, scale, mid, table, totals, defer):
    """Use exact grid membership and the fused search rounding policy."""
    acc, ext, _, _, k_lo, k_hi = table
    thr_fi = mid / scale
    thr_si = a * thr_fi + b
    if mid < k_lo or mid > k_hi:
        count, value = _count_pearson_combt(
            g1, g2, lo, hi, thr_fi, thr_si, _SIGN_CUTOFF, defer
        )
        return count, np.nan if count <= 2 else value
    index = int(mid - k_lo)
    count = int(acc[index, 0])
    if count <= 2 or ext[index, 0] == ext[index, 1] or ext[index, 2] == ext[index, 3]:
        return count, np.nan
    return _r_or_exact(
        g1,
        g2,
        lo,
        hi,
        thr_fi,
        thr_si,
        totals,
        count,
        acc[index, 1],
        acc[index, 2],
        acc[index, 3],
        acc[index, 4],
        acc[index, 5],
        _SIGN_CUTOFF,
        defer,
        hi - lo + acc.shape[0],
    )
