"""Costes threshold searches (single-threaded, cached).

The host keeps NumPy's regression reductions: tiny changes in the fitted line
can change threshold inclusion (#118). Only the repeated background searches
are compiled. Undefined searches return NaN to the wrapper, which replays the
reference for its precise RuntimeWarning. No fastmath or grid rounding is used.

For a positive fitted slope, pixels are sorted once by the exact threshold at
which they enter the background, so each candidate costs O(log n) instead of a
full rescan (see ``_sorted_background``). Bisection (M_FASTER) only probes
integer grid steps, so when the object has enough pixels relative to the scale
it buckets pixels by grid step instead of sorting (see ``_grid_background``);
that scratch scales with ``scale`` (up to ``_GRID_MAX_SCALE``), not with the
object, so it is used however large the object is.

Otherwise, a non-positive slope, or an object smaller than
``_SORTED_MIN_PIXELS`` or larger than ``_SORTED_MAX_PIXELS``, keeps the
constant-memory scan.

The dispatch thresholds below are read by ``@njit(cache=True)`` helpers, so
Numba freezes them at compile time: reassigning them at runtime (e.g. from a
test or benchmark) has no effect. Edit them here; the on-disk cache is keyed
on this file and recompiles.
"""

import numpy as np
from numba import njit

_EPS = np.finfo(np.float64).eps
# SciPy's near-constant diagnostic: scaled norm below this fraction of |mean|.
_TINY = _EPS**0.75
# Fast linear search takes 10/5/2 grid steps while r exceeds these values.
_STEP10_R = 0.45
_STEP5_R = 0.35
_STEP2_R = 0.25
# Every r at which a search decision changes (stop at <= 0, then step sizes).
_CUTOFFS = (0.0, _STEP2_R, _STEP5_R, _STEP10_R)
# The sorted path needs ~64 bytes of scratch per pixel; larger objects scan.
_SORTED_MAX_PIXELS = 1 << 22
# Below this the sort costs more than it saves (benchmarked); small objects scan.
_SORTED_MIN_PIXELS = 32


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
    if abs(covar) <= 8 * _EPS * max(xvar, yvar):
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
    """Background Pearson r; fewer than two or constant pairs are undefined."""
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
    if cnt < 2 or min1 == max1 or min2 == max2:
        return cnt, np.nan
    if cnt == 2:
        # Match SciPy's exact +/-1 two-pair result used by linear search.
        previous = -1
        for i in range(lo, hi):
            if g1[i] < thr_fi or g2[i] < thr_si:
                if previous >= 0:
                    return cnt, np.sign(g1[i] - g1[previous]) * np.sign(
                        g2[i] - g2[previous]
                    )
                previous = i
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
    if normx < _TINY * abs(m1) or normy < _TINY * abs(m2):
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
    operations_eps = 64 * cnt * _EPS
    gamma = operations_eps / (1 - operations_eps)
    conditioning = 1 + np.sqrt(cnt) * (abs(m1) / normx + abs(m2) / normy)
    error_bound = gamma * conditioning
    for cutoff in _CUTOFFS:
        if abs(r - cutoff) <= error_bound:
            return cnt, np.inf
    return cnt, min(1.0, max(-1.0, r))


@njit(cache=True, error_model="numpy")
def _bisection_scan(g1, g2, lo, hi, a, b, scale):
    """``bisection_costes`` (M_FASTER), rescanning the object per candidate."""
    left = 1.0
    right = float(scale)
    mid = ((right - left) // (6 / 5)) + left
    lastmid = 0.0
    valid = 1.0  # CP boundary fallback when no nonnegative candidate is found
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
        if cnt <= 2 or r < 0:
            left = mid - 1
        elif r >= 0:
            right = mid + 1
            valid = mid
        # A constant subset leaves the bounds unchanged, matching the reference.
        lastmid = mid
        if right - left > 6:
            mid = ((right - left) // (6 / 5)) + left
        else:
            mid = ((right - left) // 2) + left
    thr_fi_c = (valid - 1) / scale
    thr_si_c = a * thr_fi_c + b
    return thr_fi_c, thr_si_c


@njit(cache=True, error_model="numpy")
def _linear_scan(g1, g2, lo, hi, a, b, scale, accurate):
    """``linear_costes`` (M_FAST / M_ACCURATE), rescanning the object per candidate."""
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
        if cnt < 2:
            break
        if cnt != num_true:
            r = _pearson_combt(g1, g2, lo, hi, thr_fi_c, thr_si_c)
            num_true = cnt
        if np.isinf(r):
            return np.nan, np.nan  # numerical ambiguity requests reference replay
        if r <= 0:
            break
        elif accurate or cur < i_step * 10:
            cur -= i_step
        elif r > _STEP10_R:
            cur -= i_step * 10
        elif r > _STEP5_R:
            cur -= i_step * 5
        elif r > _STEP2_R:
            cur -= i_step * 2
        else:
            cur -= i_step
    return thr_fi_c, thr_si_c


# ---------------------------------------------------------------------------
# Sorted background searches (fitted slope a > 0).
#
# A pixel is background at candidate t when ``x < t or y < a*t + b``. For a > 0
# that predicate is monotone in t even under IEEE rounding, so each pixel has an
# exact *entry threshold* e (the smallest float t for which it holds) and the
# background at t is exactly ``{e <= t}``. Sorting by e once turns every
# candidate's background into a prefix: its count is a binary search and its
# Pearson r comes from prefix sums. Prefix r is only trusted far from the
# search's decision cutoffs and diagnostics; otherwise the scan kernel is rerun
# in the original pixel order, reproducing its borderline decisions exactly.
# ---------------------------------------------------------------------------

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
def _sorted_background(g1, g2, lo, hi, a, b):
    """Sort an object's pixels by entry threshold and build prefix statistics.

    Returns ``(entry, xs, ys, stats, mx, my, x_run, y_run, x_max, y_max)``:

    - ``entry``: sorted entry thresholds; the background at t is the first
      ``searchsorted(entry, t, "right")`` sorted pixels.
    - ``xs``/``ys``: the first ``min(n, 2)`` sorted pixels (for the exact
      two-pair r); later pixels are only needed through ``stats``.
    - ``stats[k]``: sums of x, y, x², y², xy over the first k pixels, shifted by
      the object means ``mx``/``my`` to limit cancellation.
    - ``x_run``/``y_run``: length of the leading run of equal values, so the
      first k pixels are constant in that channel exactly when ``k <= run``.
    - ``x_max``/``y_max``: object maxima (linear search starts above them).
    """
    n = hi - lo
    scratch = np.empty(1, dtype=np.float64)
    unsorted = np.empty(n, dtype=np.float64)
    mx = 0.0
    my = 0.0
    x_max = -np.inf
    y_max = -np.inf
    for k in range(n):
        x = g1[lo + k]
        y = g2[lo + k]
        unsorted[k] = min(np.nextafter(x, np.inf), _second_entry(y, a, b, scratch))
        mx += x
        my += y
        if x > x_max:
            x_max = x
        if y > y_max:
            y_max = y
    mx /= n
    my /= n
    order = np.argsort(unsorted)
    # One fused gather: sorted entries, leading runs, and prefix sums.
    entry = np.empty(n, dtype=np.float64)
    stats = np.empty((n + 1, 5), dtype=np.float64)
    stats[0, :] = 0.0
    m = min(n, 2)
    xs = np.empty(m, dtype=np.float64)
    ys = np.empty(m, dtype=np.float64)
    x0 = g1[lo + order[0]]
    y0 = g2[lo + order[0]]
    x_run = n
    y_run = n
    for k in range(n):
        i = lo + order[k]
        entry[k] = unsorted[order[k]]
        x = g1[i]
        y = g2[i]
        if k < m:
            xs[k] = x
            ys[k] = y
        if x_run == n and x != x0:
            x_run = k
        if y_run == n and y != y0:
            y_run = k
        dx = x - mx
        dy = y - my
        stats[k + 1, 0] = stats[k, 0] + dx
        stats[k + 1, 1] = stats[k, 1] + dy
        stats[k + 1, 2] = stats[k, 2] + dx * dx
        stats[k + 1, 3] = stats[k, 3] + dy * dy
        stats[k + 1, 4] = stats[k, 4] + dx * dy
    return entry, xs, ys, stats, mx, my, x_run, y_run, x_max, y_max


@njit(cache=True, error_model="numpy")
def _prefix_pearson(
    k, xs, ys, stats, mx, my, x_run, y_run, g1, g2, lo, hi, thr_fi, thr_si
):
    """``_count_pearson_combt``'s r for the first k sorted pixels.

    Ambiguous cases rerun the scan kernel on the original pixel order and
    thresholds, so its summation order (hence every borderline cutoff and
    near-constant decision) is reproduced exactly rather than approximately.
    """
    if k < 2 or k <= x_run or k <= y_run:
        return np.nan
    if k == 2:
        # The two-pair sign product is order independent.
        return _count_pearson_combt(xs, ys, 0, k, np.inf, np.inf)[1]
    return _pearson_from_sums(
        k,
        k,
        stats[k, 0],
        stats[k, 1],
        stats[k, 2],
        stats[k, 3],
        stats[k, 4],
        mx,
        my,
        g1,
        g2,
        lo,
        hi,
        thr_fi,
        thr_si,
    )


@njit(cache=True, error_model="numpy")
def _pearson_from_sums(
    k, nops, sx, sy, sxx, syy, sxy, mx, my, g1, g2, lo, hi, thr_fi, thr_si
):
    """r from k mean-shifted sums, each built with at most ``nops`` additions.

    Requires k > 2 and a non-constant subset. Ambiguous r reruns the scan.
    """
    cxx = sxx - sx * sx / k
    cyy = syy - sy * sy / k
    cxy = sxy - sx * sy / k
    # Recursive sums of nops terms: error <= ~nops*eps of the absolute sums,
    # and Cauchy-Schwarz bounds every cross term by sqrt(sxx*syy).
    err = 8 * nops * _EPS
    m1 = mx + sx / k
    m2 = my + sy / k
    if (
        cxx <= 64 * err * sxx
        or cyy <= 64 * err * syy
        or cxx <= 4 * (_TINY * m1) ** 2
        or cyy <= 4 * (_TINY * m2) ** 2
    ):
        # Ill-conditioned centering or near SciPy's near-constant diagnostic.
        return _count_pearson_combt(g1, g2, lo, hi, thr_fi, thr_si)[1]
    normx = np.sqrt(cxx)
    normy = np.sqrt(cyy)
    r = cxy / (normx * normy)
    kx = sxx / cxx
    ky = syy / cyy
    prefix_bound = 4 * err * (np.sqrt(kx * ky) + kx + ky)
    # Margin also covers the scan kernel's own reference-replay bound.
    operations_eps = 64 * k * _EPS
    gamma = operations_eps / (1 - operations_eps)
    scan_bound = gamma * (1 + np.sqrt(k) * (abs(m1) / normx + abs(m2) / normy))
    margin = 4 * (prefix_bound + scan_bound) + 16 * _EPS
    for cutoff in _CUTOFFS:
        if abs(r - cutoff) <= margin:
            return _count_pearson_combt(g1, g2, lo, hi, thr_fi, thr_si)[1]
    return min(1.0, max(-1.0, r))


@njit(cache=True, error_model="numpy")
def _bisection_sorted(g1, g2, lo, hi, a, b, scale):
    """``_bisection_scan`` with O(log n) background counts (requires a > 0)."""
    entry, xs, ys, stats, mx, my, x_run, y_run, _, _ = _sorted_background(
        g1, g2, lo, hi, a, b
    )
    left = 1.0
    right = float(scale)
    mid = ((right - left) // (6 / 5)) + left
    lastmid = 0.0
    valid = 1.0
    iterations = 0
    while lastmid != mid:
        iterations += 1
        if iterations > 256:
            return np.nan, np.nan
        thr_fi = mid / scale
        thr_si = a * thr_fi + b
        cnt = np.searchsorted(entry, thr_fi, side="right")
        # Same decision order as _bisection_scan: replay check, then bounds.
        r = _prefix_pearson(
            cnt, xs, ys, stats, mx, my, x_run, y_run, g1, g2, lo, hi, thr_fi, thr_si
        )
        if np.isinf(r):
            return np.nan, np.nan
        if cnt <= 2 or r < 0:
            left = mid - 1
        elif r >= 0:
            right = mid + 1
            valid = mid
        lastmid = mid
        if right - left > 6:
            mid = ((right - left) // (6 / 5)) + left
        else:
            mid = ((right - left) // 2) + left
    thr_fi_c = (valid - 1) / scale
    thr_si_c = a * thr_fi_c + b
    return thr_fi_c, thr_si_c


@njit(cache=True, error_model="numpy")
def _linear_sorted(g1, g2, lo, hi, a, b, scale, accurate):
    """``_linear_scan`` with O(log n) background counts (requires a > 0)."""
    # The full-object r keeps the scan's own reduction order for its replay
    # check; the maxima come from _sorted_background's first pass.
    if np.isinf(_pearson_combt(g1, g2, lo, hi, np.inf, np.inf)):
        return np.nan, np.nan
    entry, xs, ys, stats, mx, my, x_run, y_run, fi_max, si_max = _sorted_background(
        g1, g2, lo, hi, a, b
    )
    i_step = 1.0 / scale
    img_max = fi_max if fi_max > si_max else si_max
    cur = i_step * ((img_max // i_step) + 1)
    thr_fi_c = cur
    thr_si_c = a * cur + b
    while cur > fi_max and (a * cur + b) > si_max:
        cur -= i_step
    r = 0.0
    num_true = -1
    while cur > i_step:
        thr_fi_c = cur
        thr_si_c = a * cur + b
        cnt = np.searchsorted(entry, cur, side="right")
        if cnt < 2:
            break
        if cnt != num_true:
            r = _prefix_pearson(
                cnt,
                xs,
                ys,
                stats,
                mx,
                my,
                x_run,
                y_run,
                g1,
                g2,
                lo,
                hi,
                thr_fi_c,
                thr_si_c,
            )
            num_true = cnt
        if np.isinf(r):
            return np.nan, np.nan
        if r <= 0:
            break
        elif accurate or cur < i_step * 10:
            cur -= i_step
        elif r > _STEP10_R:
            cur -= i_step * 10
        elif r > _STEP5_R:
            cur -= i_step * 5
        elif r > _STEP2_R:
            cur -= i_step * 2
        else:
            cur -= i_step
    return thr_fi_c, thr_si_c


# ---------------------------------------------------------------------------
# Grid-bucketed bisection (fitted slope a > 0, moderate scale).
#
# Bisection only probes candidates ``mid / scale`` with integer-valued ``mid``.
# Background membership is monotone in mid (a > 0), so each pixel has a first
# grid step at which it enters. Accumulating pixel statistics per step and
# taking a cumulative sum over steps gives every candidate's background in
# O(n + scale), with no sort. ``left``/``right`` move by at most one past the
# current window per iteration and iterations are capped at 256, so every mid
# lies in [1 - 256, scale + 256]; mids outside the padded range still scan.
# ---------------------------------------------------------------------------

_GRID_PAD = 258
# Bucket scratch is ~80 bytes per grid step. Larger scales fall back to the
# sorted path, or to the scan for objects outside the sorted size limits.
_GRID_MAX_SCALE = 1 << 20
# Use grid buckets only with at most this many grid steps (scale plus padding)
# per pixel; beyond it the O(scale) bucket pass outweighs sorting (benchmarked).
_GRID_STEPS_PER_PIXEL = 2


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
def _grid_background(g1, g2, lo, hi, a, b, scale):
    """Cumulative per-grid-step background statistics.

    Returns ``(acc, ext, mx, my, k_lo, k_hi)``. Row ``j`` describes the
    background at ``mid = k_lo + j``: ``acc[j]`` holds count and sums of
    dx, dy, dx², dy², dx·dy (shifted by object means ``mx``/``my``);
    ``ext[j]`` holds min/max of x and y.
    """
    n = hi - lo
    k_lo = 1.0 - _GRID_PAD
    k_hi = float(scale) + _GRID_PAD
    nb = int(k_hi - k_lo) + 2  # last row: never background in range
    mx = 0.0
    my = 0.0
    for i in range(lo, hi):
        mx += g1[i]
        my += g2[i]
    mx /= n
    my /= n
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


@njit(cache=True, error_model="numpy")
def _grid_count_pearson(acc, ext, mx, my, k_lo, k_hi, mid, g1, g2, lo, hi, a, b, scale):
    """Bisection's ``_count_pearson_combt`` at ``thr_fi = mid / scale``.

    Matches the scan except that r is always NaN for cnt <= 2: bisection never
    reads r there, and a two-pair r is +/-1 or NaN (never the inf replay
    signal), so the exact two-pair value is not computed. Not for linear search.
    """
    thr_fi = mid / scale
    thr_si = a * thr_fi + b
    if mid < k_lo or mid > k_hi:
        cnt, r = _count_pearson_combt(g1, g2, lo, hi, thr_fi, thr_si)
        return cnt, (np.nan if cnt <= 2 else r)
    j = int(mid - k_lo)
    cnt = int(acc[j, 0])
    if cnt <= 2 or ext[j, 0] == ext[j, 1] or ext[j, 2] == ext[j, 3]:
        return cnt, np.nan
    # Each sum passed through one bucket's additions and j + 1 cumulative ones.
    r = _pearson_from_sums(
        cnt,
        cnt + j + 1,
        acc[j, 1],
        acc[j, 2],
        acc[j, 3],
        acc[j, 4],
        acc[j, 5],
        mx,
        my,
        g1,
        g2,
        lo,
        hi,
        thr_fi,
        thr_si,
    )
    return cnt, r


@njit(cache=True, error_model="numpy")
def _bisection_grid(g1, g2, lo, hi, a, b, scale):
    """``_bisection_scan`` with O(1) grid-bucket candidates (requires a > 0)."""
    acc, ext, mx, my, k_lo, k_hi = _grid_background(g1, g2, lo, hi, a, b, scale)
    left = 1.0
    right = float(scale)
    mid = ((right - left) // (6 / 5)) + left
    lastmid = 0.0
    valid = 1.0
    iterations = 0
    while lastmid != mid:
        iterations += 1
        if iterations > 256:
            return np.nan, np.nan
        cnt, r = _grid_count_pearson(
            acc, ext, mx, my, k_lo, k_hi, mid, g1, g2, lo, hi, a, b, scale
        )
        if np.isinf(r):
            return np.nan, np.nan
        if cnt <= 2 or r < 0:
            left = mid - 1
        elif r >= 0:
            right = mid + 1
            valid = mid
        lastmid = mid
        if right - left > 6:
            mid = ((right - left) // (6 / 5)) + left
        else:
            mid = ((right - left) // 2) + left
    thr_fi_c = (valid - 1) / scale
    thr_si_c = a * thr_fi_c + b
    return thr_fi_c, thr_si_c


# Dispatch thresholds are compile-time constants here; see the module docstring.
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
        and scale <= _GRID_MAX_SCALE
        and scale + 2 * _GRID_PAD <= _GRID_STEPS_PER_PIXEL * n
    )


@njit(cache=True, error_model="numpy")
def _bisection(g1, g2, lo, hi, a, b, scale):
    """``bisection_costes`` (M_FASTER): narrowing-window search for the threshold."""
    n = hi - lo
    if _use_grid(a, n, scale):
        return _bisection_grid(g1, g2, lo, hi, a, b, scale)
    if _use_sorted(a, n):
        return _bisection_sorted(g1, g2, lo, hi, a, b, scale)
    return _bisection_scan(g1, g2, lo, hi, a, b, scale)


@njit(cache=True, error_model="numpy")
def _linear(g1, g2, lo, hi, a, b, scale, accurate):
    """``linear_costes`` (M_FAST / M_ACCURATE): step the threshold down to R≈0."""
    if _use_sorted(a, hi - lo):
        return _linear_sorted(g1, g2, lo, hi, a, b, scale, accurate)
    return _linear_scan(g1, g2, lo, hi, a, b, scale, accurate)
