"""Costes compiled search and NumPy regression tests against current main.

Explicit scale=255 exercises genuine multi-iteration searches. Production API
coverage, including undefined replay and normalization, lives in
``test_costes_backend.py``.
"""

import numpy as np
import pytest
import scipy.stats

import cp_measure.core.measurecolocalization as ref
from cp_measure._detect import HAS_NUMBA

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


def _obj(seed, n=200):
    """A correlated float object in [0,1]: si ≈ fi + noise, both clipped."""
    rng = np.random.default_rng(seed)
    fi = rng.random(n)
    si = np.clip(0.7 * fi + 0.3 * rng.random(n), 0.0, 1.0)
    return np.ascontiguousarray(fi), np.ascontiguousarray(si)


@requires_numba
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_regression_ab_matches_reference(seed):
    from cp_measure.core.numba._costes import _regression_ab

    fi, si = _obj(seed)
    a, b = _regression_ab(fi, si, 0, fi.size)
    # Reference computes a,b inline; reproduce its exact expression.
    nz = (fi > 0) | (si > 0)
    xvar = np.var(fi[nz], ddof=1)
    yvar = np.var(si[nz], ddof=1)
    z = fi[nz] + si[nz]
    zvar = np.var(z, ddof=1)
    covar = 0.5 * (zvar - (xvar + yvar))
    a_ref = ((yvar - xvar) + np.sqrt((yvar - xvar) ** 2 + 4 * covar**2)) / (2 * covar)
    b_ref = si[nz].mean() - a_ref * fi[nz].mean()
    np.testing.assert_allclose(a, a_ref, rtol=1e-9)
    np.testing.assert_allclose(b, b_ref, rtol=1e-9)


@requires_numba
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pearson_combt_matches_scipy(seed):
    from cp_measure.core.numba._costes import _pearson_combt

    fi, si = _obj(seed)
    thr_fi, thr_si = 0.4, 0.35
    got = _pearson_combt(fi, si, 0, fi.size, thr_fi, thr_si)
    combt = (fi < thr_fi) | (si < thr_si)
    r_ref, _ = scipy.stats.pearsonr(fi[combt], si[combt])
    np.testing.assert_allclose(got, r_ref, rtol=1e-9, atol=1e-12)
    # full-block (inf thresholds) == pearson over everything
    got_full = _pearson_combt(fi, si, 0, fi.size, np.inf, np.inf)
    np.testing.assert_allclose(got_full, scipy.stats.pearsonr(fi, si)[0], rtol=1e-9)


@requires_numba
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_bisection_matches_reference(seed):
    from cp_measure.core.numba._costes import _bisection, _regression_ab

    fi, si = _obj(seed)
    a, b = _regression_ab(fi, si, 0, fi.size)
    thr_fi, thr_si = _bisection(fi, si, 0, fi.size, a, b, 255.0)
    thr_fi_ref, thr_si_ref = ref.bisection_costes(fi, si, fi, si, 255)
    np.testing.assert_allclose([thr_fi, thr_si], [thr_fi_ref, thr_si_ref], rtol=1e-6)


@requires_numba
@pytest.mark.parametrize("mode_name", ["Fast", "Accurate"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_linear_matches_reference(seed, mode_name):
    from cp_measure.core.numba._costes import _linear, _regression_ab

    fi, si = _obj(seed)
    a, b = _regression_ab(fi, si, 0, fi.size)
    thr_fi, thr_si = _linear(fi, si, 0, fi.size, a, b, 255.0, mode_name == "Accurate")
    thr_fi_ref, thr_si_ref = ref.linear_costes(
        fi, si, fi, si, 255, fast_costes=mode_name
    )
    np.testing.assert_allclose([thr_fi, thr_si], [thr_fi_ref, thr_si_ref], rtol=1e-6)


def _grid_obj(seed, n, scale):
    """Quantized grid object with ties, zeros and a positive or negative slope."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 0.1 * scale, n)
    rho = rng.choice([-0.5, 0.3, 0.7, 1.0])
    x = base + rng.normal(0, 0.05 * scale, n)
    y = rho * base + 0.05 * scale + rng.normal(0, 0.05 * scale, n)
    x[rng.random(n) < 0.2] = 0
    fi = np.clip(np.rint(x), 0, scale) / scale
    si = np.clip(np.rint(y), 0, scale) / scale
    return np.ascontiguousarray(fi), np.ascontiguousarray(si)


@requires_numba
@pytest.mark.parametrize("scale", [255, 65535])
@pytest.mark.parametrize("n", [3, 40, 2000])
@pytest.mark.parametrize("seed", range(6))
def test_sorted_searches_match_scan_kernels(seed, n, scale):
    """Sorted prefix searches reproduce the per-candidate scan bit for bit.

    The scale stays an ``int`` as in production (``_costes_pair``), so the
    integer-typed Numba specializations are the ones compiled and checked.
    """
    from cp_measure.core.numba import _costes as K

    fi, si = _grid_obj(seed, n, scale)
    if not ref._valid_costes_pixels(fi, si):
        pytest.skip("undefined object")
    a, b = K._regression_ab(fi, si, 0, n)
    if not np.isfinite(a):
        pytest.skip("undefined regression")
    if not a > 0:
        pytest.skip("non-positive slope dispatches to the scan kernels")
    np.testing.assert_array_equal(
        K._bisection_sorted(fi, si, 0, n, a, b, scale),
        K._bisection_scan(fi, si, 0, n, a, b, scale),
    )
    for accurate in (False, True):
        np.testing.assert_array_equal(
            K._linear_sorted(fi, si, 0, n, a, b, scale, accurate),
            K._linear_scan(fi, si, 0, n, a, b, scale, accurate),
        )


@requires_numba
def test_sorted_dispatch_requires_positive_slope_and_bounded_size():
    from cp_measure.core.numba import _costes as K

    assert K._use_sorted(0.5, 100)
    assert K._use_sorted(0.5, K._SORTED_MIN_PIXELS)
    assert not K._use_sorted(0.5, K._SORTED_MIN_PIXELS - 1)
    assert K._use_sorted(0.5, K._SORTED_MAX_PIXELS)
    assert not K._use_sorted(0.5, K._SORTED_MAX_PIXELS + 1)
    assert not K._use_sorted(0.0, 100)
    assert not K._use_sorted(-0.5, 100)


@requires_numba
def test_second_entry_is_exact_boundary():
    """Entry thresholds are the smallest float satisfying ``a*t + b > y``."""
    from cp_measure.core.numba._costes import _second_entry

    rng = np.random.default_rng(0)
    scratch = np.empty(1)
    cases = [(0.0, 1e-300, 0.0), (0.5, 1e300, -1e300), (1.0, 3.0, 0.999999999)]
    for _ in range(2000):
        y = rng.random()
        a = 10 ** rng.uniform(-8, 8)
        # b ≈ y cancels in (y - b) / a and forces the exact bisection path.
        b = y * (1 + rng.normal(0, 1e-12)) if rng.random() < 0.5 else rng.normal()
        cases.append((y, a, b))
    for y, a, b in cases:
        t = _second_entry(y, a, b, scratch)
        assert a * t + b > y
        assert not (a * np.nextafter(t, -np.inf) + b > y)


def _borderline_obj(kind, seed):
    rng = np.random.default_rng(seed)
    n = 400
    if kind == "near_constant":
        # Background varies at ~1e-14, around SciPy's near-constant diagnostic.
        fi = 0.5 + rng.normal(0, 1e-14, n)
        si = 0.4 + rng.normal(0, 1e-14, n)
        hot = rng.random(n) < 0.3
        fi[hot] = rng.uniform(0.6, 1.0, hot.sum())
        si[hot] = fi[hot] - 0.1
    else:
        # Exactly uncorrelated lattice: r == 0 for the full background.
        g = np.linspace(0.05, 0.45, 20)
        fi, si = (v.ravel() for v in np.meshgrid(g, g))
        hot = np.linspace(0.6, 1.0, 40)
        fi = np.concatenate([fi, hot])
        si = np.concatenate([si, hot - 0.05])
    fi = np.clip(fi, 0, 1)
    si = np.clip(si, 0, 1)
    return np.ascontiguousarray(fi), np.ascontiguousarray(si)


@requires_numba
@pytest.mark.parametrize("kind", ["near_constant", "zero_r"])
@pytest.mark.parametrize("seed", range(3))
def test_prefix_pearson_reproduces_scan_borderline_decisions(kind, seed):
    """Ambiguous prefix r falls back to the scan's own order and thresholds."""
    from cp_measure.core.numba import _costes as K

    fi, si = _borderline_obj(kind, seed)
    n = fi.size
    a, b = K._regression_ab(fi, si, 0, n)
    assert a > 0
    entry, xs, ys, stats, mx, my, x_run, y_run, _, _ = K._sorted_background(
        fi, si, 0, n, a, b
    )
    for t in np.linspace(0.01, 1.0, 400):
        thr_si = a * t + b
        cnt, scan = K._count_pearson_combt(fi, si, 0, n, t, thr_si)
        assert np.searchsorted(entry, t, side="right") == cnt
        got = K._prefix_pearson(
            cnt, xs, ys, stats, mx, my, x_run, y_run, fi, si, 0, n, t, thr_si
        )
        if not np.isfinite(scan):
            # NaN (undefined) and inf (reference replay) must agree exactly.
            assert got == scan or (np.isnan(got) and np.isnan(scan))
        else:
            assert np.isfinite(got)
            for cutoff in K._CUTOFFS:
                assert (got <= cutoff) == (scan <= cutoff)
                assert (got < cutoff) == (scan < cutoff)


@requires_numba
@pytest.mark.parametrize("scale", [3, 255, 4095, 65535])
@pytest.mark.parametrize("n", [3, 40, 2000])
@pytest.mark.parametrize("seed", range(6))
def test_grid_bisection_matches_scan_kernel(seed, n, scale):
    """Grid-bucketed bisection reproduces the per-candidate scan bit for bit,
    including objects too small for production to dispatch to it."""
    from cp_measure.core.numba import _costes as K

    fi, si = _grid_obj(seed, n, scale)
    if not ref._valid_costes_pixels(fi, si):
        pytest.skip("undefined object")
    a, b = K._regression_ab(fi, si, 0, n)
    if not (np.isfinite(a) and a > 0):
        pytest.skip("grid search requires a positive slope")
    np.testing.assert_array_equal(
        K._bisection_grid(fi, si, 0, n, a, b, scale),
        K._bisection_scan(fi, si, 0, n, a, b, scale),
    )


@requires_numba
@pytest.mark.parametrize("kind", ["near_constant", "zero_r"])
@pytest.mark.parametrize("seed", range(3))
def test_grid_count_pearson_reproduces_scan_decisions(kind, seed):
    """Every grid step, including padded out-of-range mids, matches the scan."""
    from cp_measure.core.numba import _costes as K

    fi, si = _borderline_obj(kind, seed)
    n = fi.size
    a, b = K._regression_ab(fi, si, 0, n)
    scale = 255
    grid = K._grid_background(fi, si, 0, n, a, b, scale)
    k_lo, k_hi = grid[4], grid[5]
    for mid in np.arange(k_lo - 3, k_hi + 4):
        thr_fi = mid / scale
        cnt, scan = K._count_pearson_combt(fi, si, 0, n, thr_fi, a * thr_fi + b)
        got_cnt, got = K._grid_count_pearson(*grid[:6], mid, fi, si, 0, n, a, b, scale)
        assert got_cnt == cnt
        if cnt <= 2:
            # Bisection never reads r here; the grid skips computing it.
            assert np.isnan(got)
        elif not np.isfinite(scan):
            assert got == scan or (np.isnan(got) and np.isnan(scan))
        else:
            assert np.isfinite(got)
            for cutoff in K._CUTOFFS:
                assert (got < cutoff) == (scan < cutoff)


@requires_numba
def test_grid_entry_is_first_background_step():
    """``_grid_entry`` returns the first integer step where the scan's test holds."""
    from cp_measure.core.numba import _costes as K

    rng = np.random.default_rng(0)
    k_lo, k_hi = -5.0, 300.0
    for _ in range(3000):
        scale = int(rng.choice([3, 255, 256]))
        x, y = rng.random(2)
        if rng.random() < 0.2:
            x = np.nan
        a = 10 ** rng.uniform(-6, 6)
        # b ≈ y cancels in (y - b) / a and forces the exact bisection path.
        b = y * (1 + rng.normal(0, 1e-12)) if rng.random() < 0.5 else rng.normal()
        k = K._grid_entry(x, y, a, b, scale, k_lo, k_hi)
        bg = lambda m: K._grid_background_at(x, y, m, a, b, scale)  # noqa: E731
        assert k_lo <= k <= k_hi + 1
        if k <= k_hi:
            assert bg(k)
        if k > k_lo:
            assert not bg(k - 1)
    # Pixels beyond the padded range land in the "never" row, not past it.
    assert K._grid_entry(1000.0, 1000.0, 1.0, 0.0, 3, k_lo, k_hi) == k_hi + 1


@requires_numba
def test_grid_dispatch_requires_positive_slope_and_enough_pixels():
    from cp_measure.core.numba import _costes as K

    assert K._use_grid(0.5, 10_000, 255)
    assert not K._use_grid(0.5, 100, 255)
    assert not K._use_grid(0.5, 10_000, 65535)
    assert K._use_grid(0.5, 100_000, 65535)
    assert not K._use_grid(0.0, 10_000, 255)
    assert not K._use_grid(-0.5, 10_000, 255)
    assert not K._use_grid(0.5, 1 << 30, K._GRID_MAX_SCALE + 1)


@requires_numba
@pytest.mark.parametrize(
    ("n", "scale", "slope", "expected"),
    [
        (10_000, 255, 1, "grid"),
        (2_000, 65535, 1, "sorted"),
        (10, 255, 1, "scan"),
        (10_000, 255, -1, "scan"),
    ],
)
def test_public_dispatchers_route_to_expected_kernel(n, scale, slope, expected):
    """_bisection/_linear pick the kernel their dispatch rules promise.

    Each kernel is checked by exact agreement with the one it should route to,
    plus the dispatch predicates for the object's actual slope and size.
    """
    from cp_measure.core.numba import _costes as K

    rng = np.random.default_rng(n + scale)
    base = rng.gamma(2.0, 0.1, n)
    fi = np.clip(np.rint(base * scale), 0, scale) / scale
    y = 0.6 + slope * 0.7 * base + rng.normal(0, 0.03, n)
    si = np.clip(np.rint(y * scale), 0, scale) / scale
    fi, si = np.ascontiguousarray(fi), np.ascontiguousarray(si)
    a, b = K._regression_ab(fi, si, 0, n)
    assert np.isfinite(a) and (a > 0) == (slope > 0)

    use_grid = K._use_grid(a, n, scale)
    use_sorted = K._use_sorted(a, n)
    routed = "grid" if use_grid else ("sorted" if use_sorted else "scan")
    assert routed == expected

    bisect = {
        "grid": K._bisection_grid,
        "sorted": K._bisection_sorted,
        "scan": K._bisection_scan,
    }[expected]
    np.testing.assert_array_equal(
        K._bisection(fi, si, 0, n, a, b, scale), bisect(fi, si, 0, n, a, b, scale)
    )
    linear = K._linear_sorted if use_sorted else K._linear_scan
    for accurate in (False, True):
        np.testing.assert_array_equal(
            K._linear(fi, si, 0, n, a, b, scale, accurate),
            linear(fi, si, 0, n, a, b, scale, accurate),
        )
