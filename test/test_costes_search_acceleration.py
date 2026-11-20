import numpy as np
import pytest

from cp_measure._detect import HAS_NUMBA
from cp_measure.core import measurecolocalization as ref

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


def _same_decisions(actual, expected, cutoffs):
    if not np.isfinite(expected):
        assert actual == expected or (np.isnan(actual) and np.isnan(expected))
    else:
        assert np.isfinite(actual)
        for cutoff in cutoffs:
            assert (actual < cutoff) == (expected < cutoff)
            assert (actual <= cutoff) == (expected <= cutoff)


@requires_numba
@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("kind", ["near_constant", "zero_r"])
@pytest.mark.parametrize("seed", range(3))
def test_sorted_and_grid_preserve_fused_boundary_policy(kind, seed, defer):
    from cp_measure.core.numba import _costes as kernel
    from cp_measure.primitives._segment_numba import numpy_sum_chunk

    first, second = _borderline_obj(kind, seed)
    size = first.size
    slope, intercept = kernel._regression_ab(first, second, 0, size, numpy_sum_chunk())
    assert slope > 0
    table = kernel._sweep_prep(first, second, slope, intercept)
    totals = table[0]
    grid = kernel._grid_background(
        first, second, 0, size, slope, intercept, 255, totals
    )
    for step in np.arange(grid[4] - 3, grid[5] + 4):
        threshold = step / 255
        other = slope * threshold + intercept
        expected_count, expected = kernel._count_pearson_combt(
            first, second, 0, size, threshold, other, kernel._SIGN_CUTOFF, defer
        )
        count, actual = kernel._grid_count_pearson(
            first, second, 0, size, slope, intercept, 255, step, grid, totals, defer
        )
        assert count == expected_count
        if count <= 2:
            assert np.isnan(actual)
        else:
            _same_decisions(actual, expected, kernel._SIGN_CUTOFF)
        expected_count, expected = kernel._count_pearson_combt(
            first, second, 0, size, threshold, other, kernel._FAST_CUTOFFS, defer
        )
        count, actual = kernel._sweep_count_pearson(
            first,
            second,
            0,
            size,
            slope,
            intercept,
            table,
            threshold,
            other,
            kernel._FAST_CUTOFFS,
            defer,
        )
        assert count == expected_count
        _same_decisions(actual, expected, kernel._FAST_CUTOFFS)


@requires_numba
@pytest.mark.parametrize(
    "slope,intercept", [(0.1, 0.5), (1.0, 0.0), (1e-12, 0.5), (-0.7, 0.9)]
)
def test_sweep_membership_at_float_neighbors(slope, intercept):
    from cp_measure.core.numba import _costes as kernel

    first = np.linspace(0, 1, 40)
    second = slope * first + intercept
    table = kernel._sweep_prep(first, second, slope, intercept)
    for threshold in first:
        for candidate in (
            np.nextafter(threshold, -np.inf),
            threshold,
            np.nextafter(threshold, np.inf),
        ):
            other = slope * candidate + intercept
            expected_count, expected = kernel._count_pearson_combt(
                first,
                second,
                0,
                first.size,
                candidate,
                other,
                kernel._FAST_CUTOFFS,
                True,
            )
            count, actual = kernel._sweep_count_pearson(
                first,
                second,
                0,
                first.size,
                slope,
                intercept,
                table,
                candidate,
                other,
                kernel._FAST_CUTOFFS,
                True,
            )
            assert count == expected_count
            _same_decisions(actual, expected, kernel._FAST_CUTOFFS)


@requires_numba
@pytest.mark.parametrize("scale", [3, 255, 4095, 65535])
@pytest.mark.parametrize("size", [3, 40, 2000])
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("defer", [False, True])
def test_bisection_paths_preserve_scan_state(monkeypatch, scale, size, seed, defer):
    from cp_measure.core.numba import _costes as kernel
    from cp_measure.primitives._segment_numba import numpy_sum_chunk

    first, second = _grid_obj(seed, size, scale)
    if ref._invalid_costes_reason(first, second) is not None:
        pytest.skip("undefined object")
    slope, intercept = kernel._regression_ab(first, second, 0, size, numpy_sum_chunk())
    if not (np.isfinite(slope) and slope > 0):
        pytest.skip("positive finite slope required")
    totals = kernel._moment_totals(first, second, 0, size)
    compiled_state = kernel._bisection_state(slope, intercept, scale)
    compiled_code = kernel._bisection_steps(
        first, second, 0, size, scale, compiled_state, totals, defer
    )
    states, codes = [], []
    for path in ("scan", "grid"):
        with monkeypatch.context() as patch:
            patch.setattr(kernel, "_use_grid", lambda *args: path == "grid")
            if path == "scan":
                patch.setattr(
                    kernel,
                    "_direct_count_pearson",
                    lambda first, second, lo, hi, totals, threshold, other, cutoffs, defer: (
                        kernel._count_pearson_combt(
                            first, second, lo, hi, threshold, other, cutoffs, defer
                        )
                    ),
                )
            state = kernel._bisection_state(slope, intercept, scale)
            code = kernel._bisection_steps.py_func(
                first, second, 0, size, scale, state, totals, defer
            )
            states.append(state)
            codes.append(code)
    assert codes[0] == codes[1]
    assert compiled_code == codes[0]
    np.testing.assert_array_equal(compiled_state, states[0])
    np.testing.assert_array_equal(states[0], states[1])


@requires_numba
@pytest.mark.parametrize("scale", [255, 4095, 65535])
@pytest.mark.parametrize("size", [3, 40, 2000])
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("accurate", [False, True])
@pytest.mark.parametrize("defer", [False, True])
def test_linear_sweep_preserves_scan_thresholds(
    monkeypatch, scale, size, seed, accurate, defer
):
    from cp_measure.core.numba import _costes as kernel
    from cp_measure.primitives._segment_numba import numpy_sum_chunk

    first, second = _grid_obj(seed, size, scale)
    if ref._invalid_costes_reason(first, second) is not None:
        pytest.skip("undefined object")
    slope, intercept = kernel._regression_ab(first, second, 0, size, numpy_sum_chunk())
    if not (np.isfinite(slope) and slope != 0):
        pytest.skip("finite nonzero slope required")
    actual = kernel._linear(
        first, second, 0, size, slope, intercept, scale, accurate, defer
    )
    monkeypatch.setattr(kernel, "_SWEEP_AFTER", np.inf)
    expected = kernel._linear.py_func(
        first, second, 0, size, slope, intercept, scale, accurate, defer
    )
    np.testing.assert_array_equal(actual, expected)


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

    assert K._use_grid(0.5, K._GRID_MIN_PIXELS, 255)
    assert not K._use_grid(0.5, K._GRID_MIN_PIXELS - 1, 255)
    assert not K._use_grid(0.5, 100, 255)
    assert not K._use_grid(0.5, 10_000, 65535)
    assert K._use_grid(0.5, 100_000, 65535)
    assert not K._use_grid(0.0, 10_000, 255)
    assert not K._use_grid(-0.5, 10_000, 255)
    assert not K._use_grid(0.5, 1 << 30, K._GRID_MAX_SCALE + 1)
