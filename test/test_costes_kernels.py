"""Costes compiled search and NumPy regression tests against current main.

Explicit scale=255 exercises genuine multi-iteration searches. Production API
coverage, including undefined replay and normalization, lives in
``test_costes_backend.py``.
"""

import numpy as np
import pytest
import scipy.stats

import cp_measure.core.measurecolocalization as ref
from cp_measure.primitives._segment_numba import numpy_sum_chunk
from cp_measure._detect import HAS_NUMBA

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


def _obj(seed, n=200):
    """A correlated float object in [0,1]: si ≈ fi + noise, both clipped."""
    rng = np.random.default_rng(seed)
    fi = rng.random(n)
    si = np.clip(0.7 * fi + 0.3 * rng.random(n), 0.0, 1.0)
    return np.ascontiguousarray(fi), np.ascontiguousarray(si)


@requires_numba
@pytest.mark.parametrize("n", [200, 9000])
@pytest.mark.parametrize("seed", [0, 1])
def test_regression_ab_equals_reference_bitwise(seed, n):
    """The kernel's regression line equals the reference's NumPy expression bit for
    bit (#118: the line decides which pixels count), past NumPy's 8192 chunking."""
    from cp_measure.core.numba._costes import _regression_ab

    fi, si = _obj(seed, n=n)
    fi[::7] = 0.0  # some pixels drop out of the nonzero subset
    si[::7] = 0.0
    a, b = _regression_ab(fi, si, 0, fi.size, numpy_sum_chunk())
    # The reference's expression (bisection_costes / linear_costes).
    nz = (fi > 0) | (si > 0)
    fz, sz = fi[nz], si[nz]
    xvar = np.var(fz, axis=0, ddof=1)
    yvar = np.var(sz, axis=0, ddof=1)
    zvar = np.var(fz + sz, axis=0, ddof=1)
    covar = 0.5 * (zvar - (xvar + yvar))
    num = (yvar - xvar) + np.sqrt((yvar - xvar) * (yvar - xvar) + 4 * (covar * covar))
    a_ref = num / (2 * covar)
    b_ref = np.mean(sz, axis=0) - a_ref * np.mean(fz, axis=0)
    assert (a, b) == (a_ref, b_ref)


@requires_numba
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pearson_combt_matches_scipy(seed):
    from cp_measure.core.numba._costes import _FAST_CUTOFFS, _pearson_combt

    fi, si = _obj(seed)
    thr_fi, thr_si = 0.4, 0.35
    got = _pearson_combt(fi, si, 0, fi.size, thr_fi, thr_si, _FAST_CUTOFFS, True)
    combt = (fi < thr_fi) | (si < thr_si)
    r_ref, _ = scipy.stats.pearsonr(fi[combt], si[combt])
    np.testing.assert_allclose(got, r_ref, rtol=1e-9, atol=1e-12)
    # full-block (inf thresholds) == pearson over everything
    got_full = _pearson_combt(fi, si, 0, fi.size, np.inf, np.inf, _FAST_CUTOFFS, True)
    np.testing.assert_allclose(got_full, scipy.stats.pearsonr(fi, si)[0], rtol=1e-9)


@requires_numba
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_bisection_matches_reference(seed):
    from cp_measure.core.numba._costes import _bisection, _regression_ab

    fi, si = _obj(seed)
    a, b = _regression_ab(fi, si, 0, fi.size, numpy_sum_chunk())
    thr_fi, thr_si, _ = _bisection(fi, si, 0, fi.size, a, b, 255.0)
    thr_fi_ref, thr_si_ref = ref.bisection_costes(fi, si, fi, si, 255)
    np.testing.assert_allclose([thr_fi, thr_si], [thr_fi_ref, thr_si_ref], rtol=1e-6)


@requires_numba
@pytest.mark.parametrize("mode_name", ["Fast", "Accurate"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_linear_matches_reference(seed, mode_name):
    from cp_measure.core.numba._costes import _linear, _regression_ab

    fi, si = _obj(seed)
    a, b = _regression_ab(fi, si, 0, fi.size, numpy_sum_chunk())
    thr_fi, thr_si, _ = _linear(
        fi, si, 0, fi.size, a, b, 255.0, mode_name == "Accurate", True
    )
    thr_fi_ref, thr_si_ref = ref.linear_costes(
        fi, si, fi, si, 255, fast_costes=mode_name
    )
    np.testing.assert_allclose([thr_fi, thr_si], [thr_fi_ref, thr_si_ref], rtol=1e-6)


@requires_numba
@pytest.mark.parametrize("mode_name", ["Fast", "Accurate"])
@pytest.mark.parametrize("seed", [0, 1])
def test_linear_long_walk_matches_reference(seed, mode_name):
    """A 12-bit search far longer than _SWEEP_AFTER steps, so it finishes on the
    prefix-sum path; thresholds must still equal the reference's exactly."""
    from cp_measure.core.numba._costes import _SWEEP_AFTER, _linear, _regression_ab

    scale = 4095
    fi, si = _obj(seed, n=1500)
    fi, si = np.round(fi * scale) / scale, np.round(si * scale) / scale
    a, b = _regression_ab(fi, si, 0, fi.size, numpy_sum_chunk())
    got = _linear(
        fi, si, 0, fi.size, a, b, float(scale), mode_name == "Accurate", True
    )[:2]
    want = ref.linear_costes(fi, si, fi, si, scale, fast_costes=mode_name)
    assert (max(fi.max(), si.max()) - got[0]) * scale > 2 * _SWEEP_AFTER
    np.testing.assert_array_equal(got, want)


@requires_numba
@pytest.mark.parametrize("n", [50, 800, 5000])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bisection_equals_reference_on_16bit_objects(n, seed):
    """Bisection answers its steps from one-pass moments and falls back to the exact
    kernel near a decision; the chosen thresholds must equal the reference's exactly."""
    from cp_measure.core.numba._costes import _bisection, _regression_ab

    scale = 65535
    fi, si = _obj(seed, n=n)
    fi, si = np.round(fi * scale) / scale, np.round(si * scale) / scale
    a, b = _regression_ab(fi, si, 0, fi.size, numpy_sum_chunk())
    got = _bisection(fi, si, 0, fi.size, a, b, float(scale))[:2]
    np.testing.assert_array_equal(got, ref.bisection_costes(fi, si, fi, si, scale))


@requires_numba
def test_bisection_ignores_linear_step_cutoffs():
    """r is exactly 0.25 here, a Fast-mode step point that bisection never branches
    on, so Faster mode measures the object instead of replaying the reference."""
    from cp_measure.core.numba._costes import REPLAY, costes_faster

    fi = np.array([3, 5, 4, 5, 3, 6, 6, 6, 6, 4]) / 255
    si = np.array([5, 5, 3, 5, 4, 5, 4, 3, 6, 2]) / 255
    c1, c2, status, _ = costes_faster(
        fi.copy(), si.copy(), np.array([0, 10]), 255.0, True, numpy_sum_chunk()
    )
    assert status[0] != REPLAY
    expected = ref.get_correlation_costes(
        fi[None], si[None], np.ones((1, 10), int), scale_max=255
    )
    np.testing.assert_array_equal(
        [c1[0], c2[0]],
        [expected["Correlation_Costes_1"][0], expected["Correlation_Costes_2"][0]],
    )
