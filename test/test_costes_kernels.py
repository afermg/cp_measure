"""Unit tests for the costes numba kernels.

The reference ``bisection_costes`` / ``linear_costes`` accept ``scale_max`` as a
parameter, so the REAL multi-iteration search can be exercised at scale=255 on
float pixels in [0,1] — without the integer dtype that would make the reference
overflow ``z = fi + si``. This is where the control flow actually runs (the
end-to-end float64 path has scale=1 and a near-trivial search).
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
