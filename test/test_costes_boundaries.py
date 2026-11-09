"""Small regressions for permissive Costes boundaries; no external data."""

import warnings

import numpy as np
import pytest

from cp_measure._detect import HAS_NUMBA
from cp_measure.core import measurecolocalization as ref


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("mode", [ref.M_FASTER, ref.M_FAST, ref.M_ACCURATE])
def test_small_background_keeps_finite_coefficients(backend, mode):
    if backend == "numba":
        if not HAS_NUMBA:
            pytest.skip("numba not installed")
        from cp_measure.core.numba import measurecolocalization as module
    else:
        module = ref
    first = np.array([[2, 3, 10, 12, 20, 30, 31, 34]]) / 255
    second = np.array([[4, 7, 9, 16, 17, 33, 28, 39]]) / 255
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = module.get_correlation_costes(
            first, second, np.ones(first.shape, int), scale_max=255, fast_costes=mode
        )
    # Checked against CP 4.2.8 on the same pixels. Final background has two
    # pixels (Faster) or one (Fast/Accurate), but coefficients remain defined.
    expected = [1.0, 1.0 if mode == ref.M_FASTER else 149 / 153]
    np.testing.assert_allclose(np.array(list(result.values())).ravel(), expected)
    assert not recorded


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("mode", [ref.M_FASTER, ref.M_FAST, ref.M_ACCURATE])
def test_no_candidate_fallback_still_checks_denominators(backend, mode):
    if backend == "numba":
        if not HAS_NUMBA:
            pytest.skip("numba not installed")
        from cp_measure.core.numba import measurecolocalization as module
    else:
        module = ref
    first = np.array([[25, 19, 15, 8, 9, 2, 3, 1]]) / 255
    second = np.array([[6, 24, 19, 27, 15, 18, 29, 22]]) / 255
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = module.get_correlation_costes(
            first, second, np.ones(first.shape, int), scale_max=255, fast_costes=mode
        )
    values = np.array(list(result.values())).ravel()
    if mode == ref.M_FASTER:
        np.testing.assert_allclose(values, [3 / 82, 1.0])
        assert not recorded
    else:
        assert np.isnan(values).all()
        assert len(recorded) == 1
        assert "zero above-threshold intensity denominator" in str(recorded[0].message)
