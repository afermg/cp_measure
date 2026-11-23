"""Statistic-only Costes Pearson must preserve SciPy decisions and warnings."""

import warnings

import numpy as np
import pytest
from scipy.stats import pearsonr

from cp_measure.core.measurecolocalization import _costes_background_correlation


@pytest.mark.parametrize("size", [2, 3, 7, 64, 513, 4096])
@pytest.mark.parametrize(
    "kind", ["random", "correlated", "near_constant", "tiny", "tied"]
)
def test_background_statistic_matches_scipy(size, kind):
    rng = np.random.default_rng(9123 + size)
    x, y = rng.random((2, size))
    if kind == "correlated":
        y = 0.8 * x + 0.2 * y
    elif kind == "near_constant":
        x, y = 0.5 + x * 1e-15, 0.2 + y * 1e-15
    elif kind == "tiny":
        x, y = x * 1e-300, y * 1e-300
    elif kind == "tied":
        x, y = np.round(x * 5), np.round(y * 5)
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected = pearsonr(x, y).statistic if np.ptp(x) and np.ptp(y) else np.nan
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual = _costes_background_correlation(x, y, np.ones(size, bool))
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15, equal_nan=True)
    assert [(w.category, str(w.message)) for w in actual_warnings] == [
        (w.category, str(w.message)) for w in expected_warnings
    ]


@pytest.mark.parametrize("boundary", [0.0, 0.25, 0.35, 0.45])
@pytest.mark.parametrize("offset", [-1e-14, 0, 1e-14])
def test_boundary_decisions_exact(boundary, offset):
    x = np.array([-1.0, -1.0, 1.0, 1.0])
    orthogonal = np.array([-1.0, 1.0, -1.0, 1.0])
    r = boundary + offset
    y = r * x + np.sqrt(1 - r * r) * orthogonal
    x, y = x * 0.1 + 0.5, y * 0.1 + 0.5
    expected = pearsonr(x, y).statistic
    assert _costes_background_correlation(x, y, np.ones(4, bool)) == expected


def test_background_selection_and_invalid_subsets():
    x = np.array([0.0, 0.0, 0.5, 1.0])
    y = np.array([0.0, 1.0, 0.2, 0.0])
    for sel in (
        np.zeros(4, bool),
        np.array([1, 0, 0, 0], bool),
        np.array([1, 1, 0, 0], bool),
    ):
        assert np.isnan(_costes_background_correlation(x, y, sel))
    sel = np.array([0, 1, 1, 1], bool)
    assert _costes_background_correlation(x, y, sel) == pytest.approx(
        pearsonr(x[sel], y[sel]).statistic
    )
