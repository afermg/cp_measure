"""Permissive threshold boundaries, independent CP fixtures, and backend parity."""

import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from cp_measure._detect import HAS_NUMBA
from cp_measure.core import measurecolocalization as ref

FIXTURES = json.loads(
    (Path(__file__).parent / "data/costes_cellprofiler_boundaries.json").read_text()
)["cases"]


def case_id(case):
    return f"{case['name']}-{case['scale_max']}-{case['mode']}"


@pytest.mark.parametrize("case", FIXTURES, ids=case_id)
def test_thresholds_match_unmodified_cp_boundary_fixtures(case):
    first, second = np.array(case["first"]), np.array(case["second"])
    helper = ref.bisection_costes if case["mode"] == ref.M_FASTER else ref.linear_costes
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        thresholds = helper(None, None, first, second, case["scale_max"], case["mode"])
    assert not recorded
    np.testing.assert_allclose(thresholds, case["thresholds"], rtol=1e-10, atol=1e-12)
    selected = (first < thresholds[0]) | (second < thresholds[1])
    assert selected.sum() == case["background_count"]


@pytest.mark.parametrize("case", FIXTURES, ids=case_id)
@pytest.mark.parametrize("backend", ["single", "numpy", "numba"])
def test_boundary_coefficients_are_not_vetoed(case, backend):
    if backend == "numba":
        if not HAS_NUMBA:
            pytest.skip("numba not installed")
        from cp_measure.core.numba.measurecolocalization import get_correlation_costes

        function = get_correlation_costes
    elif backend == "single":
        function = ref.get_correlation_costes_ind
    else:
        function = ref.get_correlation_costes
    first = np.array(case["first"]).reshape(1, -1)
    second = np.array(case["second"]).reshape(1, -1)
    masks = np.ones(first.shape, dtype=bool if backend == "single" else np.int32)
    for array in (first, second, masks):
        array.setflags(write=False)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        actual = function(
            first, second, masks, scale_max=case["scale_max"], fast_costes=case["mode"]
        )
    expected = np.array([np.nan if x is None else x for x in case["fractions"]])
    np.testing.assert_allclose(
        np.array(list(actual.values())).ravel(), expected, rtol=1e-12, atol=1e-12
    )
    if np.isnan(expected).any():
        # Label APIs count the objects per reason; the single-object API does not.
        count = "" if backend == "single" else " for 1 object"
        assert [(w.category, str(w.message)) for w in recorded] == [
            (
                RuntimeWarning,
                f"Undefined Costes result{count}: zero above-threshold intensity "
                "denominator",
            )
        ]
    else:
        assert not recorded


def test_fixtures_cover_the_removed_vetoes():
    faster = [c for c in FIXTURES if c["mode"] == ref.M_FASTER]
    finite = [c for c in faster if None not in c["fractions"]]
    assert {0, 1, 2} <= {c["background_count"] for c in finite}
    assert any(c["background_constant"] and c["background_count"] >= 3 for c in finite)
    assert any(
        c["name"] == "no_nonnegative_candidate" and c["thresholds"][0] == 0
        for c in finite
    )


def test_no_nonnegative_candidate_keeps_cp_boundary_fallback(monkeypatch):
    # Record actual candidate correlations, rather than inferring the cause from
    # a finite coefficient or a zero threshold alone.
    case = next(
        c
        for c in FIXTURES
        if c["name"] == "no_nonnegative_candidate" and c["mode"] == ref.M_FASTER
    )
    original = ref._costes_background_correlation
    correlations = []

    def recording(*args):
        value = original(*args)
        correlations.append(value)
        return value

    monkeypatch.setattr(ref, "_costes_background_correlation", recording)
    thresholds = ref.bisection_costes(
        None, None, np.array(case["first"]), np.array(case["second"]), case["scale_max"]
    )
    assert correlations and all(r < 0 for r in correlations)
    assert thresholds[0] == 0
    np.testing.assert_allclose(thresholds, case["thresholds"])


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("second,expected", [([0.1, 0.4], 1.0), ([0.4, 0.1], -1.0)])
def test_two_pair_linear_correlation_is_exact(second, expected):
    from cp_measure.core.numba._costes import _pearson_combt

    first, second = np.array([0.2, 0.3]), np.array(second)
    assert (
        ref._costes_background_correlation(first, second, np.ones(2, bool)) == expected
    )
    no_cutoffs, defer = np.zeros(0), True
    assert (
        _pearson_combt(first, second, 0, 2, np.inf, np.inf, no_cutoffs, defer)
        == expected
    )
