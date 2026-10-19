"""Scale normalization, upstream threshold fixtures, and undefined Costes inputs."""

import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from cp_measure.core import measurecolocalization as cp

FIXTURES = json.loads(
    (Path(__file__).parent / "data/costes_cellprofiler_reference.json").read_text()
)["cases"]


def arrays(case):
    dtype = np.uint8 if case["scale_max"] == 255 else np.uint16
    a = np.array(case["raw_first"], dtype=dtype).reshape(8, 16)
    b = np.array(case["raw_second"], dtype=dtype).reshape(8, 16)
    return a, b


@pytest.mark.parametrize(
    "case", FIXTURES, ids=lambda c: f"seed{c['seed']}-{c['scale_max']}-{c['mode']}"
)
def test_cellprofiler_threshold_reference(case):
    a, b = arrays(case)
    scale, mode = case["scale_max"], case["mode"]
    x, y = a.astype(float) / scale, b.astype(float) / scale
    helper = cp.bisection_costes if mode == cp.M_FASTER else cp.linear_costes
    thresholds = helper(None, None, x.ravel(), y.ravel(), scale, mode)
    np.testing.assert_allclose(thresholds, case["thresholds"], rtol=1e-10, atol=1e-12)
    got = cp.get_correlation_costes_ind(
        x, y, np.ones(a.shape, bool), fast_costes=mode, scale_max=scale
    )
    np.testing.assert_allclose(
        list(got.values()), case["fractions"], rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("case", FIXTURES)
def test_integer_float_and_single_multi_agree_without_mutation(case):
    a, b = arrays(case)
    before_a, before_b = a.copy(), b.copy()
    mask = np.ones(a.shape, dtype=bool)
    kwargs = {"fast_costes": case["mode"]}
    integer = cp.get_correlation_costes_ind(a, b, mask, **kwargs)
    floating = cp.get_correlation_costes_ind(
        a.astype(float) / case["scale_max"],
        b.astype(float) / case["scale_max"],
        mask,
        scale_max=case["scale_max"],
        **kwargs,
    )
    multiple = cp.get_correlation_costes(a, b, mask.astype(int), **kwargs)
    np.testing.assert_allclose(
        list(integer.values()), list(floating.values()), rtol=1e-12
    )
    np.testing.assert_allclose(
        list(integer.values()), np.array(list(multiple.values())).ravel(), rtol=1e-12
    )
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)


@pytest.mark.parametrize(
    "public", [cp.get_correlation_costes_ind, cp.get_correlation_costes]
)
def test_normalized_floats_require_explicit_scale(public):
    a, b = arrays(FIXTURES[0])
    mask = np.ones_like(
        a, dtype=bool if public is cp.get_correlation_costes_ind else int
    )
    with pytest.raises(ValueError, match="explicit scale_max"):
        public(a.astype(float) / 255, b.astype(float) / 255, mask)


@pytest.mark.parametrize("scale", [None, 0, 1, -1, True, 255.0, np.nan, np.inf, 2**32])
def test_invalid_float_grid_rejected(scale):
    with pytest.raises(ValueError, match="scale_max"):
        cp.get_correlation_costes_ind(
            np.arange(4) / 4, np.arange(4) / 4, np.ones(4, bool), scale_max=scale
        )


def test_lower_bit_depth_in_uint16_container():
    a, b = arrays(FIXTURES[0])
    a, b = a.astype(np.uint16) * 16, b.astype(np.uint16) * 16
    integer = cp.get_correlation_costes_ind(
        a, b, np.ones(a.shape, bool), scale_max=4095
    )
    floating = cp.get_correlation_costes_ind(
        a.astype(float) / 4095,
        b.astype(float) / 4095,
        np.ones(a.shape, bool),
        scale_max=4095,
    )
    np.testing.assert_allclose(
        list(integer.values()), list(floating.values()), equal_nan=True
    )


def test_volumetric_and_bulk_gapped_labels():
    from cp_measure.bulk import get_correlation_measurements

    a, b = arrays(FIXTURES[0])
    a, b = np.stack([a, a]), np.stack([b, b])
    labels = np.full(a.shape, 42, dtype=np.int32)
    got = get_correlation_measurements()["costes"](a, b, labels)
    expected = cp.get_correlation_costes_ind(a, b, labels == 42)
    np.testing.assert_allclose(
        np.array(list(got.values())).ravel(), list(expected.values())
    )
    np.testing.assert_array_equal(labels, 42)


def test_featurizer_forwards_scale():
    from cp_measure.featurizer import featurize, make_featurizer_config
    from conftest import ALL_OFF

    a, b = arrays(FIXTURES[0])
    image = np.stack([a, b]).astype(float) / 255
    masks = np.ones((1, *a.shape), dtype=int)
    config = make_featurizer_config(
        ["a", "b"],
        **{**ALL_OFF, "correlation_costes": True},
        correlation_costes_params={"scale_max": 255},
    )
    data, columns, rows = featurize(image, masks, config)
    expected = cp.get_correlation_costes_ind(
        image[0], image[1], masks[0].astype(bool), scale_max=255
    )
    assert rows == [(None, "object", 1)]
    for key, value in expected.items():
        assert data[0, columns.index(key + "__a__b")] == pytest.approx(value)


@pytest.mark.parametrize(
    "a,b",
    [
        (np.zeros(8), np.zeros(8)),
        (np.ones(8) / 2, np.arange(8) / 8),
        (np.arange(2) / 2, np.arange(2) / 2),
        (np.array([0.0, 0.1, np.nan, 0.8]), np.arange(4) / 4),
        (np.array([0.0, 0.1, np.inf, 0.8]), np.arange(4) / 4),
        (np.array([0.1, 0.1, 0.9, 0.9]), np.array([0.1, 0.9, 0.1, 0.9])),
        (np.array([0.0, 1, 0, 1, 0, 1]), np.array([1.0, 0, 1, 0, 1, 0])),
    ],
)
@pytest.mark.parametrize("mode", [cp.M_FASTER, cp.M_FAST, cp.M_ACCURATE])
def test_undefined_signal_warns_and_returns_nan(a, b, mode):
    with pytest.warns(RuntimeWarning, match="^Undefined Costes result:") as recorded:
        result = cp.get_correlation_costes_ind(
            a, b, np.ones(a.shape, bool), scale_max=255, fast_costes=mode
        )
    assert all(w.category is RuntimeWarning for w in recorded)
    assert np.isnan(list(result.values())).all()


def test_pixels_outside_mask_are_ignored():
    a, b = arrays(FIXTURES[0])
    a, b = a.astype(float) / 255, b.astype(float) / 255
    original = cp.get_correlation_costes_ind(
        a, b, np.ones(a.shape, bool), scale_max=255
    )
    padded_a = np.pad(a, 1, constant_values=np.nan)
    padded_b = np.pad(b, 1, constant_values=999)
    mask = np.pad(np.ones(a.shape, bool), 1)
    got = cp.get_correlation_costes_ind(padded_a, padded_b, mask, scale_max=255)
    np.testing.assert_allclose(list(got.values()), list(original.values()))


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_invalid_normalized_range_rejected(bad):
    a = np.array([bad, 0.1, 0.3, 0.7])
    with pytest.raises(ValueError, match="normalized pixels"):
        cp.get_correlation_costes_ind(a, a, np.ones(4, bool), scale_max=255)


def test_shape_dtype_mode_and_helper_grid_guards():
    a = np.array([0.1, 0.2, 0.4, 0.7])
    with pytest.raises(ValueError, match="same shape"):
        cp.get_correlation_costes_ind(a, a[:3], np.ones(4, bool), scale_max=255)
    with pytest.raises(ValueError, match="boolean"):
        cp.get_correlation_costes_ind(a, a, np.ones(4, int), scale_max=255)
    with pytest.raises(ValueError, match="mode"):
        cp.get_correlation_costes_ind(
            a, a, np.ones(4, bool), scale_max=255, fast_costes="typo"
        )
    with pytest.raises(ValueError, match="scale_max"):
        cp.bisection_costes(None, None, a, a, scale_max=1)


@pytest.mark.parametrize(
    "cutoff, at_or_below, above", [(0.25, 1, 2), (0.35, 2, 5), (0.45, 5, 10)]
)
@pytest.mark.parametrize("side", [-1, 0, 1], ids=["below", "exact", "above"])
def test_fast_linear_step_boundaries(cutoff, at_or_below, above, side):
    correlation = cutoff if side == 0 else np.nextafter(cutoff, side * np.inf)
    expected = above if side > 0 else at_or_below
    assert cp._costes_step_multiplier(correlation) == expected


def test_accurate_linear_search_never_uses_fast_step_schedule(monkeypatch):
    def unexpected_fast_step(_correlation):
        pytest.fail("Accurate mode must visit every threshold candidate")

    monkeypatch.setattr(cp, "_costes_step_multiplier", unexpected_fast_step)
    case = next(case for case in FIXTURES if case["mode"] == cp.M_ACCURATE)
    a, b = arrays(case)
    scale = case["scale_max"]
    thresholds = cp.linear_costes(
        None, None, a.ravel() / scale, b.ravel() / scale, scale, cp.M_ACCURATE
    )
    np.testing.assert_allclose(thresholds, case["thresholds"], rtol=1e-10, atol=1e-12)


def test_constant_candidates_never_reach_scipy_pearson(monkeypatch):
    original = cp.scipy.stats.pearsonr

    def guarded(a, b):
        assert len(a) >= 3 and np.ptp(a) > 0 and np.ptp(b) > 0
        return original(a, b)

    monkeypatch.setattr(cp.scipy.stats, "pearsonr", guarded)
    a = np.r_[np.zeros(12), np.arange(1, 7)] / 10
    b = np.r_[np.zeros(12), np.array([1, 3, 2, 5, 4, 6])] / 10
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        cp.get_correlation_costes_ind(a, b, np.ones(a.shape, bool), scale_max=255)
    assert all(
        w.category is RuntimeWarning
        and str(w.message).startswith("Undefined Costes result:")
        for w in recorded
    )
