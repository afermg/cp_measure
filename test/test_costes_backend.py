"""Current-main Costes API, undefined-result and batch contracts for Numba."""

import inspect
import warnings

import numpy as np
import pytest

import cp_measure
from cp_measure import bulk
from cp_measure._detect import HAS_NUMBA
from cp_measure.core import measurecolocalization as ref
from test_costes import FIXTURES, arrays

pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
MODES = (ref.M_FASTER, ref.M_FAST, ref.M_ACCURATE)


def parity(first, second, masks, **kwargs):
    from cp_measure.core.numba.measurecolocalization import get_correlation_costes

    original = [x.copy() for x in (first, second, masks)]
    results, diagnostics = [], []
    for function in (ref.get_correlation_costes, get_correlation_costes):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            results.append(function(first, second, masks, **kwargs))
        diagnostics.append([(w.category, str(w.message)) for w in recorded])
    expected, actual = results
    assert list(actual) == list(expected)
    assert diagnostics[0] == diagnostics[1]
    for key in expected:
        assert isinstance(actual[key], list)
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
        assert np.asarray(actual[key]).dtype == np.asarray(expected[key]).dtype
    for before, after in zip(original, (first, second, masks)):
        np.testing.assert_array_equal(before, after)
    return actual


@pytest.mark.parametrize("case", FIXTURES)
@pytest.mark.parametrize(
    "encoding", ["integer", "division", "float32", "signed", "mixed"]
)
def test_costes_production_scale_and_fixtures(case, encoding):
    a, b = arrays(case)
    scale = case["scale_max"]
    kwargs = {"fast_costes": case["mode"]}
    if encoding != "integer":
        kwargs["scale_max"] = scale
    if encoding in ("division", "float32"):
        dtype = np.float32 if encoding == "float32" else np.float64
        a, b = (a / scale).astype(dtype), (b / scale).astype(dtype)
    elif encoding == "signed":
        a, b = a.astype(np.int32), b.astype(np.int32)
    elif encoding == "mixed":
        b = b / scale
    parity(a, b, np.ones(a.shape, np.int32), **kwargs)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "case",
    [
        "zero",
        "constant",
        "tiny",
        "nan",
        "inf",
        "covariance",
        "anticorrelated",
        "gaps",
        "empty",
        "constant_background",
    ],
)
def test_costes_undefined_warnings_and_labels(mode, case):
    a = np.array([[0.1, 0.2, 0.4, 0.7], [0.3, 0.5, 0.8, 1.0]])
    b = a[:, ::-1].copy()
    masks = np.ones(a.shape, dtype=np.int32)
    if case == "zero":
        a[:] = b[:] = 0
    elif case == "constant":
        a[:] = 0.1
    elif case == "tiny":
        masks[:] = 0
        masks[0, :2] = 1
    elif case in ("nan", "inf"):
        a[0, 1] = np.nan if case == "nan" else np.inf
    elif case == "covariance":
        a = np.array([[0.1, 0.1, 0.9, 0.9]])
        b = np.array([[0.1, 0.9, 0.1, 0.9]])
        masks = np.ones(a.shape, dtype=int)
    elif case == "anticorrelated":
        b = 1 - a
    elif case == "gaps":
        masks[1] = 3
    elif case == "empty":
        masks[:] = 0
    elif case == "constant_background":
        a = np.r_[np.zeros(12), np.arange(1, 7)].reshape(3, 6) / 10
        b = np.r_[np.zeros(12), [1, 3, 2, 5, 4, 6]].reshape(3, 6) / 10
        masks = np.ones(a.shape, dtype=int)
    parity(a, b, masks, scale_max=255, fast_costes=mode)


@pytest.mark.parametrize("scale", [None, 0, 1, -1, True, 255.0, np.nan, np.inf, 2**32])
def test_costes_invalid_grid(scale):
    from cp_measure.core.numba.measurecolocalization import get_correlation_costes

    a = np.arange(8).reshape(2, 4) / 8
    with pytest.raises(ValueError, match="scale_max"):
        get_correlation_costes(a, a, np.ones(a.shape, int), scale_max=scale)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_costes_invalid_range(bad):
    from cp_measure.core.numba.measurecolocalization import get_correlation_costes

    a = np.array([[bad, 0.1, 0.3, 0.7]])
    with pytest.raises(ValueError, match="normalized pixels"):
        get_correlation_costes(a, a, np.ones(a.shape, int), scale_max=255)


def test_costes_signature_and_validation():
    from cp_measure.core.numba.measurecolocalization import get_correlation_costes

    assert inspect.signature(get_correlation_costes) == inspect.signature(
        ref.get_correlation_costes
    )
    a, b = arrays(FIXTURES[0])
    with pytest.raises(ValueError, match="mode"):
        get_correlation_costes(a, b, np.ones(a.shape, int), fast_costes="typo")
    with pytest.raises(ValueError, match="integer"):
        get_correlation_costes(a, b, np.ones(a.shape, bool))
    with pytest.raises(ValueError, match="same shape"):
        get_correlation_costes(a, b[None], np.ones(a.shape, int))
    with pytest.raises(ValueError, match="matching"):
        get_correlation_costes(a[None], b[None], np.ones(a.shape, int))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", ["2d", "singleton", "volume", "strided"])
def test_costes_spatial_inputs(mode, shape):
    a, b = arrays(FIXTURES[0])
    if shape == "singleton":
        a, b = a[None], b[None]
    elif shape == "volume":
        a, b = np.stack([a, a]), np.stack([b, b])
    elif shape == "strided":
        a, b = a[:, ::-2], b[:, ::-2]
    parity(a, b, np.ones(a.shape, np.uint64), fast_costes=mode)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("encoding", ["integer", "division", "multiply"])
def test_costes_pending_118_matches_current_main(mode, encoding):
    # #118 fixes equality BETWEEN encodings. Here each encoding must match main;
    # deliberately do not assert cross-encoding equality before that PR lands.
    a = np.array(
        [43, 85, 88, 36, 90, 36, 61, 81, 89, 88, 79, 79, 48, 54], np.uint8
    ).reshape(2, 7)
    b = np.array([8, 8, 8, 8, 9, 9, 8, 8, 9, 9, 9, 9, 9, 10], np.uint8).reshape(2, 7)
    if encoding == "division":
        a, b = a / 255, b / 255
    elif encoding == "multiply":
        a, b = a * (1.0 / 255), b * (1.0 / 255)
    a.setflags(write=False)
    b.setflags(write=False)
    parity(a, b, np.ones(a.shape, int), fast_costes=mode, scale_max=255)


@pytest.mark.parametrize("is_3d", [False, True])
def test_costes_bulk_and_featurizer_forwarding(is_3d):
    from cp_measure.featurizer import featurize, make_featurizer_config
    from conftest import ALL_OFF

    a, b = arrays(FIXTURES[0])
    image = np.stack([a, b]).astype(float) / 255
    masks = np.full((1, *a.shape), 42, dtype=int)
    if is_3d:
        image = np.stack([image, image], axis=1)
        masks = np.stack([masks, masks], axis=1)
    images = np.stack([image, image])
    labels = np.stack([masks, masks])
    labels[1] = 71
    labels[1, ..., : a.shape[-1] // 2] = 99
    config = make_featurizer_config(
        ["a", "b"],
        **{**ALL_OFF, "correlation_costes": True},
        correlation_costes_params={
            "scale_max": 255,
            "fast_costes": ref.M_FAST,
            "thr": 42,
        },
    )
    try:
        cp_measure.set_accelerator(None)
        expected = featurize(images, labels, config, is_3d=is_3d)
        cp_measure.set_accelerator("numba")
        assert (
            "numba"
            in bulk.get_correlation_measurements(sanitize=False)["costes"].__module__
        )
        actual = featurize(images, labels, config, is_3d=is_3d)
        assert actual[1:] == expected[1:]
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
    finally:
        cp_measure.set_accelerator(None)


@pytest.mark.parametrize("mode", MODES)
def test_costes_zero_correlation_branch_replays_reference(mode, monkeypatch):
    # Discrete seed 96: scalar Pearson sums crossed zero in the wrong direction,
    # changing C1 by 0.042 and C2 by 0.118 before reference replay was added.
    a = (
        np.array(
            [
                5,
                4,
                7,
                6,
                0,
                2,
                6,
                5,
                5,
                1,
                5,
                3,
                1,
                6,
                5,
                4,
                5,
                6,
                0,
                3,
                5,
                4,
                1,
                2,
                7,
                5,
                5,
                5,
                2,
                3,
                7,
                2,
                4,
                1,
                3,
                1,
                0,
                7,
                7,
                6,
                6,
                2,
                6,
                4,
                4,
                0,
                0,
                7,
                2,
                5,
                0,
                7,
                4,
                0,
                0,
                7,
                6,
                1,
                7,
                7,
                6,
                4,
                5,
                7,
                1,
                3,
            ],
            dtype=float,
        ).reshape(1, -1)
        / 8
    )
    b = (
        np.array(
            [
                5,
                5,
                5,
                4,
                2,
                4,
                2,
                6,
                2,
                4,
                0,
                3,
                5,
                4,
                3,
                4,
                7,
                5,
                5,
                1,
                3,
                1,
                7,
                6,
                4,
                1,
                4,
                5,
                1,
                5,
                2,
                7,
                0,
                3,
                0,
                2,
                0,
                3,
                0,
                5,
                2,
                3,
                1,
                3,
                2,
                1,
                1,
                7,
                0,
                2,
                0,
                4,
                5,
                2,
                2,
                1,
                3,
                4,
                2,
                5,
                0,
                7,
                6,
                4,
                5,
                2,
            ],
            dtype=float,
        ).reshape(1, -1)
        / 8
    )
    original = ref._costes_pair
    calls = []

    def recording(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(ref, "_costes_pair", recording)
    parity(a, b, np.ones(a.shape, int), scale_max=255, fast_costes=mode)
    assert len(calls) == 2  # reference baseline plus candidate replay


@pytest.mark.parametrize("mode", MODES)
def test_costes_near_constant_scipy_diagnostics(mode):
    rng = np.random.default_rng(13)
    a = 0.1 + rng.random((2, 12)) * 1e-14
    b = 0.1 + rng.random((2, 12)) * 1e-14
    parity(a, b, np.ones(a.shape, int), scale_max=255, fast_costes=mode)


@pytest.mark.parametrize("scale", [2, 4095, 65535, 2**32 - 1])
def test_costes_explicit_acquisition_grids(scale):
    a, b = arrays(FIXTURES[0])
    parity(a / 255, b / 255, np.ones(a.shape, int), scale_max=scale)


def test_costes_background_ignored_and_unused_threshold():
    a, b = arrays(FIXTURES[0])
    a = np.pad(a / 255, 1, constant_values=np.nan)
    b = np.pad(b / 255, 1, constant_values=999)
    masks = np.pad(np.ones((8, 16), int), 1)
    parity(a, b, masks, scale_max=255, thr=93)
