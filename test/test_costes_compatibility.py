"""Costes precision contracts; live CellProfiler comparisons live in cpm_benchmark."""

import warnings

import numpy as np
import pytest

from cp_measure._detect import HAS_NUMBA
from cp_measure.core import _costes_compat as compat
from cp_measure.core import measurecolocalization as reference


@pytest.fixture(params=["numpy", "numba"])
def measure(request):
    if request.param == "numpy":
        return reference.get_correlation_costes
    if not HAS_NUMBA:
        pytest.skip("numba not installed")
    from cp_measure.core.numba.measurecolocalization import get_correlation_costes

    return get_correlation_costes


@pytest.fixture
def images():
    random = np.random.default_rng(902)
    first = random.integers(1, 256, (8, 12)).astype(np.float32) / np.float32(255)
    second = random.integers(1, 256, first.shape).astype(np.float32) / np.float32(255)
    labels = np.ones(first.shape, np.int32)
    labels[:, 6:] = 2
    labels[0] = 0
    return first, second, labels


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16])
def test_image_fit_preserves_values_and_includes_background(
    measure, images, monkeypatch, dtype
):
    first, second, labels = images
    if dtype == np.uint16:
        first, second = (
            np.rint(array * 255).astype(dtype) for array in (first, second)
        )
        expected = tuple(
            array.astype(np.float32) / np.float32(255) for array in (first, second)
        )
    else:
        first, second = first.astype(dtype), second.astype(dtype)
        expected = first, second
    originals = [array.copy() for array in (first, second, labels)]
    calls = []

    def thresholds(actual_first, actual_second, scale, mode):
        calls.append((scale, mode))
        for actual, wanted in zip((actual_first, actual_second), expected):
            assert actual.dtype == wanted.dtype
            np.testing.assert_array_equal(actual, wanted.ravel())
        return 0.5, 0.5

    monkeypatch.setattr(compat, "_thresholds", thresholds)
    result = measure(
        first, second, labels, scale_max=255, precision="input", scope="image"
    )
    assert calls == [(255, "Faster")]
    combined = (expected[0] > 0.5) & (expected[1] > 0.5)
    for direction, pixels in enumerate(expected, 1):
        values = result[f"Correlation_Costes_{direction}"]
        assert values.dtype == np.float64
        wanted = [
            pixels[combined & (labels == label)].sum(dtype=np.float64)
            / pixels[(pixels >= 0.5) & (labels == label)].sum(dtype=np.float64)
            for label in (1, 2)
        ]
        np.testing.assert_array_equal(values, wanted)
    for actual, original in zip((first, second, labels), originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("mode", ["Faster", "Fast", "Accurate"])
@pytest.mark.parametrize("container", ["list", "stack", "volume"])
@pytest.mark.parametrize("scope", ["object", "image"])
@pytest.mark.parametrize("precision", ["float64", "input"])
def test_batch_and_volume_forward_precision(
    measure, images, mode, container, scope, precision
):
    first, second, labels = images
    kwargs = dict(scale_max=255, precision=precision, fast_costes=mode, scope=scope)
    single = measure(first, second, labels, **kwargs)
    if container == "list":
        inputs = [[array, array] for array in images]
    elif container == "stack":
        inputs = [np.stack([array[None], array[None]]) for array in images]
    else:
        inputs = [array[None] for array in images]
    actual = measure(*inputs, **kwargs)
    actual = [actual] if container == "volume" else actual
    for result in actual:
        for key in single:
            np.testing.assert_array_equal(result[key], single[key])


def test_default_path_is_unchanged(measure, images, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Default path must not use compatibility arithmetic")

    monkeypatch.setattr(compat, "image_costes", forbidden)
    expected = measure(*images, scale_max=255)
    actual = measure(
        *images, scale_max=255, precision="float64", scope="object", scipy_ties=False
    )
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("bad", ["float32", "invalid", True, 1, None])
def test_precision_requires_supported_mode(measure, images, bad):
    with pytest.raises(ValueError, match="precision"):
        measure(*images, scale_max=255, precision=bad)


def test_replaced_compatibility_keyword_is_not_accepted(measure, images):
    with pytest.raises(TypeError, match="cellprofiler"):
        measure(*images, scale_max=255, cellprofiler=True)


@pytest.mark.parametrize("scope", ["object", "image"])
def test_compatibility_uses_reference_regardless_of_tie_flag(measure, images, scope):
    expected = reference.get_correlation_costes(
        *images, scale_max=255, precision="input", scope=scope
    )
    for ties in (False, True):
        actual = measure(
            *images, scale_max=255, precision="input", scipy_ties=ties, scope=scope
        )
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("case", ["empty", "zero", "nan", "gap"])
def test_boundary_outputs_and_arrays(measure, images, case, monkeypatch):
    first, second, labels = images
    if case == "empty":
        labels[:] = 0
    elif case == "zero":
        first[:] = second[:] = 0
    elif case == "nan":
        first[:] = np.nan
    else:
        labels[labels == 2] = 3
        monkeypatch.setattr(compat, "_thresholds", lambda *args: (0.0, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = measure(
            first, second, labels, scale_max=255, precision="input", scope="image"
        )
    for values in result.values():
        assert values.dtype == np.float64
        if case == "empty":
            assert values.shape == (0,)
        elif case == "zero":
            np.testing.assert_array_equal(values, [0, 0])
        elif case == "nan":
            assert np.isnan(values).all()
        else:
            np.testing.assert_array_equal(values, [1, np.nan, 1])


def test_legacy_threshold_rounding():
    pixels = np.array([0.5], dtype=np.float32)
    threshold = np.float64(0.5) - 1e-9
    assert compat._threshold_scalar(pixels, threshold) == np.float32(0.5)
    assert not (pixels > compat._threshold_scalar(pixels, threshold)).any()


@pytest.mark.parametrize("scope", ["object", "image"])
def test_non_native_byte_order(measure, images, scope):
    first, second, labels = images
    expected = measure(*images, scale_max=255, precision="input", scope=scope)
    actual = measure(
        first.astype(first.dtype.newbyteorder("S")),
        second.astype(second.dtype.newbyteorder("S")),
        labels,
        scale_max=255,
        precision="input",
        scope=scope,
    )
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("precision", ["float64", "input"])
def test_background_changes_only_image_scope(measure, images, precision):
    first, second, labels = images
    changed = first.copy()
    changed[labels == 0] = 0
    for scope in ("object", "image"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            before = measure(*images, scale_max=255, precision=precision, scope=scope)
            after = measure(
                changed,
                second,
                labels,
                scale_max=255,
                precision=precision,
                scope=scope,
            )
        equal = [
            np.array_equal(before[key], after[key], equal_nan=True) for key in before
        ]
        assert not all(equal) if scope == "image" else all(equal)


@pytest.mark.parametrize("scope", ["object", "image"])
def test_warnings_are_deduplicated_per_image(measure, images, monkeypatch, scope):
    def warned(*args):
        for repeat in range(2):
            warnings.warn("same reason", RuntimeWarning)
        return np.ones(2), np.ones(2)

    monkeypatch.setattr(compat, "image_costes", warned)
    with pytest.warns(RuntimeWarning, match="same reason") as recorded:
        measure(
            *[[array, array] for array in images],
            scale_max=255,
            precision="input",
            scope=scope,
        )
    assert len(recorded) == 2


@pytest.mark.parametrize(
    "case", ["shape", "labels", "negative", "scale", "range", "inf", "dtype", "mode"]
)
def test_validation(measure, images, case):
    first, second, labels = images
    kwargs = dict(scale_max=255, precision="input")
    if case == "shape":
        second = second[:, :-1]
    elif case == "labels":
        labels = labels.astype(float)
    elif case == "negative":
        labels[0, 0] = -1
    elif case == "scale":
        kwargs["scale_max"] = None
    elif case == "range":
        first[0, 0] = 1.1
    elif case == "inf":
        first[0, 0] = np.inf
    elif case == "dtype":
        first = first.astype(np.float16)
    else:
        kwargs["fast_costes"] = "invalid"
    with pytest.raises(ValueError):
        measure(first, second, labels, **kwargs)


@pytest.mark.parametrize("precision", ["float64", "input"])
@pytest.mark.parametrize("scope", ["invalid", "", None])
def test_invalid_scope(measure, images, precision, scope):
    with pytest.raises(ValueError, match="scope must be"):
        measure(*images, scale_max=255, precision=precision, scope=scope)


@pytest.mark.parametrize("precision", ["float64", "input"])
@pytest.mark.parametrize("scope", ["object", "image"])
def test_scope_and_arithmetic_are_independent(
    measure, images, monkeypatch, precision, scope
):
    first, second, labels = images
    calls = []

    def thresholds(actual_first, actual_second, scale, mode):
        assert actual_first.dtype == (
            np.float32 if precision == "input" else np.float64
        )
        assert actual_second.dtype == actual_first.dtype
        calls.append(actual_first.size)
        return (0.5, 0.5) if precision == "input" else (0.5, 0.5, None)

    monkeypatch.setattr(compat, "_thresholds", thresholds)
    monkeypatch.setattr(reference, "_bisection_search", thresholds)
    result = reference.get_correlation_costes(
        *images, scale_max=255, precision=precision, scope=scope
    )
    expected_sizes = (
        [first.size]
        if scope == "image"
        else [np.count_nonzero(labels == label) for label in (1, 2)]
    )
    assert calls == expected_sizes
    if precision == "input" or scope == "image":
        actual = measure(*images, scale_max=255, precision=precision, scope=scope)
        for key in result:
            np.testing.assert_array_equal(actual[key], result[key])
    for direction, pixels in enumerate((first, second), 1):
        combined = (first > 0.5) & (second > 0.5)
        expected = [
            pixels[combined & (labels == label)].sum(dtype=np.float64)
            / pixels[(pixels >= 0.5) & (labels == label)].sum(dtype=np.float64)
            for label in (1, 2)
        ]
        np.testing.assert_array_equal(
            result[f"Correlation_Costes_{direction}"], expected
        )


@pytest.mark.parametrize("mode", ["Faster", "Fast", "Accurate"])
def test_legacy_object_missing_and_single_pixel_labels(measure, images, mode):
    first, second, labels = images
    labels[labels == 2] = 3
    labels[0, 0] = 4
    with pytest.warns(RuntimeWarning) as recorded:
        result = measure(
            first, second, labels, scale_max=255, precision="input", fast_costes=mode
        )
    assert any("missing or empty" in str(warning.message) for warning in recorded)
    assert any("fewer than two" in str(warning.message) for warning in recorded)
    for values in result.values():
        assert values.shape == (4,)
        assert np.isnan(values[[1, 3]]).all()


@pytest.mark.parametrize("case", ["empty", "constant", "nan", "gap"])
def test_native_image_boundaries(measure, images, case):
    first, second, labels = images
    if case == "empty":
        labels[:] = 0
    elif case == "constant":
        first[:] = 0.5
    elif case == "nan":
        first[0, 0] = np.nan
    else:
        labels[labels == 2] = 3
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", RuntimeWarning)
        result = measure(*images, scale_max=255, scope="image")
    for values in result.values():
        if case == "empty":
            assert values.shape == (0,)
            assert not recorded
        elif case == "gap":
            assert values.shape == (3,)
            assert np.isnan(values[1])
        else:
            assert np.isnan(values).all()
            assert len(recorded) == 1
