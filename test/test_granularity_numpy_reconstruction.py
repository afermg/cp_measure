"""Single-sort reconstruction must preserve the public-library baseline."""

import warnings

import numpy as np
import pytest
import skimage.morphology as morphology

from cp_measure.core import measuregranularity as gran


@pytest.mark.parametrize("shape", [(2, 3), (33, 41), (2, 3, 4), (9, 13, 17)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scene", ["random", "plateau", "constant", "negative"])
def test_reconstruction_exact(shape, dtype, scene):
    rng = np.random.default_rng(71)
    mask = rng.random(shape).astype(dtype)
    if scene == "plateau":
        mask = np.round(mask * 5) / 5
    elif scene == "constant":
        mask[:] = 0
    elif scene == "negative":
        mask -= 0.5
    # Exercise strided arrays as well as the repeated erosion used in production.
    mask = mask.swapaxes(0, -1)
    footprint = morphology.disk(1) if mask.ndim == 2 else morphology.ball(1)
    seed = mask.copy()
    for _ in range(4):
        seed = morphology.erosion(seed, footprint=footprint)
        expected = morphology.reconstruction(seed, mask, footprint=footprint)
        if gran.reconstruction_loop is None:
            pytest.skip(
                "Private acceleration is intentionally disabled on this version"
            )
        actual = gran._reconstruct_eroded(seed, mask)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def _outcome(mask, pixels, params):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = gran.get_granularity(mask, pixels, **params)
        except Exception as exc:
            result = (type(exc), str(exc))
    return result, [(w.category, str(w.message)) for w in caught]


def _same_result(actual, expected):
    if isinstance(expected, tuple):
        assert actual == expected
        return
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key].shape == expected[key].shape
        np.testing.assert_array_equal(np.isnan(actual[key]), np.isnan(expected[key]))
        np.testing.assert_array_equal(
            np.isposinf(actual[key]), np.isposinf(expected[key])
        )
        np.testing.assert_array_equal(
            np.isneginf(actual[key]), np.isneginf(expected[key])
        )
        assert actual[key].dtype == expected[key].dtype
        np.testing.assert_array_equal(actual[key], expected[key])
        np.testing.assert_array_equal(
            np.signbit(actual[key]), np.signbit(expected[key])
        )


@pytest.mark.parametrize("shape", [(32, 40), (8, 16, 20)])
@pytest.mark.parametrize(
    "dtype", [np.bool_, np.uint16, np.int16, np.float16, np.float32, np.float64]
)
@pytest.mark.parametrize("precision", ["input", "float64"])
@pytest.mark.parametrize("zero_signal_policy", ["nan", "epsilon_floor"])
def test_spectrum_matches_unmodified_public_path(
    monkeypatch, shape, dtype, precision, zero_signal_policy
):
    rng = np.random.default_rng(31)
    pixels = (rng.random(shape) * 100).astype(dtype)
    mask = np.zeros(shape, np.int32)
    mask[..., : shape[-1] // 2] = 1
    mask[..., shape[-1] // 2 :] = 2
    pixels[..., : shape[-1] // 2] = 0
    pixels = pixels.swapaxes(0, -1)
    mask = mask.swapaxes(0, -1)
    pixels.setflags(write=False)
    mask.setflags(write=False)
    original = pixels.copy()
    params = dict(precision=precision, zero_signal_policy=zero_signal_policy)
    actual, actual_warnings = _outcome(mask, pixels, params)
    monkeypatch.setattr(gran, "reconstruction_loop", None)
    expected, expected_warnings = _outcome(mask, pixels, params)
    _same_result(actual, expected)
    assert actual_warnings == expected_warnings
    np.testing.assert_array_equal(pixels, original)


@pytest.mark.parametrize("shape", [(160, 160), (8, 12, 16), (1, 4), (0, 4)])
@pytest.mark.parametrize("sub,bg", [(0.4, 0.25), (1, 1), (1, 0.5), (0.5, 1)])
@pytest.mark.parametrize("objects", [False, True])
def test_geometry_and_empty_cases(monkeypatch, shape, sub, bg, objects):
    pixels = np.random.default_rng(5).random(shape)
    mask = np.full(shape, int(objects), dtype=np.int32)
    params = dict(subsample_size=sub, image_sample_size=bg, element_size=1)
    actual, aw = _outcome(mask, pixels, params)
    monkeypatch.setattr(gran, "reconstruction_loop", None)
    expected, ew = _outcome(mask, pixels, params)
    _same_result(actual, expected)
    assert aw == ew


@pytest.mark.parametrize(
    "params",
    [
        {"precision": "invalid"},
        {"zero_signal_policy": "invalid"},
        {"granular_spectrum_length": 0},
        {"element_size": -1},
        {"subsample_size": 0},
        {"image_sample_size": 0},
    ],
)
def test_validation_unchanged(monkeypatch, params):
    pixels = np.ones((16, 16), np.float32)
    mask = np.ones(pixels.shape, np.int32)
    actual, aw = _outcome(mask, pixels, params)
    monkeypatch.setattr(gran, "reconstruction_loop", None)
    expected, ew = _outcome(mask, pixels, params)
    _same_result(actual, expected)
    assert aw == ew


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_uses_public_reconstruction(monkeypatch, value):
    calls = []

    def public(seed, mask, footprint):
        calls.append(mask.copy())
        return np.zeros_like(mask)

    def forbidden(*args):
        pytest.fail("Nonfinite input must not reach the private kernel")

    monkeypatch.setattr(morphology, "reconstruction", public)
    monkeypatch.setattr(gran, "_reconstruct_eroded", forbidden)
    pixels = np.full((8, 8), value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gran.get_granularity(
            np.ones((8, 8), np.int32),
            pixels,
            subsample_size=1,
            image_sample_size=1,
            element_size=1,
            granular_spectrum_length=2,
        )
    assert len(calls) == 2


@pytest.mark.parametrize(
    "version", ["0.25.2", "0.26.0", "0.19.3", "0.27.0", "0.26.0.dev0"]
)
def test_private_version_gate(version):
    def loop(ranks, prev, next, strides, current_idx, image_stride):
        raise AssertionError("Selection must never invoke the kernel")

    def older_loop(aranks, aprev, anext, astrides, current_idx, image_stride):
        raise AssertionError("Selection must never invoke the kernel")

    candidate = older_loop if version == "0.25.2" else loop
    selected = gran._compatible_reconstruction_loop(version, candidate)
    assert selected is (candidate if version in {"0.25.2", "0.26.0"} else None)


@pytest.mark.parametrize(
    "loop", [None, object(), lambda *args: None, lambda ranks: None]
)
def test_incompatible_private_api_uses_public_path(monkeypatch, loop):
    pixels = np.random.default_rng(12).random((32, 40)).astype(np.float32)
    mask = np.ones(pixels.shape, np.int32)
    params = dict(subsample_size=1, image_sample_size=1, element_size=1)
    expected, ew = _outcome(mask, pixels, params)
    selected = gran._compatible_reconstruction_loop("0.26.0", loop)
    assert selected is None
    monkeypatch.setattr(gran, "reconstruction_loop", selected)
    actual, aw = _outcome(mask, pixels, params)
    _same_result(actual, expected)
    assert aw == ew


def test_verified_kernel_failure_is_not_suppressed(monkeypatch):
    def broken(*args):
        raise RuntimeError("kernel failure")

    monkeypatch.setattr(gran, "reconstruction_loop", broken)
    with pytest.raises(RuntimeError, match="kernel failure"):
        gran._reconstruct_eroded(np.zeros((4, 4)), np.ones((4, 4)))
