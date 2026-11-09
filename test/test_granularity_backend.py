"""Correctness of the numba granularity backend vs the numpy baseline."""

import hashlib
import json
from pathlib import Path

import numpy
import pytest

import cp_measure
from cp_measure.featurizer import featurize, make_featurizer_config
from conftest import ALL_OFF

from cp_measure.core.measuregranularity import get_granularity as gran_numpy
from cp_measure.core.numba.measuregranularity import get_granularity as gran_numba

TOL = dict(rtol=1e-7, atol=1e-10, equal_nan=True)


def _scene(H=80, W=80, n=4, seed=0):
    """A textured 2D image + a labeled mask with `n` rectangular objects."""
    rng = numpy.random.default_rng(seed)
    pixels = rng.random((H, W)) + 0.3 * numpy.add.outer(
        numpy.sin(numpy.arange(H) / 4), numpy.cos(numpy.arange(W) / 5)
    )
    mask = numpy.zeros((H, W), dtype=numpy.int32)
    step = H // (n + 1)
    for k in range(n):
        y0 = step * (k + 1) - 6
        x0 = 8 + 15 * k
        mask[y0 : y0 + 12, x0 : x0 + 12] = k + 1
    return mask, pixels


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        numpy.testing.assert_allclose(a[k], b[k], **TOL, err_msg=f"key {k}")


@pytest.mark.parametrize(
    "sub,img", [(1.0, 1.0), (0.25, 0.25), (0.5, 0.25), (0.25, 1.0)]
)
@pytest.mark.parametrize("dtype", [numpy.float16, numpy.float32, numpy.float64])
def test_2d_matches_baseline(sub, img, dtype):
    mask, pixels = _scene()
    pixels = pixels.astype(dtype)
    pixels.setflags(write=False)
    params = dict(subsample_size=sub, image_sample_size=img, precision="float64")
    expected = gran_numpy(mask, pixels.astype(numpy.float64), **params)
    actual = gran_numpy(mask, pixels, **params)
    for key in expected:
        numpy.testing.assert_array_equal(actual[key], expected[key])
    _assert_same(gran_numba(mask, pixels, **params), expected)


@pytest.mark.parametrize("shape", [(0, 17), (1, 17), (3, 17), (7, 17), (17, 7)])
@pytest.mark.parametrize("sub", [0.25, 1.0])
@pytest.mark.parametrize("has_objects", [False, True])
def test_small_image_matches_baseline(shape, sub, has_objects):
    mask = numpy.full(shape, int(has_objects), dtype=numpy.int32)
    pixels = numpy.random.default_rng(0).random(shape)
    params = dict(subsample_size=sub, image_sample_size=1.0, element_size=1)
    _assert_same(gran_numba(mask, pixels, **params), gran_numpy(mask, pixels, **params))


@pytest.mark.parametrize("transpose", [False, True])
def test_subsampled_boundary_object_matches_baseline(transpose):
    # 160 * 0.4 -> 64: the final source coordinate rounds just past 63.
    mask = numpy.zeros((160, 160), numpy.int32)
    mask[153:, 30:90] = 1
    mask[10:40, 10:40] = 2
    pixels = numpy.random.default_rng(0).random(mask.shape)
    if transpose:
        mask, pixels = mask.T, pixels.T
    _assert_same(
        gran_numba(mask, pixels, subsample_size=0.4),
        gran_numpy(mask, pixels, subsample_size=0.4),
    )


@pytest.mark.parametrize(
    "case_index",
    range(3),
    ids=["resample-both", "resample-background", "no-background-resampling"],
)
def test_float32_background_matches_numpy_and_cellprofiler(case_index):
    reference = json.loads(
        (
            Path(__file__).parent
            / "data/granularity_float32_cellprofiler_reference.json"
        ).read_text()
    )
    y, x = numpy.indices((64, 80))
    raw = ((73 * y + 157 * x + ((13 * x + 29 * y) % 31) * 101) % 40000 + 1000).astype(
        numpy.uint16
    )
    pixels = raw.astype(numpy.float32) / numpy.float32(65535)
    labels = numpy.zeros(raw.shape, numpy.int32)
    labels[8:28, 8:28] = 1
    labels[32:58, 37:68] = 2
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == reference["pixels_sha256"]
    assert hashlib.sha256(labels.tobytes()).hexdigest() == reference["labels_sha256"]
    case = reference["cases"][case_index]
    params = dict(
        subsample_size=case["subsample_size"],
        image_sample_size=case["image_sample_size"],
        element_size=2,
        granular_spectrum_length=6,
    )
    before = pixels.copy()
    baseline = gran_numpy(labels, pixels, **params)
    accelerated = gran_numba(labels, pixels, **params)
    _assert_same(baseline, case["expected"])
    _assert_same(accelerated, baseline)
    numpy.testing.assert_array_equal(pixels, before)


def test_single_object():
    mask = numpy.zeros((60, 60), numpy.int32)
    mask[20:40, 20:40] = 1
    rng = numpy.random.default_rng(1)
    pixels = rng.random((60, 60))
    _assert_same(gran_numba(mask, pixels), gran_numpy(mask, pixels))


def test_one_pixel_object():
    mask = numpy.zeros((40, 40), numpy.int32)
    mask[10, 10] = 1
    mask[25, 30] = 2
    pixels = numpy.random.default_rng(2).random((40, 40))
    _assert_same(gran_numba(mask, pixels), gran_numpy(mask, pixels))


def test_label_gaps_dense_output():
    # labels 1 and 3 present, 2 absent -> dense output length 3, nan at index 1
    mask = numpy.zeros((50, 50), numpy.int32)
    mask[5:15, 5:15] = 1
    mask[30:40, 30:40] = 3
    pixels = numpy.random.default_rng(3).random((50, 50))
    out = gran_numba(mask, pixels)
    base = gran_numpy(mask, pixels)
    assert out["Granularity_1"].shape == (3,)
    _assert_same(out, base)


def test_empty_mask():
    mask = numpy.zeros((40, 40), numpy.int32)
    pixels = numpy.random.default_rng(4).random((40, 40))
    out = gran_numba(mask, pixels)
    base = gran_numpy(mask, pixels)
    assert out["Granularity_1"].shape == (0,)
    _assert_same(out, base)


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("precision", ["input", "float64"])
def test_batch_sequence_precision_matches_per_image(container, precision):
    scenes = [_scene(seed=0), _scene(n=3, seed=1)]
    masks = container([m for m, _ in scenes])
    pixels = container([p.astype(numpy.float32) for _, p in scenes])
    out = gran_numba(masks, pixels, precision=precision)
    assert isinstance(out, list) and len(out) == 2
    for m, p, got in zip(masks, pixels, out):
        _assert_same(got, gran_numba(m, p, precision=precision))
        if precision == "float64":
            _assert_same(got, gran_numpy(m, p, precision=precision))


@pytest.mark.parametrize("max_workers", [None, 1])
def test_batch_shared_mask_matches_per_image(max_workers):
    """A list repeating one mask object (the featurizer's one mask per channel),
    mixed with a different mask, equals per-image calls exactly and in order,
    whether the batch runs on threads or serially."""
    mask, pixels = _scene(seed=0)
    other_mask, other_pixels = _scene(n=3, seed=1)
    channels = [pixels, pixels[::-1].copy(), numpy.sqrt(pixels)]
    masks = [mask, mask, other_mask, mask]
    images = [channels[0], channels[1], other_pixels, channels[2]]
    out = gran_numba(masks, images, max_workers=max_workers)
    assert isinstance(out, list) and len(out) == 4
    for m, p, got in zip(masks, images, out):
        want = gran_numba(m, p)
        assert list(got) == list(want)
        for k in want:
            numpy.testing.assert_array_equal(got[k], want[k], err_msg=k)


def test_batch_list_ragged_rejected():
    a = _scene(H=80, W=80, seed=0)
    b = _scene(H=64, W=96, n=3, seed=1)
    with pytest.raises(ValueError, match="same shape"):
        gran_numba([a[0], b[0]], [a[1], b[1]])


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("precision", ["input", "float64"])
def test_mixed_dtype_batch_uses_common_precision(container, precision):
    y, x = numpy.mgrid[:80, :80]
    first = (0.5 + 0.4 * numpy.sin(x / 11) * numpy.cos(y / 13)).astype(numpy.float32)
    pixels = container([first, first.astype(numpy.float64)])
    masks = container([numpy.ones(first.shape, numpy.int32) for _ in pixels])
    for array in (*masks, *pixels):
        array.setflags(write=False)
    params = dict(subsample_size=0.3, element_size=2, precision=precision)
    actual = gran_numba(masks, pixels, **params)
    assert isinstance(actual, list) and len(actual) == len(pixels)
    common_dtype = numpy.result_type(*pixels)
    for mask, image, result in zip(masks, pixels, actual):
        expected = gran_numba(mask, image.astype(common_dtype, copy=False), **params)
        assert result.keys() == expected.keys()
        for key in expected:
            assert result[key].dtype == expected[key].dtype
            numpy.testing.assert_array_equal(result[key], expected[key])
    for key in actual[0]:
        numpy.testing.assert_array_equal(actual[0][key], actual[1][key])
    if precision == "input":
        # Whole-batch promotion differs from an independent float32 call.
        unpromoted = gran_numba(masks[0], first, **params)
        assert any(
            not numpy.array_equal(actual[0][key], unpromoted[key]) for key in actual[0]
        )


@pytest.mark.parametrize("is_3d", [False, True], ids=["2d", "3d"])
def test_featurizer_batch_matches_numpy(monkeypatch, is_3d):
    """Exercise the rebased backend through #106's explicit batch interface."""
    scenes = [_scene(seed=0), _scene(n=3, seed=1)]
    masks = numpy.stack([m for m, _ in scenes])[:, None]
    image = numpy.stack([p for _, p in scenes])[:, None]
    if is_3d:
        masks = numpy.repeat(masks[:, :, None], 4, axis=2)
        image = numpy.repeat(image[:, :, None], 4, axis=2)
        image *= numpy.arange(1, 5)[None, None, :, None, None]
    config = make_featurizer_config(
        ["DNA"],
        **{**ALL_OFF, "granularity": True},
        granularity_params={"element_size": 2, "granular_spectrum_length": 3},
    )
    image_before, masks_before = image.copy(), masks.copy()
    monkeypatch.setattr(cp_measure, "_ACCELERATOR", None)
    expected, columns, rows = featurize(
        image, masks, config, is_3d=is_3d, image_ids=["A", "B"]
    )
    monkeypatch.setattr(cp_measure, "_ACCELERATOR", "numba")
    actual, actual_columns, actual_rows = featurize(
        image, masks, config, is_3d=is_3d, image_ids=["A", "B"]
    )
    numpy.testing.assert_allclose(actual, expected, **TOL)
    assert actual_columns == columns
    assert (
        actual_rows
        == rows
        == [
            (image_id, "object", label)
            for image_id, count in [("A", 4), ("B", 3)]
            for label in range(1, count + 1)
        ]
    )
    numpy.testing.assert_array_equal(image, image_before)
    numpy.testing.assert_array_equal(masks, masks_before)


def _assert_same_or_same_error(mask, pixels, **params):
    """3D numba equals numpy bit for bit, or raises numpy's error (e.g. scipy
    rejects float16)."""
    try:
        expected = gran_numpy(mask, pixels, **params)
    except Exception as error:
        with pytest.raises(type(error), match=str(error)):
            gran_numba(mask, pixels, **params)
        return
    actual = gran_numba(mask, pixels, **params)
    assert actual.keys() == expected.keys()
    for key in expected:
        numpy.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def _volume(Z=9, H=48, W=40, seed=0):
    """A textured 3D volume and a mask with objects touching every face and an
    absent label, so the readback covers the last plane, row and column."""
    rng = numpy.random.default_rng(seed)
    z, y, x = numpy.indices((Z, H, W))
    pixels = rng.random((Z, H, W)) + 0.3 * numpy.sin(z / 2 + y / 4) * numpy.cos(x / 5)
    mask = numpy.zeros((Z, H, W), numpy.int32)
    mask[:, 5:20, 4:16] = 1
    mask[2:, 30:, 20:] = 2  # touches the last plane, row and column
    mask[0:3, 0:6, 30:] = 4  # label 3 is absent
    return mask, pixels


@pytest.mark.parametrize(
    "sub,img", [(1.0, 1.0), (0.5, 0.5), (0.5, 0.25), (0.25, 1.0), (1.0, 0.5)]
)
@pytest.mark.parametrize("dtype", [numpy.float16, numpy.float32, numpy.float64])
@pytest.mark.parametrize("precision", ["input", "float64"])
def test_3d_matches_baseline(sub, img, dtype, precision):
    if dtype is numpy.float16 and precision == "input" and sub == img == 1.0:
        pytest.skip(
            "scipy rejects float16; without a resample numba measures it anyway, "
            "as in 2D"
        )
    mask, pixels = _volume(Z=12 if sub < 0.5 else 9)
    pixels = pixels.astype(dtype)
    pixels.setflags(write=False)
    params = dict(
        subsample_size=sub,
        image_sample_size=img,
        element_size=2,
        granular_spectrum_length=6,
        precision=precision,
    )
    _assert_same_or_same_error(mask, pixels, **params)


@pytest.mark.parametrize("shape", [(1, 17, 15), (2, 17, 15), (3, 1, 15), (4, 7, 17)])
@pytest.mark.parametrize("sub", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("has_objects", [False, True])
def test_small_volume_matches_baseline(shape, sub, has_objects):
    """Thin, singleton-Z and collapsing volumes: same values, or the same error."""
    mask = numpy.full(shape, int(has_objects), dtype=numpy.int32)
    pixels = numpy.random.default_rng(0).random(shape)
    params = dict(subsample_size=sub, image_sample_size=1.0, element_size=1)
    _assert_same_or_same_error(mask, pixels, **params)


@pytest.mark.parametrize("depth", [0, 6], ids=["2d", "3d"])
@pytest.mark.parametrize("sub,img", [(1.0, 1.0), (1.0, 0.5), (0.5, 0.5)])
def test_integer_images_measure_as_float64(depth, sub, img):
    """Integer pixels are measured as their exact float64 values: the background
    subtraction must not wrap around (or round) in the integer dtype. Sparse
    bright spots make the resampled background exceed some pixels."""
    if depth:
        mask, _ = _volume(Z=depth * 2, seed=3)
    else:
        mask, _ = _scene(seed=3)
    rng = numpy.random.default_rng(3)
    pixels = rng.integers(500, 1500, mask.shape).astype(numpy.uint16)
    pixels[rng.random(mask.shape) < 0.01] = 60000
    params = dict(subsample_size=sub, image_sample_size=img, element_size=2)
    got = gran_numba(mask, pixels, **params)
    ref = gran_numba(mask, pixels.astype(numpy.float64), **params)
    for key in ref:
        numpy.testing.assert_array_equal(got[key], ref[key], err_msg=key)


def test_thin_volume_background_is_deterministic():
    """An 8-slice volume at default sizes has a 2-sample background grid along Z;
    the NumPy opening must not read outside it (mode="reflect" did), so repeated
    calls agree and match the numba backend."""
    pixels = numpy.random.default_rng(0).random((8, 16, 16))
    mask = numpy.ones(pixels.shape, numpy.int32)
    mask[:, :8] = 2
    first = gran_numpy(mask, pixels)
    for _ in range(3):
        again = gran_numpy(mask, pixels)
        for key in first:
            numpy.testing.assert_array_equal(again[key], first[key], err_msg=key)
    got = gran_numba(mask, pixels)
    for key in first:
        numpy.testing.assert_array_equal(got[key], first[key], err_msg=key)
