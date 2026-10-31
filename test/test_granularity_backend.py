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
def test_2d_matches_baseline(sub, img):
    mask, pixels = _scene()
    _assert_same(
        gran_numba(mask, pixels, subsample_size=sub, image_sample_size=img),
        gran_numpy(mask, pixels, subsample_size=sub, image_sample_size=img),
    )


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


def test_single_image_returns_dict_not_list():
    mask, pixels = _scene()
    assert isinstance(gran_numba(mask, pixels), dict)


def test_batch_4d_matches_per_image_baseline():
    scenes = [_scene(seed=s) for s in range(3)]
    masks = numpy.stack([m[numpy.newaxis] for m, _ in scenes])  # (B,1,H,W)
    pixels = numpy.stack([p[numpy.newaxis] for _, p in scenes])
    out = gran_numba(masks, pixels)
    assert isinstance(out, list) and len(out) == 3
    for m, p, got in zip(masks, pixels, out):
        _assert_same(got, gran_numpy(m, p))


@pytest.mark.parametrize("container", [list, tuple])
def test_batch_sequence_matches_per_image_baseline(container):
    a = _scene(seed=0)
    b = _scene(n=3, seed=1)
    out = gran_numba(container([a[0], b[0]]), container([a[1], b[1]]))
    assert isinstance(out, list) and len(out) == 2
    _assert_same(out[0], gran_numpy(*a))
    _assert_same(out[1], gran_numpy(*b))


def test_batch_list_ragged_rejected():
    a = _scene(H=80, W=80, seed=0)
    b = _scene(H=64, W=96, n=3, seed=1)
    with pytest.raises(ValueError, match="same shape"):
        gran_numba([a[0], b[0]], [a[1], b[1]])


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


def test_3d_volume_falls_back_to_numpy():
    # Z>1 dispatches to the numpy baseline. Use fullres to exercise morphology;
    # default subsampling would collapse Z and return NaNs instead.
    rng = numpy.random.default_rng(5)
    pixels = rng.random((3, 40, 40))
    mask = numpy.zeros((3, 40, 40), numpy.int32)
    mask[:, 10:25, 10:25] = 1
    _assert_same(
        gran_numba(mask, pixels, subsample_size=1.0, image_sample_size=1.0),
        gran_numpy(mask, pixels, subsample_size=1.0, image_sample_size=1.0),
    )
