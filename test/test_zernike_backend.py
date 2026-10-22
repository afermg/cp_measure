"""Correctness of the numba Zernike backends vs the numpy baseline."""

import numpy as np
import pytest

import cp_measure
from cp_measure.bulk import get_core_measurements
from cp_measure.core.measureobjectintensitydistribution import (
    get_radial_zernikes as radial_numpy,
)
from cp_measure.core.measureobjectsizeshape import get_zernike as shape_numpy
from cp_measure.core.numba.measureobjectintensitydistribution import (
    get_radial_zernikes as radial_numba,
)
from cp_measure.core.numba.measureobjectsizeshape import get_zernike as shape_numba

SHAPE, RADIAL = (shape_numpy, shape_numba), (radial_numpy, radial_numba)
BACKENDS = [SHAPE, RADIAL]
# Shape Zernike ignores pixels, so pixel variations only exercise the radial port.
WEIGHTS = ["random", "ones", "zeros", "signed"]
WEIGHT_CASES = [(*SHAPE, "random")] + [(*RADIAL, w) for w in WEIGHTS]
DTYPE_CASES = [(*SHAPE, np.float64)] + [
    (*RADIAL, dtype) for dtype in (np.float32, np.float64, np.uint16)
]


def assert_same(ref, got):
    assert list(ref) == list(got)
    magnitudes = [ref[k] for k in ref if "Magnitude" in k and ref[k].size]
    scale = max((np.nanmax(m, initial=0) for m in magnitudes), default=0)
    for key in ref:
        assert ref[key].shape == got[key].shape
        assert ref[key].dtype == got[key].dtype
        if "ZernikePhase" not in key:
            np.testing.assert_allclose(
                got[key], ref[key], rtol=1e-6, atol=1e-8, equal_nan=True, err_msg=key
            )
            continue
        # Phase is atan2 of round-off for a moment that is zero by symmetry, and +pi
        # vs -pi when only its real part is (that sign flips even between NumPy
        # runs). So compare angles modulo 2*pi, only above the image's noise floor.
        np.testing.assert_array_equal(np.isnan(got[key]), np.isnan(ref[key]))
        magnitude = ref[key.replace("Phase", "Magnitude")]
        defined = magnitude > 1e-9 * scale
        wrapped = np.angle(np.exp(1j * (got[key] - ref[key])[defined]))
        np.testing.assert_allclose(wrapped, 0, atol=1e-6, err_msg=key)


def mask_scene(kind):
    m = np.zeros((32, 40), np.int32)
    if kind == "empty":
        return m
    if kind == "empty_bool":  # sanitize keeps an object-free bool mask as bool
        return m.astype(bool)
    if kind == "zero_size":
        return m[:0]
    if kind == "single_pixel":
        m[3, 4] = 1
    elif kind == "two_pixels":
        m[3, 4:6] = 1
    elif kind == "line":
        m[3, 4:24] = 1
    elif kind == "square":
        m[4:20, 4:20] = 1
    elif kind == "edge_hole":
        m[:20, :15] = 1
        m[5:12, 3:8] = 0
        m[20:, 29:] = 2
    elif kind == "disconnected":
        m[2:5, 3:8] = 1
        m[25:30, 30:35] = 1
    elif kind == "irregular":
        y, x = np.mgrid[:32, :40]
        m[((y - 10.3) / 8.5) ** 2 + ((x - 11.4) / 7.3) ** 2 <= 1] = 1
        m[22:, 25:34] = 2
        m[22:25, 25:27] = 0
    else:
        raise ValueError(kind)
    return m


@pytest.mark.parametrize("ref_fn,got_fn,weights", WEIGHT_CASES)
@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "empty_bool",
        "zero_size",
        "single_pixel",
        "two_pixels",
        "line",
        "square",
        "edge_hole",
        "disconnected",
        "irregular",
    ],
)
@pytest.mark.parametrize("degree", [0, 9, 14])
def test_edge_cases(ref_fn, got_fn, weights, kind, degree):
    mask = mask_scene(kind)
    pixels = np.random.default_rng(12).random(mask.shape)
    if weights == "ones":
        pixels[:] = 1
    elif weights == "zeros":
        pixels[:] = 0
    elif weights == "signed":
        pixels -= 0.5
    old_mask, old_pixels = mask.copy(), pixels.copy()
    with np.errstate(all="ignore"):
        assert_same(ref_fn(mask, pixels, degree), got_fn(mask, pixels, degree))
    np.testing.assert_array_equal(mask, old_mask)
    np.testing.assert_array_equal(pixels, old_pixels)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("location", ["inside", "outside"])
def test_nonfinite_weights(value, location):
    mask = mask_scene("irregular")
    pixels = np.ones(mask.shape)
    pixels[10, 11] = value if location == "inside" else 1
    pixels[0, 0] = value if location == "outside" else 1
    with np.errstate(all="ignore"):
        assert_same(radial_numpy(mask, pixels), radial_numba(mask, pixels))


@pytest.mark.parametrize("ref_fn,got_fn,dtype", DTYPE_CASES)
def test_noncontiguous_inputs(ref_fn, got_fn, dtype):
    mask = mask_scene("edge_hole")[::-1, ::-1]
    pixels = (np.random.default_rng(4).random(mask.shape) * 100).astype(dtype)[:, ::-1]
    assert not mask.flags.c_contiguous
    assert not pixels.flags.c_contiguous
    assert_same(ref_fn(mask, pixels), got_fn(mask, pixels))


@pytest.mark.parametrize("ref_fn,got_fn", BACKENDS, ids=["shape", "radial"])
@pytest.mark.parametrize("depth", [1, 3])
def test_raw_volume_including_singleton_is_empty(ref_fn, got_fn, depth):
    mask = np.repeat(mask_scene("square")[None], depth, axis=0)
    pixels = np.ones(mask.shape)
    assert ref_fn(mask, pixels) == got_fn(mask, pixels) == {}
    assert got_fn([mask], [pixels]) == [{}]


@pytest.mark.parametrize("feature", ["zernike", "radial_zernikes"])
@pytest.mark.parametrize("batch", ["single", "list", "tuple", "stacked"])
def test_public_sanitation_and_batch(feature, batch):
    masks = [mask_scene("edge_hole"), mask_scene("irregular")]
    for mask in masks:
        mask[mask == 1] = 17
        mask[mask == 2] = 3000000
    pixels = [np.random.default_rng(s).random(m.shape) for s, m in enumerate(masks)]
    previous = cp_measure._ACCELERATOR
    cp_measure.set_accelerator(None)
    numpy_fn = get_core_measurements()[feature]
    refs = [numpy_fn(m, p) for m, p in zip(masks, pixels)]
    if batch == "single":
        m_arg, p_arg = masks[0], pixels[0]
    elif batch == "stacked":
        m_arg, p_arg = np.stack(masks)[:, None], np.stack(pixels)[:, None]
    elif batch == "tuple":
        m_arg, p_arg = tuple(masks), tuple(pixels)
    else:
        m_arg, p_arg = masks, pixels
    try:
        cp_measure.set_accelerator("numba")
        got = get_core_measurements()[feature](m_arg, p_arg)
    finally:
        cp_measure.set_accelerator(previous)
    if batch == "single":
        assert_same(refs[0], got)
    else:
        assert isinstance(got, list) and len(got) == 2
        for ref, result in zip(refs, got):
            assert_same(ref, result)


@pytest.mark.parametrize("feature", ["zernike", "radial_zernikes"])
def test_featurizer_preserves_columns_ids_and_values(feature):
    from conftest import ALL_OFF
    from cp_measure.featurizer import featurize, make_featurizer_config

    mask = mask_scene("edge_hole")
    mask[mask == 1] = 17
    mask[mask == 2] = 3000000
    masks = np.stack([mask, mask])[:, None]
    pixels = np.random.default_rng(42).random(masks.shape)
    config = make_featurizer_config(["DNA"], **{**ALL_OFF, feature: True})
    previous = cp_measure._ACCELERATOR
    try:
        cp_measure.set_accelerator(None)
        ref, ref_columns, ref_ids = featurize(
            pixels, masks, config, image_ids=["a", "b"]
        )
        cp_measure.set_accelerator("numba")
        got, got_columns, got_ids = featurize(
            pixels, masks, config, image_ids=["a", "b"]
        )
    finally:
        cp_measure.set_accelerator(previous)
    assert list(got_columns) == list(ref_columns)
    assert list(got_ids) == list(ref_ids)
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-8, equal_nan=True)
