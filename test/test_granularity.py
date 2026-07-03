"""Equivalence tests for the fused-upsample ``get_granularity``.

The optimisation replaces the per-spectrum-step ``scipy.ndimage.map_coordinates`` upsample +
``scipy.ndimage.mean`` (repeated with identical geometry every iteration) with a single
precomputed sparse operator. These lock **numerical fidelity to the original implementation**:
``_reference_unfused`` reproduces the pre-optimisation path, and each test asserts the fused
output matches it. Comparing the two on the *same* installed scipy/skimage isolates the change,
so the assertion holds regardless of library version (residual is sparse-accumulation order,
~1e-12).
"""

import numpy
import pytest
import scipy.ndimage
import skimage.morphology

from cp_measure.core.measuregranularity import get_granularity
from cp_measure.utils import _ensure_np_array as fix

RTOL = 1e-6  # granularity values are percentages; the fused operator matches to ~1e-12


def _reference_unfused(
    mask,
    pixels,
    subsample_size=0.25,
    image_sample_size=0.25,
    element_size=10,
    granular_spectrum_length=16,
):
    """Reproduce the pre-optimisation implementation: upsample + ndimage.mean inside the loop."""
    orig_shape = numpy.array(pixels.shape)
    orig_pixels = pixels
    orig_mask = mask
    new_shape = orig_shape.copy()
    if subsample_size < 1:
        new_shape = (orig_shape * subsample_size).astype(int)
        if pixels.ndim == 2:
            i, j = (
                numpy.mgrid[0 : new_shape[0], 0 : new_shape[1]].astype(float)
                / subsample_size
            )
            pixels = scipy.ndimage.map_coordinates(pixels, (i, j), order=1)
            mask = scipy.ndimage.map_coordinates(mask, (i, j), order=0).astype(
                orig_mask.dtype
            )
        else:
            k, i, j = (
                numpy.mgrid[
                    0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]
                ].astype(float)
                / subsample_size
            )
            pixels = scipy.ndimage.map_coordinates(pixels, (k, i, j), order=1)
            mask = scipy.ndimage.map_coordinates(mask, (k, i, j), order=0).astype(
                orig_mask.dtype
            )
    else:
        pixels = pixels.copy()
        mask = mask.copy()
    if image_sample_size < 1:
        back_shape = new_shape * image_sample_size
        if pixels.ndim == 2:
            i, j = (
                numpy.mgrid[0 : back_shape[0], 0 : back_shape[1]].astype(float)
                / image_sample_size
            )
            back_pixels = scipy.ndimage.map_coordinates(pixels, (i, j), order=1)
            back_mask = numpy.ones(back_pixels.shape, dtype=bool)
        else:
            k, i, j = (
                numpy.mgrid[
                    0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]
                ].astype(float)
                / subsample_size
            )
            back_pixels = scipy.ndimage.map_coordinates(pixels, (k, i, j), order=1)
            back_mask = numpy.ones(back_pixels.shape, dtype=bool)
    else:
        back_pixels = pixels
        back_mask = numpy.ones(back_pixels.shape, dtype=bool)
        back_shape = new_shape
    radius = element_size
    if pixels.ndim == 2:
        footprint = skimage.morphology.disk(radius, dtype=bool)
    else:
        footprint = skimage.morphology.ball(radius, dtype=bool)
    back_pixels_mask = numpy.zeros_like(back_pixels)
    back_pixels_mask[back_mask == 1] = back_pixels[back_mask == 1]
    back_pixels = skimage.morphology.erosion(back_pixels_mask, footprint=footprint)
    back_pixels_mask = numpy.zeros_like(back_pixels)
    back_pixels_mask[back_mask == 1] = back_pixels[back_mask == 1]
    back_pixels = skimage.morphology.dilation(back_pixels_mask, footprint=footprint)
    if image_sample_size < 1:
        if pixels.ndim == 2:
            i, j = numpy.mgrid[0 : new_shape[0], 0 : new_shape[1]].astype(float)
            i *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
            j *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
            back_pixels = scipy.ndimage.map_coordinates(back_pixels, (i, j), order=1)
        else:
            k, i, j = numpy.mgrid[
                0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]
            ].astype(float)
            k *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
            i *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
            j *= float(back_shape[2] - 1) / float(new_shape[2] - 1)
            back_pixels = scipy.ndimage.map_coordinates(back_pixels, (k, i, j), order=1)
    pixels -= back_pixels
    pixels[pixels < 0] = 0

    ng = granular_spectrum_length
    if pixels.ndim == 2:
        footprint = skimage.morphology.disk(1, dtype=bool)
    else:
        footprint = skimage.morphology.ball(1, dtype=bool)
    ero = pixels.copy()
    range_ = numpy.arange(1, numpy.max(orig_mask) + 1)
    current_mean = fix(scipy.ndimage.mean(orig_pixels, orig_mask, range_))
    start_mean = numpy.maximum(current_mean, numpy.finfo(float).eps)

    results = {}
    for granularity_id in range(1, ng + 1):
        ero_mask = ero.copy()
        ero = skimage.morphology.erosion(ero_mask, footprint=footprint)
        rec = skimage.morphology.reconstruction(ero, pixels, footprint=footprint)
        if pixels.ndim == 2:
            i, j = numpy.mgrid[0 : orig_shape[0], 0 : orig_shape[1]].astype(float)
            i *= float(new_shape[0] - 1) / float(orig_shape[0] - 1)
            j *= float(new_shape[1] - 1) / float(orig_shape[1] - 1)
            rec_orig = scipy.ndimage.map_coordinates(rec, (i, j), order=1)
        else:
            k, i, j = numpy.mgrid[
                0 : orig_shape[0], 0 : orig_shape[1], 0 : orig_shape[2]
            ].astype(float)
            k *= float(new_shape[0] - 1) / float(orig_shape[0] - 1)
            i *= float(new_shape[1] - 1) / float(orig_shape[1] - 1)
            j *= float(new_shape[2] - 1) / float(orig_shape[2] - 1)
            rec_orig = scipy.ndimage.map_coordinates(rec, (k, i, j), order=1)
        gss = numpy.zeros((0,))
        if range_.size:
            new_mean = fix(scipy.ndimage.mean(rec_orig, orig_mask, range_))
            gss = (current_mean - new_mean) * 100 / start_mean
            current_mean = new_mean
        results[f"Granularity_{granularity_id}"] = gss
    return results


def _assert_matches(mask, pixels, **kw):
    ref = _reference_unfused(mask, pixels, **kw)
    got = get_granularity(mask, pixels, **kw)
    assert list(got) == list(ref), "key set / order changed"
    for k in ref:
        assert got[k].shape == ref[k].shape, k
        assert numpy.allclose(got[k], ref[k], rtol=RTOL, atol=1e-9, equal_nan=True), (
            f"{k}: max|diff|={numpy.nanmax(numpy.abs(got[k] - ref[k]))}"
        )


def _generate_textured(shape, seed=0):
    rng = numpy.random.default_rng(seed)
    base = scipy.ndimage.gaussian_filter(rng.random(shape), 3)
    return (base + 0.3 * rng.random(shape)).astype(numpy.float64)


def _generate_disks(size, centers_radii):
    masks = numpy.zeros((size, size), numpy.int32)
    yy, xx = numpy.mgrid[0:size, 0:size]
    for lab, (cy, cx, r) in enumerate(centers_radii, 1):
        masks[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = lab
    return masks


@pytest.mark.parametrize(
    "kw",
    [
        {},
        dict(
            subsample_size=0.5,
            image_sample_size=0.5,
            element_size=6,
            granular_spectrum_length=8,
        ),
    ],
)
def test_granularity_matches_reference(kw):
    # Multi-object mask, default and non-default parameters.
    masks = _generate_disks(
        160, [(45, 45, 25), (45, 115, 25), (115, 45, 25), (115, 115, 25)]
    )
    _assert_matches(masks, _generate_textured((160, 160)), **kw)


def test_granularity_overshoot_boundary_object():
    # orig 160 -> new 64 (subsample 0.4) floats the last-row source coordinate just past new-1
    # (63.0000000000001), so map_coordinates(mode='constant') zeros those pixels; the fused
    # operator must do the same. Object on the last rows is the regression guard for the boundary
    # fix (pre-fix the fused mean diverged ~0.08 from the reference for such objects).
    masks = numpy.zeros((160, 160), numpy.int32)
    masks[153:160, 30:90] = 1  # touches the last row
    masks[10:40, 10:40] = 2  # interior control object
    _assert_matches(masks, _generate_textured((160, 160)), subsample_size=0.4)


def test_granularity_3d_unchanged():
    # The 3D path stays on the direct interpolation; guards that the loop restructure kept it.
    rng = numpy.random.default_rng(2)
    masks = numpy.zeros((24, 48, 48), numpy.int32)
    zz, yy, xx = numpy.mgrid[0:24, 0:48, 0:48]
    masks[(zz - 12) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2 < 100] = 1
    pixels = scipy.ndimage.gaussian_filter(rng.random((24, 48, 48)), 2).astype(
        numpy.float64
    )
    _assert_matches(masks, pixels, element_size=4, granular_spectrum_length=6)


def test_granularity_empty_mask():
    # No objects -> every feature is an empty (0,) array; _assert_matches covers keys + shapes.
    _assert_matches(numpy.zeros((64, 64), numpy.int32), _generate_textured((64, 64)))


def test_returns_nan_when_subsampling_collapses_axis():
    # #91: subsampling shrinks an axis below the 2 samples bilinear interpolation needs -> NaN.
    mask = numpy.zeros((7, 17), dtype=int)
    mask[:, 1:-1] = 1
    out = get_granularity(mask, numpy.zeros((7, 17)))
    for v in out.values():
        assert numpy.isnan(v).all()
