"""Golden + edge tests for the vectorised numpy ``get_zernike``.

The vectorised implementation avoids centrosome's full (H, W, K) scatter and the per-channel
``scipy.ndimage.sum``; it must match ``centrosome.zernike.zernike`` (called with the actual
label values) to floating-point round-off. Existing ``test_core_measurements`` already covers
shape / 3D-empty; here we lock the numerical result and the edge cases.
"""

import numpy
import pytest
from centrosome import zernike

from cp_measure.core.measureobjectsizeshape import get_zernike
from cp_measure.utils import _zernike_scores

ATOL = 1e-10  # >> the ~2e-16 round-off observed, << any real signal


def _reference(masks, zernike_numbers=9):
    """Old centrosome path, called with the real label values as indices."""
    uniq = numpy.unique(masks)
    uniq = uniq[uniq > 0]
    zidx = zernike.get_zernike_indexes(zernike_numbers + 1)
    zf = zernike.zernike(zidx, masks, uniq)
    return {f"Zernike_{n}_{m}": zf[:, i] for i, (n, m) in enumerate(zidx)}


def _assert_matches(masks, zernike_numbers=9):
    ref = _reference(masks, zernike_numbers)
    got = get_zernike(masks, None, zernike_numbers)
    assert list(got) == list(ref), "key set / order changed"
    for k in ref:
        assert got[k].shape == ref[k].shape, k
        assert numpy.allclose(got[k], ref[k], atol=ATOL, rtol=1e-10, equal_nan=True), (
            f"{k}: max|diff|={numpy.nanmax(numpy.abs(got[k] - ref[k]))}"
        )


def _generate_square_objects(size, n, gap_frac=0.75):
    masks = numpy.zeros((size, size), numpy.int32)
    step = size // n
    obj = int(step * gap_frac)
    lab = 0
    for a in range(n):
        for b in range(n):
            lab += 1
            masks[a * step : a * step + obj, b * step : b * step + obj] = lab
    return masks


@pytest.mark.parametrize("zernike_numbers", [5, 9, 14])
def test_zernike_matches_centrosome_irregular(zernike_numbers):
    # Irregular blobs of varied size/position cover the single- and multi-object cases;
    # sweeping the zernike number checks the basis truncation across degrees.
    rng = numpy.random.default_rng(0)
    masks = numpy.zeros((128, 128), numpy.int32)
    for lab, (cy, cx) in enumerate(rng.integers(20, 108, size=(6, 2)), 1):
        yy, xx = numpy.mgrid[0:128, 0:128]
        masks[(yy - cy) ** 2 + (xx - cx) ** 2 < rng.integers(40, 120)] = lab
    _assert_matches(masks, zernike_numbers)


def test_zernike_object_touching_edge():
    masks = numpy.zeros((64, 64), numpy.int32)
    masks[0:20, 0:20] = 1  # clipped at the top-left corner
    masks[40:64, 40:64] = 2  # clipped at the bottom-right corner
    _assert_matches(masks)


def test_zernike_single_pixel_object():
    # minimum_enclosing_circle radius -> 0, so both paths divide by zero identically.
    masks = numpy.zeros((32, 32), numpy.int32)
    masks[16, 16] = 1
    masks[5:15, 5:15] = 2  # a normal object alongside the degenerate one
    _assert_matches(masks)


def test_zernike_empty_mask():
    masks = numpy.zeros((40, 40), numpy.int32)
    got = get_zernike(masks, None)
    zidx = zernike.get_zernike_indexes(9 + 1)
    assert list(got) == [f"Zernike_{n}_{m}" for n, m in zidx]
    assert all(v.shape == (0,) for v in got.values())


def test_zernike_3d_returns_empty():
    assert get_zernike(numpy.zeros((4, 16, 16), numpy.int32), None) == {}


def test_zernike_scores_unit_weight_equals_unweighted():
    # The weight=None path (shape moments) used by get_zernike vs an explicit unit weight,
    # the path the intensity-weighted radial Zernikes (PR #75) take.
    masks = _generate_square_objects(128, 3)
    zidx = zernike.get_zernike_indexes(9 + 1)
    r0, i0, _rad0, c0 = _zernike_scores(masks, zidx)
    r1, i1, _rad1, c1 = _zernike_scores(masks, zidx, weight=numpy.ones(masks.shape))
    assert numpy.allclose(r0, r1) and numpy.allclose(i0, i1)
    assert numpy.array_equal(c0, c1)
