"""Golden + edge tests for the vectorised numpy ``get_zernike``.

The vectorised implementation avoids centrosome's full (H, W, K) scatter and the per-channel
``scipy.ndimage.sum``; it must match ``centrosome.zernike.zernike`` (called with the actual
label values) to floating-point round-off. Existing ``test_core_measurements`` already covers
shape / 3D-empty; here we lock the numerical result and the edge cases.
"""

import centrosome.zernike
import numpy
import scipy.ndimage

from cp_measure.core.measureobjectsizeshape import get_zernike
from cp_measure.utils import _zernike_scores

ATOL = 1e-10  # >> the ~2e-16 round-off observed, << any real signal


def _reference(masks, zernike_numbers=9):
    """Old centrosome path, called with the real label values as indices."""
    uniq = numpy.unique(masks)
    uniq = uniq[uniq > 0]
    zidx = centrosome.zernike.get_zernike_indexes(zernike_numbers + 1)
    zf = centrosome.zernike.zernike(zidx, masks, uniq)
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


def _square_objects(size, n, gap_frac=0.75):
    masks = numpy.zeros((size, size), numpy.int32)
    step = size // n
    obj = int(step * gap_frac)
    lab = 0
    for a in range(n):
        for b in range(n):
            lab += 1
            masks[a * step : a * step + obj, b * step : b * step + obj] = lab
    return masks


def test_zernike_matches_centrosome_single_object():
    masks = numpy.zeros((64, 64), numpy.int32)
    masks[20:45, 18:50] = 1
    _assert_matches(masks)


def test_zernike_matches_centrosome_multi_object():
    _assert_matches(_square_objects(256, 4))


def test_zernike_matches_centrosome_irregular():
    rng = numpy.random.default_rng(0)
    masks = numpy.zeros((128, 128), numpy.int32)
    for lab, (cy, cx) in enumerate(rng.integers(20, 108, size=(6, 2)), 1):
        yy, xx = numpy.mgrid[0:128, 0:128]
        masks[(yy - cy) ** 2 + (xx - cx) ** 2 < rng.integers(40, 120)] = lab
    _assert_matches(masks)


def test_zernike_object_touching_edge():
    masks = numpy.zeros((64, 64), numpy.int32)
    masks[0:20, 0:20] = 1  # clipped at the top-left corner
    masks[40:64, 40:64] = 2  # clipped at the bottom-right corner
    _assert_matches(masks)


def test_zernike_noncontiguous_labels():
    # labels {1, 3, 7}: the old range(1, n+1) assumption would be wrong; the new code
    # uses the real label values and must match centrosome called the same way.
    masks = numpy.zeros((96, 96), numpy.int32)
    masks[10:30, 10:30] = 1
    masks[40:60, 40:60] = 3
    masks[70:90, 70:90] = 7
    _assert_matches(masks)


def test_zernike_single_pixel_object():
    # minimum_enclosing_circle radius -> 0, so both paths divide by zero identically.
    masks = numpy.zeros((32, 32), numpy.int32)
    masks[16, 16] = 1
    masks[5:15, 5:15] = 2  # a normal object alongside the degenerate one
    _assert_matches(masks)


def test_zernike_non_default_degree():
    _assert_matches(_square_objects(128, 3), zernike_numbers=6)


def test_zernike_empty_mask():
    masks = numpy.zeros((40, 40), numpy.int32)
    got = get_zernike(masks, None)
    zidx = centrosome.zernike.get_zernike_indexes(9 + 1)
    assert list(got) == [f"Zernike_{n}_{m}" for n, m in zidx]
    assert all(v.shape == (0,) for v in got.values())


def test_zernike_3d_returns_empty():
    assert get_zernike(numpy.zeros((4, 16, 16), numpy.int32), None) == {}


def _weighted_reference(masks, pixels, zernike_numbers=9):
    """Independent weighted real/imag/count via centrosome's own basis + scipy segment-sum.

    Mirrors centrosome.zernike()'s coordinate construction but feeds the intensity image as
    ``weight`` to ``construct_zernike_polynomials`` — the path the radial Zernikes (PR #75) use.
    """
    zidx = centrosome.zernike.get_zernike_indexes(zernike_numbers + 1)
    labels = numpy.unique(masks)
    labels = labels[labels > 0]
    centers, radii = centrosome.zernike.minimum_enclosing_circle(masks, labels)
    rev = numpy.full(int(masks.max()) + 1, -1, int)
    rev[labels] = numpy.arange(len(labels))
    mask = rev[masks] != -1
    ny, nx = masks.shape
    y, x = numpy.mgrid[0:ny, 0:nx].astype(float)
    rev_ind = rev[masks[mask]]
    xc = numpy.zeros((ny, nx))
    yc = numpy.zeros((ny, nx))
    yc[mask] = (y[mask] - centers[rev_ind, 0]) / radii[rev_ind]
    xc[mask] = (x[mask] - centers[rev_ind, 1]) / radii[rev_ind]
    zf = centrosome.zernike.construct_zernike_polynomials(xc, yc, zidx, mask, weight=pixels)
    real = numpy.column_stack(
        [scipy.ndimage.sum_labels(zf[:, :, i].real, masks, labels) for i in range(len(zidx))]
    )
    imag = numpy.column_stack(
        [scipy.ndimage.sum_labels(zf[:, :, i].imag, masks, labels) for i in range(len(zidx))]
    )
    counts = numpy.asarray(scipy.ndimage.sum_labels(numpy.ones((ny, nx)), masks, labels), float)
    return real, imag, counts


def test_zernike_scores_weighted_matches_centrosome():
    # The intensity-weighted path (weight != None) + per-object counts that PR #75 reuses.
    rng = numpy.random.default_rng(1)
    masks = _square_objects(160, 3)
    pixels = rng.random(masks.shape)
    zidx = centrosome.zernike.get_zernike_indexes(9 + 1)
    real, imag, radii, counts = _zernike_scores(masks, zidx, weight=pixels)
    ref_real, ref_imag, ref_counts = _weighted_reference(masks, pixels)
    assert numpy.allclose(real, ref_real, atol=ATOL, rtol=1e-10)
    assert numpy.allclose(imag, ref_imag, atol=ATOL, rtol=1e-10)
    assert numpy.allclose(counts, ref_counts)


def test_zernike_scores_unit_weight_equals_unweighted():
    masks = _square_objects(128, 3)
    zidx = centrosome.zernike.get_zernike_indexes(9 + 1)
    r0, i0, _rad0, c0 = _zernike_scores(masks, zidx)
    r1, i1, _rad1, c1 = _zernike_scores(masks, zidx, weight=numpy.ones(masks.shape))
    assert numpy.allclose(r0, r1) and numpy.allclose(i0, i1)
    assert numpy.array_equal(c0, c1)
