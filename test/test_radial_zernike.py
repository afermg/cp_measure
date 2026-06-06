"""Golden + edge tests for the vectorised numpy ``get_radial_zernikes``.

The rewrite delegates the intensity-weighted moment sums to the shared
``cp_measure.utils._zernike_scores`` (the same masked-basis + segment-sum machinery proven
bit-exact for ``get_zernike``), then normalises by each object's pixel count. It must match a
direct ``centrosome.construct_zernike_polynomials`` reference to floating-point round-off.

The reference here indexes the enclosing-circle geometry by each label's *position* (not
``label - 1``), so it is correct for non-contiguous label sets — the case where the previous
implementation raised ``IndexError``.
"""

import centrosome.cpmorphology
import centrosome.zernike
import numpy

from cp_measure.core.measureobjectintensitydistribution import (
    M_CATEGORY,
    get_radial_zernikes,
)
from cp_measure.utils import masks_to_ijv

ATOL = 1e-9  # >> the ~1e-13 summation-order round-off, << any real signal


def _reference(labels, pixels, zernike_degree=9):
    """Direct centrosome path with correct per-label (position-based) geometry indexing."""
    zidx = centrosome.zernike.get_zernike_indexes(zernike_degree + 1)
    ul = numpy.unique(labels)
    ul = ul[ul > 0]
    out = {}
    if len(ul) == 0:
        for mp in ("Magnitude", "Phase"):
            for n, m in zidx:
                out[f"{M_CATEGORY}_Zernike{mp}_{n}_{m}"] = numpy.zeros(0)
        return out
    ij, r = centrosome.cpmorphology.minimum_enclosing_circle(labels, ul)
    ijv = masks_to_ijv(labels)
    pos = numpy.searchsorted(ul, ijv[:, 2])  # label -> row in [0, len(ul))
    yx = (ijv[:, :2] - ij[pos]) / r[pos, numpy.newaxis]
    z = centrosome.zernike.construct_zernike_polynomials(yx[:, 1], yx[:, 0], zidx)
    w = pixels[ijv[:, 0], ijv[:, 1]]
    areas = numpy.bincount(pos, minlength=len(ul)).astype(float)
    for i, (n, m) in enumerate(zidx):
        vr = numpy.bincount(pos, weights=w * z[:, i].real, minlength=len(ul))
        vi = numpy.bincount(pos, weights=w * z[:, i].imag, minlength=len(ul))
        out[f"{M_CATEGORY}_ZernikeMagnitude_{n}_{m}"] = (
            numpy.sqrt(vr * vr + vi * vi) / areas
        )
        out[f"{M_CATEGORY}_ZernikePhase_{n}_{m}"] = numpy.arctan2(vr, vi)
    return out


def _assert_matches(masks, pixels, zernike_degree=9):
    ref = _reference(masks, pixels, zernike_degree)
    got = get_radial_zernikes(masks, pixels, zernike_degree)
    assert list(got) == list(ref), "key set / order changed"
    for k in ref:
        assert got[k].shape == ref[k].shape, k
        assert numpy.allclose(got[k], ref[k], atol=ATOL, rtol=1e-9, equal_nan=True), (
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


def _pixels(shape, seed=0):
    return numpy.random.default_rng(seed).random(shape)


def test_radial_zernike_single_object():
    masks = numpy.zeros((64, 64), numpy.int32)
    masks[20:45, 18:50] = 1
    _assert_matches(masks, _pixels(masks.shape))


def test_radial_zernike_multi_object():
    masks = _square_objects(256, 4)
    _assert_matches(masks, _pixels(masks.shape))


def test_radial_zernike_irregular():
    rng = numpy.random.default_rng(0)
    masks = numpy.zeros((128, 128), numpy.int32)
    for lab, (cy, cx) in enumerate(rng.integers(20, 108, size=(6, 2)), 1):
        yy, xx = numpy.mgrid[0:128, 0:128]
        masks[(yy - cy) ** 2 + (xx - cx) ** 2 < rng.integers(40, 120)] = lab
    _assert_matches(masks, _pixels(masks.shape))


def test_radial_zernike_object_touching_edge():
    masks = numpy.zeros((64, 64), numpy.int32)
    masks[0:20, 0:20] = 1  # clipped at the top-left corner
    masks[40:64, 40:64] = 2  # clipped at the bottom-right corner
    _assert_matches(masks, _pixels(masks.shape))


def test_radial_zernike_noncontiguous_labels():
    # labels {1, 3, 7}: the previous `ij[label - 1]` indexing raised IndexError here. The
    # rewrite maps each label to its own row, so it must run and match the reference.
    masks = numpy.zeros((96, 96), numpy.int32)
    masks[10:30, 10:30] = 1
    masks[40:60, 40:60] = 3
    masks[70:90, 70:90] = 7
    _assert_matches(masks, _pixels(masks.shape))


def test_radial_zernike_single_pixel_object():
    # minimum_enclosing_circle radius -> 0, so both paths divide by zero identically.
    masks = numpy.zeros((32, 32), numpy.int32)
    masks[16, 16] = 1
    masks[5:15, 5:15] = 2  # a normal object alongside the degenerate one
    _assert_matches(masks, _pixels(masks.shape))


def test_radial_zernike_non_default_degree():
    masks = _square_objects(128, 3)
    _assert_matches(masks, _pixels(masks.shape), zernike_degree=6)


def test_radial_zernike_empty_mask():
    masks = numpy.zeros((40, 40), numpy.int32)
    got = get_radial_zernikes(masks, _pixels(masks.shape))
    zidx = centrosome.zernike.get_zernike_indexes(9 + 1)
    expected = [
        f"{M_CATEGORY}_Zernike{mp}_{n}_{m}"
        for mp in ("Magnitude", "Phase")
        for n, m in zidx
    ]
    assert sorted(got) == sorted(expected)
    assert all(v.shape == (0,) for v in got.values())


def test_radial_zernike_3d_returns_empty():
    masks = numpy.zeros((4, 16, 16), numpy.int32)
    assert get_radial_zernikes(masks, numpy.zeros((4, 16, 16))) == {}
