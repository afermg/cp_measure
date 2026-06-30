"""Equivalence tests for the vectorised numpy ``get_radial_zernikes``.

These lock **numerical fidelity to the original centrosome implementation**, not to
externally-known values: ``_reference`` reproduces the pre-vectorisation path, and each test
asserts the vectorised output matches it to floating-point round-off.
"""

import numpy
import pytest
from centrosome import cpmorphology, zernike

from cp_measure.core.measureobjectintensitydistribution import (
    M_CATEGORY,
    get_radial_zernikes,
)
from cp_measure.utils import masks_to_ijv

ATOL = 1e-9  # >> the ~1e-13 summation-order round-off, << any real signal


def _reference(labels, pixels, zernike_degree=9):
    """Reproduce the original centrosome implementation (the pre-vectorisation path)."""
    zidx = zernike.get_zernike_indexes(zernike_degree + 1)
    ul = numpy.unique(labels)
    ul = ul[ul > 0]
    out = {}
    if len(ul) == 0:  # same interleaved key order as get_radial_zernikes
        for n, m in zidx:
            out[f"{M_CATEGORY}_ZernikeMagnitude_{n}_{m}"] = numpy.zeros(0)
            out[f"{M_CATEGORY}_ZernikePhase_{n}_{m}"] = numpy.zeros(0)
        return out
    ij, r = cpmorphology.minimum_enclosing_circle(labels, ul)
    ijv = masks_to_ijv(labels)
    pos = ijv[:, 2] - 1  # contiguous 1..N: label -> row
    yx = (ijv[:, :2] - ij[pos]) / r[pos, numpy.newaxis]
    z = zernike.construct_zernike_polynomials(yx[:, 1], yx[:, 0], zidx)
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


def _pixels(shape, seed=0):
    return numpy.random.default_rng(seed).random(shape)


@pytest.mark.parametrize("zernike_degree", [5, 9, 14])
def test_radial_zernike_irregular(zernike_degree):
    # Irregular blobs of varied size/position cover the single- and multi-object cases;
    # sweeping the degree checks the basis truncation.
    rng = numpy.random.default_rng(0)
    masks = numpy.zeros((128, 128), numpy.int32)
    for lab, (cy, cx) in enumerate(rng.integers(20, 108, size=(6, 2)), 1):
        yy, xx = numpy.mgrid[0:128, 0:128]
        masks[(yy - cy) ** 2 + (xx - cx) ** 2 < rng.integers(40, 120)] = lab
    _assert_matches(masks, _pixels(masks.shape), zernike_degree)


def test_radial_zernike_object_touching_edge():
    # Frame-clipped objects are a distinct geometry the irregular test never hits (its blobs
    # stay clear of the border): the enclosing circle sits centred near the boundary.
    masks = numpy.zeros((64, 64), numpy.int32)
    masks[0:20, 0:20] = 1  # clipped at the top-left corner
    masks[40:64, 40:64] = 2  # clipped at the bottom-right corner
    _assert_matches(masks, _pixels(masks.shape))


def test_radial_zernike_single_pixel_object():
    # minimum_enclosing_circle radius -> 0, so both paths divide by zero identically.
    masks = numpy.zeros((32, 32), numpy.int32)
    masks[16, 16] = 1
    masks[5:15, 5:15] = 2  # a normal object alongside the degenerate one
    _assert_matches(masks, _pixels(masks.shape))


def test_radial_zernike_empty_mask():
    # No objects -> every feature is an empty (0,) array; _assert_matches covers keys + shapes.
    _assert_matches(numpy.zeros((40, 40), numpy.int32), _pixels((40, 40)))
