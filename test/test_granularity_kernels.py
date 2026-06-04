"""Bit-exact validation of the numba granularity morphology kernels vs skimage."""

import numpy
import pytest
import scipy.ndimage
import skimage.morphology as M

from cp_measure.core.numba._granularity import (
    _disk_halfwidths,
    bilinear_gather,
    disk_dilation_2d,
    disk_erosion_2d,
    dilation_4conn_2d,
    erosion_4conn_2d,
    reconstruction_by_dilation_2d,
)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_bilinear_gather_matches_map_coordinates(seed):
    """The gather must match scipy order-1 map_coordinates (mode=constant) to ~eps,
    including points on the array boundary (where a corner is out of range)."""
    rng = numpy.random.default_rng(seed)
    img = rng.random((37, 41))
    H, W = img.shape
    npts = 500
    sy = numpy.concatenate([rng.uniform(0, H - 1, npts), [0.0, H - 1, H - 1]])
    sx = numpy.concatenate([rng.uniform(0, W - 1, npts), [0.0, W - 1, 0.0]])
    ref = scipy.ndimage.map_coordinates(img, (sy, sx), order=1)
    y0 = numpy.floor(sy).astype(numpy.int64)
    x0 = numpy.floor(sx).astype(numpy.int64)
    got = bilinear_gather(img, y0, x0, sy - y0, sx - x0, H, W)
    numpy.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


def _images(seed):
    rng = numpy.random.default_rng(seed)
    H, W = 23, 19
    return {
        "random": rng.random((H, W)),
        "int_ties": rng.integers(0, 5, size=(H, W)).astype(numpy.float64),
        "gradient": numpy.add.outer(numpy.arange(H), numpy.arange(W)).astype(float),
        "border_hot": _border_hot(H, W),
        "single_spike": _spike(H, W),
    }


def _border_hot(H, W):
    a = numpy.zeros((H, W))
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 7.0
    return a


def _spike(H, W):
    a = numpy.zeros((H, W))
    a[H // 2, W // 2] = 5.0
    a[2, 3] = 3.0
    return a


@pytest.mark.parametrize("radius", [1, 2, 3, 5, 7, 10])
@pytest.mark.parametrize(
    "name", ["random", "int_ties", "gradient", "border_hot", "single_spike"]
)
def test_disk_erosion_bit_exact(radius, name):
    img = _images(radius)[name]
    expected = M.erosion(img, M.disk(radius))
    numpy.testing.assert_array_equal(disk_erosion_2d(img, radius), expected)


@pytest.mark.parametrize("radius", [1, 2, 3, 5, 7, 10])
@pytest.mark.parametrize(
    "name", ["random", "int_ties", "gradient", "border_hot", "single_spike"]
)
def test_disk_dilation_bit_exact(radius, name):
    img = _images(radius)[name]
    expected = M.dilation(img, M.disk(radius))
    numpy.testing.assert_array_equal(disk_dilation_2d(img, radius), expected)


@pytest.mark.parametrize(
    "name", ["random", "int_ties", "gradient", "border_hot", "single_spike"]
)
def test_4conn_erosion_bit_exact(name):
    img = _images(0)[name]
    numpy.testing.assert_array_equal(erosion_4conn_2d(img), M.erosion(img, M.disk(1)))


@pytest.mark.parametrize(
    "name", ["random", "int_ties", "gradient", "border_hot", "single_spike"]
)
def test_4conn_dilation_bit_exact(name):
    img = _images(0)[name]
    numpy.testing.assert_array_equal(dilation_4conn_2d(img), M.dilation(img, M.disk(1)))


def test_halfwidths_match_disk():
    for r in (1, 2, 3, 5, 10):
        hx = _disk_halfwidths(r)
        fp = M.disk(r)
        # per-row count of set pixels == 2*hx+1
        for dy in range(r + 1):
            row = fp[r + dy]
            assert int(row.sum()) == 2 * int(hx[dy]) + 1


@pytest.mark.parametrize("name", ["random", "int_ties", "gradient", "single_spike"])
def test_reconstruction_bit_exact(name):
    img = _images(3)[name]
    # seed must be <= mask; use an erosion of the mask as the seed (the granularity pattern)
    mask = img
    seed = M.erosion(mask, M.disk(1))
    expected = M.reconstruction(seed, mask, footprint=M.disk(1))
    got = reconstruction_by_dilation_2d(
        seed.astype(numpy.float64), mask.astype(numpy.float64)
    )
    numpy.testing.assert_array_equal(got, expected)


def test_reconstruction_cascaded_mask_chain():
    # rec_g uses rec_{g-1} as mask (cascade); each step must still match skimage
    rng = numpy.random.default_rng(7)
    pixels = rng.random((25, 25))
    ero = pixels.copy()
    recon_mask = pixels
    for _ in range(5):
        ero = erosion_4conn_2d(ero)
        rec = reconstruction_by_dilation_2d(ero, recon_mask)
        expected = M.reconstruction(ero, recon_mask, footprint=M.disk(1))
        numpy.testing.assert_array_equal(rec, expected)
        recon_mask = rec
