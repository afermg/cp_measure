"""Bit-exact validation of the numba granularity morphology kernels vs skimage."""

import numpy
import pytest
import scipy.ndimage
import skimage.morphology as M

from cp_measure.core.numba._granularity import (
    _disk_halfwidths,
    disk_dilation_2d,
    disk_erosion_2d,
    dilation_4conn_2d,
    erosion_4conn_2d,
    readback_sums,
    readback_tables,
    reconstruction_by_dilation_2d,
)


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("shape", [(37, 41), (64, 64)])
def test_readback_matches_upsample_then_label_mean(seed, shape):
    """Per-label sums of the bilinear readback equal upsampling the coarse grid with
    order-1 map_coordinates (the baseline) and summing per label, to rounding; with
    unit scales they equal a bincount of the grid at the object pixels exactly."""
    rng = numpy.random.default_rng(seed)
    labels = rng.integers(0, 6, size=shape)
    labels[labels == 3] = 0  # an absent label
    pixels = rng.random(shape)
    n = int(labels.max())

    def per_label(values):  # labels 1..n; background is never read back
        flat = numpy.bincount(labels.ravel(), weights=values.ravel(), minlength=n + 1)
        return flat[1:]

    coarse = (shape[0] // 4, shape[1] // 3)
    grid = rng.random(coarse)
    sy = (coarse[0] - 1) / (shape[0] - 1)
    sx = (coarse[1] - 1) / (shape[1] - 1)
    counts, sums, *tables = readback_tables(labels, pixels, n, sy, sx, *coarse)
    yy, xx = numpy.indices(shape).astype(float)
    upsampled = scipy.ndimage.map_coordinates(grid, (yy * sy, xx * sx), order=1)
    numpy.testing.assert_allclose(
        readback_sums(grid, *tables)[1:], per_label(upsampled), rtol=0, atol=1e-12
    )
    numpy.testing.assert_array_equal(counts[1:], per_label(numpy.ones(shape)))
    numpy.testing.assert_array_equal(sums[1:], per_label(pixels))
    # unit scales: same sums as a bincount at the object pixels, bit for bit
    fine = rng.random(shape)
    _, _, *unit = readback_tables(labels, pixels, n, 1.0, 1.0, *shape)
    numpy.testing.assert_array_equal(readback_sums(fine, *unit)[1:], per_label(fine))


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


@pytest.mark.parametrize("offset", [(0, 0), (32768, 0), (65536, 0), (0, 65536)])
def test_reconstruction_long_geodesic(offset):
    """A serpentine channel forces FIFO propagation, including beyond the old
    signed/16-bit packed-coordinate limits on either axis."""
    H, W = 41, 41
    mask = numpy.zeros((H, W))
    # carve a snake of open (high) corridors one row apart, connected at alternating ends
    mask[1:-1, 1:-1] = 9.0
    for i in range(2, H - 2, 2):
        if (i // 2) % 2 == 0:
            mask[i, 1:-2] = 0.0  # wall from the left, gap on the right
        else:
            mask[i, 2:-1] = 0.0  # wall from the right, gap on the left
    seed = numpy.zeros((H, W))
    seed[1, 1] = 9.0  # single high seed at one end of the snake
    padding = ((offset[0], 0), (offset[1], 0))
    mask = numpy.pad(mask, padding)
    seed = numpy.pad(seed, padding)
    got = reconstruction_by_dilation_2d(seed, mask)
    expected = M.reconstruction(seed, mask, footprint=M.disk(1))
    numpy.testing.assert_array_equal(got, expected)
