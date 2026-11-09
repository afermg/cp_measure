"""Bit-exact validation of the numba granularity morphology kernels vs skimage."""

import numpy
import pytest
import scipy.ndimage
import skimage.morphology as M

from cp_measure.core.numba._granularity import (
    _disk_halfwidths,
    ball_dilation_3d,
    ball_erosion_3d,
    erosion_6conn_3d,
    reconstruction_by_dilation_3d,
    upsample_label_sums_3d,
    disk_dilation_2d,
    disk_erosion_2d,
    dilation_4conn_2d,
    erosion_4conn_2d,
    label_sums,
    readback_sums,
    readback_tables,
    reconstruction_by_dilation_2d,
    spectrum_sums_2d,
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
    counts, *tables = readback_tables(labels, n, sy, sx, *coarse)
    sums = label_sums(labels, pixels, n)
    yy, xx = numpy.indices(shape).astype(float)
    upsampled = scipy.ndimage.map_coordinates(grid, (yy * sy, xx * sx), order=1)
    numpy.testing.assert_allclose(
        readback_sums(grid, *tables)[1:], per_label(upsampled), rtol=0, atol=1e-12
    )
    numpy.testing.assert_array_equal(counts[1:], per_label(numpy.ones(shape)))
    numpy.testing.assert_array_equal(sums[1:], per_label(pixels))
    # unit scales: same sums as a bincount at the object pixels, bit for bit
    fine = rng.random(shape)
    _, *unit = readback_tables(labels, n, 1.0, 1.0, *shape)
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


@pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
def test_spectrum_sums_2d_matches_step_loop(dtype):
    """The fused spectrum kernel returns, bit for bit, the readback sums of the
    erosion + cascaded reconstruction loop run step by step."""
    rng = numpy.random.default_rng(5)
    shape = (31, 27)
    pixels = rng.random(shape).astype(dtype)
    labels = rng.integers(0, 5, size=(62, 54))
    n = int(labels.max())
    _, *tables = readback_tables(labels, n, 30 / 61, 26 / 53, *shape)
    expected = []
    ero, recon_mask = pixels, pixels
    for _ in range(6):
        ero = erosion_4conn_2d(ero)
        recon_mask = reconstruction_by_dilation_2d(ero, recon_mask)
        expected.append(readback_sums(recon_mask, *tables))
    numpy.testing.assert_array_equal(spectrum_sums_2d(pixels, 6, *tables), expected)


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


def _volumes(seed, shape=(5, 17, 15)):
    rng = numpy.random.default_rng(seed)
    hot = numpy.zeros(shape)
    hot[0], hot[-1], hot[:, 0], hot[:, -1], hot[:, :, 0], hot[:, :, -1] = (7.0,) * 6
    spike = numpy.zeros(shape)
    spike[shape[0] // 2, shape[1] // 2, shape[2] // 2] = 5.0
    return {
        "random": rng.random(shape),
        "int_ties": rng.integers(0, 5, size=shape).astype(numpy.float64),
        "gradient": numpy.indices(shape).sum(0).astype(float),
        "border_hot": hot,
        "single_spike": spike,
    }


VOLUME_NAMES = ["random", "int_ties", "gradient", "border_hot", "single_spike"]


@pytest.mark.parametrize("radius", [1, 2, 3, 5])
@pytest.mark.parametrize("shape", [(5, 17, 15), (1, 12, 13), (3, 9, 30)])
@pytest.mark.parametrize("name", VOLUME_NAMES)
def test_ball_erosion_dilation_3d_bit_exact(radius, shape, name):
    img = _volumes(radius, shape)[name]
    numpy.testing.assert_array_equal(
        ball_erosion_3d(img, radius), M.erosion(img, M.ball(radius))
    )
    numpy.testing.assert_array_equal(
        ball_dilation_3d(img, radius), M.dilation(img, M.ball(radius))
    )


@pytest.mark.parametrize("shape", [(5, 17, 15), (1, 12, 13)])
@pytest.mark.parametrize("name", VOLUME_NAMES)
def test_6conn_erosion_3d_bit_exact(shape, name):
    img = _volumes(0, shape)[name]
    numpy.testing.assert_array_equal(erosion_6conn_3d(img), M.erosion(img, M.ball(1)))


@pytest.mark.parametrize("shape", [(5, 17, 15), (1, 12, 13), (7, 6, 40)])
@pytest.mark.parametrize("name", ["random", "int_ties", "gradient", "single_spike"])
def test_reconstruction_3d_cascade_bit_exact(shape, name):
    """The spectrum's cascade (rec_g is the next step's mask) matches skimage each step."""
    pixels = _volumes(3, shape)[name]
    ero, recon_mask = pixels.copy(), pixels
    for _ in range(4):
        ero = erosion_6conn_3d(ero)
        rec = reconstruction_by_dilation_3d(ero, recon_mask)
        expected = M.reconstruction(ero, recon_mask, footprint=M.ball(1))
        numpy.testing.assert_array_equal(rec, expected)
        recon_mask = rec


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
def test_upsample_label_sums_3d_bit_exact(seed, dtype):
    """Equal, bit for bit, to the baseline's trilinear map_coordinates upsample of a
    grid of that dtype followed by ndimage.mean, including the last plane, row and
    column and an absent label."""
    rng = numpy.random.default_rng(seed)
    shape = tuple(int(v) for v in rng.integers(4, 40, 3))
    coarse = tuple(max(2, int(s * f)) for s, f in zip(shape, rng.uniform(0.2, 1, 3)))
    labels = rng.integers(0, 6, size=shape)
    labels[labels == 3] = 0
    n = int(labels.max())
    grid = rng.random(coarse).astype(dtype)
    scales = [float(c - 1) / float(s - 1) for c, s in zip(coarse, shape)]
    kk, yy, xx = numpy.indices(shape).astype(float)
    upsampled = scipy.ndimage.map_coordinates(
        grid, (kk * scales[0], yy * scales[1], xx * scales[2]), order=1
    )
    expected = scipy.ndimage.mean(upsampled, labels, numpy.arange(1, n + 1))
    counts = numpy.bincount(labels.ravel(), minlength=n + 1)[1:]
    with numpy.errstate(invalid="ignore", divide="ignore"):
        got = (
            upsample_label_sums_3d(
                grid.astype(numpy.float64),
                labels,
                n,
                tuple(scales),
                dtype is numpy.float32,
            )[1:]
            / counts
        )
    numpy.testing.assert_array_equal(got, expected)
