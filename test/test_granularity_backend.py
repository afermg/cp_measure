"""Correctness of the numba granularity backend vs the numpy baseline."""

import numpy
import pytest

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
    for (m, p), got in zip(scenes, out):
        _assert_same(got, gran_numpy(m, p))


def test_batch_list_ragged():
    a = _scene(H=80, W=80, seed=0)
    b = _scene(H=64, W=96, n=3, seed=1)
    out = gran_numba([a[0], b[0]], [a[1], b[1]])
    assert isinstance(out, list) and len(out) == 2
    _assert_same(out[0], gran_numpy(*a))
    _assert_same(out[1], gran_numpy(*b))


def test_3d_volume_falls_back_to_numpy():
    # Z>1 dispatches to the numpy baseline. Use fullres: the baseline collapses Z
    # to int(Z*subsample_size) and errors when that hits 0, so a thin volume at the
    # 0.25 default cannot be subsampled at all (a baseline limitation, not ours).
    rng = numpy.random.default_rng(5)
    pixels = rng.random((3, 40, 40))
    mask = numpy.zeros((3, 40, 40), numpy.int32)
    mask[:, 10:25, 10:25] = 1
    _assert_same(
        gran_numba(mask, pixels, subsample_size=1.0, image_sample_size=1.0),
        gran_numpy(mask, pixels, subsample_size=1.0, image_sample_size=1.0),
    )
