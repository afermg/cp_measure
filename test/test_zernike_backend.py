"""Correctness of the numba Zernike backends vs the numpy baseline."""

import numpy
import pytest

from cp_measure.core.measureobjectintensitydistribution import (
    get_radial_zernikes as rz_numpy,
)
from cp_measure.core.measureobjectsizeshape import get_zernike as z_numpy
from cp_measure.core.numba.measureobjectintensitydistribution import (
    get_radial_zernikes as rz_numba,
)
from cp_measure.core.numba.measureobjectsizeshape import get_zernike as z_numba

TOL = dict(rtol=1e-6, atol=1e-8, equal_nan=True)


def _scene(H=80, W=80, n=4, seed=0):
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


def test_radial_zernikes_matches_baseline():
    mask, pixels = _scene()
    _assert_same(rz_numba(mask, pixels), rz_numpy(mask, pixels))


def test_radial_zernikes_single_object():
    mask = numpy.zeros((50, 50), numpy.int32)
    mask[15:35, 15:35] = 1
    pixels = numpy.random.default_rng(1).random((50, 50))
    _assert_same(rz_numba(mask, pixels), rz_numpy(mask, pixels))


def test_radial_zernikes_label_gaps_numba_is_robust():
    # cp_measure masks are contiguous 1..n; on a GAP the numpy baseline raises
    # (it indexes the per-object center array by `label-1`). Our compacted
    # searchsorted indexing handles gaps instead of crashing -- strictly more
    # robust, so there's no baseline value to match here; just assert it works.
    mask = numpy.zeros((60, 60), numpy.int32)
    mask[5:20, 5:20] = 1
    mask[35:55, 35:55] = 3  # gap at label 2
    pixels = numpy.random.default_rng(2).random((60, 60))
    with pytest.raises(IndexError):
        rz_numpy(mask, pixels)
    out = rz_numba(mask, pixels)
    assert (
        len(out["RadialDistribution_ZernikeMagnitude_0_0"]) == 2
    )  # two present objects


def test_radial_zernikes_single_returns_dict():
    mask, pixels = _scene()
    assert isinstance(rz_numba(mask, pixels), dict)


def test_radial_zernikes_batch_matches_per_image():
    scenes = [_scene(seed=s) for s in range(3)]
    masks = numpy.stack([m[numpy.newaxis] for m, _ in scenes])
    pixels = numpy.stack([p[numpy.newaxis] for _, p in scenes])
    out = rz_numba(masks, pixels)
    assert isinstance(out, list) and len(out) == 3
    for (m, p), got in zip(scenes, out):
        _assert_same(got, rz_numpy(m, p))


def test_radial_zernikes_3d_returns_empty():
    mask = numpy.zeros((3, 40, 40), numpy.int32)
    mask[:, 10:25, 10:25] = 1
    pixels = numpy.random.default_rng(5).random((3, 40, 40))
    assert rz_numba(mask, pixels) == {}


# --- shape zernike (get_zernike) ---


def test_zernike_matches_baseline():
    mask, pixels = _scene()
    _assert_same(z_numba(mask, pixels), z_numpy(mask, pixels))


def test_zernike_single_object():
    mask = numpy.zeros((50, 50), numpy.int32)
    mask[15:35, 15:35] = 1
    pixels = numpy.random.default_rng(1).random((50, 50))
    _assert_same(z_numba(mask, pixels), z_numpy(mask, pixels))


def test_zernike_batch_matches_per_image():
    scenes = [_scene(seed=s) for s in range(3)]
    masks = numpy.stack([m[numpy.newaxis] for m, _ in scenes])
    pixels = numpy.stack([p[numpy.newaxis] for _, p in scenes])
    out = z_numba(masks, pixels)
    assert isinstance(out, list) and len(out) == 3
    for (m, _p), got in zip(scenes, out):
        _assert_same(got, z_numpy(m, None))


def test_zernike_3d_returns_empty():
    mask = numpy.zeros((3, 40, 40), numpy.int32)
    mask[:, 10:25, 10:25] = 1
    assert z_numba(mask, None) == {}
