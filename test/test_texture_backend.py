"""Backend correctness: numba texture must match the numpy (mahotas) backend.

texture is already per-object (regionprops crops), so the numba result matches the
numpy baseline directly (no Issue-#22 isolation trick). 2D + 3D, single + batch,
default and non-default scale/gray_levels.
"""

import numpy as np
import pytest
from conftest import (
    DEPTH_3D,
    SIZE_2D,
    SIZE_3D,
    _stamp_objects_2d,
    _stamp_objects_3d,
    get_rng,
)

import cp_measure.core.measuretexture as ref
from cp_measure._detect import HAS_NUMBA

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


def _numba():
    from cp_measure.core.numba.measuretexture import get_texture

    return get_texture


def _data_2d():
    mask = np.zeros((SIZE_2D, SIZE_2D), np.int32)
    _stamp_objects_2d(mask, n_objects=3)
    return mask, get_rng().random((SIZE_2D, SIZE_2D))


def _data_3d():
    mask = np.zeros((DEPTH_3D, SIZE_3D, SIZE_3D), np.int32)
    _stamp_objects_3d(mask, n_objects=2)
    return mask, get_rng().random((DEPTH_3D, SIZE_3D, SIZE_3D))


def _assert_close(got, exp):
    assert set(got) == set(exp), set(got).symmetric_difference(exp)
    for key in exp:
        np.testing.assert_allclose(
            got[key], exp[key], rtol=1e-6, atol=1e-8, equal_nan=True, err_msg=key
        )


@requires_numba
@pytest.mark.parametrize(
    "scale,gray", [(3, 256), (2, 128)], ids=["default", "scale2gray128"]
)
def test_numba_texture_matches_numpy_2d(scale, gray):
    mask, pixels = _data_2d()
    _assert_close(
        _numba()(mask, pixels, scale=scale, gray_levels=gray),
        ref.get_texture(mask, pixels, scale=scale, gray_levels=gray),
    )


@requires_numba
def test_numba_texture_matches_numpy_3d():
    mask, pixels = _data_3d()
    _assert_close(_numba()(mask, pixels), ref.get_texture(mask, pixels))


@requires_numba
def test_batch_list_matches_per_image():
    imgs = [_data_2d(), _data_3d()]
    masks = [m for m, _ in imgs]
    pix = [p for _, p in imgs]
    got = _numba()(masks, pix)
    assert isinstance(got, list) and len(got) == 2
    for (m, p), per_image in zip(imgs, got):
        _assert_close(per_image, ref.get_texture(m, p))


@requires_numba
def test_empty_image_empty_arrays():
    mask = np.zeros((SIZE_2D, SIZE_2D), np.int32)
    pixels = get_rng().random((SIZE_2D, SIZE_2D))
    got = _numba()(mask, pixels)
    assert set(got) == set(ref.get_texture(mask, pixels))
    assert all(v.shape == (0,) for v in got.values())
