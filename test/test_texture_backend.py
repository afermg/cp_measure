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


def _data_2d(seed=42):
    mask = np.zeros((SIZE_2D, SIZE_2D), np.int32)
    _stamp_objects_2d(mask, n_objects=3)
    return mask, get_rng(seed).random((SIZE_2D, SIZE_2D))


def _data_3d():
    mask = np.zeros((DEPTH_3D, SIZE_3D, SIZE_3D), np.int32)
    _stamp_objects_3d(mask, n_objects=2)
    return mask, get_rng().random((DEPTH_3D, SIZE_3D, SIZE_3D))


def _assert_close(got, exp):
    # Variance and Correlation use the centred form where mahotas uses the raw
    # dot(px, k**2) - ux**2; they differ by up to ~1e-6 relative on low-contrast
    # objects, in our favour. Everything else tracks mahotas far more tightly.
    assert set(got) == set(exp), set(got).symmetric_difference(exp)
    for key in exp:
        loose = key.startswith(("Variance", "Correlation"))
        np.testing.assert_allclose(
            got[key],
            exp[key],
            rtol=1e-5 if loose else 1e-6,
            atol=1e-8,
            equal_nan=True,
            err_msg=key,
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
    # Same-shape images, different pixels: a per-image mix-up has to show up.
    imgs = [_data_2d(seed=1), _data_2d(seed=2)]
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


@requires_numba
def test_single_slice_volume_matches_numpy():
    """A (1, Y, X) volume is a volume, not a 2D image: 13 directions in both backends.

    The dimensionality rule lives in ``measuretexture._prep``, so the backends
    cannot drift apart here — a backend reading it off an array it has already
    normalised would see Z == 1 and measure 4 directions instead.
    """
    mask, pixels = _data_2d()
    volume_mask, volume_pixels = mask[np.newaxis], pixels[np.newaxis]
    _assert_close(
        _numba()(volume_mask, volume_pixels),
        ref.get_texture(volume_mask, volume_pixels),
    )


@requires_numba
def test_mask_and_pixels_shape_mismatch_raises():
    """A 2D mask over a 3D stack used to measure z=0 and silently drop the rest."""
    mask, _ = _data_2d()
    volume = get_rng().random((4, SIZE_2D, SIZE_2D))
    with pytest.raises(ValueError, match="same shape"):
        _numba()(mask, volume)
    with pytest.raises(ValueError, match="same shape"):
        ref.get_texture(mask, volume)


@requires_numba
@pytest.mark.parametrize("scale", [0, -3])
def test_non_positive_scale_raises(scale):
    """mahotas rejects a negative distance; reimplementing it must not accept one.

    The symmetric GLCM makes a negative scale look plausible (it walks the other
    way and files the result under a ``_-3_`` key), and scale=0 pairs every pixel
    with itself, so both backends refuse rather than return something meaningless.
    """
    mask, pixels = _data_2d()
    with pytest.raises(ValueError, match="scale"):
        _numba()(mask, pixels, scale=scale)
    with pytest.raises(ValueError, match="scale"):
        ref.get_texture(mask, pixels, scale=scale)
