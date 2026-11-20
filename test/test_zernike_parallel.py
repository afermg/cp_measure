"""zernike and radial_zernikes dispatch a 2D batch through the shared thread-pool
driver, with results byte-identical to the per-image serial path."""

import numpy as np
import pytest
from conftest import SIZE_2D, _stamp_objects_2d, get_rng

pytest.importorskip("numba")

from cp_measure.core.numba.measureobjectintensitydistribution import (  # noqa: E402
    get_radial_zernikes,
)
from cp_measure.core.numba.measureobjectsizeshape import get_zernike  # noqa: E402


def _batch(n_imgs, n_obj):
    """A batch whose images each have a distinct object count, so a slot-misordering
    bug in the batch dispatch changes per-image results and the match test catches it."""
    rng = get_rng()
    masks, pix = [], []
    for idx in range(n_imgs):
        m = np.zeros((SIZE_2D, SIZE_2D), np.int32)
        _stamp_objects_2d(m, n_objects=n_obj + idx)
        masks.append(m)
        pix.append(rng.random((SIZE_2D, SIZE_2D)))
    return masks, pix


@pytest.mark.parametrize(
    "fn", [lambda m, p: get_zernike(m), lambda m, p: get_radial_zernikes(m, p)]
)
def test_zernike_batch_matches_serial(fn):
    masks, pix = _batch(3, 5)
    batch = fn(masks, pix)
    assert isinstance(batch, list) and len(batch) == 3
    for b, (m, p) in enumerate(zip(masks, pix)):
        single = fn(m, p)
        assert set(batch[b]) == set(single)
        for k in single:
            np.testing.assert_array_equal(batch[b][k], single[k], err_msg=k)


def test_radial_zernikes_accept_non_native_byte_order():
    """Big-endian pixels and labels give the same values as native ones."""
    masks, pix = _batch(2, 4)
    swapped_masks = [m.astype(m.dtype.newbyteorder()) for m in masks]
    swapped_pix = [p.astype(p.dtype.newbyteorder()) for p in pix]
    native = get_radial_zernikes(masks, pix)
    swapped = get_radial_zernikes(swapped_masks, swapped_pix)
    for a, b in zip(native, swapped):
        for k in a:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
