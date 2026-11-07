"""zernike and radial_zernikes dispatch a 2D batch through the shared thread-pool
driver, with results byte-identical to the per-image serial path."""

import numpy as np
import pytest
from conftest import SIZE_2D, _stamp_objects_2d, get_rng

pytest.importorskip("numba")

import cp_measure.core.numba._zernike as zmod  # noqa: E402
from cp_measure.core.numba.measureobjectintensitydistribution import (  # noqa: E402
    get_radial_zernikes,
)
from cp_measure.core.numba.measureobjectsizeshape import get_zernike  # noqa: E402


def _batch(n_imgs, n_obj):
    rng = get_rng()
    masks, pix = [], []
    for _ in range(n_imgs):
        m = np.zeros((SIZE_2D, SIZE_2D), np.int32)
        _stamp_objects_2d(m, n_objects=n_obj)
        masks.append(m)
        pix.append(rng.random((SIZE_2D, SIZE_2D)))
    return masks, pix


def test_zernike_batch_uses_map_images(monkeypatch):
    calls = []
    real = getattr(zmod, "map_images", None)

    def spy(func, items, **kw):
        calls.append(len(items))
        return real(func, items, **kw)

    monkeypatch.setattr(zmod, "map_images", spy, raising=False)
    masks, pix = _batch(3, 4)
    get_zernike(masks)
    get_radial_zernikes(masks, pix)
    assert calls and all(c == 3 for c in calls), (
        f"map_images not used for batch: {calls}"
    )


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
