"""The 2D intensity prange batch kernel must be byte-identical to looping the
current serial backend (it reuses the same njit helpers)."""

import numpy as np
import pytest
from conftest import SIZE_2D, _stamp_objects_2d, get_rng

pytest.importorskip("numba")

import cp_measure.core.numba.measureobjectintensity as moi  # noqa: E402
from cp_measure.core.numba._intensity import intensity_2d_batch  # noqa: E402
from cp_measure.core.numba.measureobjectintensity import (  # noqa: E402
    _intensity_volume,
    get_intensity,
)
from cp_measure.primitives.shapes import to_bzyx  # noqa: E402
from conftest import DEPTH_3D, SIZE_3D, _stamp_objects_3d  # noqa: E402


def _batch(counts):
    rng = get_rng()
    masks, pix = [], []
    for n in counts:
        m = np.zeros((SIZE_2D, SIZE_2D), np.int32)
        if n > 0:
            _stamp_objects_2d(m, n_objects=n)
        masks.append(m)
        pix.append(rng.random((SIZE_2D, SIZE_2D)))
    return masks, pix


@pytest.mark.parametrize("legacy", [False, True], ids=["new", "legacy"])
@pytest.mark.parametrize("edge", [False, True], ids=["edge", "noedge"])
def test_batch_kernel_matches_serial(edge, legacy):
    masks, pix = _batch([3, 5, 0])  # distinct object counts, including one empty
    masks_b, pix_b, _ = to_bzyx(masks, pix)
    out, nobjs = intensity_2d_batch(masks_b, pix_b, legacy, edge)
    for b, (m, p) in enumerate(zip(masks, pix)):
        ref = _intensity_volume(
            m[np.newaxis],
            p[np.newaxis],
            edge_measurements=edge,
            legacy=legacy,
            spatial_ndim=2,
        )
        ref_vals = list(ref.values())  # insertion order == kernel column order
        n = int(m.max())
        assert nobjs[b] == n
        assert out.shape[1] == len(ref_vals)
        for f, arr in enumerate(ref_vals):
            np.testing.assert_array_equal(
                out[b, f, :n], arr, err_msg=f"img {b} feature col {f}"
            )


def _assert_dicts_equal(got, ref):
    assert set(got) == set(ref), set(got).symmetric_difference(ref)
    for k in ref:
        np.testing.assert_array_equal(got[k], ref[k], err_msg=k)


def test_list_2d_uses_the_kernel(monkeypatch):
    """A list of 2D images must actually route through the prange kernel."""
    calls = []
    real = intensity_2d_batch

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(moi, "intensity_2d_batch", spy, raising=False)
    masks, pix = _batch([3, 5])
    get_intensity(masks, pix)
    assert calls, "list-of-2D batch did not reach intensity_2d_batch"


@pytest.mark.parametrize("edge", [False, True], ids=["noedge", "edge"])
def test_get_intensity_list_batch_matches_single(edge):
    """A list of 2D images (fast path) matches per-image serial calls."""
    masks, pix = _batch([3, 5, 0])
    batch = get_intensity(masks, pix, edge_measurements=edge)
    assert isinstance(batch, list) and len(batch) == 3
    for b, (m, p) in enumerate(zip(masks, pix)):
        _assert_dicts_equal(batch[b], get_intensity(m, p, edge_measurements=edge))


def test_single_image_is_byte_identical():
    """Single 2D/3D inputs stay on the serial path, byte-identical to _intensity_volume."""
    m, p = _batch([4])[0][0], _batch([4])[1][0]
    _assert_dicts_equal(
        get_intensity(m, p),
        _intensity_volume(m[np.newaxis], p[np.newaxis], spatial_ndim=2),
    )
    m3 = np.zeros((DEPTH_3D, SIZE_3D, SIZE_3D), np.int32)
    _stamp_objects_3d(m3, n_objects=2)
    p3 = get_rng().random((DEPTH_3D, SIZE_3D, SIZE_3D))
    _assert_dicts_equal(
        get_intensity(m3, p3), _intensity_volume(m3, p3, spatial_ndim=3)
    )


def test_3d_and_4d_batches_route_serial(monkeypatch):
    """3D-volume batches must not hit the 2D kernel (different MAD fraction)."""

    def forbidden(*a, **k):
        raise AssertionError("2D kernel reached on a 3D-volume batch")

    monkeypatch.setattr(moi, "intensity_2d_batch", forbidden, raising=False)
    m3 = np.zeros((DEPTH_3D, SIZE_3D, SIZE_3D), np.int32)
    _stamp_objects_3d(m3, n_objects=2)
    p3 = get_rng().random((DEPTH_3D, SIZE_3D, SIZE_3D))
    # list of 3D volumes, and a 4D (B,1,Y,X) singleton-Z batch, both serial
    assert isinstance(get_intensity([m3, m3], [p3, p3]), list)
    masks, pix = _batch([3, 5])
    m4 = np.stack([m[np.newaxis] for m in masks])
    p4 = np.stack([p[np.newaxis] for p in pix])
    assert isinstance(get_intensity(m4, p4), list)
