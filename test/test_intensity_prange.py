"""The 2D intensity prange batch kernel must be byte-identical to looping the
current serial backend (it reuses the same njit helpers)."""

import numpy as np
import pytest
from conftest import SIZE_2D, _stamp_objects_2d, get_rng

pytest.importorskip("numba")

from cp_measure.core.numba._intensity import intensity_2d_batch  # noqa: E402
from cp_measure.core.numba.measureobjectintensity import (  # noqa: E402
    _intensity_volume,
)
from cp_measure.primitives.shapes import to_bzyx  # noqa: E402


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
