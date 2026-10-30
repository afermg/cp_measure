"""Unit tests for the radial_distribution numba kernels."""

import centrosome.cpmorphology
import centrosome.propagate
import numpy as np
import pytest
import scipy.ndimage

from cp_measure._detect import HAS_NUMBA

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


def _shapes():
    sq = np.zeros((60, 60), bool)
    sq[5:55, 5:55] = True
    L = np.zeros((60, 60), bool)
    L[5:55, 5:25] = True
    L[35:55, 5:55] = True
    U = np.zeros((60, 60), bool)
    U[5:55, 5:55] = True
    U[5:40, 20:40] = False
    yy, xx = np.mgrid[0:80, 0:80]
    r = np.sqrt((yy - 40) ** 2 + (xx - 40) ** 2)
    ring = (r > 20) & (r < 35)
    tiny = np.zeros((5, 5), bool)
    tiny[2, 2] = True
    return {"square": sq, "L": L, "U": U, "ring": ring, "tiny": tiny}


def _seed(m):
    d = scipy.ndimage.distance_transform_edt(m)
    i, j = centrosome.cpmorphology.maximum_position_of_labels(
        d, m.astype(np.int32), indices=np.array([1])
    )
    return int(i[0]), int(j[0])


@requires_numba
@pytest.mark.parametrize("name", ["square", "L", "U", "ring", "tiny"])
def test_geodesic_matches_centrosome_propagate(name):
    """The numba chamfer geodesic is bit-exact vs centrosome propagate."""
    from cp_measure.core.numba._radial import geodesic_chamfer_fifo

    m = _shapes()[name]
    si, sj = _seed(m)
    d_numba = geodesic_chamfer_fifo(np.ascontiguousarray(m), si, sj)
    center = np.zeros(m.shape, int)
    center[si, sj] = 1
    _, d_cent = centrosome.propagate.propagate(np.zeros(m.shape), center, m, 1)
    np.testing.assert_array_equal(d_numba[m], d_cent[m])


@requires_numba
def test_geodesic_leaves_disconnected_unreached():
    from cp_measure.core.numba._radial import UNREACHED, geodesic_chamfer_fifo

    m = np.zeros((20, 20), bool)
    m[2:8, 2:8] = True  # component A (holds the seed)
    m[12:18, 12:18] = True  # component B (disconnected)
    d = geodesic_chamfer_fifo(np.ascontiguousarray(m), 4, 4)
    assert d[4, 4] == 0.0
    assert np.all(d[12:18, 12:18] >= UNREACHED)  # never reached


@requires_numba
def test_radial_object_uses_first_c_order_center_on_plateau():
    """A subpixel first bin contains only the first of four tied EDT maxima."""
    from cp_measure.core.numba._radial import radial_object

    mask = np.zeros((8, 8), dtype=bool)
    mask[1:7, 1:7] = True
    distance = scipy.ndimage.distance_transform_edt(mask)
    np.testing.assert_array_equal(
        np.argwhere(distance == distance.max()), [[3, 3], [3, 4], [4, 3], [4, 4]]
    )
    pixels = np.arange(1, 65, dtype=np.float64).reshape(8, 8)
    frac, mean, sums, counts = radial_object(mask, pixels, distance, False, 1, 0.5)
    assert frac[0] == pixels[3, 3] / pixels[mask].sum()
    assert mean[0] > 0
    assert sums[0, 0] == pixels[3, 3]
    assert counts[0, 0] == counts[0].sum() == 1
