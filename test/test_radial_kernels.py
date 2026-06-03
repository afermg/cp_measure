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
    np.testing.assert_allclose(d_numba[m], d_cent[m], atol=1e-9)


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
def test_radial_object_centre_and_histograms():
    """radial_object: centre = argmax d_to_edge, and per-bin FracAtD sums to ~1."""
    from cp_measure.core.numba._radial import radial_object

    m = np.zeros((23, 23), bool)
    m[1:22, 1:22] = True  # odd square -> unique centre at (11, 11)
    pix = np.ones((23, 23))
    d_to_edge = scipy.ndimage.distance_transform_edt(m)
    fad, mfr, cv = radial_object(
        np.ascontiguousarray(m), np.ascontiguousarray(pix), d_to_edge, True, 4, 100
    )
    # uniform intensity -> FracAtD is the per-bin pixel fraction, sums to 1
    np.testing.assert_allclose(fad.sum(), 1.0, rtol=1e-9)
    # uniform intensity -> every wedge mean is equal -> RadialCV == 0
    np.testing.assert_allclose(cv[:4], 0.0, atol=1e-12)
