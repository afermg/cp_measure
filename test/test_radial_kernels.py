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
def test_radial_reduce_matches_numpy():
    """radial_reduce histograms + wedge-CV vs a direct numpy computation."""
    from cp_measure.core.numba._radial import radial_reduce

    rng = np.random.default_rng(0)
    n, bin_count = 3, 4
    M = 400
    values = rng.random(M)
    seg0 = rng.integers(0, n, M)
    bin_idx = rng.integers(0, bin_count + 1, M)
    wedge_idx = rng.integers(0, 8, M)
    fad, mfr, cv = radial_reduce(
        np.ascontiguousarray(values),
        seg0.astype(np.int64),
        bin_idx.astype(np.int64),
        wedge_idx.astype(np.int64),
        n,
        bin_count,
    )
    nb = bin_count + 1
    hist = np.zeros((n, nb))
    num = np.zeros((n, nb))
    wsum = np.zeros((n, nb, 8))
    wcnt = np.zeros((n, nb, 8))
    for v, o, b, w in zip(values, seg0, bin_idx, wedge_idx):
        hist[o, b] += v
        num[o, b] += 1
        wsum[o, b, w] += v
        wcnt[o, b, w] += 1
    eps = np.finfo(float).eps
    fad_ref = hist / hist.sum(1, keepdims=True)
    fab = num / num.sum(1, keepdims=True)
    mfr_ref = fad_ref / (fab + eps)
    np.testing.assert_allclose(fad, fad_ref, rtol=1e-9)
    np.testing.assert_allclose(mfr, mfr_ref, rtol=1e-9)
    for o in range(n):
        for b in range(nb):
            pop = wcnt[o, b] > 0
            if pop.sum() == 0:
                assert cv[o, b] == 0.0
            else:
                means = wsum[o, b, pop] / wcnt[o, b, pop]
                expected = np.std(means) / np.mean(means)
                np.testing.assert_allclose(cv[o, b], expected, rtol=1e-9)
