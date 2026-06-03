"""Lock the Zernike conventions: our basis eval must match centrosome exactly."""

import centrosome.zernike
import numpy
import pytest

from cp_measure.core.numba._zernike import (
    _zernike_basis_numpy,
    zernike_coeffs,
    zernike_moments,
)


@pytest.mark.parametrize("degree", [3, 6, 9])
def test_basis_matches_centrosome(degree):
    indexes = centrosome.zernike.get_zernike_indexes(degree + 1)
    rng = numpy.random.default_rng(degree)
    # normalised coords spanning inside, on, and outside the unit disk
    xm = rng.uniform(-1.3, 1.3, size=400)
    ym = rng.uniform(-1.3, 1.3, size=400)

    # centrosome: construct_zernike_polynomials(x=col, y=row) -> (M, K) complex
    ref = centrosome.zernike.construct_zernike_polynomials(xm, ym, indexes)

    lut, nterms, m_arr = zernike_coeffs(indexes)
    got = _zernike_basis_numpy(xm, ym, lut, nterms, m_arr)

    assert got.shape == ref.shape
    numpy.testing.assert_array_equal(got, ref)  # identical ops -> bit-exact


def test_disk_cutoff_strict():
    # r^2 == 1 exactly is KEPT (centrosome uses strict >), r^2 > 1 is zeroed
    indexes = centrosome.zernike.get_zernike_indexes(4)
    lut, nterms, m_arr = zernike_coeffs(indexes)
    xm = numpy.array([1.0, 0.0, 0.8])  # r^2 = 1.0 (kept), 0 (kept), 0.64+?
    ym = numpy.array([0.0, 1.0, 0.8])  # second r^2=1 kept; third r^2=1.28 zeroed
    got = _zernike_basis_numpy(xm, ym, lut, nterms, m_arr)
    ref = centrosome.zernike.construct_zernike_polynomials(xm, ym, indexes)
    numpy.testing.assert_array_equal(got, ref)
    # the n=0,m=0 polynomial is the radial constant: kept where r^2<=1, 0 where >1
    assert got[0, 0] != 0 and got[1, 0] != 0 and got[2, 0] == 0


def test_nterms_matches_index_structure():
    indexes = centrosome.zernike.get_zernike_indexes(10)
    _, nterms, m_arr = zernike_coeffs(indexes)
    n = indexes[:, 0]
    numpy.testing.assert_array_equal(nterms, (n - m_arr) // 2 + 1)


@pytest.mark.parametrize("weighted", [False, True], ids=["shape", "intensity"])
def test_zernike_moments_matches_basis_plus_segment_sum(weighted):
    indexes = centrosome.zernike.get_zernike_indexes(10)
    lut, nterms, m_arr = zernike_coeffs(indexes)
    rng = numpy.random.default_rng(1 + weighted)
    M, n = 600, 4
    xm = rng.uniform(-1.2, 1.2, size=M)
    ym = rng.uniform(-1.2, 1.2, size=M)
    seg0 = rng.integers(0, n, size=M).astype(numpy.int64)
    weights = rng.random(M) if weighted else numpy.ones(M)

    z = _zernike_basis_numpy(xm, ym, lut, nterms, m_arr)  # (M, K), bit-exact basis
    K = lut.shape[0]
    vr_ref = numpy.zeros((n, K))
    vi_ref = numpy.zeros((n, K))
    numpy.add.at(vr_ref, seg0, weights[:, None] * z.real)
    numpy.add.at(vi_ref, seg0, weights[:, None] * z.imag)

    vr, vi = zernike_moments(weights, xm, ym, seg0, lut, nterms, m_arr, n)
    numpy.testing.assert_allclose(vr, vr_ref, rtol=1e-9, atol=1e-12)
    numpy.testing.assert_allclose(vi, vi_ref, rtol=1e-9, atol=1e-12)


def test_zernike_moments_skips_seg_negative_and_outside_disk():
    indexes = centrosome.zernike.get_zernike_indexes(4)
    lut, nterms, m_arr = zernike_coeffs(indexes)
    # one in-disk pixel (seg 0), one outside disk (skipped), one seg -1 (skipped)
    xm = numpy.array([0.3, 0.9, 0.1])
    ym = numpy.array([0.3, 0.9, 0.1])
    seg0 = numpy.array([0, 0, -1], dtype=numpy.int64)
    weights = numpy.ones(3)
    vr, vi = zernike_moments(weights, xm, ym, seg0, lut, nterms, m_arr, 1)
    # only the first pixel contributes (r^2=0.18<=1); compare to a manual single-pixel basis
    z = _zernike_basis_numpy(xm[:1], ym[:1], lut, nterms, m_arr)
    numpy.testing.assert_allclose(vr[0], z[0].real, rtol=1e-9, atol=1e-12)
