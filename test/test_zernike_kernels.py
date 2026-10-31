"""The fused numba Zernike kernel vs centrosome's basis plus a segment sum."""

import centrosome.zernike
import numpy
import pytest

from cp_measure.core.numba._zernike import zernike_coeffs, zernike_moments


@pytest.mark.parametrize("weighted", [False, True], ids=["shape", "intensity"])
def test_zernike_moments_matches_centrosome(weighted):
    indexes = centrosome.zernike.get_zernike_indexes(10)
    rng = numpy.random.default_rng(1 + weighted)
    n = 4
    # Random points inside and outside the unit disk, plus points exactly on it:
    # centrosome keeps r^2 == 1 (strict > cutoff), so the kernel must too.
    xm = numpy.concatenate([rng.uniform(-1.2, 1.2, 600), [1.0, 0.0, -1.0, 0.0]])
    ym = numpy.concatenate([rng.uniform(-1.2, 1.2, 600), [0.0, 1.0, 0.0, -1.0]])
    seg0 = rng.integers(-1, n, size=xm.size)  # -1: background, skipped
    seg0[-4:] = numpy.arange(4)  # the on-circle points must count
    weights = rng.random(xm.size) if weighted else numpy.ones(xm.size)

    basis = centrosome.zernike.construct_zernike_polynomials(xm, ym, indexes)
    keep = seg0 >= 0
    vr_ref = numpy.zeros((n, len(indexes)))
    vi_ref = numpy.zeros((n, len(indexes)))
    numpy.add.at(vr_ref, seg0[keep], (weights[:, None] * basis.real)[keep])
    numpy.add.at(vi_ref, seg0[keep], (weights[:, None] * basis.imag)[keep])

    lut, nterms, m_arr = zernike_coeffs(indexes)
    vr, vi = zernike_moments(weights, xm, ym, seg0, lut, nterms, m_arr, n)
    numpy.testing.assert_allclose(vr, vr_ref, rtol=1e-9, atol=1e-12)
    numpy.testing.assert_allclose(vi, vi_ref, rtol=1e-9, atol=1e-12)
