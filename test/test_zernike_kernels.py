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


@pytest.mark.parametrize("seed", range(4))
def test_enclosing_circles_match_centrosome(seed):
    """The minimum enclosing circle is unique, so the numba port must return
    centrosome's circle: the same centre, and the radius to rounding. Shapes cover
    ragged and holed objects, thin diagonal lines, single pixels and pixel pairs,
    plus a label missing from the image (NaN centre, radius 0)."""
    from skimage.draw import line
    from skimage.measure import label

    from cp_measure.core.numba._zernike import enclosing_circles

    rng = numpy.random.default_rng(seed)
    for _ in range(40):
        side = int(rng.integers(4, 60))
        kind = rng.integers(3)
        if kind == 0:
            labels = label(rng.random((side, side)) < rng.uniform(0.2, 0.6))
        elif kind == 1:
            labels = numpy.zeros((side, side), int)
            for k in range(int(rng.integers(1, 5))):
                rr, cc = line(*rng.integers(0, side, 4))
                labels[rr, cc] = k + 1
        else:
            labels = numpy.zeros((side, side), int)
            pts = rng.choice(side * side - 1, int(rng.integers(1, 6)), replace=False)
            labels.flat[pts] = numpy.arange(1, pts.size + 1)
            labels.flat[pts[::2] + 1] = numpy.arange(1, pts.size + 1)[::2]
        n = int(labels.max()) + 1  # label n is absent
        want_c, want_r = centrosome.zernike.minimum_enclosing_circle(
            labels, numpy.arange(1, n + 1)
        )
        got_c, got_r = enclosing_circles(labels, n)
        numpy.testing.assert_array_equal(numpy.isnan(got_c), numpy.isnan(want_c))
        numpy.testing.assert_allclose(got_c, want_c, rtol=0, atol=1e-12)
        numpy.testing.assert_allclose(got_r, want_r, rtol=1e-12, atol=1e-12)
