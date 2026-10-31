"""Numba Zernike-moment kernel (fused basis-eval + weighted-complex segment-sum).

Both Zernike features — shape (`get_zernike`, weight=1) and intensity-weighted
(`get_radial_zernikes`, weight=pixel) — reduce to the same per-object
weighted-complex sum of the Zernike polynomial basis ``V_nm`` over the object's
pixels. This module fuses Horner basis evaluation and the segment-sum into one
single-pass ``@njit`` kernel (no ``(M, K)`` intermediate). The radial path computes
azimuthal powers with NumPy to reduce phase-rounding differences, not to guarantee
identical phases. Meaningful phases agree modulo 2π; phases of near-zero moments
are numerically unstable (see ``docs/measurement-notes.md``).

The polynomial *coefficients* come from centrosome (``construct_zernike_lookuptable``,
degree-only, cheap) via :func:`zernike_coeffs`; the per-pixel evaluation follows
``centrosome.zernike.construct_zernike_polynomials``:

- radial part ``R_nm(r)/r^m`` via Horner over ``r² = xm²+ym²`` using the LUT row,
  looping exactly ``(n-m)//2 + 1`` terms (forward order, LUT[k,0] = highest power),
- azimuthal factor ``z^m`` with ``z = ym + 1j*xm`` (ym = normalised row offset,
  xm = normalised column offset),
- strict unit-disk cutoff: zero where ``r² > 1`` (keep ``== 1``).

``minimum_enclosing_circle`` (center/radius) and label enumeration stay on the host
(:func:`cp_measure.utils._zernike_coords`, shared with the NumPy backend).
"""

import centrosome.zernike
import numpy as np
from numba import njit
from numpy.typing import NDArray

from cp_measure.primitives.shapes import to_bzyx
from cp_measure.utils import _zernike_coords


def zernike_coeffs(
    zernike_indexes: NDArray[np.integer],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Repack centrosome's radial LUT for the Horner kernel.

    Returns ``(lut, nterms, m_arr)`` where ``lut[k, :]`` are the radial-polynomial
    coefficients of index ``k`` (highest power first), ``nterms[k] = (n-m)//2 + 1``
    is exactly how many of them to consume in the Horner loop (trailing LUT entries
    are zero and must NOT be iterated — they would spuriously multiply by r²), and
    ``m_arr[k]`` is the azimuthal order ``m``.
    """
    lut = centrosome.zernike.construct_zernike_lookuptable(zernike_indexes).astype(
        np.float64
    )
    n = zernike_indexes[:, 0].astype(np.int64)
    m_arr = zernike_indexes[:, 1].astype(np.int64)
    nterms = ((n - m_arr) // 2 + 1).astype(np.int64)
    return lut, nterms, m_arr


@njit(cache=True)
def zernike_moments(weights, xm, ym, seg0, lut, nterms, m_arr, n, powers=None):
    """Fused per-object weighted-complex Zernike sum (single pass, no (M,K) buffer).

    For each pixel: evaluate every polynomial's ``V_nm`` from ``lut``/``m_arr``
    (Horner over r² + ``z^m``, ``z = ym + 1j*xm``; zero the basis when ``r² > 1``)
    and scatter-add ``weight·Re/Im`` into ``vr``/``vi`` ``(n, K)``. ``seg0[i]`` is the
    0-based object index of pixel ``i`` (``< 0`` ⇒ skip). Returns ``(vr, vi)``.
    Optional ``powers[m, i]`` supplies NumPy-computed azimuthal powers ``z^m``;
    otherwise the scalar recurrence is evaluated in the kernel.
    """
    K = lut.shape[0]
    M = xm.shape[0]
    vr = np.zeros((n, K))
    vi = np.zeros((n, K))
    max_m = 0
    for k in range(K):
        if m_arr[k] > max_m:
            max_m = m_arr[k]
    zr = np.empty(max_m + 1)
    zi = np.empty(max_m + 1)
    for i in range(M):
        seg = seg0[i]
        if seg < 0:
            continue
        x = xm[i]
        y = ym[i]
        r2 = x * x + y * y
        w = weights[i]
        zr[0] = 1.0
        zi[0] = 0.0
        if powers is None:
            for mm in range(1, max_m + 1):  # z^mm = z^(mm-1) * (y + i x)
                zr[mm] = zr[mm - 1] * y - zi[mm - 1] * x
                zi[mm] = zr[mm - 1] * x + zi[mm - 1] * y
        else:
            for mm in range(1, max_m + 1):
                zr[mm] = powers[mm, i].real
                zi[mm] = powers[mm, i].imag
        for k in range(K):
            s = 0.0
            nt = nterms[k]
            for t in range(nt):  # Horner: R_nm(r)/r^m
                s = s * r2 + lut[k, t]
            if r2 > 1.0:
                s = 0.0  # strict cutoff, before weighting (including NaN/Inf)
            m = m_arr[k]
            s *= w
            if m == 0:
                # The NumPy reference keeps the m=0 imaginary sum identically
                # zero, even for nonfinite weights or zero-radius objects.
                vr[seg, k] += s
            else:
                vr[seg, k] += s * zr[m]
                vi[seg, k] += s * zi[m]
    return vr, vi


def zernike_moments_per_object(labels, pixels, coeffs):
    """Per-object weighted-complex Zernike sums shared by both 2D backends.

    Same contract as :func:`cp_measure.utils._zernike_scores`, whose coordinate
    setup it shares: ``pixels`` is the intensity image for the weighted (radial)
    case or ``None`` for the unweighted (shape) case, ``coeffs`` is
    :func:`zernike_coeffs`. Returns ``(vr, vi, radii, counts)``.
    """
    lut, nterms, m_arr = coeffs
    seg, keep, ym, xm, radii = _zernike_coords(labels)
    n = len(radii)
    if n == 0:
        empty = np.zeros((0, len(lut)))
        return empty, empty, radii, np.zeros(0)
    weights = np.ones(len(seg)) if pixels is None else pixels[keep].astype(np.float64)
    powers = None
    if pixels is not None:
        # Mirror-symmetric objects have moments whose real (or imaginary) part is
        # zero by symmetry, so atan2 returns +pi or -pi on the sign of round-off.
        # Building z^m with NumPy's complex multiply, as _zernike_scores does,
        # reduces these sign differences but cannot guarantee identical phases:
        # even repeated NumPy calls can differ at the branch cut or near zero.
        # Only the (max_m + 1, M) powers are materialised, not the (M, K) basis.
        z = ym + 1j * xm
        powers = np.empty((int(m_arr.max()) + 1, len(z)), dtype=np.complex128)
        powers[0] = 1
        for m in range(1, len(powers)):
            powers[m] = z if m == 1 else powers[m - 1] * z
    vr, vi = zernike_moments(weights, xm, ym, seg, lut, nterms, m_arr, n, powers)
    counts = np.bincount(seg, minlength=n).astype(np.float64)
    return vr, vi, radii, counts


def zernike_per_image(labels, pixels, degree, features):
    """Batch driver shared by the numba ``get_zernike`` and ``get_radial_zernikes``.

    Accepts a single 2D image or a batch (list/tuple of images, or a 4D
    ``(B, Z, Y, X)`` array) and returns a dict or a list of dicts. Singleton-Z
    volumes use the 2D measurements; other volumes give ``{}``, consistently for
    individual inputs and batches. ``pixels`` is ``None`` for the unweighted
    shape moments. For measured entries, weighted
    inputs must have matching shapes and dimensionality, before axis promotion.
    ``features(vr, vi, radii, counts, zernike_indexes)`` names and normalises one
    image's moment sums.
    """
    is_sequence = isinstance(labels, (list, tuple))
    if not is_sequence and np.ndim(labels) == 3 and labels.shape[0] != 1:
        return {}
    weighted = pixels is not None
    masks_bzyx, pixels_bzyx, unwrap = to_bzyx(labels, pixels if weighted else labels)
    # degree-only work, hoisted out of the per-image loop (computed once per batch)
    zernike_indexes = centrosome.zernike.get_zernike_indexes(degree + 1)
    coeffs = zernike_coeffs(zernike_indexes)
    results = []
    for i, (masks_zyx, pixels_zyx) in enumerate(zip(masks_bzyx, pixels_bzyx)):
        mask_ndim = np.ndim(labels[i] if is_sequence else labels)
        if masks_zyx.shape[0] != 1:
            results.append({})
            continue
        if weighted:
            pixel_ndim = np.ndim(pixels[i] if is_sequence else pixels)
            if mask_ndim != pixel_ndim or masks_zyx.shape != pixels_zyx.shape:
                raise ValueError(
                    "radial Zernike labels and pixels must have matching shapes"
                )
        sums = zernike_moments_per_object(
            masks_zyx[0], pixels_zyx[0] if weighted else None, coeffs
        )
        results.append(features(*sums, zernike_indexes))
    return unwrap(results)
