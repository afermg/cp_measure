"""Numba Zernike-moment kernel (fused basis-eval + weighted-complex segment-sum).

Both Zernike features — shape (`get_zernike`, weight=1) and intensity-weighted
(`get_radial_zernikes`, weight=pixel) — reduce to the same per-object
weighted-complex sum of the Zernike polynomial basis ``V_nm`` over the object's
pixels. This module fuses Horner basis evaluation and the segment-sum into one
single-pass ``@njit`` kernel (no ``(M, K)`` intermediate).

The polynomial *coefficients* come from centrosome (``construct_zernike_lookuptable``,
degree-only, cheap) via :func:`zernike_coeffs`; the per-pixel evaluation follows
``centrosome.zernike.construct_zernike_polynomials``:

- radial part ``R_nm(r)/r^m`` via Horner over ``r² = xm²+ym²`` using the LUT row,
  looping exactly ``(n-m)//2 + 1`` terms (forward order, LUT[k,0] = highest power),
- azimuthal factor ``z^m`` with ``z = ym + 1j*xm`` (ym = normalised row offset,
  xm = normalised column offset),
- strict unit-disk cutoff: zero where ``r² > 1`` (keep ``== 1``).

``minimum_enclosing_circle`` (center/radius) and label enumeration stay on the host.
"""

import centrosome.cpmorphology
import centrosome.zernike
import numpy as np
from numba import njit
from numpy.typing import NDArray


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
def zernike_moments(weights, xm, ym, seg0, lut, nterms, m_arr, n):
    """Fused per-object weighted-complex Zernike sum (single pass, no (M,K) buffer).

    For each pixel: evaluate every polynomial's ``V_nm`` from ``lut``/``m_arr``
    (Horner over r² + ``z^m``, ``z = ym + 1j*xm``; zero the basis when ``r² > 1``)
    and scatter-add ``weight·Re/Im`` into ``vr``/``vi`` ``(n, K)``. ``seg0[i]`` is the
    0-based object index of pixel ``i`` (``< 0`` ⇒ skip). Returns ``(vr, vi)``.
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
        for mm in range(1, max_m + 1):  # z^mm = z^(mm-1) * (y + i x)
            zr[mm] = zr[mm - 1] * y - zi[mm - 1] * x
            zi[mm] = zr[mm - 1] * x + zi[mm - 1] * y
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
    """Per-object weighted-complex Zernike sum shared by both 2D backends.

    Does the part identical to shape and radial Zernikes: enumerate objects,
    find the per-object minimum enclosing circle, normalise each labeled pixel's
    coordinates into the unit disk (``ym`` from rows, ``xm`` from cols), and run the
    fused kernel. ``pixels`` is the intensity image for the weighted (radial) case
    or ``None`` for the unweighted (shape) case. ``coeffs`` is :func:`zernike_coeffs`.

    Returns ``(vr, vi, radii, seg0)``: ``vr``/``vi`` are ``(n, K)`` Re/Im sums,
    ``radii`` ``(n,)`` the enclosing-circle radii (shape's ``π·r²`` denominator),
    ``seg0`` ``(M,)`` the 0-based object index per pixel (radial's pixel-count
    denominator via ``bincount``). ``n == vr.shape[0]`` is 0 when there are no objects.
    """
    lut, nterms, m_arr = coeffs
    K = lut.shape[0]
    # Raw measurements use dense 1..N labels; bulk/featurizer entry points
    # sanitize arbitrary IDs before dispatch, just like the NumPy backend.
    n = int(labels.max()) if labels.size else 0
    if n == 0:
        z = np.zeros((0, K))
        return z, z, np.zeros(0), np.zeros(0, np.int64)

    centers, radii = centrosome.cpmorphology.minimum_enclosing_circle(
        labels, np.arange(1, n + 1)
    )
    radii = np.asarray(radii, dtype=np.float64)
    rows, cols = np.nonzero(labels)
    seg0 = labels[rows, cols].astype(np.int64) - 1
    # Single-pixel objects intentionally produce NaN coordinates, as in NumPy.
    with np.errstate(invalid="ignore", divide="ignore"):
        ym = (rows - centers[seg0, 0]) / radii[seg0]
        xm = (cols - centers[seg0, 1]) / radii[seg0]
    weights = (
        np.ones(seg0.shape[0])
        if pixels is None
        else pixels[rows, cols].astype(np.float64)
    )
    vr, vi = zernike_moments(weights, xm, ym, seg0, lut, nterms, m_arr, n)
    return vr, vi, radii, seg0
