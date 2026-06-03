"""Numba-accelerated radial Zernike backend.

Mirrors :func:`cp_measure.core.measureobjectintensitydistribution.get_radial_zernikes`
but replaces the per-(n,m) ``scipy.ndimage.sum_labels`` AND centrosome's per-pixel
``construct_zernike_polynomials`` with a single fused numba kernel
(:func:`cp_measure.core.numba._zernike.zernike_moments`) that evaluates the basis
and the per-object weighted-complex sum in one pass. Centrosome still provides the
radial LUT (degree-only) and ``minimum_enclosing_circle`` (host).

Batch-shaped via the canonical ``(B, Z, Y, X)`` form (single image = ``B == 1``);
2D-only (a ``Z > 1`` volume returns ``{}``, matching the baseline's ``ndim == 3``).
"""

import centrosome.cpmorphology
import centrosome.zernike
import numpy
from numpy.typing import NDArray

from cp_measure.core.measureobjectintensitydistribution import M_CATEGORY
from cp_measure.core.numba._zernike import zernike_coeffs, zernike_moments
from cp_measure.primitives.segment import label_to_idx_lut
from cp_measure.primitives.shapes import to_bzyx


def get_radial_zernikes(masks, pixels, zernike_degree: int = 9):
    """Radial Zernike magnitude/phase per object; single image/volume or batch."""
    masks_zyx, pixels_zyx, unwrap = to_bzyx(masks, pixels)
    # degree-only work, hoisted out of the per-image loop (computed once per batch)
    zernike_indexes = centrosome.zernike.get_zernike_indexes(zernike_degree + 1)
    coeffs = zernike_coeffs(zernike_indexes)
    results = [
        _radial_zernikes_2d(m, p, zernike_indexes, coeffs)
        for m, p in zip(masks_zyx, pixels_zyx)
    ]
    return unwrap(results)


def _radial_zernikes_2d(
    labels_zyx: NDArray[numpy.integer],
    pixels_zyx: NDArray[numpy.floating],
    zernike_indexes: NDArray[numpy.integer],
    coeffs: tuple,
) -> dict[str, NDArray[numpy.floating]]:
    if labels_zyx.shape[0] > 1:  # Z > 1 -> 3D volume; baseline returns {} for ndim==3
        return {}
    labels = labels_zyx[0]
    pixels = pixels_zyx[0]

    # find_objects-based enumeration (~3-5x faster than numpy.unique's full sort)
    label_lut, n = label_to_idx_lut(labels)
    unique_labels = numpy.flatnonzero(label_lut >= 0)  # present labels, ascending

    # Labeled-pixel coords in one nonzero pass (vs the baseline's per-label
    # masks_to_ijv, which scans the whole image once per label). The per-object sum
    # is order-independent, so raster vs per-label order matches baseline within tol.
    rows, cols = numpy.nonzero(labels)
    if n == 0 or rows.size == 0:  # fringe: no objects / no pixels (matches baseline)
        results: dict[str, NDArray[numpy.floating]] = {}
        for mag_or_phase in ("Magnitude", "Phase"):
            for nn, mm in zernike_indexes:
                results[f"{M_CATEGORY}_Zernike{mag_or_phase}_{nn}_{mm}"] = numpy.zeros(
                    0
                )
        return results

    ij, r = centrosome.cpmorphology.minimum_enclosing_circle(labels, unique_labels)

    seg0 = label_lut[labels[rows, cols]]  # 0-based object rank
    ym = (rows - ij[seg0, 0]) / r[seg0]  # normalised row offset
    xm = (cols - ij[seg0, 1]) / r[seg0]  # normalised column offset

    lut, nterms, m_arr = coeffs
    weights = pixels[rows, cols].astype(numpy.float64)
    vr, vi = zernike_moments(
        weights,
        numpy.ascontiguousarray(xm),
        numpy.ascontiguousarray(ym),
        seg0.astype(numpy.int64),
        lut,
        nterms,
        m_arr,
        n,
    )

    areas = numpy.bincount(seg0, minlength=n).astype(numpy.float64)
    results = {}
    for i, (nn, mm) in enumerate(zernike_indexes):
        magnitude = numpy.sqrt(vr[:, i] ** 2 + vi[:, i] ** 2) / areas
        phase = numpy.arctan2(vr[:, i], vi[:, i])  # baseline's (real, imag) arg order
        results[f"{M_CATEGORY}_ZernikeMagnitude_{nn}_{mm}"] = magnitude
        results[f"{M_CATEGORY}_ZernikePhase_{nn}_{mm}"] = phase
    return results
