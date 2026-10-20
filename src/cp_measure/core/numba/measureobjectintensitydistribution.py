"""Numba-accelerated radial Zernike backend.

Mirrors :func:`cp_measure.core.measureobjectintensitydistribution.get_radial_zernikes`
but replaces its per-moment Horner array operations and ``bincount`` reductions
with a single fused Numba kernel
(:func:`cp_measure.core.numba._zernike.zernike_moments`). Azimuthal complex powers
remain in NumPy to preserve phase under cancellation. Centrosome still provides
the radial LUT (degree-only) and ``minimum_enclosing_circle`` (host).

Batch-shaped via the canonical ``(B, Z, Y, X)`` form (single image = ``B == 1``);
2D-only (raw 3D volumes, including Z=1, return ``{}``, as in NumPy). Explicit
4D batches use Z=1 for 2D images.
"""

import centrosome.zernike
import numpy
from numpy.typing import NDArray

from cp_measure.core.measureobjectintensitydistribution import M_CATEGORY
from cp_measure.core.numba._zernike import zernike_coeffs, zernike_moments_per_object
from cp_measure.primitives.shapes import to_bzyx


def get_radial_zernikes(masks, pixels, zernike_degree: int = 9):
    """Radial Zernike magnitude/phase per object; single image/volume or batch."""
    if isinstance(masks, numpy.ndarray) and masks.ndim == 3:
        return {}  # a single volume (even Z=1), matching the NumPy contract
    masks_zyx, pixels_zyx, unwrap = to_bzyx(masks, pixels)
    # degree-only work, hoisted out of the per-image loop (computed once per batch)
    zernike_indexes = centrosome.zernike.get_zernike_indexes(zernike_degree + 1)
    coeffs = zernike_coeffs(zernike_indexes)
    results = [
        {}
        if isinstance(masks, (list, tuple)) and numpy.ndim(masks[i]) == 3
        else _radial_zernikes_2d(m, p, zernike_indexes, coeffs)
        for i, (m, p) in enumerate(zip(masks_zyx, pixels_zyx))
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

    vr, vi, _radii, seg0 = zernike_moments_per_object(
        labels_zyx[0], pixels_zyx[0], coeffs
    )
    if (
        vr.shape[0] == 0
    ):  # no objects (fringe) -> empty arrays per key, matches baseline
        return {
            f"{M_CATEGORY}_Zernike{mag_or_phase}_{nn}_{mm}": numpy.zeros(0)
            for nn, mm in zernike_indexes
            for mag_or_phase in ("Magnitude", "Phase")
        }

    areas = numpy.bincount(seg0, minlength=vr.shape[0]).astype(
        numpy.float64
    )  # px/object
    results = {}
    for i, (nn, mm) in enumerate(zernike_indexes):
        results[f"{M_CATEGORY}_ZernikeMagnitude_{nn}_{mm}"] = (
            numpy.sqrt(vr[:, i] ** 2 + vi[:, i] ** 2) / areas
        )
        # baseline's (real, imag) arctan2 arg order
        results[f"{M_CATEGORY}_ZernikePhase_{nn}_{mm}"] = numpy.arctan2(
            vr[:, i], vi[:, i]
        )
    return results
