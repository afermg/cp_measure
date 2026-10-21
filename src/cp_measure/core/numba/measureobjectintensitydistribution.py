"""Numba-accelerated radial Zernike backend.

Mirrors :func:`cp_measure.core.measureobjectintensitydistribution.get_radial_zernikes`
but replaces its per-moment Horner array operations and ``bincount`` reductions
with a single fused Numba kernel
(:func:`cp_measure.core.numba._zernike.zernike_moments`). Centrosome still provides
the radial LUT (degree-only) and ``minimum_enclosing_circle`` (host).

Batch-shaped via the canonical ``(B, Z, Y, X)`` form (single image = ``B == 1``);
2D-only (raw 3D volumes, including Z=1, return ``{}``, as in NumPy). Explicit
4D batches use Z=1 for 2D images.
"""

import centrosome.zernike
import numpy
from numpy.typing import NDArray

from cp_measure.core.measureobjectintensitydistribution import (
    _radial_zernike_features,
)
from cp_measure.core.numba._zernike import zernike_coeffs, zernike_moments_per_object
from cp_measure.primitives.shapes import to_bzyx


def get_radial_zernikes(labels, pixels, zernike_degree: int = 9):
    """Radial Zernike magnitude/phase per object; single image/volume or batch."""
    if isinstance(labels, numpy.ndarray) and labels.ndim == 3:
        return {}  # a single volume (even Z=1), matching the NumPy contract
    masks_zyx, pixels_zyx, unwrap = to_bzyx(labels, pixels)
    # degree-only work, hoisted out of the per-image loop (computed once per batch)
    zernike_indexes = centrosome.zernike.get_zernike_indexes(zernike_degree + 1)
    coeffs = zernike_coeffs(zernike_indexes)
    results = [
        {}
        if isinstance(labels, (list, tuple)) and numpy.ndim(labels[i]) == 3
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

    vr, vi, _radii, counts = zernike_moments_per_object(
        labels_zyx[0], pixels_zyx[0], coeffs
    )
    return _radial_zernike_features(vr, vi, counts, zernike_indexes)
