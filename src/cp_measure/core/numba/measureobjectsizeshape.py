"""Numba-accelerated shape-Zernike backend (`get_zernike`).

Inlines ``centrosome.zernike.zernike``'s flow (the baseline delegates wholesale to
it) but replaces the per-(n,m) ``scipy.ndimage.sum`` + centrosome's per-pixel
polynomial construction with the single fused kernel
(:func:`cp_measure.core.numba._zernike.zernike_moments`). This is the UNWEIGHTED
case (binary shape, ``weights ≡ 1``); the normalisation denominator is the
enclosing-circle area ``π·r²`` (vs the radial backend's pixel count), and only the
magnitude is emitted.

Centrosome still supplies the radial LUT (degree-only) and ``minimum_enclosing_circle``.
Batch-shaped via ``to_bzyx`` (single = ``B == 1``); 2D-only (``Z > 1`` → ``{}``).
"""

import centrosome.zernike
import numpy
from numpy.typing import NDArray

from cp_measure.core.numba._zernike import zernike_coeffs, zernike_moments_per_object
from cp_measure.primitives.shapes import to_bzyx


def get_zernike(masks, pixels=None, zernike_numbers: int = 9):
    """Shape Zernike magnitudes per object; single image/volume or batch.

    ``pixels`` is accepted for signature parity with the numpy baseline but unused
    (shape Zernike is a binary-mask descriptor), so masks alone drive the batch
    normalisation.
    """
    masks_zyx, _, unwrap = to_bzyx(masks, masks)
    # degree-only work, hoisted out of the per-image loop (computed once per batch)
    zernike_indexes = centrosome.zernike.get_zernike_indexes(zernike_numbers + 1)
    coeffs = zernike_coeffs(zernike_indexes)
    results = [_zernike_2d(m, zernike_indexes, coeffs) for m in masks_zyx]
    return unwrap(results)


def _zernike_2d(
    labels_zyx: NDArray[numpy.integer],
    zernike_indexes: NDArray[numpy.integer],
    coeffs: tuple,
) -> dict[str, NDArray[numpy.floating]]:
    if labels_zyx.shape[0] > 1:  # Z > 1 -> 3D volume; baseline returns {} for ndim==3
        return {}

    # shape Zernike is unweighted (binary mask) -> pixels=None
    vr, vi, radii, _seg0 = zernike_moments_per_object(labels_zyx[0], None, coeffs)
    if vr.shape[0] == 0:  # no objects (baseline crashes here; we return empty per key)
        return {f"Zernike_{nn}_{mm}": numpy.zeros(0) for nn, mm in zernike_indexes}

    areas = numpy.pi * numpy.asarray(radii, dtype=float) ** 2  # enclosing-circle area
    results = {}
    for i, (nn, mm) in enumerate(zernike_indexes):
        results[f"Zernike_{nn}_{mm}"] = (
            numpy.sqrt(vr[:, i] ** 2 + vi[:, i] ** 2) / areas
        )
    return results
