"""Numba-accelerated shape-Zernike backend (`get_zernike`).

Mirrors the NumPy ``_zernike_scores`` flow but replaces its per-moment array
operations and ``bincount`` reductions with the single fused kernel
(:func:`cp_measure.core.numba._zernike.zernike_moments`). This is the UNWEIGHTED
case (binary shape, ``weights ≡ 1``); the normalisation denominator is the
enclosing-circle area ``π·r²`` (vs the radial backend's pixel count), and only the
magnitude is emitted.

Centrosome still supplies the radial LUT (degree-only) and ``minimum_enclosing_circle``.
Batch-shaped via ``to_bzyx``; 2D-only (raw 3D volumes, including Z=1, → ``{}``).
An explicit 4D batch uses Z=1 for 2D images, as in the other Numba backends.
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
    # A raw 3D array is a volume, including singleton Z; NumPy returns {}.
    # Explicit 4D batches may use Z=1 to represent each 2D image.
    if isinstance(masks, numpy.ndarray) and masks.ndim == 3:
        return {}
    masks_zyx, _, unwrap = to_bzyx(masks, masks)
    # degree-only work, hoisted out of the per-image loop (computed once per batch)
    zernike_indexes = centrosome.zernike.get_zernike_indexes(zernike_numbers + 1)
    coeffs = zernike_coeffs(zernike_indexes)
    results = [
        {}
        if isinstance(masks, (list, tuple)) and numpy.ndim(masks[i]) == 3
        else _zernike_2d(m, zernike_indexes, coeffs)
        for i, m in enumerate(masks_zyx)
    ]
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
    if vr.shape[0] == 0:  # no objects -> empty per key, matching NumPy
        return {f"Zernike_{nn}_{mm}": numpy.zeros(0) for nn, mm in zernike_indexes}

    areas = numpy.pi * radii * radii  # enclosing-circle area, same order as NumPy
    results = {}
    with numpy.errstate(divide="ignore", invalid="ignore"):
        for i, (nn, mm) in enumerate(zernike_indexes):
            results[f"Zernike_{nn}_{mm}"] = (
                numpy.sqrt(vr[:, i] ** 2 + vi[:, i] ** 2) / areas
            )
    return results
