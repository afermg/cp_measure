"""Numba-accelerated shape-Zernike backend (`get_zernike`).

Mirrors the NumPy ``_zernike_scores`` flow but replaces its per-moment array
operations and ``bincount`` reductions with the single fused kernel
(:func:`cp_measure.core.numba._zernike.zernike_moments`). This is the UNWEIGHTED
case (binary shape, ``weights ≡ 1``); the normalisation denominator is the
enclosing-circle area ``π·r²`` (vs the radial backend's pixel count), and only the
magnitude is emitted.

Centrosome still supplies the radial LUT (degree-only) and ``minimum_enclosing_circle``.
Batch handling is :func:`cp_measure.core.numba._zernike.zernike_per_image`.
"""

from cp_measure.core.measureobjectsizeshape import _zernike_features
from cp_measure.core.numba._zernike import zernike_per_image


def get_zernike(masks, pixels=None, zernike_numbers: int = 9):
    """Shape Zernike magnitudes per object; single image/volume or batch.

    ``pixels`` is accepted for signature parity with the numpy baseline but unused
    (shape Zernike is a binary-mask descriptor).
    """
    return zernike_per_image(
        masks,
        None,
        zernike_numbers,
        lambda vr, vi, radii, _counts, indexes: _zernike_features(
            vr, vi, radii, indexes
        ),
    )
