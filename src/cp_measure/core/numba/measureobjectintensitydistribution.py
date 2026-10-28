"""Numba-accelerated radial Zernike backend.

Mirrors :func:`cp_measure.core.measureobjectintensitydistribution.get_radial_zernikes`
but replaces its per-moment Horner array operations and ``bincount`` reductions
with a single fused Numba kernel
(:func:`cp_measure.core.numba._zernike.zernike_moments`). Centrosome still provides
the radial LUT (degree-only) and ``minimum_enclosing_circle`` (host). Batch
handling is :func:`cp_measure.core.numba._zernike.zernike_per_image`.
"""

from cp_measure.core.measureobjectintensitydistribution import (
    _radial_zernike_features,
)
from cp_measure.core.numba._zernike import zernike_per_image


def get_radial_zernikes(labels, pixels, zernike_degree: int = 9):
    """Radial Zernike magnitude/phase per object; single image/volume or batch.

    Measured 2D entries require matching label and intensity shapes; mismatches
    raise ``ValueError`` rather than selecting an intensity plane. Compare phases
    modulo 2π; near-zero moments have numerically unstable phases.
    """
    return zernike_per_image(
        labels,
        pixels,
        zernike_degree,
        lambda vr, vi, _radii, counts, indexes: _radial_zernike_features(
            vr, vi, counts, indexes
        ),
    )
