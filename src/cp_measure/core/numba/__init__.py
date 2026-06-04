"""Numba-accelerated backend (integration branch — all lanes merged).

Selected explicitly by import or globally via ``cp_measure.set_accelerator("numba")``.
Requires the optional ``numba`` extra; availability is gated by
``cp_measure._detect.HAS_NUMBA``. This integration branch composes every numba lane
(intensity, granularity, zernike, radial_zernikes, radial_distribution, the four
colocalization features + costes, texture, feret) with the numpy implementations of
the remaining features (see ``cp_measure.bulk``).
"""

from cp_measure.core.numba._feret import get_feret
from cp_measure.core.numba.measurecolocalization import (
    get_correlation_costes,
    get_correlation_manders_fold,
    get_correlation_overlap,
    get_correlation_pearson,
    get_correlation_rwc,
)
from cp_measure.core.numba.measuregranularity import get_granularity
from cp_measure.core.numba.measureobjectintensity import get_intensity
from cp_measure.core.numba.measureobjectintensitydistribution import (
    get_radial_distribution,
    get_radial_zernikes,
)
from cp_measure.core.numba.measureobjectsizeshape import get_zernike
from cp_measure.core.numba.measuretexture import get_texture

__all__ = [
    "get_correlation_costes",
    "get_correlation_manders_fold",
    "get_correlation_overlap",
    "get_correlation_pearson",
    "get_correlation_rwc",
    "get_feret",
    "get_granularity",
    "get_intensity",
    "get_radial_distribution",
    "get_radial_zernikes",
    "get_texture",
    "get_zernike",
]
