"""Numba-accelerated backend.

Selected explicitly by import (``from cp_measure.core.numba import get_intensity``)
or globally via ``cp_measure.set_accelerator("numba")``. Requires the optional
``numba`` extra; availability is gated by ``cp_measure._detect.HAS_NUMBA``.

This backend accelerates ``intensity`` and ``granularity``; the global "numba"
This backend accelerates ``intensity`` and the colocalization features
``pearson``/``manders_fold``/``rwc``/``overlap``/``costes``; the global "numba"
This backend accelerates ``intensity`` and ``texture``; the global "numba"
accelerator composes them with the numpy implementations of every other feature
(see ``cp_measure.bulk``).
This backend accelerates ``intensity``, ``zernike``, ``radial_zernikes`` and
``radial_distribution``; the global "numba" accelerator composes them with the
numpy implementations of every other feature (see ``cp_measure.bulk``).
"""

from cp_measure.core.numba.measuregranularity import get_granularity
from cp_measure.core.numba.measurecolocalization import (
    get_correlation_costes,
    get_correlation_manders_fold,
    get_correlation_overlap,
    get_correlation_pearson,
    get_correlation_rwc,
)
from cp_measure.core.numba.measureobjectintensity import get_intensity
from cp_measure.core.numba.measureobjectintensitydistribution import (
    get_radial_distribution,
    get_radial_zernikes,
)
from cp_measure.core.numba.measureobjectsizeshape import get_zernike

__all__ = ["get_granularity", "get_intensity"]
__all__ = [
    "get_intensity",
    "get_radial_distribution",
    "get_radial_zernikes",
    "get_zernike",
__all__ = [
    "get_correlation_costes",
    "get_correlation_manders_fold",
    "get_correlation_overlap",
    "get_correlation_pearson",
    "get_correlation_rwc",
    "get_intensity",
]
from cp_measure.core.numba.measuretexture import get_texture

__all__ = ["get_intensity", "get_texture"]
