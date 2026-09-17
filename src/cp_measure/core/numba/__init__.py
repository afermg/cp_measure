"""Numba-accelerated backend.

Selected explicitly by import (``from cp_measure.core.numba import get_intensity``)
or globally via ``cp_measure.set_accelerator("numba")``. Requires the optional
``numba`` extra; availability is gated by ``cp_measure._detect.HAS_NUMBA``.

This backend currently accelerates ``intensity`` and ``feret``; the global
"numba" accelerator composes them with the numpy implementations of every other
feature (see ``cp_measure.bulk``).
"""

from cp_measure.core.numba._feret import get_feret
from cp_measure.core.numba.measureobjectintensity import get_intensity

__all__ = ["get_feret", "get_intensity"]
