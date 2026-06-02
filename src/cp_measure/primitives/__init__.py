"""Shared primitive layer.

Backend-agnostic building blocks that the per-backend feature implementations
(``cp_measure.core`` = numpy, ``cp_measure.core.numba`` = numba, ...) compose.

The host helpers here flatten a labeled image into a 1-D *segment* representation
(values + 0-based segment index + per-axis coordinates). All spatial structure
(2D vs 3D) and any future batch/image axis are encoded in that flat segment
index, so a single set of segment kernels covers every case without a rewrite.
"""

from cp_measure.primitives.segment import (
    flatten_labeled,
    label_to_idx_lut,
)

__all__ = [
    "flatten_labeled",
    "label_to_idx_lut",
]
