"""Shared primitive layer.

Backend-agnostic building blocks that the per-backend feature implementations
(``cp_measure.core`` = numpy, ``cp_measure.core.numba`` = numba, ...) compose.

A labeled image is reduced to a flat *segment* representation (values + 0-based
segment index + per-axis coordinates) which the segment kernels reduce. All
spatial structure (2D vs 3D) and any future batch/image axis are encoded in that
flat segment index, so a single set of segment kernels covers every case without
a rewrite. The numpy ``label_to_idx_lut`` (in ``segment``) builds the
label→index lookup; the flattening + reductions themselves live in
``_segment_numba``.

This is an internal layer with no curated public API: import its building
blocks directly from the concrete modules (``primitives.segment``,
``primitives._segment_numba``) rather than re-exporting them here.
"""
