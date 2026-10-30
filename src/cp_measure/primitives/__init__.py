"""Shared primitive layer.

Backend-agnostic building blocks that the per-backend feature implementations
(``cp_measure.core`` = numpy, ``cp_measure.core.numba`` = numba, ...) compose.

A labeled image is reduced to a flat *segment* representation (values + 0-based
segment index + per-axis coordinates) which the segment kernels reduce. All
spatial structure (2D vs 3D) and any future batch/image axis are encoded in that
flat segment index, so a single set of segment kernels covers every case without
a rewrite. Labels are the contiguous ``1..N`` cp_measure guarantees (see
:mod:`cp_measure._sanitize`), so the segment index is ``label - 1``; the
flattening + reductions live in ``_segment_numba``.

This is an internal layer with no curated public API: import its building
blocks directly from the concrete modules (``primitives._segment_numba``)
rather than re-exporting them here.
"""
