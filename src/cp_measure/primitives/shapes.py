"""Canonical ``(B, Z, Y, X)`` input normalisation (numpy, backend-agnostic).

Every optimised feature works on a batch of volumes, represented as a single
``(B, Z, Y, X)`` array — one ``(Z, Y, X)`` volume per image, with a single image
being just ``B == 1``.

Normalisation rules (the only public entry, :func:`to_bzyx`):

================================ ============================ ========
input                            yields                       batch?
================================ ============================ ========
2D ``(H, W)`` ndarray            ``(1, 1, H, W)``             no
3D ``(Z, Y, X)`` ndarray         ``(1, Z, Y, X)``             no
4D ``(B, Z, Y, X)`` ndarray      ``(B, Z, Y, X)``             yes
list/tuple of 2D/3D arrays       ``(B, Z, Y, X)``             yes
================================ ============================ ========

A 3D ndarray is therefore ALWAYS one volume, never a batch — this preserves the
existing single-volume semantics. To pass a batch of 2D images as an array, use
``(B, 1, H, W)``. ``unwrap`` then re-shapes the per-image results back to a
single dict (single input) or the list (batch).

All images in a batch must share one shape: a list is stacked into a single
array, so ragged (differently-sized) batches are rejected with an informative
error. Normalise your images to a common shape first.

This is a pure batch normaliser: ``masks`` and ``pixels`` must share the same
batch/ndim structure. It does NOT broadcast a lower-dimensional mask over a
volume (e.g. a 2D mask applied to a 3D stack) — per-element ndim handling is the
caller's job (each backend dispatches its own 2D vs 3D path).
"""

import numpy
from numpy.typing import NDArray


def _to_zyx(arr: NDArray) -> NDArray:
    """Promote a single 2D/3D image to ``(Z, Y, X)`` (2D gets a unit Z axis)."""
    a = numpy.asarray(arr)
    if a.ndim == 2:
        return a[numpy.newaxis]
    if a.ndim == 3:
        return a
    raise ValueError(f"expected a 2D or 3D image, got ndim={a.ndim}")


def _stack(seq, what: str) -> NDArray:
    """Stack a list/tuple of equal-shape images into one array; reject ragged."""
    try:
        arr = numpy.asarray(seq)
    except ValueError:
        arr = None
    if arr is None or arr.dtype == object:
        raise ValueError(
            f"all {what} in a batch must have the same shape; got a ragged batch. "
            "Normalise your images to a common shape first — ragged batches are "
            "not supported."
        )
    return arr


def to_bzyx(masks, pixels):
    """Normalise ``(masks, pixels)`` to the canonical batch-of-volumes form.

    Returns ``(masks_bzyx, pixels_bzyx, unwrap)`` where ``masks_bzyx`` and
    ``pixels_bzyx`` are ``(B, Z, Y, X)`` arrays (one ``(Z, Y, X)`` volume per
    image), and ``unwrap(results)`` maps a length-``B`` list of per-image results
    back to a single result (non-batch input) or the list itself (batch input).
    """
    masks_is_seq = isinstance(masks, (list, tuple))
    pixels_is_seq = isinstance(pixels, (list, tuple))
    if masks_is_seq != pixels_is_seq:
        raise ValueError("masks and pixels must both be sequences, or both arrays")

    if masks_is_seq:
        m, p = _stack(masks, "masks"), _stack(pixels, "pixels")
        is_batch = True
    else:
        m, p = numpy.asarray(masks), numpy.asarray(pixels)
        if (m.ndim == 4) != (p.ndim == 4):
            raise ValueError("masks and pixels must both be 4D for a stacked batch")
        is_batch = m.ndim == 4

    if is_batch:
        masks_bzyx = numpy.stack([_to_zyx(x) for x in m])
        pixels_bzyx = numpy.stack([_to_zyx(x) for x in p])
    else:
        masks_bzyx = _to_zyx(m)[numpy.newaxis]
        pixels_bzyx = _to_zyx(p)[numpy.newaxis]

    if len(masks_bzyx) != len(pixels_bzyx):
        raise ValueError(
            f"batch size mismatch: {len(masks_bzyx)} masks vs {len(pixels_bzyx)} images"
        )

    def unwrap(results):
        return results if is_batch else results[0]

    return masks_bzyx, pixels_bzyx, unwrap
