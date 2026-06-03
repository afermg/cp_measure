"""Canonical ``(B, Z, Y, X)`` input normalisation (numpy, backend-agnostic).

Every optimised feature works on a batch of volumes. Rather than a single dense
``(B, Z, Y, X)`` array (which cannot hold ragged, differently-sized images), the
batch is represented as a **list of B ``(Z, Y, X)`` arrays** — equal-size and
ragged batches share one code path, and a single image is just ``B == 1``.

Normalisation rules (the only public entry, :func:`to_bzyx`):

================================ ============================ ========
input                            yields                       batch?
================================ ============================ ========
2D ``(H, W)`` ndarray            ``[(1, H, W)]``              no
3D ``(Z, Y, X)`` ndarray         ``[(Z, Y, X)]`` (one volume) no
4D ``(B, Z, Y, X)`` ndarray      ``[(Z, Y, X)] * B``          yes
list/tuple of 2D/3D arrays       one ``(Z, Y, X)`` per item   yes
================================ ============================ ========

A 3D ndarray is therefore ALWAYS one volume, never a batch — this preserves the
existing single-volume semantics. To pass a batch of 2D images as an array, use
``(B, 1, H, W)``; for ragged sizes, pass a list. ``unwrap`` then re-shapes the
per-image results back to a single dict (single input) or the list (batch).
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


def to_bzyx(masks, pixels):
    """Normalise ``(masks, pixels)`` to the canonical batch-of-volumes form.

    Returns ``(masks_zyx, pixels_zyx, unwrap)`` where ``masks_zyx`` and
    ``pixels_zyx`` are length-``B`` lists of ``(Z, Y, X)`` arrays (one per image),
    and ``unwrap(results)`` maps a length-``B`` list of per-image results back to
    a single result (non-batch input) or the list itself (batch input).
    """
    masks_is_seq = isinstance(masks, (list, tuple))
    pixels_is_seq = isinstance(pixels, (list, tuple))
    if masks_is_seq != pixels_is_seq:
        raise ValueError("masks and pixels must both be sequences, or both arrays")

    if masks_is_seq:
        masks_zyx = [_to_zyx(m) for m in masks]
        pixels_zyx = [_to_zyx(p) for p in pixels]
        is_batch = True
    else:
        m = numpy.asarray(masks)
        p = numpy.asarray(pixels)
        if (m.ndim == 4) != (p.ndim == 4):
            raise ValueError("masks and pixels must both be 4D for a stacked batch")
        if m.ndim == 4:
            masks_zyx = list(m)
            pixels_zyx = list(p)
            is_batch = True
        else:
            masks_zyx = [_to_zyx(m)]
            pixels_zyx = [_to_zyx(p)]
            is_batch = False

    if len(masks_zyx) != len(pixels_zyx):
        raise ValueError(
            f"batch size mismatch: {len(masks_zyx)} masks vs {len(pixels_zyx)} images"
        )

    def unwrap(results):
        return results if is_batch else results[0]

    return masks_zyx, pixels_zyx, unwrap
