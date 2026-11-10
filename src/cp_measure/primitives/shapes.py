"""Canonical ``(B, Z, Y, X)`` input normalisation (numpy, backend-agnostic).

Every optimised feature works on a batch of volumes, represented as a single
``(B, Z, Y, X)`` array — one ``(Z, Y, X)`` volume per image, with a single image
being just ``B == 1``.

Normalisation rules for :func:`to_bzyx`:

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

All images in a batch must share one shape. :func:`to_bzyx` stacks a list into
a single array; :func:`each_volume` instead returns per-item views. Both reject
ragged (differently-sized) batches. Normalise your images to a common shape first.

This is a pure batch normaliser: ``masks`` and ``pixels`` must share the same
batch/ndim structure. It does NOT broadcast a lower-dimensional mask over a
volume (e.g. a 2D mask applied to a 3D stack) — per-element ndim handling is the
caller's job (each backend dispatches its own 2D vs 3D path).
"""

import numpy
from numpy.typing import NDArray


def as_2d(masks: NDArray) -> NDArray | None:
    """View the only Z slice for a 2D-only measurement; skip other volumes.

    This is a measurement eligibility rule, not a change to input dimensionality.
    Genuinely 3D measurements must retain the caller's spatial rank.
    """
    if masks.ndim == 3:
        return masks[0] if masks.shape[0] == 1 else None
    if masks.ndim != 2:
        raise ValueError("expected one 2D image or 3D volume")
    return masks


def as_2d_pair(masks: NDArray, pixels: NDArray) -> tuple[NDArray, NDArray] | None:
    """Apply :func:`as_2d` to matching labels/intensities without broadcasting."""
    measured_masks = as_2d(masks)
    if measured_masks is None:
        return None
    if masks.shape != pixels.shape:
        raise ValueError("labels and pixels must have matching shapes")
    return measured_masks, pixels[0] if pixels.ndim == 3 else pixels


def _to_zyx(arr: NDArray) -> NDArray:
    """Promote a single 2D/3D image to ``(Z, Y, X)`` (2D gets a unit Z axis)."""
    a = numpy.asarray(arr)
    if a.ndim == 2:
        return a[numpy.newaxis]
    if a.ndim == 3:
        return a
    raise ValueError(f"expected a 2D or 3D image, got ndim={a.ndim}")


def _reject_ragged(shapes, what: str) -> None:
    """Reject a batch whose images do not all share one shape (incl. rank)."""
    if any(s != shapes[0] for s in shapes[1:]):
        raise ValueError(
            f"all {what} in a batch must have the same shape; got a ragged batch. "
            "Normalise your images to a common shape first — ragged batches are "
            "not supported."
        )


def _batch_items(masks, pixels):
    """Detect the batch structure shared by :func:`each_volume` and :func:`to_bzyx`.

    Returns ``(pairs, is_batch, unwrap)`` where ``pairs`` is a list of
    ``(mask, pixels)`` at each image's original rank (a 3D array is one volume; a
    4D array or a list/tuple is a batch), ``is_batch`` says whether the input was a
    batch, and ``unwrap(results)`` maps a length-``B`` result list back to a single
    result (non-batch input) or the list itself (batch input). Ragged batches are
    rejected; no lower-rank mask is broadcast over a volume.
    """
    masks_is_seq = isinstance(masks, (list, tuple))
    pixels_is_seq = isinstance(pixels, (list, tuple))
    if masks_is_seq != pixels_is_seq:
        raise ValueError("masks and pixels must both be sequences, or both arrays")

    if masks_is_seq:
        if len(masks) != len(pixels):
            raise ValueError(
                f"batch size mismatch: {len(masks)} masks vs {len(pixels)} images"
            )
        if len(masks) == 0:
            raise ValueError("a batch must contain at least one image")
        _reject_ragged([numpy.shape(p) for p in pixels], "images")
        pairs = list(zip(masks, pixels))
        is_batch = True
    else:
        m, p = numpy.asarray(masks), numpy.asarray(pixels)
        if (m.ndim == 4) != (p.ndim == 4):
            raise ValueError("masks and pixels must both be 4D for a stacked batch")
        is_batch = m.ndim == 4
        if is_batch:
            if len(m) != len(p):
                raise ValueError(
                    f"batch size mismatch: {len(m)} masks vs {len(p)} images"
                )
            if len(m) == 0:
                raise ValueError("a batch must contain at least one image")
            pairs = list(zip(m, p))
        else:
            pairs = [(m, p)]

    def unwrap(results):
        return results if is_batch else results[0]

    return pairs, is_batch, unwrap


def each_volume(masks, pixels):
    """Per-item ``(Z, Y, X)`` views of a batch, WITHOUT stacking, so no whole-batch
    copy is made.

    Returns ``(items, unwrap)`` where ``items`` is a list of
    ``(mask_zyx, pixels_zyx, spatial_ndim)``: both arrays are views (a 2D image
    gains a unit Z axis) and ``spatial_ndim`` is the image's original rank (2 or 3).
    ``unwrap`` is as :func:`_batch_items`.

    Batch structure (matching structure, ragged rejection, no broadcast) comes from
    :func:`_batch_items`; in addition each image's labels and pixels must match in
    shape and rank. Preserves each array's dtype and makes no whole-batch
    allocation.
    """
    pairs, _, unwrap = _batch_items(masks, pixels)
    items = []
    for m, p in pairs:
        if numpy.shape(m) != numpy.shape(p):
            raise ValueError("labels and pixels must have matching shapes and rank")
        items.append((_to_zyx(m), _to_zyx(p), numpy.ndim(m)))
    return items, unwrap


def to_bzyx(masks, pixels):
    """Normalise ``(masks, pixels)`` to the canonical batch-of-volumes form.

    Returns ``(masks_bzyx, pixels_bzyx, unwrap)`` where ``masks_bzyx`` and
    ``pixels_bzyx`` are ``(B, Z, Y, X)`` arrays (one ``(Z, Y, X)`` volume per
    image), and ``unwrap`` is as :func:`_batch_items`.

    Thin wrapper over :func:`each_volume`, sharing its batch contract and ragged
    rejection. A real batch is stacked into one array; a single image keeps its
    view (``_to_zyx`` of the caller's array, plus a unit batch axis), because
    callers index ``[0]`` and rely on it sharing the caller's memory.
    """
    items, unwrap = each_volume(masks, pixels)
    if is_batch(masks):
        return (
            numpy.stack([m for m, _, _ in items]),
            numpy.stack([p for _, p, _ in items]),
            unwrap,
        )
    m, p, _ = items[0]
    return m[numpy.newaxis], p[numpy.newaxis], unwrap


def memory_key(array) -> tuple:
    """The memory an array views (address, shape, strides, dtype). Batches use it
    to share per-mask work: the featurizer passes one view of a mask per channel."""
    return (
        array.__array_interface__["data"][0],
        array.shape,
        array.strides,
        array.dtype.str,
    )


def native(array):
    """``array`` in native byte order, which numba kernels require; no copy when it
    already is (big-endian data comes from e.g. FITS, N5 or some zarr stores)."""
    array = numpy.asarray(array)
    if array.dtype.isnative:
        return array
    return array.astype(array.dtype.newbyteorder("="))


def is_batch(masks) -> bool:
    """Whether ``masks`` is a batch: a list/tuple of images or a 4D array."""
    return isinstance(masks, (list, tuple)) or numpy.ndim(masks) == 4


def per_image(masks, pixels) -> list[tuple]:
    """The ``(mask, pixels)`` pairs of a batch, each at its original rank (a 2D image
    stays 2D). ``pixels`` may be ``None`` for the shape-only measurements.

    Uses :func:`each_volume`, so it shares its batch contract (matching structure,
    no ragged batches, no broadcasting) and makes no batch copy.
    """
    items, _ = each_volume(masks, masks if pixels is None else pixels)
    return [
        (m if ndim == 3 else m[0], None if pixels is None else p if ndim == 3 else p[0])
        for m, p, ndim in items
    ]


def apply_per_image(func, masks, pixels, **kwargs) -> list:
    """Run a single-image ``func(mask, pixels, **kwargs)`` over each image of a batch
    (see :func:`per_image`). Returns one result per image."""
    return [func(m, p, **kwargs) for m, p in per_image(masks, pixels)]


def apply_per_pair(func, pixels_1, pixels_2, masks, **kwargs) -> list:
    """:func:`apply_per_image` for the two-channel colocalization functions,
    ``func(pixels_1, pixels_2, masks, **kwargs)``. Returns one result per image."""
    first, _ = each_volume(masks, pixels_1)
    second, _ = each_volume(masks, pixels_2)
    return [
        func(*(x if ndim == 3 else x[0] for x in (p1, p2, m)), **kwargs)
        for (m, p1, ndim), (_, p2, _) in zip(first, second)
    ]
