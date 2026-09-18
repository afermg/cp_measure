"""Relabel non-contiguous mask label IDs (gaps or arbitrary values) to contiguous
``1..N`` without mutating the caller's array, applied at the entry points. See
:func:`sanitize`.
"""

import functools
import inspect
from typing import Callable

import numpy
from numpy.typing import NDArray
from skimage.segmentation import relabel_sequential

# Argument names that hold the label image across the ``get_*`` functions.
_MASK_PARAMS = ("masks", "labels", "mask")


def sanitize_masks(masks: NDArray) -> tuple[NDArray, NDArray[numpy.int64]]:
    """Relabel positive labels to contiguous ``1..N``.

    Returns ``(clean, ids)`` where ``clean`` has labels ``1..N`` (the input is
    returned unchanged when already contiguous) and ``ids[i]`` is the original
    label of rank ``i + 1``. Never mutates the input. Raises ``ValueError`` for
    non-integer (non-boolean) or negative labels.
    """
    if masks.dtype != bool and not numpy.issubdtype(masks.dtype, numpy.integer):
        raise ValueError(f"labels must be an integer array, got dtype {masks.dtype!r}")
    if masks.min(initial=0) < 0:
        raise ValueError("labels must be non-negative")
    mx = int(masks.max(initial=0))
    if mx == 0:
        return masks, numpy.empty(0, dtype=numpy.int64)
    # bincount is faster than unique but needs a bounded range; fall back for huge labels.
    if mx <= masks.size:
        ids = numpy.flatnonzero(numpy.bincount(masks.ravel(), minlength=mx + 1))
    else:
        ids = numpy.unique(masks)
    ids = ids[ids > 0].astype(numpy.int64)
    if ids.size == mx:
        # already 1..N: no copy (cast only a bool mask up to an integer dtype)
        return (masks if masks.dtype != bool else masks.astype(numpy.intp)), ids
    clean, _forward, _inverse = relabel_sequential(masks)
    return clean, ids


def _sanitize_arg(masks):
    """Relabel one image, or every image of a batch.

    A batch is a list/tuple of images or a 4D ``(B, Z, Y, X)`` array (3D is a
    single volume — the convention :func:`cp_measure.primitives.shapes.to_bzyx`
    uses). Each image is relabelled on its own, because results are per image:
    relabelling a batch as a whole would leave an image whose labels start above
    1, and the measurements index their output by label. Dense input is returned
    unchanged, so an already-clean batch costs one ``bincount`` per image and no
    copy.
    """
    if isinstance(masks, (list, tuple)):
        return [sanitize_masks(m)[0] for m in masks]
    if numpy.ndim(masks) == 4:
        images = list(masks)  # views, no copy
        clean = [sanitize_masks(m)[0] for m in images]
        if all(c is m for c, m in zip(clean, images)):
            return masks
        return numpy.stack(clean)
    return sanitize_masks(masks)[0]


def sanitize(func: Callable) -> Callable:
    """Wrap a ``get_*`` function to relabel its label argument (named in
    :data:`_MASK_PARAMS`) to ``1..N`` before the call — per image when the
    argument is a batch (see :func:`_sanitize_arg`); functions with no such
    argument are returned unchanged. Use this only to call a raw measurement
    function directly with gapped IDs — the entry points already sanitize.
    """
    sig = inspect.signature(func)
    param = next((name for name in _MASK_PARAMS if name in sig.parameters), None)
    if param is None:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.arguments[param] = _sanitize_arg(bound.arguments[param])
        return func(*bound.args, **bound.kwargs)

    return wrapper
