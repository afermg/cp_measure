"""Central sanitation of non-contiguous mask label IDs.

cp_measure's measurement functions assume labels are the contiguous integers
``1..N`` (see :func:`cp_measure.featurizer.featurize`). Real segmentations may
use gaps (``{1, 5, 17}``) or arbitrary values; this maps them to ``1..N`` before
any math runs and reports results against the original IDs, without mutating the
caller's array.
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
    mx = int(masks.max(initial=0))
    if mx == 0:
        return masks, numpy.empty(0, dtype=numpy.int64)
    # Cheap contiguity check (~4x faster than numpy.unique); bincount needs a
    # bounded range, so fall back to unique for pathologically large labels.
    if mx <= masks.size:
        ids = numpy.flatnonzero(numpy.bincount(masks.ravel(), minlength=mx + 1))
    else:
        ids = numpy.unique(masks)
    ids = ids[ids > 0].astype(numpy.int64)
    if ids.size == mx:
        return masks, ids  # already 1..N: no copy
    clean, _forward, _inverse = relabel_sequential(masks)
    return clean, ids


def sanitize(func: Callable) -> Callable:
    """Wrap a ``get_*`` function so its label argument (found by name in
    :data:`_MASK_PARAMS`) is relabelled to ``1..N`` before the call; functions
    with no such argument (e.g. the two-mask multimask functions) are returned
    unchanged.

    Measurement functions are *not* sanitized by default — the bulk entry
    points (:func:`cp_measure.bulk.get_core_measurements` and friends) apply
    this for you, and :func:`cp_measure.featurizer.featurize` sanitizes once up
    front. Apply it yourself only when calling a raw function directly with
    gapped or arbitrary label IDs.
    """
    sig = inspect.signature(func)
    param = next((name for name in _MASK_PARAMS if name in sig.parameters), None)
    if param is None:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.arguments[param], _ids = sanitize_masks(bound.arguments[param])
        return func(*bound.args, **bound.kwargs)

    return wrapper
