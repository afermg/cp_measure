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
    labels = numpy.unique(masks)
    if labels.size and labels[0] < 0:
        raise ValueError("labels must be non-negative")
    ids = labels[labels > 0].astype(numpy.int64)
    if ids.size == 0 or (ids[0] == 1 and ids[-1] == ids.size):
        return masks, ids  # already contiguous (or empty): no copy
    clean, _forward, _inverse = relabel_sequential(masks)
    return clean, ids


def sanitize_labels(func: Callable) -> Callable:
    """Decorate a ``get_*`` function to sanitize its label argument (found by
    name in :data:`_MASK_PARAMS`); functions with no such argument (e.g. the
    two-mask multimask functions) are returned unchanged."""
    sig = inspect.signature(func)
    param = next((name for name in _MASK_PARAMS if name in sig.parameters), None)
    if param is None:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.arguments[param], _ids = sanitize_masks(bound.arguments[param])
        return func(*bound.args, **bound.kwargs)

    wrapper._sanitized = True  # type: ignore[attr-defined]
    return wrapper
