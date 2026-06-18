"""Central input sanitation for non-contiguous mask label IDs.

cp_measure's measurement functions assume object labels are the contiguous
integers ``1..N`` (the convention documented on
:func:`cp_measure.featurizer.featurize`). Real segmentations do not always
honour that: labels may have gaps (``{1, 5, 17}``) or arbitrary values. This
module provides the single policy that maps arbitrary positive-integer labels to
``1..N`` *before* any math runs, so every downstream function can assume clean
labels and focus on the calculation.

Two pieces:

* :func:`sanitize_masks` — the policy. Returns ``(clean, ids)`` where ``clean``
  has labels ``1..N`` and ``ids[i]`` is the *original* label of rank ``i + 1``.
  It never mutates the caller's array (the relabel path returns a fresh copy; the
  already-clean fast path returns the input unchanged).
* :func:`sanitize_labels` — a thin decorator that applies the policy to whichever
  argument of a ``get_*`` function holds the label image, so direct callers get
  the same guarantee as the featurizer.

Relabelling is cheap relative to a single feature (<1 % of a featurized image),
so the featurizer sanitizes once up front and the per-function decorator is a
cheap idempotent guard (a single :func:`numpy.unique`) for direct callers.
"""

import functools
import inspect
from typing import Callable

import numpy
from numpy.typing import NDArray
from skimage.segmentation import relabel_sequential

# Argument names that, across the ``get_*`` functions, hold the label image.
_MASK_PARAMS = ("masks", "labels", "mask")


def _is_contiguous(masks: NDArray) -> bool:
    """Are the positive labels exactly ``1..N``?

    Uses :func:`numpy.unique` rather than a dense ``bincount`` so the check is
    safe and bounded for any dtype, negative values, or large label values.
    """
    unique = numpy.unique(masks)
    positive = unique[unique > 0]
    return positive.size == 0 or (positive[0] == 1 and positive[-1] == positive.size)


def sanitize_masks(masks: NDArray) -> tuple[NDArray, NDArray[numpy.int64]]:
    """Relabel arbitrary positive labels to contiguous ``1..N``.

    Parameters
    ----------
    masks
        Integer label array (any number of dimensions). Background is ``0``.

    Returns
    -------
    clean
        Array with labels ``1..N`` in ascending original-label order. The
        already-contiguous input is returned unchanged (no copy); otherwise a
        fresh relabelled copy is returned. The input is never mutated.
    ids
        ``ids[i]`` is the original label whose sanitized value is ``i + 1``
        (ascending). Use it to report results against the caller's IDs.

    Raises
    ------
    ValueError
        If ``masks`` is not an integer array or contains negative values.
    """
    if not numpy.issubdtype(masks.dtype, numpy.integer):
        raise ValueError(f"labels must be an integer array, got dtype {masks.dtype!r}")
    if masks.size and masks.min() < 0:
        raise ValueError("labels must be non-negative")

    if _is_contiguous(masks):
        n = int(masks.max(initial=0))
        return masks, numpy.arange(1, n + 1, dtype=numpy.int64)

    clean, _forward, _inverse = relabel_sequential(masks)
    ids = numpy.unique(masks)
    ids = ids[ids > 0].astype(numpy.int64)
    return clean, ids


def sanitize_labels(func: Callable) -> Callable:
    """Decorate a ``get_*`` function to sanitize its label argument.

    The label argument is detected by name (:data:`_MASK_PARAMS`), so the
    decorator is position-independent. Functions without a recognised label
    argument (e.g. the two-mask ``multimask`` functions, which take
    ``masks1``/``masks2``) are returned unchanged.
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

    wrapper._sanitized = True  # type: ignore[attr-defined]
    return wrapper
