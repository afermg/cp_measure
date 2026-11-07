"""Shared per-image batch driver for the numba backends.

A measurement's batch dispatch builds a list of per-image work items, then calls
:func:`map_images` to run the per-image function over them. For a real batch the
work runs across a thread pool; for a single image it runs serially, with no pool
overhead. Order is preserved and exceptions propagate.

Parallelism only materialises when ``func``'s heavy work releases the GIL — the
numba kernels it calls must be ``@njit(nogil=True)`` (numpy's large array ops also
release it). A ``ThreadPoolExecutor`` shares memory with the caller, so no image
data is copied to workers, unlike a process pool.
"""

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_images(
    func: Callable[[T], R], items: Sequence[T], *, max_workers: int | None = None
) -> list[R]:
    """Return ``[func(x) for x in items]``, parallel across a thread pool for
    batches of two or more, serial for zero or one item.

    Order matches ``items``. ``max_workers`` defaults to ``os.cpu_count()``.
    Parallelism requires ``func`` to release the GIL during its heavy work.
    """
    if len(items) <= 1:
        return [func(x) for x in items]
    with ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
        return list(executor.map(func, items))
