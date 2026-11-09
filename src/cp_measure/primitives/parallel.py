"""Shared per-image batch driver for the numba backends.

A measurement's batch dispatch builds a list of per-image work items, then calls
:func:`map_images` to run the per-image function over them. For a real batch the
work runs across a thread pool; for a single image it runs serially, with no pool
overhead. Order is preserved and exceptions propagate. First used by the zernike /
radial_zernikes backend; the other batch backends can adopt it as they are wired.

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

# Pixels of batch work per worker thread. Below that, the Python parts of a
# measurement (which hold the GIL) outweigh its kernels, and more threads only
# queue for the GIL: on 256 px images, 20 threads were slower than 4 or a serial
# loop (HPC, 32 cores).
PIXELS_PER_WORKER = 512 * 512


def _usable_cpus() -> int:
    """CPUs this process may use, as scirpy's ``_get_usable_cpus``: the affinity set
    (which Slurm and cgroups restrict, unlike ``os.cpu_count()``), else the machine's
    count (macOS has no ``sched_getaffinity``), capped by ``NUMBA_NUM_THREADS``."""
    from numba import config

    try:
        usable = len(os.sched_getaffinity(0))
    except AttributeError:
        usable = os.cpu_count() or 1
    return min(usable, config.NUMBA_NUM_THREADS)


def map_images(
    func: Callable[[T], R],
    items: Sequence[T],
    *,
    max_workers: int | None = None,
    pixels: int | None = None,
) -> list[R]:
    """Return ``[func(x) for x in items]``, parallel across a thread pool for
    batches of two or more, serial for zero or one item.

    Order matches ``items``. ``max_workers`` defaults to the CPUs this process may
    use (see :func:`_usable_cpus`) and is
    capped at ``len(items)`` so a batch never spawns more threads than images (an
    explicit ``0`` is passed through and rejected by ``ThreadPoolExecutor``).
    Without ``max_workers``, ``pixels`` (the batch's total pixel count) also caps
    it at one worker per :data:`PIXELS_PER_WORKER`; one worker runs serially.
    Parallelism requires ``func`` to release the GIL during its heavy work.
    """
    workers = max_workers if max_workers is not None else _usable_cpus()
    if max_workers is None and pixels is not None:
        workers = min(workers, max(1, pixels // PIXELS_PER_WORKER))
    if workers:
        workers = min(workers, len(items))
    if len(items) <= 1 or workers == 1:
        return [func(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, items))
