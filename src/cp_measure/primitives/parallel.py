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

import functools
import os
import threading
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


_THREAD_PREFIX = "cp_measure"


@functools.cache
def _pool(pid: int) -> ThreadPoolExecutor:
    """The process's thread pool, created on first use and kept for later calls
    (starting threads per call held the GIL for about 5% of a 32-core batch).
    Keyed by process id: a forked child gets its own pool, as the parent's threads
    do not exist there. Sized to the CPUs usable when it is first created."""
    return ThreadPoolExecutor(_usable_cpus(), thread_name_prefix=_THREAD_PREFIX)


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
    capped at ``len(items)``, and at the size of the process's shared pool (an
    explicit ``0`` is rejected).
    Without ``max_workers``, ``pixels`` (the batch's total pixel count) also caps
    it at one worker per :data:`PIXELS_PER_WORKER`; one worker runs serially.
    A call from inside a pool task runs serially on that task's thread, so a full
    pool never waits on itself. Parallelism requires ``func`` to release the GIL
    during its heavy work.
    """
    if max_workers == 0:
        raise ValueError("max_workers must be greater than 0")
    workers = max_workers if max_workers is not None else _usable_cpus()
    if max_workers is None and pixels is not None:
        workers = min(workers, max(1, pixels // PIXELS_PER_WORKER))
    if workers:
        workers = min(workers, len(items))
    nested = threading.current_thread().name.startswith(_THREAD_PREFIX)
    if len(items) <= 1 or workers == 1 or nested:
        return [func(x) for x in items]
    # ``workers`` runner tasks take the next item until none is left, so this call
    # uses at most ``workers`` pool threads and slow images do not hold up a queue.
    results: list = [None] * len(items)
    indices = iter(range(len(items)))
    lock = threading.Lock()

    def runner() -> None:
        while True:
            with lock:
                i = next(indices, None)
            if i is None:
                return
            results[i] = func(items[i])

    pool = _pool(os.getpid())
    for future in [pool.submit(runner) for _ in range(workers)]:
        future.result()  # raises the runner's exception, if any
    return results
