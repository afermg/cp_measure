"""The shared batch driver maps a per-image function over a batch, parallel for
2+ items, serial for 0 or 1, preserving order and propagating exceptions."""

import pytest

from cp_measure.primitives.parallel import map_images


@pytest.mark.parametrize("n", [0, 1, 2, 5, 17])
def test_map_images_matches_serial_and_preserves_order(n):
    items = list(range(n))
    assert map_images(lambda x: x * x, items) == [x * x for x in items]


def test_map_images_propagates_exceptions():
    def boom(x):
        if x == 2:
            raise ValueError("boom")
        return x

    with pytest.raises(ValueError, match="boom"):
        map_images(boom, [0, 1, 2, 3])


def test_map_images_raises_the_lowest_index_exception():
    """Several failing items surface the lowest-index one deterministically, not
    whichever worker finished first; the serial path raises in item order too."""

    def f(x):
        if x in (3, 7):
            raise ValueError(f"fail-{x}")
        return x

    # Size the batch so the threaded path is used when the process may use 2+ CPUs.
    with pytest.raises(ValueError, match="fail-3"):
        map_images(f, list(range(8)), pixels=8 * 512 * 512)


def test_map_images_sizes_the_pool_by_pixels():
    """A batch with less than two workers' worth of pixels runs on the calling
    thread; a larger one uses threads (when this process may use 2+ CPUs)."""
    import threading

    from cp_measure.primitives.parallel import PIXELS_PER_WORKER, _usable_cpus

    def thread_ids(pixels):
        ids = map_images(lambda _: threading.get_ident(), range(8), pixels=pixels)
        return set(ids)

    assert thread_ids(PIXELS_PER_WORKER) == {threading.get_ident()}
    if _usable_cpus() > 1:
        assert threading.get_ident() not in thread_ids(8 * PIXELS_PER_WORKER)


def test_map_images_reuses_its_threads_across_calls():
    """Repeated calls run on one pool's threads instead of starting new ones."""
    import threading

    from cp_measure.primitives.parallel import PIXELS_PER_WORKER, _usable_cpus

    names = set()
    for _ in range(5):
        names |= set(
            map_images(
                lambda _: threading.current_thread().name,
                range(64),
                pixels=64 * PIXELS_PER_WORKER,
            )
        )
    assert len(names) <= _usable_cpus()


def test_map_images_nested_in_a_pool_thread_runs_serially():
    """A map_images call from inside a pool task runs on that task's thread, so a
    full pool cannot deadlock waiting on itself."""
    import threading

    from cp_measure.primitives.parallel import PIXELS_PER_WORKER

    def inner(_):
        here = threading.get_ident()
        ids = map_images(
            lambda _: threading.get_ident(), range(4), pixels=4 * PIXELS_PER_WORKER
        )
        return set(ids) == {here}

    big = 64 * PIXELS_PER_WORKER
    assert all(map_images(inner, range(64), pixels=big))


def test_usable_cpus_without_numba(monkeypatch):
    """The pool size does not need numba, which is optional: the NumPy paths
    (e.g. the featurizer's mask relabelling) use the pool too."""
    import sys

    from cp_measure.primitives.parallel import _usable_cpus

    monkeypatch.setitem(sys.modules, "numba", None)  # makes `import numba` fail
    assert _usable_cpus() >= 1


def test_pool_is_created_once_under_concurrent_first_use():
    """Threads that ask for a process's pool at the same moment all get one pool."""
    import threading

    from cp_measure.primitives.parallel import _pool

    for trial in range(20):
        key = -1 - trial  # a fresh "process id" per trial, so each one is a first use
        start = threading.Barrier(16)
        got = []

        def first_use():
            start.wait()
            got.append(_pool(key))

        threads = [threading.Thread(target=first_use) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({id(p) for p in got}) == 1
