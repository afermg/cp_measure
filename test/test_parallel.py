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
