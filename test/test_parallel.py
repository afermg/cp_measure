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
