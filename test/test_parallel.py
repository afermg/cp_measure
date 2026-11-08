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
