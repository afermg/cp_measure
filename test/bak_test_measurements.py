from itertools import cycle, product

import numpy
import pytest

from cp_measure.bulk import get_all_measurements


def get_pixels(size: int = 789, seed: int = 42) -> numpy.ndarray:
    rng = numpy.random.default_rng(seed)
    random_pixels = rng.integers(low=0, high=10, size=(size, size))

    return random_pixels


def get_masks(size: int = 789, mask_width: int = 9) -> tuple[numpy.ndarray]:
    full = numpy.ones((size, size), dtype=bool)
    empty = ~full

    center = empty.copy()
    center[mask_width : -mask_width - 1, mask_width : -mask_width - 1] = True

    edges = ~center

    corner = empty.copy()
    corner[:mask_width, :mask_width] = True
    corners = [numpy.rot90(corner, k=i, axes=(0, 1)) for i in range(4)]

    one = empty.copy()
    one[size // 2, size // 2] = True

    two = one + numpy.roll(one, shift=1)
    return {
        "full": full,
        "center": center,
        "edges": edges,
        **{f"corner_{i}": corners[i] for i in range(4)},
        "one": one,
        "two": two,
    }


# These are cases where we expect everything to be zero


# for (mask_name, mask), img in zip(get_masks().items(), cycle((get_pixels(),))):
@pytest.mark.parametrize("named_mask", get_masks().items())
@pytest.mark.parametrize("pixels", [get_pixels()])
def test_measurements(named_mask: tuple[str, numpy.ndarray], pixels: numpy.ndarray):
    exceptions = (
        ("one", "ferret"),
        ("one", "zernike"),
        *list(  # Downsampling means trouble for the 'thin' masks
            product(
                ("one", "two", "edges", *[f"corner_{i}" for i in range(4)]),
                ("radial_distribution", "radial_zernikes", "texture", "granularity"),
            )
        ),
    )
    mask_name, mask = named_mask
    for name, v in get_all_measurements().items():
        result = v(mask, pixels)
        if (mask_name, name) not in exceptions:
            text = f"Feature {name} returned zero/null on mask {mask_name}"
            if isinstance(result, dict):
                assert any(
                    [x != 0 and not numpy.isnan(x) for x in result.values()]
                ), text
            else:
                assert result != 0 and not numpy.isnan(result), text
