from itertools import product

import numpy
import pytest

from cp_measure.bulk import get_core_measurements
from cp_measure.core.measurecolocalization import get_correlation_overlap
from cp_measure.examples import get_masks, get_pixels


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
    for name, v in get_core_measurements().items():
        result = v(mask, pixels.copy())
        if (mask_name, name) not in exceptions:
            text = f"Feature {name} returned zero/null on mask {mask_name}"
            if isinstance(result, dict):
                # Test that at least one item contains a valid value
                assert any(
                    [any(~(x == 0 | numpy.isnan(x))) for x in result.values()]
                ), text
                # Test that the output and number of masks match
                assert all([len(x) == mask.max() for x in result.values()]), (
                    f"Input-Output size does not match: Feature {name}, mask {mask_name}"
                )
            else:
                assert result != 0 and not numpy.isnan(result), text


def test_correlation_overlap():
    size = 240
    rng = numpy.random.default_rng(42)
    pixels = rng.integers(low=1, high=255, size=(size, size, 2))

    # Create two similar-sized objects
    masks = numpy.zeros((size, size), dtype=int)
    masks[50:100, 50:100] = 1  # First square 50x50
    masks[80:120, 90:120] = 1  # Major asymmetries on bottom right edge
    masks[150:200, 150:200] = 2  # Second square 50x50
    masks[175:180, 180:210] = 2  # Minor asymmetries on bottom right edge
    get_correlation_overlap(
        pixels_1=pixels[..., 0], pixels_2=pixels[..., 0], masks=masks
    )
