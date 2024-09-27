"""
Test functions that operate multiple pixel arrays (e.g., correlations).
"""
import numpy
import pytest

from cp_measure.bulk import get_correlation_measurements
from cp_measure.examples import get_pixels, get_masks

@pytest.mark.parametrize("named_mask", get_masks().items())
@pytest.mark.parametrize("pixels", [get_pixels()])
def test_measurements(named_mask: tuple[str, numpy.ndarray], pixels: numpy.ndarray):
    exceptions = (("one", "costes"),
                  ("one", "pearson"),
                  ("two", "costes"))
    mask_name, mask = named_mask
    for name, v in get_correlation_measurements().items():
        if (mask_name, name) not in exceptions:
            result = v(pixels.copy(), pixels.copy(), mask)
            text = f"Feature {name} returned zero/null on mask {mask_name}"

            # Test that at least one item contains a valid value
            assert all(
                        [any(~(x == 0 | numpy.isnan(x))) for x in result.values()]
                    ), text
            # Test that the output and number of masks match
            assert all([len(x)==mask.max() for x in result.values()]), f"Input-Output size does not match: Feature {name}, mask {mask_name}"
