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
    mask_name, mask = named_mask
    for name, v in get_correlation_measurements().items():
        result = v(pixels.copy(), pixels.copy(), mask)
        text = f"Feature {name} returned zero/null on mask {mask_name}"
        
