from itertools import product

import numpy
import pytest

from cp_measure.bulk import get_core_measurements, get_core_measurements_3d
from cp_measure.core.measurecolocalization import get_correlation_overlap
from cp_measure.core.measureobjectintensity import (
    INTEGRATED_INTENSITY_EDGE,
    INTENSITY,
    MAX_INTENSITY_EDGE,
    MEAN_INTENSITY_EDGE,
    MIN_INTENSITY_EDGE,
    STD_INTENSITY_EDGE,
    get_intensity,
)
from cp_measure.examples import get_masks, get_pixels


@pytest.mark.parametrize("named_mask", get_masks().items())
@pytest.mark.parametrize("pixels", [get_pixels()])
def test_measurements(named_mask: tuple[str, numpy.ndarray], pixels: numpy.ndarray):
    exceptions = (
        ("one", "feret"),
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


def test_3d_measurements():
    """Test 3D support: 2D-only measurements return empty, 3D ones produce valid output."""
    size = 240
    rng = numpy.random.default_rng(42)
    pixels = rng.integers(low=1, high=255, size=(32, size, size))

    masks = numpy.zeros_like(pixels)
    masks[:, 50:100, 50:100] = 1
    masks[:, 80:120, 90:120] = 1
    masks[:, 150:200, 150:200] = 2
    masks[:, 175:180, 180:210] = 2

    # 2D-only measurements should return empty dict for 3D input
    only_2d = {"radial_distribution", "radial_zernikes", "zernike", "feret"}
    for name, v in get_core_measurements().items():
        result = v(masks, pixels)
        assert isinstance(result, dict), f"{name} did not return a dict"
        if name in only_2d:
            assert result == {}, f"{name} should return empty dict for 3D input"
        else:
            assert any(any(~(x == 0 | numpy.isnan(x))) for x in result.values()), (
                f"{name} returned zero/null on 3D input"
            )
            assert all(len(x) == masks.max() for x in result.values()), (
                f"{name}: output length doesn't match number of objects"
            )

    # get_core_measurements_3d should return only 3D-compatible measurements
    measurements_3d = get_core_measurements_3d()
    assert set(measurements_3d.keys()) == set(get_core_measurements().keys()) - only_2d
    for name, v in measurements_3d.items():
        result = v(masks, pixels)
        assert len(result) > 0, f"{name} returned empty dict"


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


def test_get_intensity_edge_measurements_flag():
    """With edge_measurements=True (default) edge keys are present; with False they are omitted."""
    masks = get_masks()["one"]
    pixels = get_pixels()
    n_objects = int(masks.max())

    edge_keys = [
        f"{INTENSITY}_{INTEGRATED_INTENSITY_EDGE}",
        f"{INTENSITY}_{MEAN_INTENSITY_EDGE}",
        f"{INTENSITY}_{STD_INTENSITY_EDGE}",
        f"{INTENSITY}_{MIN_INTENSITY_EDGE}",
        f"{INTENSITY}_{MAX_INTENSITY_EDGE}",
    ]

    result_default = get_intensity(masks, pixels.copy())
    for key in edge_keys:
        assert key in result_default, f"default (edge_measurements=True) should include {key}"

    result_with_edge = get_intensity(masks, pixels.copy(), edge_measurements=True)
    for key in edge_keys:
        assert key in result_with_edge, f"edge_measurements=True should include {key}"
        assert len(result_with_edge[key]) == n_objects

    result_without_edge = get_intensity(masks, pixels.copy(), edge_measurements=False)
    for key in edge_keys:
        assert key not in result_without_edge, (
            f"edge_measurements=False should omit {key}"
        )
    assert "Intensity_IntegratedIntensity" in result_without_edge
    assert "Intensity_MeanIntensity" in result_without_edge
    assert all(len(v) == n_objects for v in result_without_edge.values())
