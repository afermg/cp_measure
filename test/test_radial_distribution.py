"""Regression tests for per-label radial-distribution independence."""

import numpy
import pytest

from cp_measure.core.measureobjectintensitydistribution import (
    _maximum_position_of_labels,
    get_radial_distribution,
)


def test_radial_center_uses_first_c_order_maximum_per_label():
    """Unique and tied maxima have an explicit, deterministic center policy."""
    image = numpy.array(
        [
            [0.0, 1.0, 3.0, 0.0],
            [0.0, 2.0, 1.0, 0.0],
            [0.0, 4.0, 4.0, 0.0],
            [0.0, 4.0, 4.0, 0.0],
        ]
    )
    labels = numpy.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 2, 2, 0],
            [0, 2, 2, 0],
        ]
    )

    rows, columns = _maximum_position_of_labels(image, labels, nobjects=2)
    numpy.testing.assert_array_equal(rows, [0, 2])
    numpy.testing.assert_array_equal(columns, [2, 1])


@pytest.mark.parametrize(
    ("scaled", "maximum_radius"),
    [(True, 100), (False, 10)],
    ids=["scaled", "unscaled-with-overflow"],
)
def test_radial_distribution_is_independent_of_other_labels(scaled, maximum_radius):
    """Issue #22: measuring objects together must equal measuring each alone."""
    size = 240
    pixels = numpy.random.default_rng(42).integers(1, 255, size=(size, size))

    labels = numpy.zeros_like(pixels)
    labels[50:100, 50:100] = 1
    labels[80:120, 90:120] = 1
    labels[150:200, 150:200] = 2
    labels[175:180, 180:210] = 2

    # Empty angular wedges in unscaled outer bins are masked downstream.
    with numpy.errstate(divide="ignore", invalid="ignore"):
        together = get_radial_distribution(
            labels, pixels, scaled=scaled, maximum_radius=maximum_radius
        )
        separate = [
            get_radial_distribution(
                labels == label,
                pixels,
                scaled=scaled,
                maximum_radius=maximum_radius,
            )
            for label in (1, 2)
        ]

    for key, actual in together.items():
        expected = numpy.concatenate([result[key] for result in separate])
        numpy.testing.assert_array_equal(actual, expected, err_msg=key)
