"""Shared minimum precision for explicitly opted-in measurement arithmetic."""

from typing import Literal

import numpy
from numpy.typing import NDArray


Precision = Literal["input", "float64"]


def validate_precision(precision: Precision) -> None:
    if precision not in ("input", "float64"):
        raise ValueError("precision must be 'input' or 'float64'")


def promote_float_pixels(pixels: NDArray, precision: Precision = "input") -> NDArray:
    """Widen float16/32 values exactly, without mutating or renormalizing them.

    ``input`` preserves the pre-policy dtype path; it does not force all
    intermediates to the input dtype or promise CellProfiler parity.
    ``float64`` widens float16/32 only. Integer conventions are unchanged.
    """
    validate_precision(precision)
    if (
        precision == "float64"
        and pixels.dtype.kind == "f"
        and pixels.dtype.itemsize < 8
    ):
        return pixels.astype(numpy.float64)
    return pixels
