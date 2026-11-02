"""Shared minimum precision for explicitly opted-in measurement arithmetic."""

import numpy
from numpy.typing import NDArray


def promote_float_pixels(pixels: NDArray) -> NDArray:
    """Widen float16/32 values exactly, without mutating or renormalizing them.

    Float64/wider floats and integers are returned unchanged. Integer overflow
    conventions and acquisition-scale handling are separate policies.
    """
    if pixels.dtype.kind == "f" and pixels.dtype.itemsize < 8:
        return pixels.astype(numpy.float64)
    return pixels
