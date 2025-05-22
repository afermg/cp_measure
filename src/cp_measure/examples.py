"""
Convenience functions for tests and demos.
"""

import numpy


def get_pixels(size: int = 789, seed: int = 42) -> numpy.ndarray:
    rng = numpy.random.default_rng(seed)
    random_pixels = rng.integers(low=0, high=10, size=(size, size))

    return random_pixels


def get_masks(size: int = 789, mask_width: int = 9) -> tuple[numpy.ndarray]:
    full = numpy.ones(
        (size, size), dtype=numpy.uint16
    )  # A mask that covers the entire image
    empty = numpy.zeros_like(full)

    # A mask in the center
    center = empty.copy()
    center[mask_width : -mask_width - 1, mask_width : -mask_width - 1] = True

    # Everything but a mask in the center
    edges = (~center.astype(bool)).astype(numpy.uint16)

    # Only corners
    corner = empty.copy()
    corner[:mask_width, :mask_width] = True
    corners = [numpy.rot90(corner, k=i, axes=(0, 1)) for i in range(4)]

    # One pixel mask
    one = empty.copy()
    one[size // 2, size // 2] = True

    # Two pixels' mask
    two = one + numpy.roll(one, shift=1)

    # Two masks
    full_two_masks = full.copy()
    full_two_masks[size // 2 : -1, :] = 2

    return {
        "full": full,
        "full_2": full_two_masks,
        "center": center,
        "edges": edges,
        **{f"corner_{i}": corners[i] for i in range(4)},
        "one": one,
        "two": two,
    }
