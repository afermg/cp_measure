"""Numba-accelerated MeasureTexture (Haralick) backend.

Drop-in for :func:`cp_measure.core.measuretexture.get_texture`. The ~99% cost —
``mahotas.features.haralick`` per object (the step-0 profile) — is replaced by a
fused numba GLCM + 13-Haralick-feature kernel
(:func:`cp_measure.core.numba._texture.haralick_object`), bit-exact to mahotas. The
host prep (``skimage.util.img_as_ubyte`` + mask-zeroing + optional ``gray_levels``
rescale + ``skimage.measure.regionprops``) stays host-side (~1%, exact).

Batch-shaped via the canonical ``(B, Z, Y, X)`` form (single image = ``B == 1``).
2D (``Z == 1`` -> 4 directions) and 3D (``Z > 1`` -> 13 directions) are both
supported; the ``Z == 1`` choice follows the ``to_bzyx`` convention (a 2D image and
a single-slice volume are indistinguishable once normalised) and matches the common
case. texture is already per-object (regionprops crops), so results match the numpy
baseline directly — no Issue-#22 analogue.
"""

import numpy
import skimage.exposure
import skimage.measure
import skimage.util
from numpy.typing import NDArray

from cp_measure.core.measuretexture import F_HARALICK
from cp_measure.core.numba._texture import DELTAS_2D, DELTAS_3D, haralick_object
from cp_measure.primitives.shapes import to_bzyx


def get_texture(
    masks: NDArray[numpy.integer],
    pixels: NDArray[numpy.floating],
    scale: int = 3,
    gray_levels: int = 256,
) -> dict[str, NDArray[numpy.floating]]:
    """Haralick texture features per object; single image/volume or batch."""
    masks_zyx, pixels_zyx, unwrap = to_bzyx(masks, pixels)
    results = [
        _texture_image(m, p, scale, gray_levels) for m, p in zip(masks_zyx, pixels_zyx)
    ]
    return unwrap(results)


def _texture_keys(scale, gray_levels, n_directions):
    return [
        "{}_{:d}_{:02d}_{:d}".format(feature, scale, direction, gray_levels)
        for direction in range(n_directions)
        for feature in F_HARALICK
    ]


def _texture_image(
    masks_zyx: NDArray[numpy.integer],
    pixels_zyx: NDArray[numpy.floating],
    scale: int,
    gray_levels: int,
) -> dict[str, NDArray[numpy.floating]]:
    two_d = masks_zyx.shape[0] == 1
    masks = masks_zyx[0] if two_d else masks_zyx
    pixels = pixels_zyx[0] if two_d else pixels_zyx
    deltas = DELTAS_2D if two_d else DELTAS_3D
    n_directions = deltas.shape[0]

    unique_labels = numpy.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]
    if len(unique_labels) == 0:
        return {
            k: numpy.zeros(0) for k in _texture_keys(scale, gray_levels, n_directions)
        }

    # Host prep — identical to the numpy reference (kept on scipy/skimage).
    pix = skimage.util.img_as_ubyte(pixels, force_copy=True)
    pix[~masks.astype(bool)] = 0
    if gray_levels != 256:
        pix = skimage.exposure.rescale_intensity(
            pix, in_range=(0, 255), out_range=(0, gray_levels - 1)
        ).astype(numpy.uint8)
    props = skimage.measure.regionprops(masks, pix)

    offsets = numpy.ascontiguousarray(scale * deltas)
    features = numpy.empty((n_directions, 13, len(unique_labels)))
    for index, prop in enumerate(props):
        crop = numpy.ascontiguousarray(prop["intensity_image"], dtype=numpy.int64)
        if two_d:
            crop = crop[numpy.newaxis]  # (h, w) -> (1, h, w)
        features[:, :, index] = haralick_object(crop, offsets)

    results = {}
    for direction_i, direction_features in enumerate(features):
        for feature_name, values in zip(F_HARALICK, direction_features):
            results[
                "{}_{:d}_{:02d}_{:d}".format(
                    feature_name, scale, direction_i, gray_levels
                )
            ] = values
    return results
