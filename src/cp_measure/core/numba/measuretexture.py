"""Numba-accelerated MeasureTexture (Haralick) backend.

Drop-in for :func:`cp_measure.core.measuretexture.get_texture`, which documents
what the measurement is. Almost all of that function's cost is
``mahotas.features.haralick`` per object (mahotas accumulates its entropies in
Python); this backend replaces it with one fused GLCM + 13-feature numba kernel,
:func:`cp_measure.core.numba._texture.haralick_object`.

Host preparation and output packing are the numpy backend's ``_prep``/``_pack``,
so both backends resolve dimensionality and name their features identically.
Values agree with mahotas to ~1e-5 relative — see ``_texture`` for the two
expressions where they differ, and why.
"""

import numpy
from numpy.typing import NDArray

from cp_measure.core.measuretexture import _pack, _prep
from cp_measure.core.numba._texture import DELTAS_2D, DELTAS_3D, haralick_object


def get_texture(
    masks: NDArray[numpy.integer],
    pixels: NDArray[numpy.floating],
    scale: int = 3,
    gray_levels: int = 256,
) -> dict[str, NDArray[numpy.floating]] | list[dict[str, NDArray[numpy.floating]]]:
    """Haralick texture features per object; a single image/volume, or a batch.

    Each image is 2D ``(Y, X)`` or 3D ``(Z, Y, X)``, mask and pixels the same
    shape, as in the numpy backend. A batch is a list/tuple of such images or a
    4D ``(B, Z, Y, X)`` array, and yields one result dict per image — matching
    the ``to_bzyx`` convention that 3D is a single volume, never a batch. Images
    are measured as given rather than
    normalised to ``(Z, Y, X)`` first: the direction count follows
    ``pixels.ndim``, and normalising would make a 2D image indistinguishable from
    a single-slice volume.
    """
    masks_batched = isinstance(masks, (list, tuple)) or numpy.ndim(masks) == 4
    pixels_batched = isinstance(pixels, (list, tuple)) or numpy.ndim(pixels) == 4
    if masks_batched != pixels_batched:
        raise ValueError("masks and pixels must both be batches, or both single images")
    if not masks_batched:
        return _texture_image(masks, pixels, scale, gray_levels)
    if len(masks) != len(pixels):
        raise ValueError(
            f"batch size mismatch: {len(masks)} masks vs {len(pixels)} images"
        )
    return [_texture_image(m, p, scale, gray_levels) for m, p in zip(masks, pixels)]


def _texture_image(
    masks: NDArray[numpy.integer],
    pixels: NDArray[numpy.floating],
    scale: int,
    gray_levels: int,
) -> dict[str, NDArray[numpy.floating]]:
    props, n_directions = _prep(masks, pixels, scale, gray_levels)
    deltas = DELTAS_2D if n_directions == 4 else DELTAS_3D
    offsets = numpy.ascontiguousarray(scale * deltas)

    features = numpy.empty((n_directions, 13, len(props)))
    for index, prop in enumerate(props):
        crop = numpy.ascontiguousarray(prop.image_intensity)  # uint8 from regionprops
        if crop.ndim == 2:
            crop = crop[numpy.newaxis]  # the kernel always takes (Z, Y, X)
        features[:, :, index] = haralick_object(crop, offsets)
    return _pack(features, scale, gray_levels)
