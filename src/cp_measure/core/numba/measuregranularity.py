"""Numba-accelerated granularity backend (2D).

Mirrors :func:`cp_measure.core.measuregranularity.get_granularity` but swaps the
skimage morphology (background opening, per-iteration disk(1) erosion, geodesic
reconstruction) for the bit-exact numba kernels in :mod:`._granularity`, and the
16 ``scipy.ndimage.mean`` calls for a precomputed ``bincount`` per-object mean
sampled by a sparse point-query of the reconstructed image (no full-resolution
upsample). Resampling stays on ``scipy.ndimage.map_coordinates``, as in every
backend.

Batch-shaped via the canonical ``(B, Z, Y, X)`` form (see
:func:`cp_measure.primitives.shapes.to_bzyx`): a single image is ``B == 1``. Each
``(Z, Y, X)`` element with ``Z == 1`` runs the numba 2D path; ``Z > 1`` (a true
3D volume) falls back to the numpy baseline.
"""

import numpy
import scipy.ndimage
from numpy.typing import NDArray

from cp_measure.core.measuregranularity import get_granularity as _get_granularity_numpy
from cp_measure.core.numba._granularity import (
    disk_dilation_2d,
    disk_erosion_2d,
    erosion_4conn_2d,
    reconstruction_by_dilation_2d,
)
from cp_measure.primitives.shapes import to_bzyx


def get_granularity(
    masks,
    pixels,
    subsample_size: float = 0.25,
    image_sample_size: float = 0.25,
    element_size: int = 10,
    granular_spectrum_length: int = 16,
):
    """Granularity spectrum per object; accepts a single image/volume or a batch.

    Returns a single ``{Granularity_k: array}`` dict for a lone 2D image or 3D
    volume, or a list of such dicts for a batch (4D ``(B,Z,Y,X)`` array or a list
    of images). Output arrays are indexed densely by label ``1..max(mask)``.
    """
    masks_zyx, pixels_zyx, unwrap = to_bzyx(masks, pixels)
    results = [
        _granularity_one(
            m, p, subsample_size, image_sample_size, element_size, granular_spectrum_length
        )
        for m, p in zip(masks_zyx, pixels_zyx)
    ]
    return unwrap(results)


def _granularity_one(
    mask_zyx, pixels_zyx, subsample_size, image_sample_size, element_size, ng
):
    """Dispatch one ``(Z, Y, X)`` element: numba 2D path, else numpy baseline (3D)."""
    if mask_zyx.shape[0] == 1:  # Z == 1 -> 2D image
        return _granularity_2d(
            mask_zyx[0], pixels_zyx[0], subsample_size, image_sample_size, element_size, ng
        )
    return _get_granularity_numpy(
        mask_zyx, pixels_zyx, subsample_size, image_sample_size, element_size, ng
    )


def _granularity_2d(
    orig_mask: NDArray[numpy.integer],
    orig_pixels: NDArray[numpy.floating],
    subsample_size: float,
    image_sample_size: float,
    element_size: int,
    ng: int,
) -> dict[str, NDArray[numpy.floating]]:
    orig_shape = numpy.array(orig_pixels.shape)
    new_shape = orig_shape.copy()

    # 1. Subsample image (scipy map_coordinates, as baseline). The baseline also
    # resamples the mask here but never reads it (the spectrum loop uses orig_mask),
    # so we skip that dead order-0 resample.
    if subsample_size < 1:
        new_shape = (orig_shape * subsample_size).astype(int)
        i, j = numpy.mgrid[0 : new_shape[0], 0 : new_shape[1]].astype(float) / subsample_size
        pixels = scipy.ndimage.map_coordinates(orig_pixels, (i, j), order=1)
    else:
        pixels = orig_pixels.astype(numpy.float64, copy=True)

    # 2. Background subtraction via greyscale opening (numba disk erosion+dilation)
    if image_sample_size < 1:
        back_shape = new_shape * image_sample_size
        i, j = numpy.mgrid[0 : back_shape[0], 0 : back_shape[1]].astype(float) / image_sample_size
        back_pixels = scipy.ndimage.map_coordinates(pixels, (i, j), order=1)
    else:
        back_pixels = pixels
        back_shape = new_shape
    radius = element_size
    # back_mask is all-ones (full-frame), so the baseline's masked re-zeroing is a no-op
    back_pixels = disk_erosion_2d(back_pixels, radius)
    back_pixels = disk_dilation_2d(back_pixels, radius)
    if image_sample_size < 1:
        i, j = numpy.mgrid[0 : new_shape[0], 0 : new_shape[1]].astype(float)
        i *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
        j *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
        back_pixels = scipy.ndimage.map_coordinates(back_pixels, (i, j), order=1)
    pixels = pixels - back_pixels
    pixels[pixels < 0] = 0

    # 3. Per-object readback structures (dense labels 1..max, like baseline)
    flat_mask = orig_mask.ravel()
    in_obj = flat_mask > 0
    max_label = int(orig_mask.max()) if orig_mask.size else 0
    results: dict[str, NDArray[numpy.floating]] = {}
    if max_label == 0:  # no objects -> baseline returns empty arrays
        empty = numpy.zeros((0,))
        for granularity_id in range(1, ng + 1):
            results[f"Granularity_{granularity_id}"] = empty
        return results

    flat_pos = numpy.flatnonzero(in_obj)
    labels_in = flat_mask[flat_pos]
    counts = numpy.bincount(labels_in, minlength=max_label + 1)[1:].astype(numpy.float64)

    needs_resize = not numpy.array_equal(new_shape, orig_shape)
    if needs_resize:
        yy, xx = numpy.unravel_index(flat_pos, tuple(orig_shape))
        sy = yy * (float(new_shape[0] - 1) / float(orig_shape[0] - 1))
        sx = xx * (float(new_shape[1] - 1) / float(orig_shape[1] - 1))

    def _label_mean(values_at_in_obj):
        sums = numpy.bincount(labels_in, weights=values_at_in_obj, minlength=max_label + 1)[1:]
        with numpy.errstate(invalid="ignore", divide="ignore"):
            return sums / counts  # 0/0 -> nan for absent labels, matching scipy.ndimage.mean

    orig_valid = orig_pixels.ravel()[flat_pos].astype(numpy.float64)
    current_mean = _label_mean(orig_valid)
    start_mean = numpy.maximum(current_mean, numpy.finfo(float).eps)

    # 4. Granular spectrum loop (numba disk(1) erosion + reconstruction, cascaded mask)
    ero = pixels
    recon_mask = pixels
    for granularity_id in range(1, ng + 1):
        ero = erosion_4conn_2d(ero)
        if ero.max() == 0:
            rec = ero
        else:
            rec = reconstruction_by_dilation_2d(ero, recon_mask)
            recon_mask = rec  # cascade: rec_g <= rec_{g-1} <= pixels (exact)

        if needs_resize:
            rec_valid = scipy.ndimage.map_coordinates(rec, (sy, sx), order=1)
        else:
            rec_valid = rec.ravel()[flat_pos]

        new_mean = _label_mean(rec_valid)
        results[f"Granularity_{granularity_id}"] = (current_mean - new_mean) * 100 / start_mean
        current_mean = new_mean

    return results
