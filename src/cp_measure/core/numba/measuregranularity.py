"""Numba-accelerated granularity backend (2D).

Mirrors :func:`cp_measure.core.measuregranularity.get_granularity` but swaps the
skimage morphology (background opening, per-iteration disk(1) erosion, geodesic
reconstruction) for the bit-exact numba kernels in :mod:`._granularity`, and the
16 ``scipy.ndimage.mean`` calls (each on a full-resolution order-1 upsample of the
reconstructed image) for per-object sums of fixed bilinear weights: one pass over
the mask (:func:`._granularity.readback_tables`) sums each object's corner weights
on a block over its bounding box in the working grid, and each spectrum step then
reads only those blocks (:func:`._granularity.readback_sums`). This matches the
upsample-then-mean to floating-point summation order (~1e-12 absolute on the
spectrum, inside this backend's ``rtol=1e-7, atol=1e-10`` contract), and is
bit-exact when no resampling is needed. The three one-off grid resamples
(subsample / background) stay on ``map_coordinates``.

Batch-shaped via the canonical ``(B, Z, Y, X)`` form (see
:func:`cp_measure.primitives.shapes.to_bzyx`): a single image is ``B == 1``. Each
``(Z, Y, X)`` element with ``Z == 1`` runs the numba 2D path; ``Z > 1`` (a true
3D volume) falls back to the numpy baseline.
"""

import numpy
import scipy.ndimage
from numpy.typing import NDArray

from cp_measure._precision import Precision, promote_float_pixels, validate_precision
from cp_measure.core.measuregranularity import get_granularity as _get_granularity_numpy
from cp_measure.core.numba._granularity import (
    disk_dilation_2d,
    disk_erosion_2d,
    erosion_4conn_2d,
    label_sums,
    readback_sums,
    readback_tables,
    reconstruction_by_dilation_2d,
)
from cp_measure.primitives.parallel import map_images
from cp_measure.primitives.shapes import to_bzyx


def get_granularity(
    masks,
    pixels,
    subsample_size: float = 0.25,
    image_sample_size: float = 0.25,
    element_size: int = 10,
    granular_spectrum_length: int = 16,
    *,
    precision: Precision = "input",
    max_workers: int | None = None,
):
    """Granularity spectrum per object; accepts a single image/volume or a batch.

    Returns a single ``{Granularity_k: array}`` dict for a lone 2D image or 3D
    volume, or a list of such dicts for a batch (4D ``(B,Z,Y,X)`` array or a list
    of same-shaped images). Ragged batches are rejected by the shared normalizer.
    These raw/bulk shapes are distinct from the featurizer's explicit batch and
    channel axes. Output arrays are indexed densely by label ``1..max(mask)``.
    ``precision="input"`` retains the pre-policy mixed-precision path;
    ``"float64"`` widens float16/32 before resampling/background operations.
    A batch runs on a thread pool; ``max_workers`` caps it (default
    ``os.cpu_count()``), e.g. when nesting under an outer parallel map.
    """
    validate_precision(precision)
    masks_zyx, pixels_zyx, unwrap = to_bzyx(masks, pixels)
    is_sequence = isinstance(masks, (list, tuple))
    batch = range(len(masks_zyx))
    ndims = [
        numpy.ndim(masks[i]) if is_sequence else min(numpy.ndim(masks), 3)
        for i in batch
    ]
    # The readback tables depend only on the mask, so a list that repeats one mask
    # object (the featurizer passes one mask per channel) builds them once.
    keys = [id(masks[i]) if is_sequence else i for i in batch]
    readbacks = {}
    for i in batch:
        if ndims[i] == 2 and keys[i] not in readbacks:
            readbacks[keys[i]] = _readback(masks_zyx[i][0], subsample_size)

    def one(i):
        return _granularity_one(
            masks_zyx[i],
            pixels_zyx[i],
            subsample_size,
            image_sample_size,
            element_size,
            granular_spectrum_length,
            precision=precision,
            spatial_ndim=ndims[i],
            readback=readbacks.get(keys[i]),
        )

    # Images are independent; map_images runs a batch on a thread pool (the
    # granularity kernels release the GIL) and a single image serially.
    return unwrap(map_images(one, batch, max_workers=max_workers))


def _granularity_one(
    mask_zyx,
    pixels_zyx,
    subsample_size,
    image_sample_size,
    element_size,
    ng,
    *,
    spatial_ndim,
    precision: Precision = "input",
    readback=None,
):
    """Preserve original spatial rank, including singleton-Z volumes (NumPy 3D)."""
    # Per image, before interpolation and background operations; no batch-wide
    # float64 copy and no changes to labels or dimensional dispatch.
    pixels_zyx = promote_float_pixels(pixels_zyx, precision)
    if spatial_ndim == 2:
        return _granularity_2d(
            mask_zyx[0],
            pixels_zyx[0],
            subsample_size,
            image_sample_size,
            element_size,
            ng,
            readback,
        )
    return _get_granularity_numpy(
        mask_zyx,
        pixels_zyx,
        subsample_size,
        image_sample_size,
        element_size,
        ng,
        precision=precision,
    )


def _granularity_2d(
    orig_mask: NDArray[numpy.integer],
    orig_pixels: NDArray[numpy.floating],
    subsample_size: float,
    image_sample_size: float,
    element_size: int,
    ng: int,
    readback=None,
) -> dict[str, NDArray[numpy.floating]]:
    orig_shape = numpy.array(orig_pixels.shape)
    effective_shape = (
        (orig_shape * subsample_size).astype(int) if subsample_size < 1 else orig_shape
    )
    if (effective_shape < 2).any():
        # Match the baseline's undefined-spectrum contract before resampling or
        # padding an empty/single-sample axis (issue #90).
        n_objects = int(orig_mask.max()) if orig_mask.size else 0
        return {
            f"Granularity_{i}": numpy.full(n_objects, numpy.nan)
            for i in range(1, ng + 1)
        }
    new_shape = orig_shape.copy()

    # 1. Subsample image (scipy map_coordinates, as baseline). The baseline also
    # resamples the mask here but never reads it (the spectrum loop uses orig_mask),
    # so we skip that dead order-0 resample.
    if subsample_size < 1:
        new_shape = (orig_shape * subsample_size).astype(int)
        i, j = (
            numpy.mgrid[0 : new_shape[0], 0 : new_shape[1]].astype(float)
            / subsample_size
        )
        pixels = scipy.ndimage.map_coordinates(orig_pixels, (i, j), order=1)
    else:
        pixels = orig_pixels.astype(numpy.float64, copy=True)

    # 2. Background subtraction via greyscale opening (numba disk erosion+dilation)
    if image_sample_size < 1:
        back_shape = new_shape * image_sample_size
        i, j = (
            numpy.mgrid[0 : back_shape[0], 0 : back_shape[1]].astype(float)
            / image_sample_size
        )
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

    # 3. Per-object readback (dense labels 1..max, like baseline)
    max_label = int(orig_mask.max()) if orig_mask.size else 0
    results: dict[str, NDArray[numpy.floating]] = {}
    if max_label == 0:  # no objects -> baseline returns empty arrays
        empty = numpy.zeros((0,))
        for granularity_id in range(1, ng + 1):
            results[f"Granularity_{granularity_id}"] = empty
        return results

    if readback is None:
        readback = _readback(orig_mask, subsample_size)
    mask, counts, tables = readback
    # exact widening (any input dtype, incl. float16) for the starting means
    pixels_f64 = numpy.ascontiguousarray(orig_pixels, dtype=numpy.float64)
    sums = label_sums(mask, pixels_f64, max_label)

    def _label_mean(label_sums):
        with numpy.errstate(invalid="ignore", divide="ignore"):
            # 0/0 -> nan for absent labels, matching scipy.ndimage.mean
            return label_sums[1:] / counts

    current_mean = _label_mean(sums)
    start_mean = numpy.maximum(current_mean, numpy.finfo(float).eps)

    # 4. Granular spectrum loop (numba disk(1) erosion + reconstruction, cascaded mask)
    ero = pixels
    recon_mask = pixels
    for granularity_id in range(1, ng + 1):
        ero = erosion_4conn_2d(ero)
        # reconstruction of an all-zero seed returns zeros cheaply (no FIFO seeds),
        # so no all-zero early-out is needed (it would only add a full ero.max() scan)
        rec = reconstruction_by_dilation_2d(ero, recon_mask)
        recon_mask = rec  # cascade: rec_g <= rec_{g-1} <= pixels (exact)

        new_mean = _label_mean(readback_sums(rec, *tables))
        results[f"Granularity_{granularity_id}"] = (
            (current_mean - new_mean) * 100 / start_mean
        )
        current_mean = new_mean

    return results


def _readback(orig_mask, subsample_size):
    """Mask-only readback tables for the 2D spectrum: ``(mask, counts, tables)``.

    Object pixel ``(y, x)`` samples the working grid at ``(y * scale_y, x *
    scale_x)``, the baseline's order-1 upsample of the grid back to the original
    shape (see :func:`._granularity.readback_tables`). ``None`` when there is no
    spectrum to read back (undefined grid or no objects).
    """
    orig_shape = numpy.array(orig_mask.shape)
    new_shape = (
        (orig_shape * subsample_size).astype(int) if subsample_size < 1 else orig_shape
    )
    max_label = int(orig_mask.max()) if orig_mask.size else 0
    if (new_shape < 2).any() or max_label == 0:
        return None
    mask = numpy.ascontiguousarray(orig_mask, dtype=numpy.intp)
    counts, *tables = readback_tables(
        mask,
        max_label,
        float(new_shape[0] - 1) / float(orig_shape[0] - 1),
        float(new_shape[1] - 1) / float(orig_shape[1] - 1),
        int(new_shape[0]),
        int(new_shape[1]),
    )
    return mask, counts[1:], tables
