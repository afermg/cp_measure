"""Numba-accelerated granularity backend (2D and 3D).

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
2D image runs the numba 2D path, and a 3D volume (including singleton Z, as in
NumPy) the same scheme in 3D: ``ball`` footprints, 6-connectivity and trilinear
per-object means that follow the baseline's 3D upsample voxel by voxel
(:func:`_granularity_3d`).
"""

import numpy
import scipy.ndimage
from numpy.typing import NDArray

from cp_measure._precision import Precision, promote_float_pixels, validate_precision
from cp_measure.core.numba._granularity import (
    ball_dilation_3d,
    ball_erosion_3d,
    disk_dilation_2d,
    disk_erosion_2d,
    erosion_6conn_3d,
    label_sums,
    readback_tables,
    reconstruction_by_dilation_3d,
    spectrum_sums_2d,
    upsample_label_sums_3d,
)
from cp_measure.primitives.parallel import map_images
from cp_measure.primitives.shapes import memory_key, native, to_bzyx


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
    is_sequence = isinstance(masks, (list, tuple))
    # The kernels index the pixels with the mask's geometry: reject mismatches.
    pairs = zip(masks, pixels) if is_sequence else [(masks, pixels)]
    if any(numpy.shape(m) != numpy.shape(p) for m, p in pairs):
        raise ValueError("labels and pixels must have matching shapes and rank")
    masks_zyx, pixels_zyx, unwrap = to_bzyx(masks, pixels)
    batch = range(len(masks_zyx))
    ndims = [
        numpy.ndim(masks[i]) if is_sequence else min(numpy.ndim(masks), 3)
        for i in batch
    ]
    # The readback tables depend only on the mask, so a list that repeats one mask
    # object (the featurizer passes one mask per channel) builds them once.
    # keyed on the caller's arrays: to_bzyx's stack gives every item its own copy
    keys = [memory_key(numpy.asarray(masks[i])) if is_sequence else i for i in batch]
    readbacks = {}
    for i in batch:
        if keys[i] not in readbacks:
            mask = masks_zyx[i][0] if ndims[i] == 2 else masks_zyx[i]
            readbacks[keys[i]] = _readback(mask, subsample_size)

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
    pixels_zyx = promote_float_pixels(native(pixels_zyx), precision)
    # Integer images are measured as their exact float64 values: the float-dtype
    # handling below (#123) would otherwise subtract the background in the integer
    # dtype, which wraps around (or rounds) where it exceeds a pixel.
    if pixels_zyx.dtype.kind in "biu":
        pixels_zyx = pixels_zyx.astype(numpy.float64)
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
    return _granularity_3d(
        mask_zyx,
        pixels_zyx,
        subsample_size,
        image_sample_size,
        element_size,
        ng,
        readback,
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
        pixels = orig_pixels.copy()

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
    background_dtype = back_pixels.dtype
    back_pixels = disk_erosion_2d(back_pixels, radius)
    back_pixels = disk_dilation_2d(back_pixels, radius)
    # The kernels select extrema in float64, but the NumPy opening preserves
    # its input dtype. Restore it before interpolation and subtraction: promoting
    # these stages changes float32 rounding and the resulting spectrum (#123).
    back_pixels = back_pixels.astype(background_dtype, copy=False)
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

    # 4. Granular spectrum: disk(1) erosion + cascaded reconstruction, all steps in
    # one kernel call
    step_sums = spectrum_sums_2d(pixels, ng, *tables)
    for granularity_id in range(1, ng + 1):
        new_mean = _label_mean(step_sums[granularity_id - 1])
        results[f"Granularity_{granularity_id}"] = (
            (current_mean - new_mean) * 100 / start_mean
        )
        current_mean = new_mean

    return results


def _granularity_3d(
    orig_mask: NDArray[numpy.integer],
    orig_pixels: NDArray[numpy.floating],
    subsample_size: float,
    image_sample_size: float,
    element_size: int,
    ng: int,
    readback=None,
) -> dict[str, NDArray[numpy.floating]]:
    """:func:`_granularity_2d` for a ``(Z, Y, X)`` volume, following the baseline's
    3D path step by step. That includes its background resample, which reuses the
    subsample grid (``new_shape`` and ``subsample_size``) instead of the background
    grid, as CellProfiler does; only a corner of it is read back."""
    orig_shape = numpy.array(orig_pixels.shape)
    effective_shape = (
        (orig_shape * subsample_size).astype(int) if subsample_size < 1 else orig_shape
    )
    if (effective_shape < 2).any():  # undefined spectrum, as the baseline (issue #90)
        n_objects = int(orig_mask.max()) if orig_mask.size else 0
        return {
            f"Granularity_{i}": numpy.full(n_objects, numpy.nan)
            for i in range(1, ng + 1)
        }
    new_shape = orig_shape.copy()

    # 1. Subsample (the baseline's mask resample is never read, so it is skipped)
    if subsample_size < 1:
        new_shape = (orig_shape * subsample_size).astype(int)
        k, i, j = (
            numpy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(
                float
            )
            / subsample_size
        )
        pixels = scipy.ndimage.map_coordinates(orig_pixels, (k, i, j), order=1)
    else:
        pixels = orig_pixels.copy()

    # 2. Background subtraction via greyscale ball opening
    if image_sample_size < 1:
        back_shape = new_shape * image_sample_size
        k, i, j = (
            numpy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(
                float
            )
            / subsample_size
        )
        back_pixels = scipy.ndimage.map_coordinates(pixels, (k, i, j), order=1)
    else:
        back_pixels = pixels
        back_shape = new_shape
    # As in 2D: the kernels select extrema in float64; the NumPy opening keeps the
    # input dtype, so restore it before interpolation and subtraction (#123).
    background_dtype = back_pixels.dtype
    back_pixels = ball_erosion_3d(back_pixels, element_size)
    back_pixels = ball_dilation_3d(back_pixels, element_size)
    back_pixels = back_pixels.astype(background_dtype, copy=False)
    if image_sample_size < 1:
        k, i, j = numpy.mgrid[
            0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]
        ].astype(float)
        k *= (back_shape[0] - 1) / (new_shape[0] - 1)
        i *= (back_shape[1] - 1) / (new_shape[1] - 1)
        j *= (back_shape[2] - 1) / (new_shape[2] - 1)
        back_pixels = scipy.ndimage.map_coordinates(back_pixels, (k, i, j), order=1)
    pixels -= back_pixels
    pixels[pixels < 0] = 0

    # 3. Per-object readback (dense labels 1..max, like baseline)
    max_label = int(orig_mask.max()) if orig_mask.size else 0
    if max_label == 0:
        return {f"Granularity_{g}": numpy.zeros((0,)) for g in range(1, ng + 1)}
    if readback is None:
        readback = _readback(orig_mask, subsample_size)
    mask, counts, scales = readback
    # The baseline's upsample keeps the reconstruction's dtype (that of ``pixels``),
    # so a float32 volume's samples are rounded to float32 before the mean.
    round_float32 = pixels.dtype == numpy.float32
    pixels_f64 = numpy.ascontiguousarray(orig_pixels, dtype=numpy.float64)
    # label_sums reads in raster order, so a (Z * Y, X) view sums the volume.
    sums = label_sums(
        mask.reshape(-1, mask.shape[-1]),
        pixels_f64.reshape(-1, pixels_f64.shape[-1]),
        max_label,
    )
    with numpy.errstate(invalid="ignore", divide="ignore"):
        current_mean = sums[1:] / counts  # 0/0 -> nan for absent labels
    start_mean = numpy.maximum(current_mean, numpy.finfo(float).eps)

    # 4. Granular spectrum (ball(1) erosion + 6-connectivity reconstruction, cascaded)
    results: dict[str, NDArray[numpy.floating]] = {}
    ero = pixels
    recon_mask = pixels
    for granularity_id in range(1, ng + 1):
        ero = erosion_6conn_3d(ero)
        rec = reconstruction_by_dilation_3d(ero, recon_mask)
        recon_mask = rec
        with numpy.errstate(invalid="ignore", divide="ignore"):
            new_mean = (
                upsample_label_sums_3d(rec, mask, max_label, scales, round_float32)[1:]
                / counts
            )
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
    scales = tuple(float(n - 1) / float(o - 1) for n, o in zip(new_shape, orig_shape))
    if mask.ndim == 3:  # 3D reads every voxel per step: counts and grid scales only
        counts = numpy.bincount(mask.ravel(), minlength=max_label + 1).astype(float)
        return mask, counts[1:], scales
    counts, *tables = readback_tables(
        mask, max_label, *scales, *(int(n) for n in new_shape)
    )
    return mask, counts[1:], tables
