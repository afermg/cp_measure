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
(subsample / background) run in numba too, bit-exact with ``map_coordinates``
order 1 (:func:`._granularity.bilinear_resample_2d` and
:func:`._granularity.trilinear_resample_3d`), so each image or volume is three
GIL-free kernel calls.

Batch-shaped via the canonical ``(B, Z, Y, X)`` form (see
:func:`cp_measure.primitives.shapes.to_bzyx`): a single image is ``B == 1``. Each
2D image runs the numba 2D path, and a 3D volume (including singleton Z, as in
NumPy) the same scheme in 3D: ``ball`` footprints, 6-connectivity and trilinear
per-object means that follow the baseline's 3D upsample voxel by voxel
(:func:`_granularity_3d`).
"""

import math

import numpy
from numpy.typing import NDArray

from cp_measure._precision import Precision, promote_float_pixels, validate_precision
from cp_measure.core.numba._granularity import (
    bilinear_resample_2d,
    granularity_spectrum_2d,
    granularity_spectrum_3d,
    readback_tables,
    subtract_background_2d,
    subtract_background_3d,
    trilinear_resample_3d,
)
from cp_measure.primitives.parallel import map_images
from cp_measure.primitives.shapes import each_volume, memory_key, native


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
    # Views of the caller's images: no whole-batch copy (stacking a featurizer
    # batch copied every image and channel on the calling thread).
    volumes, unwrap = each_volume(masks, pixels)
    # A mixed-dtype batch is measured at its common dtype, as when it was stacked;
    # each image is cast on its own worker thread.
    dtype = numpy.result_type(*(image.dtype for _, image, _ in volumes))
    # The readback tables depend only on the mask, so items that view one mask (the
    # featurizer passes one mask per channel) share them; they are built on threads.
    keys = [memory_key(mask) for mask, _, _ in volumes]
    first = {key: i for i, key in reversed(list(enumerate(keys)))}

    def readback(i):
        mask, _, ndim = volumes[i]
        return _readback(mask[0] if ndim == 2 else mask, subsample_size)

    pixel_count = sum(mask.size for mask, _, _ in volumes)
    built = map_images(
        readback,
        list(first.values()),
        max_workers=max_workers,
        pixels=pixel_count * len(first) // len(volumes),
    )
    readbacks = dict(zip(first, built))

    def one(i):
        mask_zyx, pixels_zyx, ndim = volumes[i]
        return _granularity_one(
            mask_zyx,
            pixels_zyx.astype(dtype, copy=False),
            subsample_size,
            image_sample_size,
            element_size,
            granular_spectrum_length,
            precision=precision,
            spatial_ndim=ndim,
            readback=readbacks[keys[i]],
        )

    # Images are independent; map_images runs a batch on a thread pool (the
    # granularity kernels release the GIL) and a single image serially.
    return unwrap(
        map_images(
            one, range(len(volumes)), max_workers=max_workers, pixels=pixel_count
        )
    )


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
    # Shapes as Python numbers: under a batch, every small NumPy call on the
    # calling thread waits for the GIL.
    orig_shape = orig_pixels.shape
    new_shape = (
        tuple(int(n * subsample_size) for n in orig_shape)
        if subsample_size < 1
        else orig_shape
    )
    if min(new_shape) < 2:
        # Match the baseline's undefined-spectrum contract before resampling or
        # padding an empty/single-sample axis (issue #90).
        n_objects = int(orig_mask.max()) if orig_mask.size else 0
        return {
            f"Granularity_{i}": numpy.full(n_objects, numpy.nan)
            for i in range(1, ng + 1)
        }
    resample = subsample_size < 1 or image_sample_size < 1
    if resample and orig_pixels.dtype not in (numpy.float32, numpy.float64):
        # scipy's map_coordinates error, which the baseline raises here (float16)
        raise RuntimeError("data type not supported")
    if readback is None:
        readback = _readback(orig_mask, subsample_size)
    if readback is None:  # no objects -> baseline returns empty arrays
        return {f"Granularity_{g}": numpy.zeros((0,)) for g in range(1, ng + 1)}

    # 1. Subsample image (bit-exact with the baseline's order-1 map_coordinates).
    # The baseline also resamples the mask here but never reads it (the spectrum
    # loop uses orig_mask), so we skip that dead order-0 resample.
    if subsample_size < 1:
        pixels = bilinear_resample_2d(
            orig_pixels, *new_shape, subsample_size, subsample_size, True
        )
    else:
        pixels = orig_pixels

    # 2. Background subtraction via greyscale disk opening, on the baseline's
    # background grid: ``mgrid`` over the float shape ``new_shape *
    # image_sample_size`` has ``ceil`` samples per axis, and the way back scales by
    # that float shape. The whole step stays in the input dtype, as the NumPy
    # opening does: promoting it changes float32 rounding (#123). The baseline's
    # background mask is all ones, so its masked re-zeroing is a no-op.
    back_shape = [n * image_sample_size for n in new_shape]
    pixels = subtract_background_2d(
        pixels,
        element_size,
        image_sample_size,
        *(math.ceil(b) for b in back_shape),
        *(float(b - 1) / float(n - 1) for b, n in zip(back_shape, new_shape)),
    )

    # 3. Spectrum: disk(1) erosion + cascaded reconstruction and the per-object
    # readback (dense labels 1..max, like the baseline), all in one kernel call
    mask, counts, tables = readback
    spectrum = granularity_spectrum_2d(pixels, ng, mask, orig_pixels, counts, *tables)
    return {f"Granularity_{g}": spectrum[g - 1] for g in range(1, ng + 1)}


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
    orig_shape = orig_pixels.shape
    new_shape = (
        tuple(int(n * subsample_size) for n in orig_shape)
        if subsample_size < 1
        else orig_shape
    )
    if min(new_shape) < 2:  # undefined spectrum, as the baseline (issue #90)
        n_objects = int(orig_mask.max()) if orig_mask.size else 0
        return {
            f"Granularity_{i}": numpy.full(n_objects, numpy.nan)
            for i in range(1, ng + 1)
        }
    resample = subsample_size < 1 or image_sample_size < 1
    if resample and orig_pixels.dtype not in (numpy.float32, numpy.float64):
        # scipy's map_coordinates error, which the baseline raises here (float16)
        raise RuntimeError("data type not supported")
    if readback is None:
        readback = _readback(orig_mask, subsample_size)
    if readback is None:  # no objects -> baseline returns empty arrays
        return {f"Granularity_{g}": numpy.zeros((0,)) for g in range(1, ng + 1)}

    # 1. Subsample (the baseline's mask resample is never read, so it is skipped)
    if subsample_size < 1:
        pixels = trilinear_resample_3d(
            orig_pixels, *new_shape, *(subsample_size,) * 3, True
        )
    else:
        pixels = orig_pixels

    # 2. Background subtraction via greyscale ball opening, in the input dtype as
    # the NumPy opening (#123). The way back scales by the float background shape
    # ``new_shape * image_sample_size``.
    pixels = subtract_background_3d(
        pixels,
        element_size,
        image_sample_size < 1,
        subsample_size,
        *((n * image_sample_size - 1) / (n - 1) for n in new_shape),
    )

    # 3. Spectrum (ball(1) erosion + 6-connectivity reconstruction, cascaded) and
    # per-object readback (dense labels 1..max, like baseline), in one kernel call
    mask, counts, scales = readback
    # The baseline's upsample keeps the reconstruction's dtype (that of ``pixels``),
    # so a float32 volume's samples are rounded to float32 before the mean.
    spectrum = granularity_spectrum_3d(
        pixels, ng, mask, orig_pixels, counts, scales, pixels.dtype == numpy.float32
    )
    return {f"Granularity_{g}": spectrum[g - 1] for g in range(1, ng + 1)}


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
