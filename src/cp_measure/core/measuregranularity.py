"""
MeasureGranularity
==================
**MeasureGranularity** outputs spectra of size measurements of the
textures in the image.

Image granularity is a texture measurement that tries to fit a series of
structure elements of increasing size into the texture of the image and outputs a spectrum of measures
based on how well they fit.
Granularity is measured as described by Ilya Ravkin (references below).

Basically, MeasureGranularity:
1 - Downsamples the image (if you tell it to). This is set in
**Subsampling factor for granularity measurements** or **Subsampling factor for background reduction**.
2 - Background subtracts anything larger than the radius in pixels set in
**Radius of structuring element.**
3 - For as many times as you set in **Range of the granular spectrum**, it gets rid of bright areas
that are only 1 pixel across, reports how much signal was lost by doing that, then repeats.
i.e. The first time it removes one pixel from all bright areas in the image,
(effectively deleting those that are only 1 pixel in size) and then reports what % of the signal was lost.
It then takes the first-iteration image and repeats the removal and reporting (effectively reporting
the amount of signal that is two pixels in size). etc.

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Granularity:* The module returns one measurement for each instance
   of the granularity spectrum set in **Range of the granular spectrum**.

References
^^^^^^^^^^

-  Serra J. (1989) *Image Analysis and Mathematical Morphology*, Vol. 1.
   Academic Press, London
-  Maragos P. “Pattern spectrum and multiscale shape representation”,
   *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11,
   N 7, pp. 701-716, 1989
-  Vincent L. (2000) “Granulometries and Opening Trees”, *Fundamenta
   Informaticae*, 41, No. 1-2, pp. 57-90, IOS Press, 2000.
-  Vincent L. (1992) “Morphological Area Opening and Closing for
   Grayscale Images”, *Proc. NATO Shape in Picture Workshop*,
   Driebergen, The Netherlands, pp. 197-208.
-  Ravkin I, Temov V. (1988) “Bit representation techniques and image
   processing”, *Applied Informatics*, v.14, pp. 41-90, Finances and
   Statistics, Moskow, (in Russian)
"""

from typing import Callable

import numpy
import scipy.ndimage
import scipy.sparse
import skimage.morphology
from cp_measure.utils import _ensure_np_array as fix
from numpy.typing import NDArray


def _make_fused_upsample_mean(
    orig_shape, new_shape, orig_mask
) -> Callable[[NDArray[numpy.floating]], NDArray[numpy.floating]]:
    """Build a callable ``rec -> per-object mean of rec restored to the original scale``.

    Each granular-spectrum step restores the downsampled reconstruction ``rec`` to the original
    scale with a bilinear ``scipy.ndimage.map_coordinates`` (order=1) and then takes the
    per-object mean with ``scipy.ndimage.mean``. Both depend only on ``rec`` (the sampling
    geometry and the labels are fixed across iterations), and both are linear in ``rec``: every
    original pixel is a fixed convex combination of four downsampled pixels, and the per-object
    mean sums those contributions. So the whole "upsample then average per object" is one sparse
    ``(n_labels x n_downsampled)`` operator, built once here and applied as a single mat-vec per
    step instead of a full-resolution interpolation + label reduction every iteration.

    The result matches the direct ``map_coordinates`` + ``ndimage.mean`` path to floating-point
    round-off (~1e-12, sparse-accumulation order), including objects on the last row/column where
    the source coordinate floats just outside the grid. 2D only; the rarely used 3D path keeps the
    direct interpolation.
    """
    # Fractional source coordinates of every original pixel (same grid map_coordinates uses).
    i, j = numpy.mgrid[0 : orig_shape[0], 0 : orig_shape[1]].astype(float)
    i *= float(new_shape[0] - 1) / float(orig_shape[0] - 1)
    j *= float(new_shape[1] - 1) / float(orig_shape[1] - 1)
    # Drop background up front: those pixels carry no object and would be filtered out anyway,
    # so restrict the four-neighbour expansion to foreground pixels only.
    labels = orig_mask.ravel()
    foreground = labels > 0
    fg_labels = labels[foreground]
    r, c = i.ravel()[foreground], j.ravel()[foreground]
    # scipy.ndimage.map_coordinates(mode='constant', cval=0) returns 0 for any source coordinate
    # outside [0, new-1] — the float-rounded scale can push the last row/col just past new-1
    # (e.g. 63.0000000000001 for orig 160 -> new 64). Reproduce that by keeping only in-bounds
    # pixels in the operator; `counts` below still spans ALL foreground, so a dropped pixel
    # contributes 0 to the numerator but 1 to the denominator, exactly as the direct path does.
    in_bounds = (
        (r >= 0)
        & (r <= new_shape[0] - 1)
        & (c >= 0)
        & (c <= new_shape[1] - 1)
    )
    r, c, op_labels = r[in_bounds], c[in_bounds], fg_labels[in_bounds]
    r0, c0 = numpy.floor(r).astype(int), numpy.floor(c).astype(int)
    fr, fc = r - r0, c - c0
    r1 = numpy.minimum(r0 + 1, new_shape[0] - 1)
    c1 = numpy.minimum(c0 + 1, new_shape[1] - 1)
    ncols = int(new_shape[1])
    neighbours = numpy.concatenate(
        [r0 * ncols + c0, r0 * ncols + c1, r1 * ncols + c0, r1 * ncols + c1]
    )
    weights = numpy.concatenate(
        [(1 - fr) * (1 - fc), (1 - fr) * fc, fr * (1 - fc), fr * fc]
    )
    rows = numpy.tile(op_labels, 4)
    max_label = int(orig_mask.max())
    objects = slice(
        1, max_label + 1
    )  # operator/count rows for real labels; row 0 is background
    # Row L accumulates sum_{pixel in L} weight; dividing the mat-vec by the pixel count gives
    # the per-object mean.
    operator = scipy.sparse.coo_matrix(
        (weights, (rows, neighbours)),
        shape=(max_label + 1, int(new_shape[0]) * ncols),
    ).tocsr()
    counts = numpy.bincount(labels, minlength=max_label + 1)[objects].astype(float)

    def fused(rec):
        # counts is 0 for labels absent from the mask, yielding NaN exactly as ndimage.mean does.
        with numpy.errstate(invalid="ignore"):
            return (operator @ rec.ravel())[objects] / counts

    return fused


def get_granularity(
    mask: NDArray[numpy.integer],
    pixels: NDArray[numpy.floating],
    subsample_size: float = 0.25,
    image_sample_size: float = 0.25,
    element_size: int = 10,
    granular_spectrum_length: int = 16,
) -> dict[str, NDArray[numpy.floating]]:
    """
    1. (Outcommented) Subsample image
    2.  Remove background pixels using a greyscale tophat filter
    3.  Calculate granular spectrum (size distribution) for all masks

    Parameters
    ----------
    subsample_size : float, optional
        Subsampling factor for granularity measurements.
        If the textures of interest are larger than a few pixels, we recommend
        you subsample the image with a factor <1 to speed up the processing.
        Downsampling the image will let you detect larger structures with a
        smaller sized structure element. A factor >1 might increase the accuracy
        but also require more processing time. Images are typically of higher
        resolution than is required for granularity measurements, so the default
        value is 0.25. For low-resolution images, increase the subsampling
        fraction; for high-resolution images, decrease the subsampling fraction.
        Subsampling by 1/4 reduces computation time by (1/4) :sup:`3` because the
        size of the image is (1/4) :sup:`2` of original and the range of granular
        spectrum can be 1/4 of original. Moreover, the results are sometimes
        actually a little better with subsampling, which is probably because
        with subsampling the individual granular spectrum components can be used
        as features, whereas without subsampling a feature should be a sum of
        several adjacent granular spectrum components. The recommendation on the
        numerical value cannot be determined in advance; an analysis as in this
        reference may be required before running the whole set. See this `pdf`_,
        slides 27-31, 49-50.

        .. _pdf:     http://www.ravkin.net/presentations/Statistical%20properties%20of%20algorithms%20for%20analysis%20of%20cell%20images.pdf"

    image_sample_size : float, optional
        Subsampling factor for background reduction.
        It is important to remove low frequency image background variations as
        they will affect the final granularity measurement. Any method can be
        used as a pre-processing step prior to this module; we have chosen to
        simply subtract a highly open image. To do it quickly, we subsample the
        image first. The subsampling factor for background reduction is usually
        [0.125 – 0.25]. This is highly empirical, but a small factor should be
        used if the structures of interest are large. The significance of
        background removal in the context of granulometry is that image volume
        at certain granular size is normalized by total image volume, which
        depends on how the background was removed.

    element_size : int, optional
        Radius of structuring element.
        This radius should correspond to the radius of the textures of interest
        *after* subsampling; i.e., if textures in the original image scale have
        a radius of 40 pixels, and a subsampling factor of 0.25 is used, the
        structuring element size should be 10 or slightly smaller, and the range
        of the spectrum defined below will cover more sizes.

    granular_spectrum_length : int, optional
        Range of the granular spectrum.
        You may need a trial run to see which granular
        spectrum range yields informative measurements. Start by using a wide
        spectrum and narrow it down to the informative range to save time.

    Returns
    -------
    Dictionary of 1-d arrays where each value contains the specific granularity feature for a given cells.

    Examples

    """
    #
    # Downsample the image and mask
    #
    orig_shape = numpy.array(pixels.shape)
    orig_pixels = pixels  # original, non-background-subtracted, used for start_mean
    orig_mask = mask
    new_shape = orig_shape.copy()
    if subsample_size < 1:
        new_shape = (orig_shape * subsample_size).astype(int)
        if pixels.ndim == 2:
            i, j = (
                numpy.mgrid[0 : new_shape[0], 0 : new_shape[1]].astype(float)
                / subsample_size
            )
            pixels = scipy.ndimage.map_coordinates(pixels, (i, j), order=1)
            mask = scipy.ndimage.map_coordinates(mask, (i, j), order=0).astype(
                orig_mask.dtype
            )
        else:
            k, i, j = (
                numpy.mgrid[
                    0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]
                ].astype(float)
                / subsample_size
            )
            pixels = scipy.ndimage.map_coordinates(pixels, (k, i, j), order=1)
            mask = scipy.ndimage.map_coordinates(mask, (k, i, j), order=0).astype(
                orig_mask.dtype
            )
    else:
        pixels = pixels.copy()
        mask = mask.copy()
    #
    # Remove background pixels using a greyscale tophat filter
    #
    if image_sample_size < 1:
        back_shape = new_shape * image_sample_size
        if pixels.ndim == 2:
            i, j = (
                numpy.mgrid[0 : back_shape[0], 0 : back_shape[1]].astype(float)
                / image_sample_size
            )
            back_pixels = scipy.ndimage.map_coordinates(pixels, (i, j), order=1)
            # Use all pixels for background estimation (matches CellProfiler which
            # uses a binary image-validity mask, True everywhere for full-frame images).
            # Using object labels here would restrict background to within-object
            # regions only, causing incomplete background subtraction.
            back_mask = numpy.ones(back_pixels.shape, dtype=bool)
        else:
            k, i, j = (
                numpy.mgrid[
                    0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]
                ].astype(float)
                / subsample_size
            )
            back_pixels = scipy.ndimage.map_coordinates(pixels, (k, i, j), order=1)
            back_mask = numpy.ones(back_pixels.shape, dtype=bool)
    else:
        back_pixels = pixels
        back_mask = numpy.ones(back_pixels.shape, dtype=bool)
        back_shape = new_shape
    radius = element_size
    if pixels.ndim == 2:
        footprint = skimage.morphology.disk(radius, dtype=bool)
    else:
        footprint = skimage.morphology.ball(radius, dtype=bool)
    back_pixels_mask = numpy.zeros_like(back_pixels)
    back_pixels_mask[back_mask == 1] = back_pixels[back_mask == 1]
    back_pixels = skimage.morphology.erosion(back_pixels_mask, footprint=footprint)
    back_pixels_mask = numpy.zeros_like(back_pixels)
    back_pixels_mask[back_mask == 1] = back_pixels[back_mask == 1]
    back_pixels = skimage.morphology.dilation(back_pixels_mask, footprint=footprint)
    if image_sample_size < 1:
        if pixels.ndim == 2:
            i, j = numpy.mgrid[0 : new_shape[0], 0 : new_shape[1]].astype(float)
            #
            # Make sure the mapping only references the index range of
            # back_pixels.
            #
            i *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
            j *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
            back_pixels = scipy.ndimage.map_coordinates(back_pixels, (i, j), order=1)
        else:
            k, i, j = numpy.mgrid[
                0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]
            ].astype(float)
            k *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
            i *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
            j *= float(back_shape[2] - 1) / float(new_shape[2] - 1)
            back_pixels = scipy.ndimage.map_coordinates(back_pixels, (k, i, j), order=1)
    pixels -= back_pixels
    pixels[pixels < 0] = 0

    # Transcribed from the Matlab module: granspectr function
    #
    # CALCULATES GRANULAR SPECTRUM, ALSO KNOWN AS SIZE DISTRIBUTION,
    # GRANULOMETRY, AND PATTERN SPECTRUM, SEE REF.:
    # J.Serra, Image Analysis and Mathematical Morphology, Vol. 1. Academic Press, London, 1989
    # Maragos,P. "Pattern spectrum and multiscale shape representation", IEEE Transactions on Pattern Analysis and Machine Intelligence, 11, N 7, pp. 701-716, 1989
    # L.Vincent "Granulometries and Opening Trees", Fundamenta Informaticae, 41, No. 1-2, pp. 57-90, IOS Press, 2000.
    # L.Vincent "Morphological Area Opening and Closing for Grayscale Images", Proc. NATO Shape in Picture Workshop, Driebergen, The Netherlands, pp. 197-208, 1992.
    # I.Ravkin, V.Temov "Bit representation techniques and image processing", Applied Informatics, v.14, pp. 41-90, Finances and Statistics, Moskow, 1988 (in Russian)
    # THIS IMPLEMENTATION INSTEAD OF OPENING USES EROSION FOLLOWED BY RECONSTRUCTION
    #
    ng = granular_spectrum_length

    if pixels.ndim == 2:
        footprint = skimage.morphology.disk(1, dtype=bool)
    else:
        footprint = skimage.morphology.ball(1, dtype=bool)

    ero = pixels.copy()

    # Per-object stats use original-scale labels so the cell boundaries are exact.
    # start_mean uses the raw (non-background-subtracted) original pixels, matching
    # CellProfiler's ObjectRecord initialisation which also uses im_pixel_data directly.
    unique_labels = numpy.unique(orig_mask)
    unique_labels = unique_labels[unique_labels > 0]
    range_ = numpy.arange(1, numpy.max(orig_mask) + 1)

    current_mean = fix(scipy.ndimage.mean(orig_pixels, orig_mask, range_))
    start_mean = numpy.maximum(current_mean, numpy.finfo(float).eps)

    # The per-step "restore to original scale + per-object mean" is linear in the
    # reconstruction and uses fixed geometry, so precompute it once as a sparse operator
    # (2D only; the 3D path keeps the direct interpolation below).
    fused_mean = None
    if pixels.ndim == 2 and unique_labels.any():
        fused_mean = _make_fused_upsample_mean(orig_shape, new_shape, orig_mask)

    results: dict[str, NDArray[numpy.floating]] = {}
    for granularity_id in range(1, ng + 1):
        ero_mask = ero.copy()
        # Shrink bright regions
        ero = skimage.morphology.erosion(ero_mask, footprint=footprint)
        # Reconstruct: undo erosion for pixels that were already small
        rec = skimage.morphology.reconstruction(ero, pixels, footprint=footprint)

        # Calculate the means for the objects
        gss = numpy.zeros((0,))
        if unique_labels.any():
            if fused_mean is not None:
                new_mean = fused_mean(rec)
            else:
                # 3D: restore reconstructed image to original scale, then per-object means.
                k, i, j = numpy.mgrid[
                    0 : orig_shape[0], 0 : orig_shape[1], 0 : orig_shape[2]
                ].astype(float)
                k *= float(new_shape[0] - 1) / float(orig_shape[0] - 1)
                i *= float(new_shape[1] - 1) / float(orig_shape[1] - 1)
                j *= float(new_shape[2] - 1) / float(orig_shape[2] - 1)
                rec_orig = scipy.ndimage.map_coordinates(rec, (k, i, j), order=1)
                new_mean = fix(scipy.ndimage.mean(rec_orig, orig_mask, range_))
            gss = (current_mean - new_mean) * 100 / start_mean
            current_mean = new_mean  # update running mean for next iteration

        results[f"Granularity_{granularity_id}"] = gss

    return results
