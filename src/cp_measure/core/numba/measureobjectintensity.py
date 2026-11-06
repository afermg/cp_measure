"""Numba-backed MeasureObjectIntensity.

Drop-in for :func:`cp_measure.core.measureobjectintensity.get_intensity`,
producing the identical dict of features for 2D and 3D input. The per-label
reductions — including ``Location_MaxIntensity_*`` — run entirely as fused
single-pass numba kernels over a flat segment representation
(:mod:`cp_measure.primitives`). The max position uses a deterministic
``>=``-last rule that is bit-identical to ``scipy.ndimage.maximum_position`` on
real (tie-free) data; only exact-value ties can differ (scipy's tie pick is
quicksort-dependent and not stable across numpy versions).
"""

import numpy
import skimage.segmentation
from numpy.typing import NDArray

from cp_measure.core.measureobjectintensity import (
    C_LOCATION,
    INTEGRATED_INTENSITY,
    INTEGRATED_INTENSITY_EDGE,
    INTENSITY,
    LOC_CMI_X,
    LOC_CMI_Y,
    LOC_CMI_Z,
    LOC_MAX_X,
    LOC_MAX_Y,
    LOC_MAX_Z,
    LOWER_QUARTILE_INTENSITY,
    MAD_INTENSITY,
    MASS_DISPLACEMENT,
    MAX_INTENSITY,
    MAX_INTENSITY_EDGE,
    MEAN_INTENSITY,
    MEAN_INTENSITY_EDGE,
    MEDIAN_INTENSITY,
    MIN_INTENSITY,
    MIN_INTENSITY_EDGE,
    STD_INTENSITY,
    STD_INTENSITY_EDGE,
    UPPER_QUARTILE_INTENSITY,
)
from cp_measure.core.numba._intensity import intensity_2d_batch
from cp_measure.primitives.shapes import to_bzyx
from cp_measure.primitives._segment_numba import (
    flatten_numba,
    inner_boundary,
    segment_moments,
    segment_quantiles,
    segment_resid_sumsq,
    segment_stats,
)

# Column order of the prange kernel output, matching ``measurement_names`` below.
_BASE_KEYS = (
    f"{INTENSITY}_{INTEGRATED_INTENSITY}",
    f"{INTENSITY}_{MEAN_INTENSITY}",
    f"{INTENSITY}_{STD_INTENSITY}",
    f"{INTENSITY}_{MIN_INTENSITY}",
    f"{INTENSITY}_{MAX_INTENSITY}",
    f"{INTENSITY}_{MASS_DISPLACEMENT}",
    f"{INTENSITY}_{LOWER_QUARTILE_INTENSITY}",
    f"{INTENSITY}_{MEDIAN_INTENSITY}",
    f"{INTENSITY}_{MAD_INTENSITY}",
    f"{INTENSITY}_{UPPER_QUARTILE_INTENSITY}",
    f"{C_LOCATION}_{LOC_CMI_X}",
    f"{C_LOCATION}_{LOC_CMI_Y}",
    f"{C_LOCATION}_{LOC_CMI_Z}",
    f"{C_LOCATION}_{LOC_MAX_X}",
    f"{C_LOCATION}_{LOC_MAX_Y}",
    f"{C_LOCATION}_{LOC_MAX_Z}",
)
_EDGE_KEYS = (
    f"{INTENSITY}_{INTEGRATED_INTENSITY_EDGE}",
    f"{INTENSITY}_{MEAN_INTENSITY_EDGE}",
    f"{INTENSITY}_{STD_INTENSITY_EDGE}",
    f"{INTENSITY}_{MIN_INTENSITY_EDGE}",
    f"{INTENSITY}_{MAX_INTENSITY_EDGE}",
)


def _assemble_intensity(out, b, nobj, edge):
    """One per-image dict from the kernel output, with independent arrays."""
    keys = _BASE_KEYS + _EDGE_KEYS if edge else _BASE_KEYS
    return {k: out[b, f, :nobj].copy() for f, k in enumerate(keys)}


def get_intensity(
    masks,
    pixels,
    edge_measurements: bool = True,
    legacy: bool = False,
):
    """Per-object intensity features for 2D or 3D images, individually or in batches.

    Returns a feature dict for a single image, or a list of dicts for a batch
    (4D ``(B, Z, Y, X)`` array or a list/tuple of images). See
    :func:`cp_measure.primitives.shapes.to_bzyx` for the accepted shapes.
    Labels must be contiguous ``1..N`` and match the pixels in shape and rank.
    ``legacy=True`` retains the original quartiles and rank-dependent MAD;
    a singleton-Z image uses the 3D convention.
    """
    masks_is_seq = isinstance(masks, (list, tuple))
    pixels_is_seq = isinstance(pixels, (list, tuple))
    if masks_is_seq != pixels_is_seq:
        raise ValueError("masks and pixels must both be sequences, or both arrays")
    if pixels_is_seq:
        if len(masks) != len(pixels):
            raise ValueError("masks and pixels must have matching batch sizes")
        if any(numpy.shape(m) != numpy.shape(p) for m, p in zip(masks, pixels)):
            raise ValueError("labels and pixels must have matching shapes and rank")
    elif numpy.shape(masks) != numpy.shape(pixels):
        raise ValueError("labels and pixels must have matching shapes and rank")

    is_batch = pixels_is_seq or numpy.ndim(pixels) == 4
    if is_batch:
        if len(pixels) == 0:
            raise ValueError("intensity batches must not be empty")
        if pixels_is_seq and any(
            numpy.shape(p) != numpy.shape(pixels[0]) for p in pixels
        ):
            raise ValueError("all images in a batch must have the same shape")
        pairs = zip(masks, pixels)
    else:
        pairs = zip([masks], [pixels])

    # Fast path: a list/tuple batch of 2D images runs the parallel prange kernel.
    # A 4D (B, 1, Y, X) array is a batch of singleton-Z volumes (spatial rank 3),
    # so it stays serial; the 2D kernel's MAD fraction (1/2) would be wrong for it.
    # The validation above already enforced the per-item shape/rank and ragged
    # contract, so the stacked batch is well-formed.
    if masks_is_seq and all(numpy.ndim(m) == 2 for m in masks):
        masks_b = numpy.stack([numpy.asarray(m)[numpy.newaxis] for m in masks])
        pixels_b = numpy.stack([numpy.asarray(p)[numpy.newaxis] for p in pixels])
        out, nobjs = intensity_2d_batch(masks_b, pixels_b, legacy, edge_measurements)
        return [
            _assemble_intensity(out, b, int(nobjs[b]), edge_measurements)
            for b in range(len(masks))
        ]

    results = []
    for mask, image in pairs:
        spatial_ndim = numpy.ndim(image)
        if spatial_ndim not in (2, 3):
            raise ValueError("expected a 2D or 3D image per batch item")
        # Normalize one image at a time: these are views, not whole-batch copies.
        # The per-volume kernel is serial, so materializing a stack buys nothing.
        masks_zyx, pixels_zyx, _ = to_bzyx(mask, image)
        results.append(
            _intensity_volume(
                masks_zyx[0],
                pixels_zyx[0],
                edge_measurements,
                legacy,
                spatial_ndim=spatial_ndim,
            )
        )
    return results if is_batch else results[0]


def _intensity_volume(
    masks: NDArray[numpy.integer],
    pixels: NDArray[numpy.floating],
    edge_measurements: bool = True,
    legacy: bool = False,
    *,
    spatial_ndim: int,
) -> dict[str, NDArray[numpy.floating]]:
    """Intensity features for one ``(Z, Y, X)`` volume (a 2D image is ``Z == 1``).

    ``masks`` is a labeled array where 0 is background and its shape matches
    ``pixels``. ``spatial_ndim`` preserves whether the caller supplied a
    2D image or a 3D volume after both are normalized to ``(Z, Y, X)``; in
    particular, a genuine singleton-Z volume retains the NumPy baseline's 1/3
    legacy MAD fraction rather than being treated as 2D.
    """
    nobjects = int(masks.max())

    integrated_intensity = numpy.zeros(nobjects)
    mean_intensity = numpy.zeros(nobjects)
    std_intensity = numpy.zeros(nobjects)
    min_intensity = numpy.zeros(nobjects)
    max_intensity = numpy.zeros(nobjects)
    mass_displacement = numpy.zeros(nobjects)
    lower_quartile_intensity = numpy.zeros(nobjects)
    median_intensity = numpy.zeros(nobjects)
    mad_intensity = numpy.zeros(nobjects)
    upper_quartile_intensity = numpy.zeros(nobjects)
    cmi_x = numpy.zeros(nobjects)
    cmi_y = numpy.zeros(nobjects)
    cmi_z = numpy.zeros(nobjects)
    max_x = numpy.zeros(nobjects)
    max_y = numpy.zeros(nobjects)
    max_z = numpy.zeros(nobjects)

    values, seg0, xc, yc, zc = flatten_numba(
        numpy.ascontiguousarray(masks),
        numpy.ascontiguousarray(pixels),
    )
    has_objects = values.size > 0

    if has_objects:
        (
            count,
            sumI,
            minI,
            maxI,
            max_x,
            max_y,
            max_z,
            sx,
            sy,
            sz,
            sxI,
            syI,
            szI,
        ) = segment_moments(values, seg0, xc, yc, zc, nobjects)
        cnt = count.astype(numpy.float64)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            integrated_intensity = sumI
            mean_intensity = sumI / cnt
            ss = segment_resid_sumsq(values, seg0, nobjects, mean_intensity)
            std_intensity = numpy.sqrt(ss / cnt)
            min_intensity = minI
            max_intensity = maxI

            cm_x = sx / cnt
            cm_y = sy / cnt
            cm_z = sz / cnt
            cmi_x = sxI / sumI
            cmi_y = syI / sumI
            cmi_z = szI / sumI
            mass_displacement = numpy.sqrt(
                (cm_x - cmi_x) ** 2 + (cm_y - cmi_y) ** 2 + (cm_z - cmi_z) ** 2
            )

        (
            lower_quartile_intensity,
            median_intensity,
            upper_quartile_intensity,
            mad_intensity,
        ) = segment_quantiles(values, seg0, count, nobjects, 1.0 / spatial_ndim, legacy)

    if edge_measurements:
        integrated_intensity_edge = numpy.zeros(nobjects)
        mean_intensity_edge = numpy.zeros(nobjects)
        std_intensity_edge = numpy.zeros(nobjects)
        min_intensity_edge = numpy.zeros(nobjects)
        max_intensity_edge = numpy.zeros(nobjects)

        # 2D plane (Z==1): numba inner-boundary kernel, bit-identical to skimage
        # mode="inner" but ~12-27x faster. True 3D keeps skimage (6-neighbourhood).
        if masks.shape[0] == 1:
            emask = inner_boundary(numpy.ascontiguousarray(masks[0]))[numpy.newaxis] > 0
        else:
            emask = skimage.segmentation.find_boundaries(masks, mode="inner") > 0
        e_values = pixels[emask].astype(numpy.float64)
        e_seg0 = masks[emask] - 1

        if e_values.size > 0:
            ecount, esum, emin, emax = segment_stats(e_values, e_seg0, nobjects)
            edge_obj = ecount > 0
            ecnt = ecount.astype(numpy.float64)
            with numpy.errstate(invalid="ignore", divide="ignore"):
                emean = esum / ecnt
                ess = segment_resid_sumsq(e_values, e_seg0, nobjects, emean)
                estd = numpy.sqrt(ess / ecnt)
            integrated_intensity_edge[edge_obj] = esum[edge_obj]
            mean_intensity_edge[edge_obj] = emean[edge_obj]
            std_intensity_edge[edge_obj] = estd[edge_obj]
            min_intensity_edge[edge_obj] = emin[edge_obj]
            max_intensity_edge[edge_obj] = emax[edge_obj]

    measurement_names = [
        (INTENSITY, INTEGRATED_INTENSITY, integrated_intensity),
        (INTENSITY, MEAN_INTENSITY, mean_intensity),
        (INTENSITY, STD_INTENSITY, std_intensity),
        (INTENSITY, MIN_INTENSITY, min_intensity),
        (INTENSITY, MAX_INTENSITY, max_intensity),
        (INTENSITY, MASS_DISPLACEMENT, mass_displacement),
        (INTENSITY, LOWER_QUARTILE_INTENSITY, lower_quartile_intensity),
        (INTENSITY, MEDIAN_INTENSITY, median_intensity),
        (INTENSITY, MAD_INTENSITY, mad_intensity),
        (INTENSITY, UPPER_QUARTILE_INTENSITY, upper_quartile_intensity),
        (C_LOCATION, LOC_CMI_X, cmi_x),
        (C_LOCATION, LOC_CMI_Y, cmi_y),
        (C_LOCATION, LOC_CMI_Z, cmi_z),
        (C_LOCATION, LOC_MAX_X, max_x),
        (C_LOCATION, LOC_MAX_Y, max_y),
        (C_LOCATION, LOC_MAX_Z, max_z),
    ]
    if edge_measurements:
        measurement_names.extend(
            [
                (INTENSITY, INTEGRATED_INTENSITY_EDGE, integrated_intensity_edge),
                (INTENSITY, MEAN_INTENSITY_EDGE, mean_intensity_edge),
                (INTENSITY, STD_INTENSITY_EDGE, std_intensity_edge),
                (INTENSITY, MIN_INTENSITY_EDGE, min_intensity_edge),
                (INTENSITY, MAX_INTENSITY_EDGE, max_intensity_edge),
            ]
        )

    return {
        "{}_{}".format(category, feature_name): measurement
        for category, feature_name, measurement in measurement_names
    }
