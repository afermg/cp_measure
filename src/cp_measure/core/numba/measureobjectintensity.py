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

from cp_measure._sanitize import sanitize_labels

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
from cp_measure.primitives.segment import label_to_idx_lut
from cp_measure.primitives._segment_numba import (
    flatten_numba,
    inner_boundary,
    segment_moments,
    segment_quantiles,
    segment_resid_sumsq,
    segment_stats,
)


@sanitize_labels
def get_intensity(
    masks: NDArray[numpy.integer],
    pixels: NDArray[numpy.floating],
    edge_measurements: bool = True,
    legacy: bool = False,
) -> dict[str, NDArray[numpy.floating]]:
    """masks is a labeled array where 0 are background.

    ``legacy`` mirrors the numpy backend: False (default) uses ``numpy.percentile``
    'linear' quartiles + textbook median MAD; True reproduces the original
    CellProfiler ``n*q`` quartiles + ``(1/ndim)``-quantile MAD.
    """
    orig_ndim = pixels.ndim

    masked_image = pixels
    if pixels.ndim == 2:
        masked_image = pixels.reshape(1, *pixels.shape)
        if masks.ndim == 2:
            masks = masks.reshape(1, *masks.shape)
    elif pixels.ndim == 3 and masks.ndim == 2:  # 3D image, 2D mask
        masks = masks.reshape(1, *masks.shape)

    lut, nobjects = label_to_idx_lut(masks)

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
        numpy.ascontiguousarray(masked_image),
        lut,
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
        ) = segment_quantiles(values, seg0, count, nobjects, 1.0 / orig_ndim, legacy)

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
        e_values = masked_image[emask].astype(numpy.float64)
        e_seg0 = lut[masks[emask]]

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
