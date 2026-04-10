"""
MeasureObjectIntensityDistribution
===================================

**MeasureObjectIntensityDistribution** measures the distribution of
intensities within each object, producing radial distribution and
Zernike moment features.

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

See also **MeasureObjectIntensity** and **MeasureTexture**.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *FracAtD:* Fraction of total stain in an object at a given radius.
-  *MeanFrac:* Mean fractional intensity at a given radius; calculated
   as fraction of total intensity normalized by fraction of pixels at a
   given radius.
-  *RadialCV:* Coefficient of variation of intensity within a ring,
   calculated across 8 slices.
-  *Zernike:* The Zernike features characterize the distribution of
   intensity across the object. For instance, Zernike 1,1 has a high
   value if the intensity is low on one side of the object and high on
   the other. The ZernikeMagnitude feature records the rotationally
   invariant degree magnitude of the moment and the ZernikePhase feature
   gives the moment’s orientation.
"""

import centrosome.cpmorphology
import centrosome.propagate
import centrosome.zernike
import numpy
import numpy.ma
import scipy.ndimage
import scipy.sparse
from cp_measure.utils import masks_to_ijv

Z_NONE = "None"
Z_MAGNITUDES = "Magnitudes only"
Z_MAGNITUDES_AND_PHASE = "Magnitudes and phase"
Z_ALL = [Z_NONE, Z_MAGNITUDES, Z_MAGNITUDES_AND_PHASE]

M_CATEGORY = "RadialDistribution"
F_FRAC_AT_D = "FracAtD"
F_MEAN_FRAC = "MeanFrac"
F_RADIAL_CV = "RadialCV"
F_ALL = [F_FRAC_AT_D, F_MEAN_FRAC, F_RADIAL_CV]

FF_SCALE = "%dof%d"
FF_OVERFLOW = "Overflow"
FF_GENERIC = FF_SCALE

MF_FRAC_AT_D = "_".join((M_CATEGORY, F_FRAC_AT_D, FF_GENERIC))
MF_MEAN_FRAC = "_".join((M_CATEGORY, F_MEAN_FRAC, FF_GENERIC))
MF_RADIAL_CV = "_".join((M_CATEGORY, F_RADIAL_CV, FF_GENERIC))
OF_FRAC_AT_D = "_".join((M_CATEGORY, F_FRAC_AT_D, FF_OVERFLOW))
OF_MEAN_FRAC = "_".join((M_CATEGORY, F_MEAN_FRAC, FF_OVERFLOW))
OF_RADIAL_CV = "_".join((M_CATEGORY, F_RADIAL_CV, FF_OVERFLOW))

"""# of settings aside from groups"""
SETTINGS_STATIC_COUNT = 3
"""# of settings in image group"""
SETTINGS_IMAGE_GROUP_COUNT = 1
"""# of settings in object group"""
SETTINGS_OBJECT_GROUP_COUNT = 3
"""# of settings in bin group, v1"""
SETTINGS_BIN_GROUP_COUNT_V1 = 1
"""# of settings in bin group, v2"""
SETTINGS_BIN_GROUP_COUNT_V2 = 3
SETTINGS_BIN_GROUP_COUNT = 3
"""# of settings in heatmap group, v4"""
SETTINGS_HEATMAP_GROUP_COUNT_V4 = 7
SETTINGS_HEATMAP_GROUP_COUNT = 7
"""Offset of center choice in object group"""
SETTINGS_CENTER_CHOICE_OFFSET = 1

A_FRAC_AT_D = "Fraction at Distance"
A_MEAN_FRAC = "Mean Fraction"
A_RADIAL_CV = "Radial CV"
MEASUREMENT_CHOICES = [A_FRAC_AT_D, A_MEAN_FRAC, A_RADIAL_CV]

MEASUREMENT_ALIASES = {
    A_FRAC_AT_D: MF_FRAC_AT_D,
    A_MEAN_FRAC: MF_MEAN_FRAC,
    A_RADIAL_CV: MF_RADIAL_CV,
}


def get_radial_distribution(
    labels: numpy.ndarray,
    pixels: numpy.ndarray,
    scaled: bool = True,
    bin_count: int = 4,
    maximum_radius: int = 100,
):
    """Measure the radial distribution of intensity within labeled objects (2D only).

    Computes fraction at distance, mean fraction, and radial coefficient of
    variation features for concentric rings radiating from each object's center.

    Parameters
    ----------
    labels : numpy.ndarray
        Labeled 2D mask array where each positive integer identifies an object.
        Returns an empty dict for 3D inputs.
    pixels : numpy.ndarray
        Grayscale intensity image with the same shape as ``labels``.
    scaled : bool, optional
        If True, divide each object radially into ``bin_count`` bins scaled to
        the object size. If False, use absolute distance bins up to
        ``maximum_radius``, by default True.
    bin_count : int, optional
        Number of concentric rings for the radial distribution, by default 4.
    maximum_radius : int, optional
        Maximum radius in pixels for unscaled bins, by default 100. Only used
        when ``scaled`` is False.

    Returns
    -------
    dict of {str: numpy.ndarray}
        Dictionary mapping feature names (FracAtD, MeanFrac, RadialCV) to
        1-D arrays of per-object measurements.
    """

    if labels.ndim == 3:
        return {}

    if labels.dtype == bool:
        labels = labels.astype(numpy.integer)

    unique_labels = numpy.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    nobjects = len(unique_labels)
    d_to_edge = centrosome.cpmorphology.distance_to_edge(labels)

    # Find the point in each object farthest away from the edge.
    # This does better than the centroid:
    # * The center is within the object
    # * The center tends to be an interesting point, like the
    #   center of the nucleus or the center of one or the other
    #   of two touching cells.
    #
    # MODIFICATION: Delegated label indices to maximum_position_of_labels
    # This should not affect this one-mask/object function
    i, j = centrosome.cpmorphology.maximum_position_of_labels(
        # d_to_edge, labels, indices=[1]
        d_to_edge,
        labels,
        indices=unique_labels,
    )

    center_labels = numpy.zeros(labels.shape, int)

    center_labels[i, j] = labels[i, j]

    #
    # Use the coloring trick here to process touching objects
    # in separate operations
    #
    colors = centrosome.cpmorphology.color_labels(labels)

    ncolors = numpy.max(colors)

    d_from_center = numpy.zeros(labels.shape)

    cl = numpy.zeros(labels.shape, int)

    for color in range(1, ncolors + 1):
        mask = colors == color
        l_, d = centrosome.propagate.propagate(
            numpy.zeros(center_labels.shape), center_labels, mask, 1
        )

        d_from_center[mask] = d[mask]

        cl[mask] = l_[mask]

    good_mask = cl > 0

    i_center = numpy.zeros(cl.shape)

    i_center[good_mask] = i[cl[good_mask] - 1]

    j_center = numpy.zeros(cl.shape)

    j_center[good_mask] = j[cl[good_mask] - 1]

    normalized_distance = numpy.zeros(labels.shape)

    if scaled:
        total_distance = d_from_center + d_to_edge

        normalized_distance[good_mask] = d_from_center[good_mask] / (
            total_distance[good_mask] + 0.001
        )
    else:
        normalized_distance[good_mask] = d_from_center[good_mask] / maximum_radius

    ngood_pixels = numpy.sum(good_mask)

    good_labels = labels[good_mask]

    bin_indexes = (normalized_distance * bin_count).astype(int)

    bin_indexes[bin_indexes > bin_count] = bin_count

    labels_and_bins = (good_labels - 1, bin_indexes[good_mask])

    histogram = scipy.sparse.coo_matrix(
        (pixels[good_mask], labels_and_bins), (nobjects, bin_count + 1)
    ).toarray()

    sum_by_object = numpy.sum(histogram, 1)

    sum_by_object_per_bin = numpy.dstack([sum_by_object] * (bin_count + 1))[0]

    fraction_at_distance = histogram / sum_by_object_per_bin

    number_at_distance = scipy.sparse.coo_matrix(
        (numpy.ones(ngood_pixels), labels_and_bins), (nobjects, bin_count + 1)
    ).toarray()

    sum_by_object = numpy.sum(number_at_distance, 1)

    sum_by_object_per_bin = numpy.dstack([sum_by_object] * (bin_count + 1))[0]

    fraction_at_bin = number_at_distance / sum_by_object_per_bin

    mean_pixel_fraction = fraction_at_distance / (
        fraction_at_bin + numpy.finfo(float).eps
    )

    # Anisotropy calculation.  Split each cell into eight wedges, then
    # compute coefficient of variation of the wedges' mean intensities
    # in each ring.
    #
    # Compute each pixel's delta from the center object's centroid
    i, j = numpy.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]

    imask = i[good_mask] > i_center[good_mask]

    jmask = j[good_mask] > j_center[good_mask]

    absmask = abs(i[good_mask] - i_center[good_mask]) > abs(
        j[good_mask] - j_center[good_mask]
    )

    radial_index = imask.astype(int) + jmask.astype(int) * 2 + absmask.astype(int) * 4

    results = {}

    for bin in range(bin_count + (0 if scaled else 1)):
        bin_mask = good_mask & (bin_indexes == bin)

        bin_pixels = numpy.sum(bin_mask)

        bin_labels = labels[bin_mask]

        bin_radial_index = radial_index[bin_indexes[good_mask] == bin]

        labels_and_radii = (bin_labels - 1, bin_radial_index)

        radial_values = scipy.sparse.coo_matrix(
            (pixels[bin_mask], labels_and_radii), (nobjects, 8)
        ).toarray()

        pixel_count = scipy.sparse.coo_matrix(
            (numpy.ones(bin_pixels), labels_and_radii), (nobjects, 8)
        ).toarray()

        mask = pixel_count == 0

        radial_means = numpy.ma.masked_array(radial_values / pixel_count, mask)

        radial_cv = numpy.std(radial_means, 1) / numpy.mean(radial_means, 1)

        radial_cv[numpy.sum(~mask, 1) == 0] = 0

        for measurement, feature, overflow_feature in (
            (fraction_at_distance[:, bin], MF_FRAC_AT_D, OF_FRAC_AT_D),
            (mean_pixel_fraction[:, bin], MF_MEAN_FRAC, OF_MEAN_FRAC),
            (numpy.array(radial_cv), MF_RADIAL_CV, OF_RADIAL_CV),
        ):
            if bin == bin_count:
                measurement_name = overflow_feature
            else:
                measurement_name = feature % (bin + 1, bin_count)

            results[measurement_name] = measurement

    return results


def get_radial_zernikes(
    labels: numpy.ndarray, pixels: numpy.ndarray, zernike_degree: int = 9
):
    """Compute radial Zernike moment features for labeled objects (2D only).

    Zernike polynomials characterize the spatial distribution of intensity
    within each object, producing magnitude and phase features for each
    polynomial order.

    Parameters
    ----------
    labels : numpy.ndarray
        Labeled 2D mask array where each positive integer identifies an object.
        Returns an empty dict for 3D inputs.
    pixels : numpy.ndarray
        Grayscale intensity image with the same shape as ``labels``.
    zernike_degree : int, optional
        Maximum radial degree for Zernike polynomials, by default 9.

    Returns
    -------
    dict of {str: numpy.ndarray}
        Dictionary mapping Zernike feature names
        (``RadialDistribution_ZernikeMagnitude_n_m`` and
        ``RadialDistribution_ZernikePhase_n_m``) to 1-D arrays of per-object
        measurements.
    """
    # Radial Zernike features (2D only)
    if labels.ndim == 3:
        return {}
    zernike_indexes = centrosome.zernike.get_zernike_indexes(zernike_degree + 1)

    unique_labels = numpy.unique(labels)  # Will be used later for scipy.ndimage.sum
    unique_labels = unique_labels[unique_labels > 0]
    # MODIFIED: Delegate index generation to the minimum_enclosing_circle
    # MODIFIED: We assume non-overlapping labels for now
    # TODO Support label overlap (i.e., format in ijv)
    # MODIFIED: Delegate indexes to minimum_enclosing_circle
    ij, r = centrosome.cpmorphology.minimum_enclosing_circle(labels, unique_labels)

    #
    # Then compute x and y, the position of each labeled pixel
    # within a unit circle around the object
    #
    ijv = masks_to_ijv(labels)

    l_ = ijv[:, 2]  # (N,1) vector with labels

    yx = (ijv[:, :2] - ij[l_ - 1, :]) / r[l_ - 1, numpy.newaxis]

    z = centrosome.zernike.construct_zernike_polynomials(
        yx[:, 1], yx[:, 0], zernike_indexes
    )

    # Filter ijv-formatted items to keep the ones inside the pixels boundary
    ijv_mask = (ijv[:, 0] < pixels.shape[0]) & (ijv[:, 1] < pixels.shape[1])
    # ijv_mask[ijv_mask] = pixels[ijv[ijv_mask,0], ijv[ijv_mask, 1]]

    yx = yx[ijv_mask, :]
    l_ = l_[ijv_mask]
    z_ = z[ijv_mask, :]

    if len(l_) == 0:
        # Cover fringe case in which all labels were filtered out
        results = {}
        for mag_or_phase in ("Magnitude", "Phase"):
            for n, m in zernike_indexes:
                name = f"{M_CATEGORY}_Zernike{mag_or_phase}_{n}_{m}"
                results[name] = numpy.zeros(0)
    else:
        # MODIFIED: Replaced sum with the updated sum_labels
        areas = scipy.ndimage.sum_labels(
            numpy.ones(l_.shape, int), labels=l_, index=unique_labels
        )

        #
        # Results will be formatted in a dictionary with the following keys:
        # Zernike{Magniture|Phase}_{n}_{m}
        # n - the radial moment of the Zernike
        # m - the azimuthal moment of the Zernike
        #
        results = {}
        for i, (n, m) in enumerate(zernike_indexes):
            vr = scipy.ndimage.sum_labels(
                pixels[ijv[:, 0], ijv[:, 1]] * z_[:, i].real,
                labels=l_,
                index=unique_labels,
            )

            vi = scipy.ndimage.sum_labels(
                pixels[ijv[:, 0], ijv[:, 1]] * z[:, i].imag,
                labels=l_,
                index=unique_labels,
            )

            magnitude = numpy.sqrt(vr * vr + vi * vi) / areas
            phase = numpy.arctan2(vr, vi)

            results[f"{M_CATEGORY}_ZernikeMagnitude_{n}_{m}"] = magnitude
            results[f"{M_CATEGORY}_ZernikePhase_{n}_{m}"] = phase

    return results
