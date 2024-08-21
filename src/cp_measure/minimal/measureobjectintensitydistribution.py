import centrosome.cpmorphology
import centrosome.propagate
import centrosome.zernike
import numpy
import numpy.ma
import scipy.ndimage
import scipy.sparse

from cp_measure.minimal.utils import boolean_mask_to_ijv

""""
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
    pixels: numpy.ndarray,
    mask: numpy.ndarray,
    scaled: bool = True,
    bin_count: int = 4,
    maximum_radius: int = 100,
):
    """
    zernike_degree : int
        Maximum zernike moment.

        This is the maximum radial moment that will be calculated. There are
        increasing numbers of azimuthal moments as you increase the radial
        moment, so higher values are increasingly expensive to calculate.

    scaled : bool
        Scale the bins?

        When True divide the object radially into the number of bins
        that you specify. Otherwise create the number of bins you specify
        based on distance. If True, it will use a maximum distance so
        that each object will have the same measurements (which might be zero
        for small objects) and so that the measurements can be taken without
        knowing the maximum object radius before the run starts.

    bin_count : int
        Number of bins

        Specify the number of bins that you want to use to measure the distribution.
        Radial distribution is measured with respect to a series of concentric
        rings starting from the object center (or more generally, between contours
        at a normalized distance from the object center). This number specifies
        the number of rings into which the distribution is to be divided.

    maximum_radius : int
        Maximum radius

        Specify the maximum radius for the unscaled bins. The unscaled binning method
        creates the number of bins that you specify and creates equally spaced bin
        boundaries up to the maximum radius. Parts of the object that are beyond this
        radius will be counted in an overflow bin. The radius is measured in pixels.
    """

    labels = mask.astype(int)
    nobjects = 1
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
        d_to_edge, labels, indices=[1]
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
        l, d = centrosome.propagate.propagate(
            numpy.zeros(center_labels.shape), center_labels, mask, 1
        )

        d_from_center[mask] = d[mask]

        cl[mask] = l[mask]

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
        print(bin)
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

            results[measurement_name] = measurement[0]

    return results


def get_radial_zernikes(pixels, mask, zernike_degree: int = 9):
    zernike_indexes = centrosome.zernike.get_zernike_indexes(zernike_degree + 1)

    labels = mask.astype(int)  # Convert boolean mask to labels
    l_ = [1]  # Will be used later for scipy.ndimage.sum

    # MODIFIED: Delegate index generation to the minimum_enclosing_circle
    # TODO: Check that this has the correct dimensions
    ij, r = centrosome.cpmorphology.minimum_enclosing_circle(labels)

    #
    # Then compute x and y, the position of each labeled pixel
    # within a unit circle around the object
    #
    ijv = boolean_mask_to_ijv(mask)

    yx = (ijv[:, :2] - ij) / r

    z = centrosome.zernike.construct_zernike_polynomials(
        yx[:, 1], yx[:, 0], zernike_indexes
    )

    area = mask.sum()

    #
    # Results will be formatted in a dictionary with the following keys:
    # Zernike{Magniture|Phase}_{n}_{m}
    # n - the radial moment of the Zernike
    # m - the azimuthal moment of the Zernike
    #
    results = {}
    for i, (n, m) in enumerate(zernike_indexes):
        vr = scipy.ndimage.sum(
            pixels[ijv[:, 0], ijv[:, 1]] * z[:, i].real,
            labels=l_,
        )

        vi = scipy.ndimage.sum(
            pixels[ijv[:, 0], ijv[:, 1]] * z[:, i].imag,
            labels=l_,
        )

        magnitude = numpy.sqrt(vr * vr + vi * vi) / area
        phase = numpy.arctan2(vr, vi)

        results[f"{M_CATEGORY}_ZernikeMagnitude_{n}_{m}"] = magnitude
        results[f"{M_CATEGORY}_ZernikePhase_{n}_{m}"] = phase

    return results
