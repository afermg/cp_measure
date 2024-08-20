import centrosome.cpmorphology
import centrosome.filter
import centrosome.outline
import numpy
import scipy.ndimage
import skimage.segmentation

__doc__ = """
MeasureObjectIntensity
======================

**MeasureObjectIntensity** measures several intensity features for
identified objects.

Given an image with objects identified (e.g., nuclei or cells), this
module extracts intensity features for each object based on one or more
corresponding grayscale images. Measurements are recorded for each
object.

Intensity measurements are made for all combinations of the images and
objects entered. If you want only specific image/object measurements,
you can use multiple MeasureObjectIntensity modules for each group of
measurements desired.

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

See also
^^^^^^^^

See also **NamesAndTypes**, **MeasureImageIntensity**.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *IntegratedIntensity:* The sum of the pixel intensities within an
   object.
-  *MeanIntensity:* The average pixel intensity within an object.
-  *StdIntensity:* The standard deviation of the pixel intensities
   within an object.
-  *MaxIntensity:* The maximal pixel intensity within an object.
-  *MinIntensity:* The minimal pixel intensity within an object.
-  *IntegratedIntensityEdge:* The sum of the edge pixel intensities of
   an object.
-  *MeanIntensityEdge:* The average edge pixel intensity of an object.
-  *StdIntensityEdge:* The standard deviation of the edge pixel
   intensities of an object.
-  *MaxIntensityEdge:* The maximal edge pixel intensity of an object.
-  *MinIntensityEdge:* The minimal edge pixel intensity of an object.
-  *MassDisplacement:* The distance between the centers of gravity in
   the gray-level representation of the object and the binary
   representation of the object.
-  *LowerQuartileIntensity:* The intensity value of the pixel for which
   25% of the pixels in the object have lower values.
-  *MedianIntensity:* The median intensity value within the object.
-  *MADIntensity:* The median absolute deviation (MAD) value of the
   intensities within the object. The MAD is defined as the
   median(\|x\ :sub:`i` - median(x)\|).
-  *UpperQuartileIntensity:* The intensity value of the pixel for which
   75% of the pixels in the object have lower values.
-  *Location\_CenterMassIntensity\_X, Location\_CenterMassIntensity\_Y:*
   The (X,Y) coordinates of the intensity weighted centroid (=
   center of mass = first moment) of all pixels within the object.
-  *Location\_MaxIntensity\_X, Location\_MaxIntensity\_Y:* The
   (X,Y) coordinates of the pixel with the maximum intensity within the
   object.
"""

INTENSITY = "Intensity"
INTEGRATED_INTENSITY = "IntegratedIntensity"
MEAN_INTENSITY = "MeanIntensity"
STD_INTENSITY = "StdIntensity"
MIN_INTENSITY = "MinIntensity"
MAX_INTENSITY = "MaxIntensity"
INTEGRATED_INTENSITY_EDGE = "IntegratedIntensityEdge"
MEAN_INTENSITY_EDGE = "MeanIntensityEdge"
STD_INTENSITY_EDGE = "StdIntensityEdge"
MIN_INTENSITY_EDGE = "MinIntensityEdge"
MAX_INTENSITY_EDGE = "MaxIntensityEdge"
MASS_DISPLACEMENT = "MassDisplacement"
LOWER_QUARTILE_INTENSITY = "LowerQuartileIntensity"
MEDIAN_INTENSITY = "MedianIntensity"
MAD_INTENSITY = "MADIntensity"
UPPER_QUARTILE_INTENSITY = "UpperQuartileIntensity"
LOC_CMI_X = "CenterMassIntensity_X"
LOC_CMI_Y = "CenterMassIntensity_Y"
LOC_CMI_Z = "CenterMassIntensity_Z"
LOC_MAX_X = "MaxIntensity_X"
LOC_MAX_Y = "MaxIntensity_Y"
LOC_MAX_Z = "MaxIntensity_Z"

ALL_MEASUREMENTS = [
    INTEGRATED_INTENSITY,
    MEAN_INTENSITY,
    STD_INTENSITY,
    MIN_INTENSITY,
    MAX_INTENSITY,
    INTEGRATED_INTENSITY_EDGE,
    MEAN_INTENSITY_EDGE,
    STD_INTENSITY_EDGE,
    MIN_INTENSITY_EDGE,
    MAX_INTENSITY_EDGE,
    MASS_DISPLACEMENT,
    LOWER_QUARTILE_INTENSITY,
    MEDIAN_INTENSITY,
    MAD_INTENSITY,
    UPPER_QUARTILE_INTENSITY,
]
ALL_LOCATION_MEASUREMENTS = [
    LOC_CMI_X,
    LOC_CMI_Y,
    LOC_CMI_Z,
    LOC_MAX_X,
    LOC_MAX_Y,
    LOC_MAX_Z,
]


def get_intensity(pixels: numpy.ndarray, mask: numpy.ndarray):
    masked_image = pixels
    image_mask = numpy.ones_like(pixels, dtype=bool)

    if pixels.ndim == 2:
        img = pixels.reshape(1, *pixels.shape)
        masked_image = img

    # MODIFIED: Provided directly
    nobjects = 1
    integrated_intensity = numpy.zeros((nobjects,))
    integrated_intensity_edge = numpy.zeros((nobjects,))
    mean_intensity = numpy.zeros((nobjects,))
    mean_intensity_edge = numpy.zeros((nobjects,))
    std_intensity = numpy.zeros((nobjects,))
    std_intensity_edge = numpy.zeros((nobjects,))
    min_intensity = numpy.zeros((nobjects,))
    min_intensity_edge = numpy.zeros((nobjects,))
    max_intensity = numpy.zeros((nobjects,))
    max_intensity_edge = numpy.zeros((nobjects,))
    mass_displacement = numpy.zeros((nobjects,))
    lower_quartile_intensity = numpy.zeros((nobjects,))
    median_intensity = numpy.zeros((nobjects,))
    mad_intensity = numpy.zeros((nobjects,))
    upper_quartile_intensity = numpy.zeros((nobjects,))
    cmi_x = numpy.zeros((nobjects,))
    cmi_y = numpy.zeros((nobjects,))
    cmi_z = numpy.zeros((nobjects,))
    max_x = numpy.zeros((nobjects,))
    max_y = numpy.zeros((nobjects,))
    max_z = numpy.zeros((nobjects,))

    # Modified: Passed single mask directly
    result = {}
    for labels, lindexes in ((mask.astype(int), numpy.array([1])),):
        lindexes = lindexes[lindexes != 0]

        if pixels.ndim == 2:
            labels = labels.reshape(1, *labels.shape)

        outlines = skimage.segmentation.find_boundaries(labels, mode="inner")

        masked_labels = labels
        masked_outlines = outlines

        lmask = masked_labels > 0 & numpy.isfinite(img)  # Ignore NaNs, Infs
        has_objects = numpy.any(lmask)
        if has_objects:
            limg = img[lmask]

            llabels = labels[lmask]

            mesh_z, mesh_y, mesh_x = numpy.mgrid[
                0 : masked_image.shape[0],
                0 : masked_image.shape[1],
                0 : masked_image.shape[2],
            ]

            mesh_x = mesh_x[lmask]
            mesh_y = mesh_y[lmask]
            mesh_z = mesh_z[lmask]

            lcount = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.sum(numpy.ones(len(limg)), llabels, lindexes)
            )

            integrated_intensity[lindexes - 1] = (
                centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.sum(limg, llabels, lindexes)
                )
            )

            mean_intensity[lindexes - 1] = integrated_intensity[lindexes - 1] / lcount

            std_intensity[lindexes - 1] = numpy.sqrt(
                centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.mean(
                        (limg - mean_intensity[llabels - 1]) ** 2,
                        llabels,
                        lindexes,
                    )
                )
            )

            min_intensity[lindexes - 1] = (
                centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.minimum(limg, llabels, lindexes)
                )
            )

            max_intensity[lindexes - 1] = (
                centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.maximum(limg, llabels, lindexes)
                )
            )

            # Compute the position of the intensity maximum
            max_position = numpy.array(
                centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.maximum_position(limg, llabels, lindexes)
                ),
                dtype=int,
            )
            max_position = numpy.reshape(max_position, (max_position.shape[0],))

            max_x[lindexes - 1] = mesh_x[max_position]
            max_y[lindexes - 1] = mesh_y[max_position]
            max_z[lindexes - 1] = mesh_z[max_position]

            # The mass displacement is the distance between the center
            # of mass of the binary image and of the intensity image. The
            # center of mass is the average X or Y for the binary image
            # and the sum of X or Y * intensity / integrated intensity
            cm_x = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.mean(mesh_x, llabels, lindexes)
            )
            cm_y = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.mean(mesh_y, llabels, lindexes)
            )
            cm_z = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.mean(mesh_z, llabels, lindexes)
            )

            i_x = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.sum(mesh_x * limg, llabels, lindexes)
            )
            i_y = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.sum(mesh_y * limg, llabels, lindexes)
            )
            i_z = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.sum(mesh_z * limg, llabels, lindexes)
            )

            cmi_x[lindexes - 1] = i_x / integrated_intensity[lindexes - 1]
            cmi_y[lindexes - 1] = i_y / integrated_intensity[lindexes - 1]
            cmi_z[lindexes - 1] = i_z / integrated_intensity[lindexes - 1]

            diff_x = cm_x - cmi_x[lindexes - 1]
            diff_y = cm_y - cmi_y[lindexes - 1]
            diff_z = cm_z - cmi_z[lindexes - 1]

            mass_displacement[lindexes - 1] = numpy.sqrt(
                diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
            )

            #
            # Sort the intensities by label, then intensity.
            # For each label, find the index above and below
            # the 25%, 50% and 75% mark and take the weighted
            # average.
            #
            order = numpy.lexsort((limg, llabels))
            areas = lcount.astype(int)
            indices = numpy.cumsum(areas) - areas
            for dest, fraction in (
                (lower_quartile_intensity, 1.0 / 4.0),
                (median_intensity, 1.0 / 2.0),
                (upper_quartile_intensity, 3.0 / 4.0),
            ):
                qindex = indices.astype(float) + areas * fraction
                qfraction = qindex - numpy.floor(qindex)
                qindex = qindex.astype(int)
                qmask = qindex < indices + areas - 1
                qi = qindex[qmask]
                qf = qfraction[qmask]
                dest[lindexes[qmask] - 1] = (
                    limg[order[qi]] * (1 - qf) + limg[order[qi + 1]] * qf
                )

                #
                # In some situations (e.g., only 3 points), there may
                # not be an upper bound.
                #
                qmask = (~qmask) & (areas > 0)
                dest[lindexes[qmask] - 1] = limg[order[qindex[qmask]]]

            #
            # Once again, for the MAD
            #
            madimg = numpy.abs(limg - median_intensity[llabels - 1])
            order = numpy.lexsort((madimg, llabels))
            qindex = indices.astype(float) + areas / pixels.ndim
            qfraction = qindex - numpy.floor(qindex)
            qindex = qindex.astype(int)
            qmask = qindex < indices + areas - 1
            qi = qindex[qmask]
            qf = qfraction[qmask]
            mad_intensity[lindexes[qmask] - 1] = (
                madimg[order[qi]] * (1 - qf) + madimg[order[qi + 1]] * qf
            )
            qmask = (~qmask) & (areas > 0)
            mad_intensity[lindexes[qmask] - 1] = madimg[order[qindex[qmask]]]

            # END OF ITERATIONS

    emask = masked_outlines > 0
    eimg = img[emask]
    elabels = labels[emask]
    has_edge = len(eimg) > 0

    if has_edge:
        ecount = centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scipy.ndimage.sum(numpy.ones(len(eimg)), elabels, lindexes)
        )

        integrated_intensity_edge[lindexes - 1] = (
            centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.sum(eimg, elabels, lindexes)
            )
        )

        mean_intensity_edge[lindexes - 1] = (
            integrated_intensity_edge[lindexes - 1] / ecount
        )

        std_intensity_edge[lindexes - 1] = numpy.sqrt(
            centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.mean(
                    (eimg - mean_intensity_edge[elabels - 1]) ** 2,
                    elabels,
                )
            )
        )

    min_intensity_edge[lindexes - 1] = (
        centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scipy.ndimage.minimum(eimg, elabels, lindexes)
        )
    )

    max_intensity_edge[lindexes - 1] = (
        centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scipy.ndimage.maximum(eimg, elabels, lindexes)
        )
    )
    for category, feature_name, measurement in (
        (INTENSITY, INTEGRATED_INTENSITY, integrated_intensity),
        (INTENSITY, MEAN_INTENSITY, mean_intensity),
        (INTENSITY, STD_INTENSITY, std_intensity),
        (INTENSITY, MIN_INTENSITY, min_intensity),
        (INTENSITY, MAX_INTENSITY, max_intensity),
        (INTENSITY, INTEGRATED_INTENSITY_EDGE, integrated_intensity_edge),
        (INTENSITY, MEAN_INTENSITY_EDGE, mean_intensity_edge),
        (INTENSITY, STD_INTENSITY_EDGE, std_intensity_edge),
        (INTENSITY, MIN_INTENSITY_EDGE, min_intensity_edge),
        (INTENSITY, MAX_INTENSITY_EDGE, max_intensity_edge),
        (INTENSITY, MASS_DISPLACEMENT, mass_displacement),
        (INTENSITY, LOWER_QUARTILE_INTENSITY, lower_quartile_intensity),
        (INTENSITY, MEDIAN_INTENSITY, median_intensity),
        (INTENSITY, MAD_INTENSITY, mad_intensity),
        (INTENSITY, UPPER_QUARTILE_INTENSITY, upper_quartile_intensity),
        # FIXME: Find out what this is, maybe location?
        # (C_LOCATION, LOC_CMI_X, cmi_x),
        # (C_LOCATION, LOC_CMI_Y, cmi_y),
        # (C_LOCATION, LOC_CMI_Z, cmi_z),
        # (C_LOCATION, LOC_MAX_X, max_x),
        # (C_LOCATION, LOC_MAX_Y, max_y),
        # (C_LOCATION, LOC_MAX_Z, max_z),
    ):
        measurement_name = "{}_{}".format(category, feature_name)
        # MODIFIED: Selected first
        result[measurement_name] = measurement[0]

    return result
