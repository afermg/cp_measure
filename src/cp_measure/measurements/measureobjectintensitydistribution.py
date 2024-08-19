import cellprofiler.gui.help.content
import centrosome.cpmorphology
import centrosome.propagate
import centrosome.zernike
import matplotlib.cm
import numpy
import numpy.ma
import scipy.ndimage
import scipy.sparse
# from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
# from cellprofiler_core.image import Image
# from cellprofiler_core.module import Module
# from cellprofiler_core.preferences import get_default_colormap
# from cellprofiler_core.setting import (
#     Binary,
#     Divider,
#     HiddenCount,
#     SettingsGroup,
#     ValidationError,
# )
# from cellprofiler_core.setting.choice import Choice, Colormap
# from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
# from cellprofiler_core.setting.subscriber import (
#     ImageListSubscriber,
#     ImageSubscriber,
#     LabelSubscriber,
# )
# from cellprofiler_core.setting.text import ImageName, Integer
# from cellprofiler_core.utilities.core.object import (
#     crop_labels_and_image,
#     size_similarly,
# )

# |
"""
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

C_SELF = "These objects"
C_CENTERS_OF_OTHER_V2 = "Other objects"
C_CENTERS_OF_OTHER = "Centers of other objects"
C_EDGES_OF_OTHER = "Edges of other objects"
C_ALL = [C_SELF, C_CENTERS_OF_OTHER, C_EDGES_OF_OTHER]

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
FF_GENERIC = "_%s_" + FF_SCALE
FF_FRAC_AT_D = F_FRAC_AT_D + FF_GENERIC
FF_MEAN_FRAC = F_MEAN_FRAC + FF_GENERIC
FF_RADIAL_CV = F_RADIAL_CV + FF_GENERIC

FF_ZERNIKE_MAGNITUDE = "ZernikeMagnitude"
FF_ZERNIKE_PHASE = "ZernikePhase"

MF_FRAC_AT_D = "_".join((M_CATEGORY, FF_FRAC_AT_D))
MF_MEAN_FRAC = "_".join((M_CATEGORY, FF_MEAN_FRAC))
MF_RADIAL_CV = "_".join((M_CATEGORY, FF_RADIAL_CV))
OF_FRAC_AT_D = "_".join((M_CATEGORY, F_FRAC_AT_D, "%s", FF_OVERFLOW))
OF_MEAN_FRAC = "_".join((M_CATEGORY, F_MEAN_FRAC, "%s", FF_OVERFLOW))
OF_RADIAL_CV = "_".join((M_CATEGORY, F_RADIAL_CV, "%s", FF_OVERFLOW))

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


class MeasureObjectIntensityDistribution(Module):
    def create_settings(self):
      self.wants_zernikes = Choice(
            "Calculate intensity Zernikes?",
            Z_ALL,
            doc="""\
This setting determines whether the intensity Zernike moments are
calculated. Choose *{Z_NONE}* to save computation time by not
calculating the Zernike moments. Choose *{Z_MAGNITUDES}* to only save
the magnitude information and discard information related to the
object’s angular orientation. Choose *{Z_MAGNITUDES_AND_PHASE}* to
save the phase information as well. The last option lets you recover
each object’s rough appearance from the Zernikes but may not contribute
useful information for classifying phenotypes.

|MeasureObjectIntensityDistribution_image0|

.. |MeasureObjectIntensityDistribution_image0| image:: {MeasureObjectIntensityDistribution_Magnitude_Phase}
""".format(
                **{
                    "Z_NONE": Z_NONE,
                    "Z_MAGNITUDES": Z_MAGNITUDES,
                    "Z_MAGNITUDES_AND_PHASE": Z_MAGNITUDES_AND_PHASE,
                    "MeasureObjectIntensityDistribution_Magnitude_Phase": MeasureObjectIntensityDistribution_Magnitude_Phase,
                }
            ),
        )

        self.zernike_degree = Integer(
            "Maximum zernike moment",
            value=9,
            minval=1,
            maxval=20,
            doc="""\
(*Only if "{wants_zernikes}" is "{Z_MAGNITUDES}" or "{Z_MAGNITUDES_AND_PHASE}"*)

This is the maximum radial moment that will be calculated. There are
increasing numbers of azimuthal moments as you increase the radial
moment, so higher values are increasingly expensive to calculate.
""".format(
                **{
                    "wants_zernikes": self.wants_zernikes.text,
                    "Z_MAGNITUDES": Z_MAGNITUDES,
                    "Z_MAGNITUDES_AND_PHASE": Z_MAGNITUDES_AND_PHASE,
                }
            ),
        )

        self.spacer_1 = Divider()

        self.add_object_button = DoSomething("", "Add another object", self.add_object)

        self.spacer_2 = Divider()

        self.add_bin_count_button = DoSomething(
            "", "Add another set of bins", self.add_bin_count
        )

        self.spacer_3 = Divider()

        self.add_heatmap_button = DoSomething(
            "",
            "Add another heatmap display",
            self.add_heatmap,
            doc="""\
Press this button to add a display of one of the radial distribution
measurements. Each radial band of the object is colored using a
heatmap according to the measurement value for that band.
""",
        )

        self.add_object(can_remove=False)

        self.add_bin_count(can_remove=False)

        group.append(
            "center_choice",
            Choice(
                "Object to use as center?",
                C_ALL,
                doc="""\
There are three ways to specify the center of the radial measurement:

-  *{C_SELF}:* Use the centers of these objects for the radial
   measurement.
-  *{C_CENTERS_OF_OTHER}:* Use the centers of other objects for the
   radial measurement.
-  *{C_EDGES_OF_OTHER}:* Measure distances from the edge of the other
   object to each pixel outside of the centering object. Do not include
   pixels within the centering object in the radial measurement
   calculations.

For example, if measuring the radial distribution in a Cell object, you
can use the center of the Cell objects (*{C_SELF}*) or you can use
previously identified Nuclei objects as the centers
(*{C_CENTERS_OF_OTHER}*).""".format(
                    **{
                        "C_SELF": C_SELF,
                        "C_CENTERS_OF_OTHER": C_CENTERS_OF_OTHER,
                        "C_EDGES_OF_OTHER": C_EDGES_OF_OTHER,
                        "MeasureObjectIntensityDistribution_Edges_Centers": MeasureObjectIntensityDistribution_Edges_Centers,
                    }
                ),
            ),
        )

       group.append(
            "wants_scaled",
            Binary(
                "Scale the bins?",
                True,
                doc="""\
Select *{YES}* to divide the object radially into the number of bins
that you specify.

Select *{NO}* to create the number of bins you specify based on
distance. For this option, you will be asked to specify a maximum
distance so that each object will have the same measurements (which
might be zero for small objects) and so that the measurements can be
taken without knowing the maximum object radius before the run starts.
""".format(**{"YES": "Yes", "NO": "No"}),
            ),
        )

        group.append(
            "bin_count",
            Integer(
                "Number of bins",
                4,
                2,
                doc="""\
Specify the number of bins that you want to use to measure the
distribution. Radial distribution is measured with respect to a series
of concentric rings starting from the object center (or more generally,
between contours at a normalized distance from the object center). This
number specifies the number of rings into which the distribution is to
be divided. Additional ring counts can be specified by clicking the *Add
another set of bins* button.""",
            ),
        )

        group.append(
            "maximum_radius",
            Integer(
                "Maximum radius",
                100,
                minval=1,
                doc="""\
Specify the maximum radius for the unscaled bins. The unscaled binning method creates the number of
bins that you specify and creates equally spaced bin boundaries up to the maximum radius. Parts of
the object that are beyond this radius will be counted in an overflow bin. The radius is measured
in pixels.
""",
            ),
        )

        group.can_remove = can_remove

        if can_remove:
            group.append(
                "remover",
                RemoveSettingButton(
                    "", "Remove this set of bins", self.bin_counts, group
                ),
            )

        self.bin_counts.append(group)

    def settings(self):
        result = [
            self.images_list,
            self.object_count,
            self.bin_counts_count,
            self.heatmap_count,
            self.wants_zernikes,
            self.zernike_degree,
        ]

    def run(self, workspace):
        header = (
            "Image",
            "Objects",
            "Bin # (innermost=1)",
            "Bin count",
            "Fraction",
            "Intensity",
            "COV",
        )

        stats = []

        d = {}

        for image in self.images_list.value:
            for o in self.objects:
                for bin_count_settings in self.bin_counts:
                    stats += self.do_measurements(
                        workspace,
                        image,
                        o.object_name.value,
                        o.center_object_name.value
                        if o.center_choice != C_SELF
                        else None,
                        o.center_choice.value,
                        bin_count_settings,
                        d,
                    )

        if self.wants_zernikes != Z_NONE:
            self.calculate_zernikes(workspace)

        if self.show_window:
            workspace.display_data.header = header

            workspace.display_data.stats = stats

            workspace.display_data.heatmaps = []

        for heatmap in self.heatmaps:
            heatmap_img = d.get(id(heatmap))

            if heatmap_img is not None:
                if self.show_window or heatmap.wants_to_save_display:
                    labels = workspace.object_set.get_objects(
                        heatmap.object_name.get_objects_name()
                    ).segmented

                if self.show_window:
                    workspace.display_data.heatmaps.append((heatmap_img, labels != 0))

                if heatmap.wants_to_save_display:
                    colormap = heatmap.colormap.value

                    if colormap == matplotlib.cm.gray.name:
                        output_pixels = heatmap_img
                    else:
                        if colormap == "Default":
                            colormap = get_default_colormap()

                        cm = matplotlib.cm.ScalarMappable(cmap=colormap)

                        output_pixels = cm.to_rgba(heatmap_img)[:, :, :3]

                        output_pixels[labels == 0, :] = 0

                    parent_image = workspace.image_set.get_image(
                        heatmap.image_name.get_image_name()
                    )

                    output_img = Image(output_pixels, parent_image=parent_image)

                    img_name = heatmap.display_name.value

                    workspace.image_set.add(img_name, output_img)

    def do_measurements(
        self,
        workspace,
        image_name,
        object_name,
        center_object_name,
        center_choice,
        bin_count_settings,
        dd,
    ):
        """Perform the radial measurements on the image set

        workspace - workspace that holds images / objects
        image_name - make measurements on this image
        object_name - make measurements on these objects
        center_object_name - use the centers of these related objects as
                      the centers for radial measurements. None to use the
                      objects themselves.
        center_choice - the user's center choice for this object:
                      C_SELF, C_CENTERS_OF_OBJECTS or C_EDGES_OF_OBJECTS.
        bin_count_settings - the bin count settings group
        d - a dictionary for saving reusable partial results

        returns one statistics tuple per ring.
        """
        bin_count = bin_count_settings.bin_count.value

        wants_scaled = bin_count_settings.wants_scaled.value

        maximum_radius = bin_count_settings.maximum_radius.value

        image = workspace.image_set.get_image(image_name, must_be_grayscale=True)

        objects = workspace.object_set.get_objects(object_name)

        labels, pixel_data = crop_labels_and_image(objects.segmented, image.pixel_data)

        nobjects = numpy.max(objects.segmented)

        measurements = workspace.measurements

        name = (
            object_name
            if center_object_name is None
            else "{}_{}".format(object_name, center_object_name)
        )

        if name in dd:
            normalized_distance, i_center, j_center, good_mask = dd[name]
        else:
            d_to_edge = centrosome.cpmorphology.distance_to_edge(labels)

            if center_object_name is not None:
                #
                # Use the center of the centering objects to assign a center
                # to each labeled pixel using propagation
                #
                center_objects = workspace.object_set.get_objects(center_object_name)

                center_labels, cmask = size_similarly(labels, center_objects.segmented)

                pixel_counts = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.sum(
                        numpy.ones(center_labels.shape),
                        center_labels,
                        numpy.arange(
                            1, numpy.max(center_labels) + 1, dtype=numpy.int32
                        ),
                    )
                )

                good = pixel_counts > 0

                i, j = (
                    centrosome.cpmorphology.centers_of_labels(center_labels) + 0.5
                ).astype(int)

                ig = i[good]

                jg = j[good]

                lg = numpy.arange(1, len(i) + 1)[good]

                if center_choice == C_CENTERS_OF_OTHER:
                    #
                    # Reduce the propagation labels to the centers of
                    # the centering objects
                    #
                    center_labels = numpy.zeros(center_labels.shape, int)

                    center_labels[ig, jg] = lg

                cl, d_from_center = centrosome.propagate.propagate(
                    numpy.zeros(center_labels.shape), center_labels, labels != 0, 1
                )

                #
                # Erase the centers that fall outside of labels
                #
                cl[labels == 0] = 0

                #
                # If objects are hollow or crescent-shaped, there may be
                # objects without center labels. As a backup, find the
                # center that is the closest to the center of mass.
                #
                missing_mask = (labels != 0) & (cl == 0)

                missing_labels = numpy.unique(labels[missing_mask])

                if len(missing_labels):
                    all_centers = centrosome.cpmorphology.centers_of_labels(labels)

                    missing_i_centers, missing_j_centers = all_centers[
                        :, missing_labels - 1
                    ]

                    di = missing_i_centers[:, numpy.newaxis] - ig[numpy.newaxis, :]

                    dj = missing_j_centers[:, numpy.newaxis] - jg[numpy.newaxis, :]

                    missing_best = lg[numpy.argsort(di * di + dj * dj)[:, 0]]

                    best = numpy.zeros(numpy.max(labels) + 1, int)

                    best[missing_labels] = missing_best

                    cl[missing_mask] = best[labels[missing_mask]]

                    #
                    # Now compute the crow-flies distance to the centers
                    # of these pixels from whatever center was assigned to
                    # the object.
                    #
                    iii, jjj = numpy.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]

                    di = iii[missing_mask] - i[cl[missing_mask] - 1]

                    dj = jjj[missing_mask] - j[cl[missing_mask] - 1]

                    d_from_center[missing_mask] = numpy.sqrt(di * di + dj * dj)
            else:
                # Find the point in each object farthest away from the edge.
                # This does better than the centroid:
                # * The center is within the object
                # * The center tends to be an interesting point, like the
                #   center of the nucleus or the center of one or the other
                #   of two touching cells.
                #
                i, j = centrosome.cpmorphology.maximum_position_of_labels(
                    d_to_edge, labels, objects.indices
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

            if wants_scaled:
                total_distance = d_from_center + d_to_edge

                normalized_distance[good_mask] = d_from_center[good_mask] / (
                    total_distance[good_mask] + 0.001
                )
            else:
                normalized_distance[good_mask] = (
                    d_from_center[good_mask] / maximum_radius
                )

            dd[name] = [normalized_distance, i_center, j_center, good_mask]

        ngood_pixels = numpy.sum(good_mask)

        good_labels = labels[good_mask]

        bin_indexes = (normalized_distance * bin_count).astype(int)

        bin_indexes[bin_indexes > bin_count] = bin_count

        labels_and_bins = (good_labels - 1, bin_indexes[good_mask])

        histogram = scipy.sparse.coo_matrix(
            (pixel_data[good_mask], labels_and_bins), (nobjects, bin_count + 1)
        ).toarray()

        sum_by_object = numpy.sum(histogram, 1)

        sum_by_object_per_bin = numpy.dstack([sum_by_object] * (bin_count + 1))[0]

        fraction_at_distance = histogram / sum_by_object_per_bin

        number_at_distance = scipy.sparse.coo_matrix(
            (numpy.ones(ngood_pixels), labels_and_bins), (nobjects, bin_count + 1)
        ).toarray()

        object_mask = number_at_distance > 0

        sum_by_object = numpy.sum(number_at_distance, 1)

        sum_by_object_per_bin = numpy.dstack([sum_by_object] * (bin_count + 1))[0]

        fraction_at_bin = number_at_distance / sum_by_object_per_bin

        mean_pixel_fraction = fraction_at_distance / (
            fraction_at_bin + numpy.finfo(float).eps
        )

        masked_fraction_at_distance = numpy.ma.masked_array(
            fraction_at_distance, ~object_mask
        )

        masked_mean_pixel_fraction = numpy.ma.masked_array(
            mean_pixel_fraction, ~object_mask
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

        radial_index = (
            imask.astype(int) + jmask.astype(int) * 2 + absmask.astype(int) * 4
        )

        statistics = []

        for bin in range(bin_count + (0 if wants_scaled else 1)):
            bin_mask = good_mask & (bin_indexes == bin)

            bin_pixels = numpy.sum(bin_mask)

            bin_labels = labels[bin_mask]

            bin_radial_index = radial_index[bin_indexes[good_mask] == bin]

            labels_and_radii = (bin_labels - 1, bin_radial_index)

            radial_values = scipy.sparse.coo_matrix(
                (pixel_data[bin_mask], labels_and_radii), (nobjects, 8)
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
                    measurement_name = overflow_feature % image_name
                else:
                    measurement_name = feature % (image_name, bin + 1, bin_count)

                measurements.add_measurement(object_name, measurement_name, measurement)

                if feature in heatmaps:
                    heatmaps[feature][bin_mask] = measurement[bin_labels - 1]

            radial_cv.mask = numpy.sum(~mask, 1) == 0

            bin_name = str(bin + 1) if bin < bin_count else "Overflow"

            statistics += [
                (
                    image_name,
                    object_name,
                    bin_name,
                    str(bin_count),
                    numpy.round(numpy.mean(masked_fraction_at_distance[:, bin]), 4),
                    numpy.round(numpy.mean(masked_mean_pixel_fraction[:, bin]), 4),
                    numpy.round(numpy.mean(radial_cv), 4),
                )
            ]

        return statistics

def calculate_zernikes(pixels, mask, zernike_degree: int):
    zernike_indexes = centrosome.zernike.get_zernike_indexes(
        zernike_degree + 1
    )

    labels = mask.astype(int) # Convert boolean mask to labels
    for o in self.objects:
        object_name = o.object_name.value

        objects = workspace.object_set.get_objects(object_name)

        #
        # First, get a table of centers and radii of minimum enclosing
        # circles per object
        #
        # ij = numpy.zeros((objects.count + 1, 2))

        # r = numpy.zeros(objects.count + 1)

        # MODIFIED: Delegate index generation to the minimum_enclosing_circle
        # TODO: Check that this has the correct dimensions
        ij, r = centrosome.cpmorphology.minimum_enclosing_circle(labels)

            # ij[indexes] = ij_

            # r[indexes] = r_

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

        results = {}
        for i, (n, m) in enumerate(zernike_indexes):
            vr = scipy.ndimage.sum(
                pixels[ijv[:, 0], ijv[:, 1]] * z_[:, i].real,
                # FIXME replace functionality of labels and index
                # labels=l_,
                # index=objects.indices,
            )

            vi = scipy.ndimage.sum(
                pixels[ijv[mask, 0], ijv[mask, 1]] * z_[:, i].imag,
                # labels=l_,
                # index=objects.indices,
            )

            magnitude = numpy.sqrt(vr * vr + vi * vi) / area

            # ftr = self.get_zernike_magnitude_name(image_name, n, m)

            # meas[object_name, ftr] = magnitude
            results[f"ZernikeMagnitude_{n}_{m}"] = magnitude
             
            if self.wants_zernikes == Z_MAGNITUDES_AND_PHASE:
                phase = numpy.arctan2(vr, vi)

                ftr = self.get_zernike_phase_name(image_name, n, m)

                meas[object_name, ftr] = phase
        return results
                        
def boolean_mask_to_ijv(mask:numpy.ndarray) -> numpy.ndarray:
    """
    input: 2d boolean array
    output: (n, 3) integer array following (i,j,1)
    """

    # Extract coordinates of object from boolean mask
    i,j = numpy.where(mask)
    n = len(i)
    ijv = np.ones((n,n,n), dtype=int)
    ijv[:,0] = i
    ijv[:,1] = j
    return ijv
            

def get_zernike_magnitude_name(image_name, n, m):
    """The feature name of the magnitude of a Zernike moment

    image_name - the name of the image being measured
    n - the radial moment of the Zernike
    m - the azimuthal moment of the Zernike
    """
    return "_".join((M_CATEGORY, FF_ZERNIKE_MAGNITUDE, image_name, str(n), str(m)))

def get_zernike_phase_name(image_name, n, m):
    """The feature name of the phase of a Zernike moment

    image_name - the name of the image being measured
    n - the radial moment of the Zernike
    m - the azimuthal moment of the Zernike
    """
    return "_".join((M_CATEGORY, FF_ZERNIKE_PHASE, image_name, str(n), str(m)))

    def get_measurement_columns(self, pipeline):
        columns = []

        for image_name in self.images_list.value:
            for o in self.objects:
                object_name = o.object_name.value

                for bin_count_obj in self.bin_counts:
                    bin_count = bin_count_obj.bin_count.value

                    wants_scaling = bin_count_obj.wants_scaled.value

                    for feature, ofeature in (
                        (MF_FRAC_AT_D, OF_FRAC_AT_D),
                        (MF_MEAN_FRAC, OF_MEAN_FRAC),
                        (MF_RADIAL_CV, OF_RADIAL_CV),
                    ):
                        for bin in range(1, bin_count + 1):
                            columns.append(
                                (
                                    object_name,
                                    feature % (image_name, bin, bin_count),
                                    COLTYPE_FLOAT,
                                )
                            )

                        if not wants_scaling:
                            columns.append(
                                (
                                    object_name,
                                    ofeature % image_name,
                                    COLTYPE_FLOAT,
                                )
                            )

                    if self.wants_zernikes != Z_NONE:
                        name_fns = [self.get_zernike_magnitude_name]

                        if self.wants_zernikes == Z_MAGNITUDES_AND_PHASE:
                            name_fns.append(self.get_zernike_phase_name)

                        max_n = self.zernike_degree.value

                        for name_fn in name_fns:
                            for n, m in centrosome.zernike.get_zernike_indexes(
                                max_n + 1
                            ):
                                ftr = name_fn(image_name, n, m)

                                columns.append(
                                    (
                                        object_name,
                                        ftr,
                                        COLTYPE_FLOAT,
                                    )
                                )

        return columns

    def get_categories(self, pipeline, object_name):
        if object_name in [x.object_name.value for x in self.objects]:
            return [M_CATEGORY]

        return []

    def get_measurements(self, pipeline, object_name, category):
        if category in self.get_categories(pipeline, object_name):
            if self.wants_zernikes == Z_NONE:
                return F_ALL

            if self.wants_zernikes == Z_MAGNITUDES:
                return F_ALL + [FF_ZERNIKE_MAGNITUDE]

            return F_ALL + [FF_ZERNIKE_MAGNITUDE, FF_ZERNIKE_PHASE]

        return []

    def get_measurement_images(self, pipeline, object_name, category, feature):
        if feature in self.get_measurements(pipeline, object_name, category):
            return self.images_list.value
        return []

    def get_measurement_scales(
        self, pipeline, object_name, category, feature, image_name
    ):
        if image_name in self.get_measurement_images(
            pipeline, object_name, category, feature
        ):
            if feature in (FF_ZERNIKE_MAGNITUDE, FF_ZERNIKE_PHASE):
                n_max = self.zernike_degree.value

                result = [
                    "{}_{}".format(n, m)
                    for n, m in centrosome.zernike.get_zernike_indexes(n_max + 1)
                ]
            else:
                result = [
                    FF_SCALE % (bin, bin_count.bin_count.value)
                    for bin_count in self.bin_counts
                    for bin in range(1, bin_count.bin_count.value + 1)
                ]

                if any(
                    [not bin_count.wants_scaled.value for bin_count in self.bin_counts]
                ):
                    result += [FF_OVERFLOW]

            return result

        return []
