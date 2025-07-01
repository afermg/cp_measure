"""
MeasureObjectNeighbors
======================

**MeasureObjectNeighbors** calculates how many neighbors each object
has and records various properties about the neighbors’ relationships,
including the percentage of an object’s edge pixels that touch a
neighbor. Please note that the distances reported for object
measurements are center-to-center distances, not edge-to-edge distances.

Given an image with objects identified (e.g., nuclei or cells), this
module determines how many neighbors each object has. You can specify
the distance within which objects should be considered neighbors, or
that objects are only considered neighbors if they are directly
touching.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

See also
^^^^^^^^

See also the **Identify** modules.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Object measurements**

-  *NumberOfNeighbors:* Number of neighbor objects.
-  *PercentTouching:* Percent of the object’s boundary pixels that touch
   neighbors, after the objects have been expanded to the specified
   distance.
-  *FirstClosestObjectNumber:* The index of the closest object.
-  *FirstClosestDistance:* The distance to the closest object (in units
   of pixels), measured between object centers.
-  *SecondClosestObjectNumber:* The index of the second closest object.
-  *SecondClosestDistance:* The distance to the second closest object (in units
   of pixels), measured between object centers.
-  *AngleBetweenNeighbors:* The angle formed with the object center as
   the vertex and the first and second closest object centers along the
   vectors.

**Object relationships:** The identity of the neighboring objects, for
each object. Since per-object output is one-to-one and neighbors
relationships are often many-to-one, they may be saved as a separate
file in **ExportToSpreadsheet** by selecting *Object relationships* from
the list of objects to export.

Technical notes
^^^^^^^^^^^^^^^

Objects discarded via modules such as **IdentifyPrimaryObjects** or
**IdentifySecondaryObjects** will still register as neighbors for the
purposes of accurate measurement. For instance, if an object touches a
single object and that object had been discarded, *NumberOfNeighbors*
will be positive, but there may not be a corresponding
*ClosestObjectNumber*. This can be disabled in module settings.

"""

from typing import Union

import numpy
import scipy
import scipy.ndimage
import scipy.signal
import skimage.morphology
from centrosome.cpmorphology import centers_of_labels, strel_disk
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from centrosome.outline import outline

D_ADJACENT = "Adjacent"
D_EXPAND = "Expand until adjacent"
D_WITHIN = "Within a specified distance"
D_ALL = [D_ADJACENT, D_EXPAND, D_WITHIN]

M_NUMBER_OF_NEIGHBORS = "NumberOfNeighbors"
M_PERCENT_TOUCHING = "PercentTouching"
M_FIRST_CLOSEST_OBJECT_NUMBER = "FirstClosestObjectNumber"
M_FIRST_CLOSEST_DISTANCE = "FirstClosestDistance"
M_SECOND_CLOSEST_OBJECT_NUMBER = "SecondClosestObjectNumber"
M_SECOND_CLOSEST_DISTANCE = "SecondClosestDistance"
M_ANGLE_BETWEEN_NEIGHBORS = "AngleBetweenNeighbors"
M_ALL = [
    M_NUMBER_OF_NEIGHBORS,
    M_PERCENT_TOUCHING,
    M_FIRST_CLOSEST_OBJECT_NUMBER,
    M_FIRST_CLOSEST_DISTANCE,
    M_SECOND_CLOSEST_OBJECT_NUMBER,
    M_SECOND_CLOSEST_DISTANCE,
    M_ANGLE_BETWEEN_NEIGHBORS,
]

C_NEIGHBORS = "Neighbors"

S_EXPANDED = "Expanded"
S_ADJACENT = "Adjacent"


def measureobjectneighbors(
    masks1: numpy.ndarray,
    masks2: numpy.ndarray,
    distance_method: str = D_EXPAND,
    distance: int = 5,
) -> dict[str, list[Union[float, int]]]:
    """
    Calculate neighbors of objects based on different methods. Supports 3D masks.

    Parameters
    ----------
    masks1 : (array of integers) label masks to be used as reference
    masks2 : (array of integers) label masks to be used as potential neighbors
    distance_method : str
        There are several methods by which to determine whether objects are
        neighbors:

        -  *%(D_ADJACENT)s:* In this mode, two objects must have adjacent
           boundary pixels to be neighbors.
        -  *%(D_EXPAND)s:* The objects are expanded until all pixels on the
           object boundaries are touching another. Two objects are neighbors if
           any of their boundary pixels are adjacent after expansion.
        -  *%(D_WITHIN)s:* Each object is expanded by the number of pixels you
           specify. Two objects are neighbors if they have adjacent pixels after
           expansion. Note that *all* objects are expanded by this amount (e.g.,
           if this distance is set to 10, a pair of objects will count as
           neighbors if their edges are 20 pixels apart or closer).

        For *%(D_ADJACENT)s* and *%(D_EXPAND)s*, the
        *%(M_PERCENT_TOUCHING)s* measurement is the percentage of pixels on
        the boundary of an object that touch adjacent objects. For
        *%(D_WITHIN)s*, two objects are touching if any of their boundary
        pixels are adjacent after expansion and *%(M_PERCENT_TOUCHING)s*
        measures the percentage of boundary pixels of an *expanded* object that
        touch adjacent objects.
    distance : int
        Neighbor distance (used only when “D_WITHIN” is selected).
        The Neighbor distance is the number of pixels that each object is
        expanded for the neighbor calculation. Expanded objects that touch are
        considered neighbors.

    Returns
    -------
    List of dictionaries with overlap features.
    """
    dimensions = masks1.ndim
    # has_pixels = masks.any()
    labels = masks1
    neighbor_labels = masks2
    # neighbor_objects = workspace.object_set.get_objects(self.neighbors_name.value)
    # neighbor_labels = neighbor_objects.small_removed_segmented
    # # neighbor_kept_labels = neighbor_objects.segmented
    # assert isinstance(neighbor_objects, Objects)

    nobjects = numpy.max(labels)
    nneighbors = numpy.max(neighbor_labels)

    neighbor_count = numpy.zeros((nobjects,))
    pixel_count = numpy.zeros((nobjects,))
    first_object_number = numpy.zeros((nobjects,), int)
    second_object_number = numpy.zeros((nobjects,), int)
    first_x_vector = numpy.zeros((nobjects,))
    second_x_vector = numpy.zeros((nobjects,))
    first_y_vector = numpy.zeros((nobjects,))
    second_y_vector = numpy.zeros((nobjects,))
    angle = numpy.zeros((nobjects,))
    percent_touching = numpy.zeros((nobjects,))
    # expanded_labels = None

    if distance_method == D_EXPAND:
        # Find the i,j coordinates of the nearest foreground point
        # to every background point
        if dimensions == 2:
            i, j = scipy.ndimage.distance_transform_edt(
                labels == 0, return_distances=False, return_indices=True
            )
            # Assign each background pixel to the label of its nearest
            # foreground pixel. Assign label to label for foreground.
            labels = labels[i, j]
        else:
            k, i, j = scipy.ndimage.distance_transform_edt(
                labels == 0, return_distances=False, return_indices=True
            )
            labels = labels[k, i, j]
        # expanded_labels = labels  # for display
        distance = 1  # dilate once to make touching edges overlap
    # These scale assignments are never used
    # scale = S_EXPANDED
    # if neighbors_are_objects:
    #     neighbor_labels = labels.copy()
    elif distance_method == D_WITHIN:
        distance = distance
        # scale = str(distance)
    elif distance_method == D_ADJACENT:
        distance = 1
        # scale = S_ADJACENT
    else:
        raise ValueError("Unknown distance method: %s" % distance_method)
    # if nneighbors > (1 if neighbors_are_objects else 0):
    if nneighbors > 1:
        first_objects = []
        second_objects = []
        object_indexes = numpy.arange(nobjects, dtype=numpy.int32) + 1
        #
        # First, compute the first and second nearest neighbors,
        # and the angles between self and the first and second
        # nearest neighbors
        #
        ocenters = centers_of_labels(labels).transpose()
        ncenters = centers_of_labels(neighbor_labels).transpose()
        # This is not used even in original implementation
        # areas = fix(scipy.ndimage.sum(numpy.ones(labels.shape), labels, object_indexes))
        perimeter_outlines = outline(labels)
        perimeters = fix(
            scipy.ndimage.sum(
                numpy.ones(labels.shape), perimeter_outlines, object_indexes
            )
        )

        #
        # order[:,0] should be arange(nobjects)
        # order[:,1] should be the nearest neighbor
        # order[:,2] should be the next nearest neighbor
        #
        order = numpy.zeros((nobjects, min(nneighbors, 3)), dtype=numpy.uint32)
        j = numpy.arange(nneighbors)
        # (0, 1, 2) unless there are less than 3 neighbors
        partition_keys = tuple(range(min(nneighbors, 3)))
        for i in range(nobjects):
            dr = numpy.sqrt(
                (ocenters[i, 0] - ncenters[j, 0]) ** 2
                + (ocenters[i, 1] - ncenters[j, 1]) ** 2
            )
            order[i, :] = numpy.argpartition(dr, partition_keys)[:3]

        # first_neighbor = 1 if neighbors_are_objects else 0
        # neighbors are objects because
        first_neighbor = 1
        first_object_index = order[:, first_neighbor]
        first_x_vector = ncenters[first_object_index, 1] - ocenters[:, 1]
        first_y_vector = ncenters[first_object_index, 0] - ocenters[:, 0]
        if nneighbors > first_neighbor + 1:
            second_object_index = order[:, first_neighbor + 1]
            second_x_vector = ncenters[second_object_index, 1] - ocenters[:, 1]
            second_y_vector = ncenters[second_object_index, 0] - ocenters[:, 0]
            v1 = numpy.array((first_x_vector, first_y_vector))
            v2 = numpy.array((second_x_vector, second_y_vector))
            #
            # Project the unit vector v1 against the unit vector v2
            #
            dot = numpy.sum(v1 * v2, 0) / numpy.sqrt(
                numpy.sum(v1**2, 0) * numpy.sum(v2**2, 0)
            )
            angle = numpy.arccos(dot) * 180.0 / numpy.pi

        # Make the structuring element for dilation
        if dimensions == 2:
            strel = strel_disk(distance)
        else:
            strel = skimage.morphology.ball(distance)
        #
        # A little bigger one to enter into the border with a structure
        # that mimics the one used to create the outline
        #
        if dimensions == 2:
            strel_touching = strel_disk(distance + 0.5)
        else:
            strel_touching = skimage.morphology.ball(distance + 0.5)
        #
        # Get the extents for each object and calculate the patch
        # that excises the part of the image that is "distance"
        # away
        if dimensions == 2:
            i, j = numpy.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]

            minimums_i, maximums_i, _, _ = scipy.ndimage.extrema(
                i, labels, object_indexes
            )
            minimums_j, maximums_j, _, _ = scipy.ndimage.extrema(
                j, labels, object_indexes
            )

            minimums_i = numpy.maximum(fix(minimums_i) - distance, 0).astype(int)
            maximums_i = numpy.minimum(
                fix(maximums_i) + distance + 1, labels.shape[0]
            ).astype(int)
            minimums_j = numpy.maximum(fix(minimums_j) - distance, 0).astype(int)
            maximums_j = numpy.minimum(
                fix(maximums_j) + distance + 1, labels.shape[1]
            ).astype(int)
        else:
            k, i, j = numpy.mgrid[
                0 : labels.shape[0], 0 : labels.shape[1], 0 : labels.shape[2]
            ]

            minimums_k, maximums_k, _, _ = scipy.ndimage.extrema(
                k, labels, object_indexes
            )
            minimums_i, maximums_i, _, _ = scipy.ndimage.extrema(
                i, labels, object_indexes
            )
            minimums_j, maximums_j, _, _ = scipy.ndimage.extrema(
                j, labels, object_indexes
            )

            minimums_k = numpy.maximum(fix(minimums_k) - distance, 0).astype(int)
            maximums_k = numpy.minimum(
                fix(maximums_k) + distance + 1, labels.shape[0]
            ).astype(int)
            minimums_i = numpy.maximum(fix(minimums_i) - distance, 0).astype(int)
            maximums_i = numpy.minimum(
                fix(maximums_i) + distance + 1, labels.shape[1]
            ).astype(int)
            minimums_j = numpy.maximum(fix(minimums_j) - distance, 0).astype(int)
            maximums_j = numpy.minimum(
                fix(maximums_j) + distance + 1, labels.shape[2]
            ).astype(int)
        #
        # Loop over all objects
        # Calculate which ones overlap "index"
        # Calculate how much overlap there is of others to "index"
        #

        # This originally linked labels to post-filter labels
        # TODO replace with simpler procedure
        _, object_numbers = relate_labels(labels, labels)
        _, neighbor_numbers = relate_labels(neighbor_labels, neighbor_labels)
        for object_number in object_numbers:
            if object_number == 0:
                #
                # No corresponding object in small-removed. This means
                # that the object has no pixels, e.g., not renumbered.
                #
                continue
            index = object_number - 1
            if dimensions == 2:
                patch = labels[
                    minimums_i[index] : maximums_i[index],
                    minimums_j[index] : maximums_j[index],
                ]
                npatch = neighbor_labels[
                    minimums_i[index] : maximums_i[index],
                    minimums_j[index] : maximums_j[index],
                ]
            else:
                patch = labels[
                    minimums_k[index] : maximums_k[index],
                    minimums_i[index] : maximums_i[index],
                    minimums_j[index] : maximums_j[index],
                ]
                npatch = neighbor_labels[
                    minimums_k[index] : maximums_k[index],
                    minimums_i[index] : maximums_i[index],
                    minimums_j[index] : maximums_j[index],
                ]

            #
            # Find the neighbors
            #
            patch_mask = patch == (index + 1)
            if distance <= 5:
                extended = scipy.ndimage.binary_dilation(patch_mask, strel)
            else:
                extended = (
                    scipy.signal.fftconvolve(patch_mask, strel, mode="same") > 0.5
                )
            neighbors = numpy.unique(npatch[extended])
            neighbors = neighbors[neighbors != 0]
            # if neighbors_are_objects:
            # neighbors = neighbors[neighbors != object_number]
            neighbors = neighbors[neighbors != object_number]
            nc = len(neighbors)
            neighbor_count[index] = nc
            if nc > 0:
                first_objects.append(numpy.ones(nc, int) * object_number)
                second_objects.append(neighbors)
            #
            # Find the # of overlapping pixels. Dilate the neighbors
            # and see how many pixels overlap our image. Use a 3x3
            # structuring element to expand the overlapping edge
            # into the perimeter.
            #
            if dimensions == 2:
                outline_patch = (
                    perimeter_outlines[
                        minimums_i[index] : maximums_i[index],
                        minimums_j[index] : maximums_j[index],
                    ]
                    == object_number
                )
            else:
                outline_patch = (
                    perimeter_outlines[
                        minimums_k[index] : maximums_k[index],
                        minimums_i[index] : maximums_i[index],
                        minimums_j[index] : maximums_j[index],
                    ]
                    == object_number
                )
            # if neighbors_are_objects:
            extendme = (patch != 0) & (patch != object_number)
            if distance <= 5:
                extended = scipy.ndimage.binary_dilation(extendme, strel_touching)
            else:
                extended = (
                    scipy.signal.fftconvolve(extendme, strel_touching, mode="same")
                    > 0.5
                )
            # else:
            #     if distance <= 5:
            #         extended = scipy.ndimage.binary_dilation(
            #             (npatch != 0), strel_touching
            #         )
            #     else:
            #         extended = (
            #             scipy.signal.fftconvolve(
            #                 (npatch != 0), strel_touching, mode="same"
            #             )
            #             > 0.5
            #         )
            overlap = numpy.sum(outline_patch & extended)
            pixel_count[index] = overlap
        if sum([len(x) for x in first_objects]) > 0:
            first_objects = numpy.hstack(first_objects)
            reverse_object_numbers = numpy.zeros(
                max(numpy.max(object_numbers), numpy.max(first_objects)) + 1, int
            )
            reverse_object_numbers[object_numbers] = (
                numpy.arange(len(object_numbers)) + 1
            )
            first_objects = reverse_object_numbers[first_objects]

            second_objects = numpy.hstack(second_objects)
            reverse_neighbor_numbers = numpy.zeros(
                max(numpy.max(neighbor_numbers), numpy.max(second_objects)) + 1, int
            )
            reverse_neighbor_numbers[neighbor_numbers] = (
                numpy.arange(len(neighbor_numbers)) + 1
            )
            second_objects = reverse_neighbor_numbers[second_objects]
            to_keep = (first_objects > 0) & (second_objects > 0)
            first_objects = first_objects[to_keep]
            second_objects = second_objects[to_keep]
        else:
            first_objects = numpy.zeros(0, int)
            second_objects = numpy.zeros(0, int)
        percent_touching = pixel_count * 100 / perimeters
        object_indexes = object_numbers - 1
        neighbor_indexes = neighbor_numbers - 1
        #
        # Have to recompute nearest
        #
        # first_object_number = numpy.zeros(nkept_objects, int)
        # second_object_number = numpy.zeros(nkept_objects, int)
        first_object_number = numpy.zeros(nobjects, int)
        second_object_number = numpy.zeros(nobjects, int)
        # if nkept_objects > (1 if neighbors_are_objects else 0):
        di = (
            ocenters[object_indexes[:, numpy.newaxis], 0]
            - ncenters[neighbor_indexes[numpy.newaxis, :], 0]
        )
        dj = (
            ocenters[object_indexes[:, numpy.newaxis], 1]
            - ncenters[neighbor_indexes[numpy.newaxis, :], 1]
        )
        distance_matrix = numpy.sqrt(di * di + dj * dj)
        # distance_matrix[~has_pixels, :] = numpy.inf
        # distance_matrix[:, ~neighbor_has_pixels] = numpy.inf
        #
        # order[:,0] should be arange(nobjects)
        # order[:,1] should be the nearest neighbor
        # order[:,2] should be the next nearest neighbor
        #
        order = numpy.lexsort([distance_matrix]).astype(first_object_number.dtype)
        # Outcomments are due to these conditions being assumed
        # if neighbors_are_objects:
        # first_object_number[has_pixels] = order[has_pixels, 1] + 1
        # TODO check conditions in which they should be 0 and 1 or 0 and 2
        first_object_number = order[:, 0] + 1
        if order.shape[1] > 1:
            second_object_number = order[:, 1] + 1
        # else:
        #     first_object_number[has_pixels] = order[has_pixels, 0] + 1
        #     if order.shape[1] > 1:
        #         second_object_number[has_pixels] = order[has_pixels, 1] + 1
    else:
        object_indexes = object_numbers - 1
        neighbor_indexes = neighbor_numbers - 1
        first_objects = numpy.zeros(0, int)
        second_objects = numpy.zeros(0, int)
    #
    # Now convert all measurements from the small-removed to
    # the final number set.
    #
    neighbor_count = neighbor_count[object_indexes]
    # neighbor_count[~has_pixels] = 0
    percent_touching = percent_touching[object_indexes]
    # percent_touching[~has_pixels] = 0
    first_x_vector = first_x_vector[object_indexes]
    second_x_vector = second_x_vector[object_indexes]
    first_y_vector = first_y_vector[object_indexes]
    second_y_vector = second_y_vector[object_indexes]
    angle = angle[object_indexes]
    #
    # Record the measurements
    #
    # assert isinstance(workspace, Workspace)
    # m = workspace.measurements
    # assert isinstance(m, Measurements)
    # image_set = workspace.image_set
    features_and_data = [
        (M_NUMBER_OF_NEIGHBORS, neighbor_count),
        (M_FIRST_CLOSEST_OBJECT_NUMBER, first_object_number),
        (
            M_FIRST_CLOSEST_DISTANCE,
            numpy.sqrt(first_x_vector**2 + first_y_vector**2),
        ),
        (M_SECOND_CLOSEST_OBJECT_NUMBER, second_object_number),
        (
            M_SECOND_CLOSEST_DISTANCE,
            numpy.sqrt(second_x_vector**2 + second_y_vector**2),
        ),
        (M_ANGLE_BETWEEN_NEIGHBORS, angle),
        (M_PERCENT_TOUCHING, percent_touching),
    ]
    result = {}
    for feature_name, data in features_and_data:
        result[get_measurement_name(feature_name, distance_method, distance)] = data
    # TODO add related measurements
    # if len(first_objects) > 0:
    #     m.add_relate_measurement(
    #         self.module_num,
    #         NEIGHBORS,
    #         self.object_name.value,
    #         self.object_name.value
    #         if neighbors_are_objects
    #         else self.neighbors_name.value,
    #         m.image_set_number * numpy.ones(first_objects.shape, int),
    #         first_objects,
    #         m.image_set_number * numpy.ones(second_objects.shape, int),
    #         second_objects,
    #     )

    # labels = kept_labels
    # neighbor_labels = neighbor_kept_labels
    return result


def get_measurement_name(feature: str, distance_method: str, distance: int = 5):
    if distance_method == D_EXPAND:
        scale = S_EXPANDED
    elif distance_method == D_WITHIN:
        scale = str(distance)
    elif distance_method == D_ADJACENT:
        scale = S_ADJACENT
    return "_".join((C_NEIGHBORS, feature, scale))


def find_label_overlaps(parent_labels: numpy.ndarray, child_labels: numpy.ndarray):
    """
    Produces a matrix in which each row is a parent and each column is a child.


    Ported and modified from https://github.com/cellprofiler/CellProfiler/blob/main/src/subpackages/library/cellprofiler_library/functions/segmentation.py#L654
    """

    parent_count = numpy.max(parent_labels)
    child_count = numpy.max(child_labels)

    #
    # Only look at points that are labeled in parent and child
    #
    not_zero = (parent_labels > 0) & (child_labels > 0)
    not_zero_count = numpy.sum(not_zero)

    #
    # each row (axis = 0) is a parent
    # each column (axis = 1) is a child
    #
    return scipy.sparse.coo_matrix(
        (
            numpy.ones((not_zero_count,)),
            (parent_labels[not_zero], child_labels[not_zero]),
        ),
        shape=(parent_count + 1, child_count + 1),
    )


def relate_histogram(histogram: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return child counts and parents of children given a histogram

    histogram - histogram from histogram_from_ijv or histogram_from_labels
    """
    parent_count = histogram.shape[0] - 1

    parents_of_children = numpy.asarray(histogram.argmax(axis=0))
    if len(parents_of_children.shape) == 2:
        parents_of_children = numpy.squeeze(parents_of_children, axis=0)
    #
    # Create a histogram of # of children per parent
    children_per_parent = numpy.histogram(
        parents_of_children[1:], numpy.arange(parent_count + 2)
    )[0][1:]

    #
    # Make sure to remove the background elements at index 0
    #
    return children_per_parent, parents_of_children[1:]


def relate_labels(
    parent_labels: numpy.ndarray, child_labels: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    relate the object numbers in one label to those in another

    parent_labels - 2d label matrix of parent labels

    child_labels - 2d label matrix of child labels

    Returns two 1-d arrays. The first gives the number of children within
    each parent. The second gives the mapping of each child to its parent's
    object number.

    Ported from https://github.com/cellprofiler/CellProfiler/blob/main/src/subpackages/core/cellprofiler_core/object/_objects.py#L283
    """
    return relate_histogram(find_label_overlaps(parent_labels, child_labels))
