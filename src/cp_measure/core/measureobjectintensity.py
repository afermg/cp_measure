import numpy
import scipy.ndimage
import skimage.segmentation
from numpy.typing import NDArray

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
C_LOCATION = "Location"
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


def get_intensity(
    masks: NDArray[numpy.integer],
    pixels: NDArray[numpy.floating],
    edge_measurements: bool = True,
) -> dict[str, NDArray[numpy.floating]]:
    """Per-object intensity features.

    Walks each object on its `scipy.ndimage.find_objects` bounding box
    rather than the full image, and uses batched `scipy.ndimage` calls
    for whole-image stats (max position, intensity-weighted and geometric
    centers of mass). 2D inputs are promoted to ``(1, Y, X)`` so the same
    code path handles 2D and 3D; for 3D inputs Z-axis locations are filled
    in `CenterMassIntensity_Z` / `MaxIntensity_Z`.
    """
    # Normalize to (Z, Y, X). Broadcast 2D masks across a 3D pixel stack
    # to match the original behavior.
    if pixels.ndim == 3 and masks.ndim == 2:
        masks = numpy.broadcast_to(masks, pixels.shape)
    if pixels.ndim == 2:
        pixels = pixels[numpy.newaxis, ...]
    if masks.ndim == 2:
        masks = masks[numpy.newaxis, ...]

    if not numpy.issubdtype(masks.dtype, numpy.integer):
        masks = masks.astype(numpy.intp, copy=False)

    # find_objects returns one slice per label in 1..max(masks), with None
    # for missing labels. We use it both for bbox crops below and to derive
    # the present-label list — np.unique on a multi-MB mask is much slower
    # (e.g. ~200 ms on a 32x240x240 volume).
    bboxes = scipy.ndimage.find_objects(masks)
    present = [(i + 1, sl) for i, sl in enumerate(bboxes) if sl is not None]
    nobjects = len(present)

    integrated_intensity = numpy.zeros(nobjects)
    mean_intensity = numpy.zeros(nobjects)
    std_intensity = numpy.zeros(nobjects)
    min_intensity = numpy.zeros(nobjects)
    max_intensity = numpy.zeros(nobjects)
    lower_quartile_intensity = numpy.zeros(nobjects)
    median_intensity = numpy.zeros(nobjects)
    mad_intensity = numpy.zeros(nobjects)
    upper_quartile_intensity = numpy.zeros(nobjects)

    cmi_z = numpy.zeros(nobjects)
    cmi_y = numpy.zeros(nobjects)
    cmi_x = numpy.zeros(nobjects)
    max_z = numpy.zeros(nobjects)
    max_y = numpy.zeros(nobjects)
    max_x = numpy.zeros(nobjects)
    mass_displacement = numpy.zeros(nobjects)

    for out_i, (label_value, sl) in enumerate(present):
        mask_crop = masks[sl] == label_value
        pixels_crop = pixels[sl]
        finite = mask_crop & numpy.isfinite(pixels_crop)
        if not finite.any():
            continue
        vals = pixels_crop[finite]

        integrated = vals.sum()
        integrated_intensity[out_i] = integrated
        mean_intensity[out_i] = vals.mean()
        std_intensity[out_i] = vals.std()

        q = numpy.percentile(vals, [0, 25, 50, 75, 100])
        min_intensity[out_i] = q[0]
        lower_quartile_intensity[out_i] = q[1]
        median_intensity[out_i] = q[2]
        upper_quartile_intensity[out_i] = q[3]
        max_intensity[out_i] = q[4]
        mad_intensity[out_i] = numpy.median(numpy.abs(vals - q[2]))

        # Positions / centroids on the bbox crop, then shift by the bbox
        # offset. This avoids any full-image scipy.ndimage scans.
        coords = numpy.argwhere(finite)
        offset = numpy.array([s.start for s in sl], dtype=coords.dtype)
        coords = coords + offset

        argmax = int(numpy.argmax(vals))
        max_z[out_i] = coords[argmax, 0]
        max_y[out_i] = coords[argmax, 1]
        max_x[out_i] = coords[argmax, 2]

        cm = coords.mean(axis=0)
        if integrated != 0:
            cmi = (coords * vals[:, numpy.newaxis]).sum(axis=0) / integrated
        else:
            cmi = numpy.full(coords.shape[1], numpy.nan)
        cmi_z[out_i] = cmi[0]
        cmi_y[out_i] = cmi[1]
        cmi_x[out_i] = cmi[2]

        diff = cmi - cm
        mass_displacement[out_i] = numpy.sqrt((diff * diff).sum())

    result: dict[str, NDArray[numpy.floating]] = {
        f"{INTENSITY}_{INTEGRATED_INTENSITY}": integrated_intensity,
        f"{INTENSITY}_{MEAN_INTENSITY}": mean_intensity,
        f"{INTENSITY}_{STD_INTENSITY}": std_intensity,
        f"{INTENSITY}_{MIN_INTENSITY}": min_intensity,
        f"{INTENSITY}_{MAX_INTENSITY}": max_intensity,
        f"{INTENSITY}_{MASS_DISPLACEMENT}": mass_displacement,
        f"{INTENSITY}_{LOWER_QUARTILE_INTENSITY}": lower_quartile_intensity,
        f"{INTENSITY}_{MEDIAN_INTENSITY}": median_intensity,
        f"{INTENSITY}_{MAD_INTENSITY}": mad_intensity,
        f"{INTENSITY}_{UPPER_QUARTILE_INTENSITY}": upper_quartile_intensity,
        f"{C_LOCATION}_{LOC_CMI_X}": cmi_x,
        f"{C_LOCATION}_{LOC_CMI_Y}": cmi_y,
        f"{C_LOCATION}_{LOC_CMI_Z}": cmi_z,
        f"{C_LOCATION}_{LOC_MAX_X}": max_x,
        f"{C_LOCATION}_{LOC_MAX_Y}": max_y,
        f"{C_LOCATION}_{LOC_MAX_Z}": max_z,
    }

    if edge_measurements:
        integrated_intensity_edge = numpy.zeros(nobjects)
        mean_intensity_edge = numpy.zeros(nobjects)
        std_intensity_edge = numpy.zeros(nobjects)
        min_intensity_edge = numpy.zeros(nobjects)
        max_intensity_edge = numpy.zeros(nobjects)

        # One pass to compute the inner boundary of every object, then
        # index into the bbox crop per label.
        boundaries = skimage.segmentation.find_boundaries(masks, mode="inner")

        for out_i, (label_value, sl) in enumerate(present):
            mask_crop = masks[sl] == label_value
            pixels_crop = pixels[sl]
            edge = mask_crop & boundaries[sl] & numpy.isfinite(pixels_crop)
            if not edge.any():
                continue
            evals = pixels_crop[edge]
            integrated_intensity_edge[out_i] = evals.sum()
            mean_intensity_edge[out_i] = evals.mean()
            std_intensity_edge[out_i] = evals.std()
            min_intensity_edge[out_i] = evals.min()
            max_intensity_edge[out_i] = evals.max()

        result[f"{INTENSITY}_{INTEGRATED_INTENSITY_EDGE}"] = integrated_intensity_edge
        result[f"{INTENSITY}_{MEAN_INTENSITY_EDGE}"] = mean_intensity_edge
        result[f"{INTENSITY}_{STD_INTENSITY_EDGE}"] = std_intensity_edge
        result[f"{INTENSITY}_{MIN_INTENSITY_EDGE}"] = min_intensity_edge
        result[f"{INTENSITY}_{MAX_INTENSITY_EDGE}"] = max_intensity_edge

    return result
