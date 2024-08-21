__doc__ = """\
MeasureTexture
==============

**MeasureTexture** measures the degree and nature of textures within
images and objects to quantify their roughness and smoothness.

This module measures intensity variations in grayscale images. An object or
entire image without much texture has a smooth appearance; an object or
image with a lot of texture will appear rough and show a wide variety of
pixel intensities.

Note that any input objects specified will have their texture measured
against *all* input images specified, which may lead to image-object
texture combinations that are unnecessary. If you do not want this
behavior, use multiple **MeasureTexture** modules to specify the
particular image-object measures that you want.

Note also that CellProfiler in all 2.X versions increased speed by binning 
the image into only 8 grayscale levels before calculating Haralick features;
in all 3.X CellProfiler versions the images were binned into 256 grayscale
levels. CellProfiler 4 allows you to select your own preferred number of
grayscale levels, but note that since we use a slightly different
implementation than CellProfiler 2 we do not guarantee concordance with
CellProfiler 2.X-generated texture values.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Haralick Features:* Haralick texture features are derived from the
   co-occurrence matrix, which contains information about how image
   intensities in pixels with a certain position in relation to each
   other occur together. **MeasureTexture** can measure textures at
   different scales; the scale you choose determines how the
   co-occurrence matrix is constructed. For example, if you choose a
   scale of 2, each pixel in the image (excluding some border pixels)
   will be compared against the one that is two pixels to the right.

   Thirteen measurements are then calculated for the image by performing
   mathematical operations on the co-occurrence matrix (the formulas can
   be found `here`_):

   -  *AngularSecondMoment:* Measure of image homogeneity. A higher
      value of this feature indicates that the intensity varies less in
      an image. Has a value of 1 for a uniform image.
   -  *Contrast:* Measure of local variation in an image, with 0 for a
      uniform image and a high value indicating a high degree of local
      variation.
   -  *Correlation:* Measure of linear dependency of intensity values in
      an image. For an image with large areas of similar intensities,
      correlation is much higher than for an image with noisier,
      uncorrelated intensities. Has a value of 1 or -1 for a perfectly
      positively or negatively correlated image, respectively.
   -  *Variance:* Measure of the variation of image intensity values.
      For an image with uniform intensity, the texture variance would be
      zero.
   -  *InverseDifferenceMoment:* Another feature to represent image
      contrast. Has a low value for inhomogeneous images, and a
      relatively higher value for homogeneous images.
   -  *SumAverage:* The average of the normalized grayscale image in the
      spatial domain.
   -  *SumVariance:* The variance of the normalized grayscale image in
      the spatial domain.
   -  *SumEntropy:* A measure of randomness within an image.
   -  *Entropy:* An indication of the complexity within an image. A
      complex image produces a high entropy value.
   -  *DifferenceVariance:* The image variation in a normalized
      co-occurrence matrix.
   -  *DifferenceEntropy:* Another indication of the amount of
      randomness in an image.
   -  *InfoMeas1:* A measure of the total amount of information contained
      within a region of pixels derived from the recurring spatial
      relationship between specific intensity values.
   -  *InfoMeas2:* An additional measure of the total amount of information
      contained within a region of pixels derived from the recurring spatial
      relationship between specific intensity values. It is a complementary
      value to InfoMeas1 and is on a different scale.

**Note**: each of the above measurements are computed for different 
'directions' in the image, specified by a series of correspondence vectors. 
These are indicated in the results table in the *scale* column as n_00, n_01,
n_02... for each scale *n*. In 2D, the directions and correspondence vectors *(y, x)* 
for each measurement are given below:

- _00 = horizontal -, 0 degrees   (0, 1)
- _01 = diagonal \\\\, 135 degrees or NW-SE   (1, 1)
- _02 = vertical \|, 90 degrees   (1, 0)
- _03 = diagonal /, 45 degrees or NE-SW  (1, -1)

When analyzing 3D images, there are 13 correspondence vectors *(y, x, z)*:

- (1, 0, 0)
- (1, 1, 0)
- (0, 1, 0)
- (1,-1, 0)
- (0, 0, 1)
- (1, 0, 1)
- (0, 1, 1)
- (1, 1, 1)
- (1,-1, 1)
- (1, 0,-1)
- (0, 1,-1)
- (1, 1,-1)
- (1,-1,-1)

In this case, an image makes understanding their directions easier. 
Imagine the origin (0, 0, 0) is at the upper left corner of the first image
in your z-stack. Yellow vectors fall along the axes, and pairs of vectors with 
matching colors are reflections of each other across the x axis. The two
images represent two views of the same vectors. Images made in `GeoGebra`_.

|MT_image0| |MT_image1|

Technical notes
^^^^^^^^^^^^^^^

To calculate the Haralick features, **MeasureTexture** normalizes the
co-occurrence matrix at the per-object level by basing the intensity
levels of the matrix on the maximum and minimum intensity observed
within each object. This is beneficial for images in which the maximum
intensities of the objects vary substantially because each object will
have the full complement of levels.

References
^^^^^^^^^^

-  Haralick RM, Shanmugam K, Dinstein I. (1973), “Textural Features for
   Image Classification” *IEEE Transaction on Systems Man, Cybernetics*,
   SMC-3(6):610-621. `(link) <https://doi.org/10.1109/TSMC.1973.4309314>`__

.. _here: http://murphylab.web.cmu.edu/publications/boland/boland_node26.html
.. _GeoGebra: https://www.geogebra.org/ 
.. |MT_image0| image:: {MEASURE_TEXTURE_3D_INFO}
.. |MT_image1| image:: {MEASURE_TEXTURE_3D_INFO2}
"""

import mahotas.features
import numpy
import skimage.exposure
import skimage.measure
import skimage.util

F_HARALICK = """AngularSecondMoment Contrast Correlation Variance
InverseDifferenceMoment SumAverage SumVariance SumEntropy Entropy
DifferenceVariance DifferenceEntropy InfoMeas1 InfoMeas2""".split()


def get_texture(
    pixels: numpy.ndarray,
    mask: numpy.ndarray = None,  # MODIFIED: if mask is not provided do whole image
    scale: int = 3,
    gray_levels: int = 256,
):
    """
    Parameters
    ----------
    gray_levels : int, optional (default is 256)
        Number of gray levels. Measuring at more levels gives you _potentially_
        more detailed information about your image, but at the cost of somewhat
        decreased processing speed (default is 256).
    texture_scale : int, optional (default is 3)
        You can specify the scale of texture to be measured, in pixel units; the
        texture scale is the distance between correlated intensities in the
        image. A higher number for the scale of texture measures larger patterns
        of texture whereas smaller numbers measure more localized patterns of
        texture. It is best to measure texture on a scale smaller than your
        objects’ sizes, so be sure that the value entered for scale of texture
        is smaller than most of your objects. For very small objects (smaller
        than the scale of texture you are measuring), the texture cannot be
        measured and will result in a undefined value in the output file.

    Returns
    -------
    Dictionary of floats.

    Notes
    -----
    Before processing, your image will be rescaled from its current pixel values
    to 0 - [gray levels - 1]. The texture features will then be calculated.

    In all CellProfiler 2 versions, this value was fixed at 8; in all
    CellProfiler 3 versions it was fixed at 256.  The minimum number of levels is
    2, the maximum is 256.
    """

    # Modified use pixels nality to determine the number of directions
    n_directions = 13 if pixels.ndim > 2 else 4

    # MODIFIED: We assume that the mask provided has the same shape
    # as pixels, thus no cropping is performed
    # MODIFIED: Perform image-wide operation if no mask is provided
    if mask is not None:
        pixels[~mask] = 0

    # mahotas.features.haralick bricks itself when provided a
    # dtype larger than uint8 (version 1.4.3)
    pixels = skimage.util.img_as_ubyte(pixels)
    if gray_levels != 256:
        pixels = skimage.exposure.rescale_intensity(
            pixels, in_range=(0, 255), out_range=(0, gray_levels - 1)
        ).astype(numpy.uint8)
    # MODIFIED: We assume only one mask/object
    # MODIFIED: Given that we are only using one mask we do not need a third dimension
    features = numpy.empty((n_directions, 13))

    try:
        features[:, :] = mahotas.features.haralick(
            pixels, distance=scale, ignore_zeros=True
        )
    except ValueError:
        features[:, :] = numpy.nan

        # MODIFIED: Reconstructed name:
        # Texture_{X}_{scale}_{distance_id}_{graylevels}
    results = {}
    for feature_name, values in zip(F_HARALICK, features):
        for distance_i, value in enumerate(values):
            results[
                "{}_{:d}_{:02d}_{:d}".format(
                    feature_name,
                    scale,
                    distance_i,
                    gray_levels,
                )
            ] = value

    return results
