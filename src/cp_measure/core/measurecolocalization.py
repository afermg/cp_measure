"""
MeasureColocalization
=====================

**MeasureColocalization** measures the colocalization and correlation
between intensities in different images (e.g., different color channels)
on a pixel-by-pixel basis, within identified objects or across an entire
image.

Given two or more images, this module calculates the correlation &
colocalization (Overlap, Manders, Costes’ Automated Threshold & Rank
Weighted Colocalization) between the pixel intensities. The correlation
/ colocalization can be measured for entire images, or a correlation
measurement can be made within each individual object. Correlations /
Colocalizations will be calculated between all pairs of images that are
selected in the module, as well as between selected objects. For
example, if correlations are to be measured for a set of red, green, and
blue images containing identified nuclei, measurements will be made
between the following:

-  The blue and green, red and green, and red and blue images.
-  The nuclei in each of the above image pairs.

A good primer on colocalization theory can be found on the `SVI website`_.

You can find a helpful review on colocalization from Aaron *et al*. `here`_.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Correlation:* The correlation between a pair of images *I* and *J*,
   calculated as Pearson’s correlation coefficient. The formula is
   covariance(\ *I* ,\ *J*)/[std(\ *I* ) × std(\ *J*)].
-  *Slope:* The slope of the least-squares regression between a pair of
   images I and J. Calculated using the model *A* × *I* + *B* = *J*, where *A* is the slope.
-  *Overlap coefficient:* The overlap coefficient is a modification of
   Pearson’s correlation where average intensity values of the pixels are
   not subtracted from the original intensity values. For a pair of
   images R and G, the overlap coefficient is measured as r = sum(Ri \*
   Gi) / sqrt (sum(Ri\*Ri)\*sum(Gi\*Gi)).
-  *Manders coefficient:* The Manders coefficient for a pair of images R
   and G is measured as M1 = sum(Ri_coloc)/sum(Ri) and M2 =
   sum(Gi_coloc)/sum(Gi), where Ri_coloc = Ri when Gi > 0, 0 otherwise
   and Gi_coloc = Gi when Ri >0, 0 otherwise.
-  *Manders coefficient (Costes Automated Threshold):* Costes’ automated
   threshold estimates maximum threshold of intensity for each image
   based on correlation. Manders coefficient is applied on thresholded
   images as Ri_coloc = Ri when Gi > Gthr and Gi_coloc = Gi when Ri >
   Rthr where Gthr and Rthr are thresholds calculated using Costes’
   automated threshold method.
-  *Rank Weighted Colocalization coefficient:* The RWC coefficient for a
   pair of images R and G is measured as RWC1 =
   sum(Ri_coloc\*Wi)/sum(Ri) and RWC2 = sum(Gi_coloc\*Wi)/sum(Gi),
   where Wi is Weight defined as Wi = (Rmax - Di)/Rmax where Rmax is the
   maximum of Ranks among R and G based on the max intensity, and Di =
   abs(Rank(Ri) - Rank(Gi)) (absolute difference in ranks between R and
   G) and Ri_coloc = Ri when Gi > 0, 0 otherwise and Gi_coloc = Gi
   when Ri >0, 0 otherwise. (Singan et al. 2011, BMC Bioinformatics
   12:407).

References
^^^^^^^^^^

-  Aaron JS, Taylor AB, Chew TL. Image co-localization - co-occurrence versus correlation.
   J Cell Sci. 2018;131(3):jcs211847. Published 2018 Feb 8. doi:10.1242/jcs.211847



.. _SVI website: http://svi.nl/ColocalizationTheory
.. _here: https://jcs.biologists.org/content/joces/131/3/jcs211847.full.pdf
"""

from functools import partial

import numpy
import scipy.ndimage
import scipy.stats
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from scipy.linalg import lstsq

from cp_measure.utils import labels_to_binmasks

M_IMAGES = "Across entire image"
M_OBJECTS = "Within objects"
M_IMAGES_AND_OBJECTS = "Both"

M_FAST = "Fast"
M_FASTER = "Faster"
M_ACCURATE = "Accurate"

"""Feature name format for the correlation measurement"""
# Modified Correlation-Pearson
F_CORRELATION_FORMAT = "Correlation_Pearson"

"""Feature name format for the slope measurement"""
F_SLOPE_FORMAT = "Correlation_Slope"

"""Feature name format for the overlap coefficient measurement"""
F_OVERLAP_FORMAT = "Correlation_Overlap"

"""Feature name format for the Manders Coefficient measurement"""
F_K_FORMAT = "Correlation_K"

"""Feature name format for the Manders Coefficient measurement"""
F_KS_FORMAT = "Correlation_KS"

"""Feature name format for the Manders Coefficient measurement"""
F_MANDERS_FORMAT = "Correlation_Manders"

"""Feature name format for the RWC Coefficient measurement"""
F_RWC_FORMAT = "Correlation_RWC"

"""Feature name format for the Costes Coefficient measurement"""
F_COSTES_FORMAT = "Correlation_Costes"


"""
thr : int or float, optional
    Set threshold as percentage of maximum intensity for the images (default 15).
    You may choose to measure colocalization metrics only for those pixels above 
    a certain threshold. Select the threshold as a percentage of the maximum intensity 
    of the above image [0-99].
    This value is used by the Overlap, Manders, and Rank Weighted Colocalization 
    measurements.

fast_costes : {M_FASTER, M_FAST, M_ACCURATE}, optional
    Method for Costes thresholding (default M_FASTER).
    This setting determines the method used to calculate the threshold for use within the
    Costes calculations. The *{M_FAST}* and *{M_ACCURATE}* modes will test candidate thresholds
    in descending order until the optimal threshold is reached. Selecting *{M_FAST}* will attempt 
    to skip candidates when results are far from the optimal value being sought. Selecting *{M_ACCURATE}* 
    will test every possible threshold value. When working with 16-bit images these methods can be extremely 
    time-consuming. Selecting *{M_FASTER}* will use a modified bisection algorithm to find the threshold 
    using a shrinking window of candidates. This is substantially faster but may produce slightly lower 
    thresholds in exceptional circumstances.
    In the vast majority of instances the results of all strategies should be identical. We recommend using 
    *{M_FAST}* mode when working with 8-bit images and *{M_FASTER}* mode when using 16-bit images.
    Alternatively, you may want to disable these specific measurements entirely 
    (available when "*Run All Metrics?*" is set to "*No*").
"""


def extract_pixels(
    pixels_1: numpy.ndarray, pixels_2: numpy.ndarray, mask: numpy.ndarray
):
    fi = pixels_1[mask]
    si = pixels_2[mask]
    labels = mask.astype(numpy.uint32)[mask]
    lrange = numpy.arange(labels.max(), dtype=numpy.int32) + 1
    return fi, si, labels, lrange


def calculate_threshold(
    pixels_1: numpy.ndarray,
    pixels_2: numpy.ndarray,
    mask: numpy.ndarray,
    thr: int,
):
    # Threshold as percentage of maximum intensity of objects in each channel
    first_pixels, second_pixels, labels, lrange = extract_pixels(
        pixels_1, pixels_2, mask
    )
    tff = (thr / 100) * fix(scipy.ndimage.maximum(first_pixels, labels, lrange))
    tss = (thr / 100) * fix(scipy.ndimage.maximum(second_pixels, labels, lrange))
    combined_thresh = (first_pixels >= tff[labels - 1]) & (
        second_pixels >= tss[labels - 1]
    )
    fi_thresh = first_pixels[combined_thresh]
    si_thresh = second_pixels[combined_thresh]
    tot_fi_thr = scipy.ndimage.sum(
        first_pixels[first_pixels >= tff[labels - 1]],
        labels[first_pixels >= tff[labels - 1]],
        lrange,
    )
    tot_si_thr = scipy.ndimage.sum(
        second_pixels[second_pixels >= tss[labels - 1]],
        labels[second_pixels >= tss[labels - 1]],
        lrange,
    )
    return fi_thresh, si_thresh, tot_fi_thr, tot_si_thr, combined_thresh


def get_correlation_pearson_ind(
    pixels_1: numpy.ndarray, pixels_2: numpy.ndarray, mask: numpy.ndarray
) -> dict[str, float]:
    fi, si, _, _ = extract_pixels(pixels_1, pixels_2, mask)
    #
    # Perform the correlation, which returns:
    # [ [ii, ij],
    #   [ji, jj] ]
    #
    corr = numpy.corrcoef((fi, si))[1, 0]
    #
    # Find the slope as a linear regression to
    # A * i1 + B = i2
    #
    coeffs = lstsq(numpy.array((fi, numpy.ones_like(fi))).transpose(), si)[0]
    slope = coeffs[0]
    return {F_CORRELATION_FORMAT: corr, F_SLOPE_FORMAT: slope}


def get_correlation_manders_fold_ind(
    pixels_1: numpy.ndarray, pixels_2: numpy.ndarray, mask: numpy.ndarray, thr: int = 15
) -> dict[str, float]:
    first_pixels, second_pixels, labels, lrange = extract_pixels(
        pixels_1, pixels_2, mask
    )
    fi_thresh, si_thresh, tot_fi_thr, tot_si_thr, combined_thresh = calculate_threshold(
        pixels_1, pixels_2, mask, thr
    )
    # Manders Coefficient
    M1 = 0.0
    M2 = 0.0
    if combined_thresh.any():
        M1 = numpy.array(
            scipy.ndimage.sum(fi_thresh, labels[combined_thresh], lrange)
        ) / numpy.array(tot_fi_thr)
        M2 = numpy.array(
            scipy.ndimage.sum(si_thresh, labels[combined_thresh], lrange)
        ) / numpy.array(tot_si_thr)

        # TODO remove this to support multiple labels
        M1 = M1[0]
        M2 = M2[0]

    return {
        f"{F_MANDERS_FORMAT}_1": M1,
        f"{F_MANDERS_FORMAT}_2": M2,
    }


def get_correlation_rwc_ind(
    pixels_1: numpy.ndarray, pixels_2: numpy.ndarray, mask: numpy.ndarray, thr: int = 15
) -> dict[str, float]:
    first_pixels, second_pixels, labels, lrange = extract_pixels(
        pixels_1, pixels_2, mask
    )
    fi_thresh, si_thresh, tot_fi_thr, tot_si_thr, combined_thresh = calculate_threshold(
        pixels_1, pixels_2, mask, thr
    )

    # RWC Coefficient
    RWC1 = numpy.zeros(len(lrange))
    RWC2 = numpy.zeros(len(lrange))
    [Rank1] = numpy.lexsort(([labels], [first_pixels]))
    [Rank2] = numpy.lexsort(([labels], [second_pixels]))
    Rank1_U = numpy.hstack(
        [
            [False],
            first_pixels[Rank1[:-1]] != first_pixels[Rank1[1:]],
        ]
    )
    Rank2_U = numpy.hstack(
        [
            [False],
            second_pixels[Rank2[:-1]] != second_pixels[Rank2[1:]],
        ]
    )
    Rank1_S = numpy.cumsum(Rank1_U)
    Rank2_S = numpy.cumsum(Rank2_U)
    Rank_im1 = numpy.zeros(first_pixels.shape, dtype=int)
    Rank_im2 = numpy.zeros(second_pixels.shape, dtype=int)
    Rank_im1[Rank1] = Rank1_S
    Rank_im2[Rank2] = Rank2_S

    R = max(Rank_im1.max(), Rank_im2.max()) + 1
    Di = abs(Rank_im1 - Rank_im2)
    weight = (R - Di) * 1.0 / R
    weight_thresh = weight[combined_thresh]

    RWC1 = 0.0
    RWC2 = 0.0
    if combined_thresh.any():  # TODO adjust this to support multiple labels
        RWC1 = numpy.array(
            scipy.ndimage.sum(
                fi_thresh * weight_thresh, labels[combined_thresh], lrange
            )
        ) / numpy.array(tot_fi_thr)
        RWC2 = numpy.array(
            scipy.ndimage.sum(
                si_thresh * weight_thresh, labels[combined_thresh], lrange
            )
        ) / numpy.array(tot_si_thr)
        RWC1 = RWC1[0]
        RWC2 = RWC2[0]

    return {
        f"{F_RWC_FORMAT}_1": RWC1,
        f"{F_RWC_FORMAT}_2": RWC2,
    }


def get_correlation_costes_ind(
    pixels_1: numpy.ndarray,
    pixels_2: numpy.ndarray,
    mask: numpy.ndarray,
    fast_costes: str = M_FASTER,
    thr: int = 15,
):
    # Orthogonal Regression for Costes' automated threshold
    first_pixels, second_pixels, labels, lrange = extract_pixels(
        pixels_1, pixels_2, mask
    )
    fi_thresh, si_thresh, tot_fi_thr, tot_si_thr, combined_thresh = calculate_threshold(
        pixels_1, pixels_2, mask, thr
    )
    # MODIFIED: Removed the section that combines both images
    # MODIFIED: Added dimension because algorithms expect fi,si to be 3D
    scale = infer_scale(pixels_1)
    if fast_costes == M_FASTER:
        thr_fi_c, thr_si_c = bisection_costes(
            pixels_1, pixels_2, first_pixels, second_pixels, scale, fast_costes
        )
    else:
        thr_fi_c, thr_si_c = linear_costes(
            pixels_1, pixels_2, first_pixels, second_pixels, scale
        )

    # Costes' thershold for entire image is applied to each object
    fi_above_thr = first_pixels > thr_fi_c
    si_above_thr = second_pixels > thr_si_c
    combined_thresh_c = fi_above_thr & si_above_thr
    fi_thresh_c = first_pixels[combined_thresh_c]
    si_thresh_c = second_pixels[combined_thresh_c]

    tot_fi_thr_c = numpy.zeros(len(lrange))
    tot_si_thr_c = numpy.zeros(len(lrange))

    if numpy.any(fi_above_thr):
        tot_fi_thr_c = scipy.ndimage.sum(
            first_pixels[first_pixels >= thr_fi_c],
            labels[first_pixels >= thr_fi_c],
            lrange,
        )

    if numpy.any(si_above_thr):
        tot_si_thr_c = scipy.ndimage.sum(
            second_pixels[second_pixels >= thr_si_c],
            labels[second_pixels >= thr_si_c],
            lrange,
        )

    # Costes Automated Threshold
    C1, C2 = ([0.0], [0.0])  # Cover fringe case of no pixels above threshold
    if len(fi_thresh_c) and len(si_thresh_c):
        C1 = numpy.array(
            scipy.ndimage.sum(fi_thresh_c, labels[combined_thresh_c], lrange)
        ) / numpy.array(tot_fi_thr_c)
        C2 = numpy.array(
            scipy.ndimage.sum(si_thresh_c, labels[combined_thresh_c], lrange)
        ) / numpy.array(tot_si_thr_c)

    return {
        f"{F_COSTES_FORMAT}_1": C1[0],
        f"{F_COSTES_FORMAT}_2": C2[0],
    }


def linear_costes(
    first_pixels: numpy.ndarray,
    second_pixels: numpy.ndarray,
    fi: numpy.ndarray,
    si: numpy.ndarray,
    scale_max: int = 255,
    fast_costes: str = M_FASTER,
):
    """
    Finds the Costes Automatic Threshold for colocalization using a linear algorithm.
    Candiate thresholds are gradually decreased until Pearson R falls below 0.
    If "Fast" mode is enabled the "steps" between tested thresholds will be increased
    when Pearson R is much greater than 0.
    """
    i_step = 1 / scale_max
    non_zero = (fi > 0) | (si > 0)
    xvar = numpy.var(fi[non_zero], axis=0, ddof=1)
    yvar = numpy.var(si[non_zero], axis=0, ddof=1)

    xmean = numpy.mean(fi[non_zero], axis=0)
    ymean = numpy.mean(si[non_zero], axis=0)

    z = fi[non_zero] + si[non_zero]
    zvar = numpy.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + numpy.sqrt(
        (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar)
    )
    a = num / denom
    b = ymean - a * xmean

    # Start at 1 step above the maximum value
    img_max = max(fi.max(), si.max())
    i = i_step * ((img_max // i_step) + 1)

    num_true = None
    fi_max = fi.max()
    si_max = si.max()

    # Initialise without a threshold
    costReg, _ = scipy.stats.pearsonr(fi, si)
    thr_fi_c = i
    thr_si_c = (a * i) + b
    while i > fi_max and (a * i) + b > si_max:
        i -= i_step
    while i > i_step:
        thr_fi_c = i
        thr_si_c = (a * i) + b
        combt = (fi < thr_fi_c) | (si < thr_si_c)
        try:
            # Only run pearson if the input has changed.
            if (positives := numpy.count_nonzero(combt)) != num_true:
                costReg, _ = scipy.stats.pearsonr(fi[combt], si[combt])
                num_true = positives

            if costReg <= 0:
                break
            elif fast_costes == M_ACCURATE or i < i_step * 10:
                i -= i_step
            elif costReg > 0.45:
                # We're way off, step down 10x
                i -= i_step * 10
            elif costReg > 0.35:
                # Still far from 0, step 5x
                i -= i_step * 5
            elif costReg > 0.25:
                # Step 2x
                i -= i_step * 2
            else:
                i -= i_step
        except ValueError:
            break
    return thr_fi_c, thr_si_c


def bisection_costes(
    first_pixels: numpy.ndarray,
    second_pixels: numpy.ndarray,
    fi: numpy.ndarray,
    si: numpy.ndarray,
    scale_max: int = 255,
    fast_costes: str = M_FASTER,
):
    """
    Finds the Costes Automatic Threshold for colocalization using a bisection algorithm.
    Candidate thresholds are selected from within a window of possible intensities,
    this window is narrowed based on the R value of each tested candidate.
    We're looking for the first point below 0, and R value can become highly variable
    at lower thresholds in some samples. Therefore the candidate tested in each
    loop is 1/6th of the window size below the maximum value (as opposed to the midpoint).
    """

    non_zero = (fi > 0) | (si > 0)
    xvar = numpy.var(fi[non_zero], axis=0, ddof=1)
    yvar = numpy.var(si[non_zero], axis=0, ddof=1)

    xmean = numpy.mean(fi[non_zero], axis=0)
    ymean = numpy.mean(si[non_zero], axis=0)

    z = fi[non_zero] + si[non_zero]
    zvar = numpy.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + numpy.sqrt(
        (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar)
    )
    a = num / denom
    b = ymean - a * xmean

    # Initialise variables
    left = 1
    right = scale_max
    mid = ((right - left) // (6 / 5)) + left
    lastmid = 0
    # Marks the value with the last positive R value.
    valid = 1

    while lastmid != mid:
        thr_fi_c = mid / scale_max
        thr_si_c = (a * thr_fi_c) + b
        combt = (fi < thr_fi_c) | (si < thr_si_c)
        if numpy.count_nonzero(combt) <= 2:
            # Can't run pearson with only 2 values.
            left = mid - 1
        else:
            try:
                costReg, _ = scipy.stats.pearsonr(fi[combt], si[combt])
                if costReg < 0:
                    left = mid - 1
                elif costReg >= 0:
                    right = mid + 1
                    valid = mid
            except ValueError:
                # Catch misc Pearson errors with low sample numbers
                left = mid - 1
        lastmid = mid
        if right - left > 6:
            mid = ((right - left) // (6 / 5)) + left
        else:
            mid = ((right - left) // 2) + left

    thr_fi_c = (valid - 1) / scale_max
    thr_si_c = (a * thr_fi_c) + b

    return thr_fi_c, thr_si_c


def get_correlation_overlap_ind(
    pixels_1: numpy.ndarray,
    pixels_2: numpy.ndarray,
    mask: numpy.ndarray,
    thr: int = 15,
):
    first_pixels, second_pixels, labels, lrange = extract_pixels(
        pixels_1, pixels_2, mask
    )
    _, _, _, _, combined_thresh = calculate_threshold(pixels_1, pixels_2, mask, thr)
    # Overlap Coefficient
    K1 = 0.0
    K2 = 0.0
    if combined_thresh.any():  # TODO adjust for multiple labels
        fpsq = scipy.ndimage.sum(
            first_pixels[combined_thresh] ** 2,
            labels[combined_thresh],
            lrange,
        )
        spsq = scipy.ndimage.sum(
            second_pixels[combined_thresh] ** 2,
            labels[combined_thresh],
            lrange,
        )
        pdt = numpy.sqrt(numpy.array(fpsq) * numpy.array(spsq))

        overlap = fix(
            scipy.ndimage.sum(
                first_pixels[combined_thresh] * second_pixels[combined_thresh],
                labels[combined_thresh],
                lrange,
            )
            / pdt
        )
        K1 = fix(
            (
                scipy.ndimage.sum(
                    first_pixels[combined_thresh] * second_pixels[combined_thresh],
                    labels[combined_thresh],
                    lrange,
                )
            )
            / (numpy.array(fpsq))
        )
        K2 = fix(
            scipy.ndimage.sum(
                first_pixels[combined_thresh] * second_pixels[combined_thresh],
                labels[combined_thresh],
                lrange,
            )
            / numpy.array(spsq)
        )

        K1 = K1[0]
        K2 = K2[0]
    is_scalar = numpy.isscalar(K1)
    return {
        F_OVERLAP_FORMAT: overlap[0],
        f"{F_K_FORMAT}_1": K1 if is_scalar else K1[0],
        f"{F_K_FORMAT}_2": K2 if is_scalar else K2[0],
    }


# MODIFIED: This reproduces the behaviour of the block at
#  https://github.com/cellprofiler/CellProfiler/blob/450abdc2eaa0332cb6d1d4aaed4bf0a4b843368d/src/subpackages/core/cellprofiler_core/image/abstract_image/file/_file_image.py#L396-L405
def infer_scale(data: numpy.ndarray) -> int:
    if data.dtype in [numpy.int8, numpy.uint8]:
        scale = 255
    elif data.dtype in [numpy.int16, numpy.uint16]:
        scale = 65535
    elif data.dtype == numpy.int32:
        scale = 2**32 - 1
    elif data.dtype == numpy.uint32:
        scale = 2**32
    else:
        scale = 1

    return scale


## Define functions using skimage style
# The implementation of these correlation functions makes it irrelevant to vectorize the input.
def get_correlation_pearson(
    pixels_1: numpy.ndarray, pixels_2: numpy.ndarray, masks: numpy.ndarray
):
    return apply_correlation_fun(get_correlation_pearson_ind, pixels_1, pixels_2, masks)


def get_correlation_manders_fold(
    pixels_1: numpy.ndarray,
    pixels_2: numpy.ndarray,
    masks: numpy.ndarray,
    thr: int = 15,
):
    return apply_correlation_fun(
        get_correlation_manders_fold_ind, pixels_1, pixels_2, masks, thr=thr
    )


def get_correlation_rwc(
    pixels_1: numpy.ndarray,
    pixels_2: numpy.ndarray,
    masks: numpy.ndarray,
    thr: int = 15,
):
    return apply_correlation_fun(
        get_correlation_rwc_ind, pixels_1, pixels_2, masks, thr=thr
    )


def get_correlation_costes(
    pixels_1: numpy.ndarray,
    pixels_2: numpy.ndarray,
    masks: numpy.ndarray,
    fast_costes: str = M_FASTER,
    thr: int = 15,
):
    return apply_correlation_fun(
        get_correlation_costes_ind,
        pixels_1,
        pixels_2,
        masks,
        fast_costes=fast_costes,
        thr=thr,
    )


def get_correlation_overlap(
    pixels_1: numpy.ndarray,
    pixels_2: numpy.ndarray,
    masks: numpy.ndarray,
    thr: int = 15,
):
    return apply_correlation_fun(get_correlation_overlap_ind, pixels_1, pixels_2, masks)


# Helper functions


def apply_correlation_fun(
    corr_function,
    pixels_1: numpy.ndarray,
    pixels_2: numpy.ndarray,
    masks: numpy.ndarray,
    **kwargs,
):
    """
    Apply `corr_function` to the subsequent args and kwargs. It assumes that pixels (arrays containing images) are passed, as well as
    masks in a 2-d labels format. Any kwargs are passed to corr_functions.
    """
    results = []
    partial_corr_fun = partial(corr_function, pixels_1, pixels_2)
    for mask in labels_to_binmasks(masks):
        results.append(partial_corr_fun(mask, **kwargs))

    return {k: [item[k] for item in results] for k in results[0]}
