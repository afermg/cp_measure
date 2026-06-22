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

from typing import Iterator

import numpy
from numpy.typing import NDArray
import scipy.ndimage
import scipy.stats
from scipy.linalg import lstsq


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


# ---------------------------------------------------------------------------
# Per-pair primitives (operate on already-extracted 1D pixel arrays)
# ---------------------------------------------------------------------------


def _pearson_pair(
    fi: NDArray[numpy.floating], si: NDArray[numpy.floating]
) -> tuple[float, float]:
    corr = numpy.corrcoef(fi, si)[0, 1]
    coeffs = lstsq(numpy.array((fi, numpy.ones_like(fi))).transpose(), si)[0]
    return float(corr), float(coeffs[0])


def _manders_pair(
    fi: NDArray[numpy.floating], si: NDArray[numpy.floating], thr: float
) -> tuple[float, float]:
    tff = (thr / 100) * fi.max()
    tss = (thr / 100) * si.max()
    combined = (fi >= tff) & (si >= tss)
    if not combined.any():
        return 0.0, 0.0
    tot_fi = fi[fi >= tff].sum()
    tot_si = si[si >= tss].sum()
    # Match historical behaviour: when tot is 0, propagate NaN.
    with numpy.errstate(invalid="ignore", divide="ignore"):
        M1 = float(fi[combined].sum() / tot_fi)
        M2 = float(si[combined].sum() / tot_si)
    return M1, M2


def _overlap_pair(
    fi: NDArray[numpy.floating], si: NDArray[numpy.floating], thr: float
) -> tuple[float, float, float]:
    tff = (thr / 100) * fi.max()
    tss = (thr / 100) * si.max()
    combined = (fi >= tff) & (si >= tss)
    if not combined.any():
        return 0.0, 0.0, 0.0
    fi_t = fi[combined]
    si_t = si[combined]
    fpsq = float(numpy.sum(fi_t * fi_t))
    spsq = float(numpy.sum(si_t * si_t))
    prod = float(numpy.sum(fi_t * si_t))
    pdt = numpy.sqrt(fpsq * spsq)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        overlap = prod / pdt
        K1 = prod / fpsq
        K2 = prod / spsq
    return overlap, K1, K2


def _rwc_pair(
    fi: NDArray[numpy.floating], si: NDArray[numpy.floating], thr: float
) -> tuple[float, float]:
    Rank1 = numpy.lexsort([fi])
    Rank2 = numpy.lexsort([si])
    Rank1_U = numpy.hstack([[False], fi[Rank1[:-1]] != fi[Rank1[1:]]])
    Rank2_U = numpy.hstack([[False], si[Rank2[:-1]] != si[Rank2[1:]]])
    Rank1_S = numpy.cumsum(Rank1_U)
    Rank2_S = numpy.cumsum(Rank2_U)
    Rank_im1 = numpy.zeros(fi.shape, dtype=int)
    Rank_im2 = numpy.zeros(si.shape, dtype=int)
    Rank_im1[Rank1] = Rank1_S
    Rank_im2[Rank2] = Rank2_S

    R = max(Rank_im1.max(), Rank_im2.max()) + 1
    Di = abs(Rank_im1 - Rank_im2)
    weight = (R - Di) * 1.0 / R

    tff = (thr / 100) * fi.max()
    tss = (thr / 100) * si.max()
    combined = (fi >= tff) & (si >= tss)
    if not combined.any():
        return 0.0, 0.0
    fi_t = fi[combined]
    si_t = si[combined]
    w_t = weight[combined]
    tot_fi = fi[fi >= tff].sum()
    tot_si = si[si >= tss].sum()
    with numpy.errstate(invalid="ignore", divide="ignore"):
        RWC1 = float((fi_t * w_t).sum() / tot_fi)
        RWC2 = float((si_t * w_t).sum() / tot_si)
    return RWC1, RWC2


def _costes_pair(
    fi: NDArray[numpy.floating],
    si: NDArray[numpy.floating],
    scale: int,
    fast_costes: str,
) -> tuple[float, float]:
    if fast_costes == M_FASTER:
        thr_fi, thr_si = bisection_costes(None, None, fi, si, scale, fast_costes)
    else:
        thr_fi, thr_si = linear_costes(None, None, fi, si, scale, fast_costes)
    fi_above = fi > thr_fi
    si_above = si > thr_si
    combined = fi_above & si_above
    if not combined.any():
        return 0.0, 0.0
    tot_fi = fi[fi >= thr_fi].sum()
    tot_si = si[si >= thr_si].sum()
    with numpy.errstate(invalid="ignore", divide="ignore"):
        C1 = float(fi[combined].sum() / tot_fi)
        C2 = float(si[combined].sum() / tot_si)
    return C1, C2


# ---------------------------------------------------------------------------
# Per-label pixel iteration using find_objects (bounding box slicing)
# ---------------------------------------------------------------------------


def _iter_label_pixels(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
) -> Iterator[
    tuple[NDArray[numpy.floating], NDArray[numpy.floating]] | tuple[None, None]
]:
    """Yield (fi, si) per label using bounding boxes; (None, None) for empty labels."""
    objects = scipy.ndimage.find_objects(masks)
    for label_idx, sl in enumerate(objects, start=1):
        if sl is None:
            yield None, None
            continue
        local = masks[sl] == label_idx
        fi = pixels_1[sl][local]
        si = pixels_2[sl][local]
        yield fi, si


# ---------------------------------------------------------------------------
# Public single-mask (binary) entry points — kept for API compatibility.
# These accept a boolean mask and return scalar features for that single object.
# ---------------------------------------------------------------------------


def extract_pixels(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    mask: NDArray[numpy.integer],
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    fi = pixels_1[mask]
    si = pixels_2[mask]
    labels = mask.astype(numpy.uint32)[mask]
    labels.setflags(write=False)
    lrange = numpy.arange(labels.max(), dtype=numpy.int32) + 1
    return fi, si, labels, lrange


def get_correlation_pearson_ind(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    mask: NDArray[numpy.integer],
) -> dict[str, float]:
    fi = pixels_1[mask]
    si = pixels_2[mask]
    corr, slope = _pearson_pair(fi, si)
    return {F_CORRELATION_FORMAT: corr, F_SLOPE_FORMAT: slope}


def get_correlation_manders_fold_ind(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    mask: NDArray[numpy.integer],
    thr: int = 15,
) -> dict[str, float]:
    fi = pixels_1[mask]
    si = pixels_2[mask]
    M1, M2 = _manders_pair(fi, si, thr)
    return {f"{F_MANDERS_FORMAT}_1": M1, f"{F_MANDERS_FORMAT}_2": M2}


def get_correlation_rwc_ind(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    mask: NDArray[numpy.integer],
    thr: int = 15,
) -> dict[str, float]:
    fi = pixels_1[mask]
    si = pixels_2[mask]
    RWC1, RWC2 = _rwc_pair(fi, si, thr)
    return {f"{F_RWC_FORMAT}_1": RWC1, f"{F_RWC_FORMAT}_2": RWC2}


def get_correlation_costes_ind(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    mask: NDArray[numpy.integer],
    fast_costes: str = M_FASTER,
    thr: int = 15,
) -> dict[str, float]:
    fi = pixels_1[mask]
    si = pixels_2[mask]
    scale = infer_scale(pixels_1)
    C1, C2 = _costes_pair(fi, si, scale, fast_costes)
    return {f"{F_COSTES_FORMAT}_1": C1, f"{F_COSTES_FORMAT}_2": C2}


def get_correlation_overlap_ind(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    mask: NDArray[numpy.integer],
    thr: int = 15,
) -> dict[str, float]:
    fi = pixels_1[mask]
    si = pixels_2[mask]
    overlap, K1, K2 = _overlap_pair(fi, si, thr)
    return {
        F_OVERLAP_FORMAT: overlap,
        f"{F_K_FORMAT}_1": K1,
        f"{F_K_FORMAT}_2": K2,
    }


# ---------------------------------------------------------------------------
# Costes' automated threshold (unchanged algorithms; first_pixels/second_pixels
# arguments are unused historically but kept for API compatibility).
# ---------------------------------------------------------------------------


def linear_costes(
    first_pixels: NDArray[numpy.floating] | None,
    second_pixels: NDArray[numpy.floating] | None,
    fi: NDArray[numpy.floating],
    si: NDArray[numpy.floating],
    scale_max: int = 255,
    fast_costes: str = M_FASTER,
) -> tuple[float, float]:
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
    return float(thr_fi_c), float(thr_si_c)


def bisection_costes(
    first_pixels: NDArray[numpy.floating] | None,
    second_pixels: NDArray[numpy.floating] | None,
    fi: NDArray[numpy.floating],
    si: NDArray[numpy.floating],
    scale_max: int = 255,
    fast_costes: str = M_FASTER,
) -> tuple[float, float]:
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
    left: float = 1
    right: float = scale_max
    mid: float = ((right - left) // (6 / 5)) + left
    lastmid: float = 0
    # Marks the value with the last positive R value.
    valid: float = 1

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

    return float(thr_fi_c), float(thr_si_c)


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


# ---------------------------------------------------------------------------
# Public multi-label entry points. These iterate per label using find_objects
# (bounding-box slicing) and use plain numpy reductions per label rather than
# scipy.ndimage.sum / maximum across the full label image.
# ---------------------------------------------------------------------------


def get_correlation_pearson(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
) -> dict[str, list[float]]:
    """Per-object Pearson correlation and slope between two channels.

    Assumes labels are the contiguous integers ``1..N``; call via a
    :mod:`cp_measure.bulk` ``get_*`` entry point or wrap with
    :func:`cp_measure._sanitize.sanitize` to handle gapped IDs.
    """
    corrs: list[float] = []
    slopes: list[float] = []
    for fi, si in _iter_label_pixels(pixels_1, pixels_2, masks):
        if fi is None or si is None or fi.size == 0:
            corrs.append(0.0)
            slopes.append(0.0)
            continue
        corr, slope = _pearson_pair(fi, si)
        corrs.append(corr)
        slopes.append(slope)
    return {F_CORRELATION_FORMAT: corrs, F_SLOPE_FORMAT: slopes}


def get_correlation_manders_fold(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    thr: int = 15,
) -> dict[str, list[float]]:
    """Per-object Manders fold coefficients between two channels.

    Assumes labels are the contiguous integers ``1..N``; call via a
    :mod:`cp_measure.bulk` ``get_*`` entry point or wrap with
    :func:`cp_measure._sanitize.sanitize` to handle gapped IDs.
    """
    m1_list: list[float] = []
    m2_list: list[float] = []
    for fi, si in _iter_label_pixels(pixels_1, pixels_2, masks):
        if fi is None or si is None or fi.size == 0:
            m1_list.append(0.0)
            m2_list.append(0.0)
            continue
        M1, M2 = _manders_pair(fi, si, thr)
        m1_list.append(M1)
        m2_list.append(M2)
    return {f"{F_MANDERS_FORMAT}_1": m1_list, f"{F_MANDERS_FORMAT}_2": m2_list}


def get_correlation_rwc(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    thr: int = 15,
) -> dict[str, list[float]]:
    """Per-object rank-weighted colocalization coefficients between two channels.

    Assumes labels are the contiguous integers ``1..N``; call via a
    :mod:`cp_measure.bulk` ``get_*`` entry point or wrap with
    :func:`cp_measure._sanitize.sanitize` to handle gapped IDs.
    """
    r1: list[float] = []
    r2: list[float] = []
    for fi, si in _iter_label_pixels(pixels_1, pixels_2, masks):
        if fi is None or si is None or fi.size == 0:
            r1.append(0.0)
            r2.append(0.0)
            continue
        RWC1, RWC2 = _rwc_pair(fi, si, thr)
        r1.append(RWC1)
        r2.append(RWC2)
    return {f"{F_RWC_FORMAT}_1": r1, f"{F_RWC_FORMAT}_2": r2}


def get_correlation_costes(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    fast_costes: str = M_FASTER,
    thr: int = 15,
) -> dict[str, list[float]]:
    """Per-object Costes colocalization coefficients between two channels.

    Assumes labels are the contiguous integers ``1..N``; call via a
    :mod:`cp_measure.bulk` ``get_*`` entry point or wrap with
    :func:`cp_measure._sanitize.sanitize` to handle gapped IDs.
    """
    scale = infer_scale(pixels_1)
    c1: list[float] = []
    c2: list[float] = []
    for fi, si in _iter_label_pixels(pixels_1, pixels_2, masks):
        if fi is None or si is None or fi.size == 0:
            c1.append(0.0)
            c2.append(0.0)
            continue
        C1, C2 = _costes_pair(fi, si, scale, fast_costes)
        c1.append(C1)
        c2.append(C2)
    return {f"{F_COSTES_FORMAT}_1": c1, f"{F_COSTES_FORMAT}_2": c2}


def get_correlation_overlap(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    thr: int = 15,
) -> dict[str, list[float]]:
    """Per-object overlap and k1/k2 colocalization coefficients between two channels.

    Assumes labels are the contiguous integers ``1..N``; call via a
    :mod:`cp_measure.bulk` ``get_*`` entry point or wrap with
    :func:`cp_measure._sanitize.sanitize` to handle gapped IDs.
    """
    overlap_list: list[float] = []
    k1_list: list[float] = []
    k2_list: list[float] = []
    for fi, si in _iter_label_pixels(pixels_1, pixels_2, masks):
        if fi is None or si is None or fi.size == 0:
            overlap_list.append(0.0)
            k1_list.append(0.0)
            k2_list.append(0.0)
            continue
        overlap, K1, K2 = _overlap_pair(fi, si, thr)
        overlap_list.append(overlap)
        k1_list.append(K1)
        k2_list.append(K2)
    return {
        F_OVERLAP_FORMAT: overlap_list,
        f"{F_K_FORMAT}_1": k1_list,
        f"{F_K_FORMAT}_2": k2_list,
    }
