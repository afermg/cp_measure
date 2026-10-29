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

from numbers import Integral
from typing import Iterator
import warnings

import numpy
from numpy.typing import NDArray
import scipy.ndimage
import scipy.stats

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
    # OLS slope of si on fi is cov/var — avoids a per-object lstsq + design-matrix build.
    slope = numpy.cov(fi, si)[0, 1] / numpy.var(fi, ddof=1)
    return float(corr), float(slope)


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


def _undefined_costes(reason: str) -> tuple[float, float]:
    warnings.warn(f"Undefined Costes result: {reason}", RuntimeWarning, stacklevel=3)
    return numpy.nan, numpy.nan


def _validate_costes_scale(scale_max: int) -> int:
    # A one-step grid silently collapses the search. Limit grids to the integer
    # precisions historically supported here and representable by float64.
    if (
        isinstance(scale_max, (bool, numpy.bool_))
        or not isinstance(scale_max, Integral)
        or not 2 <= scale_max <= 2**32 - 1
    ):
        raise ValueError("scale_max must be an integer between 2 and 2**32 - 1")
    return int(scale_max)


def _prepare_costes_inputs(pixels_1, pixels_2, scale_max):
    """Keep acquisition scale independent of the normalized array dtype."""
    first, second = numpy.asarray(pixels_1), numpy.asarray(pixels_2)
    if first.shape != second.shape:
        raise ValueError("Costes images must have the same shape")
    if scale_max is None:
        if first.dtype != second.dtype or first.dtype.kind != "u":
            raise ValueError(
                "Costes requires explicit scale_max for float, signed, or mixed-dtype "
                "inputs (e.g. scale_max=65535 for normalized 16-bit images)"
            )
        scale_max = numpy.iinfo(first.dtype).max
    scale = _validate_costes_scale(scale_max)
    converted = []
    for image in (first, second):
        if image.dtype.kind not in "uif":
            raise ValueError(
                "Costes pixels must be real integer or floating-point arrays"
            )
        values = image.astype(numpy.float64, copy=False)
        if image.dtype.kind in "ui":
            values = values / scale
        converted.append(values)
    return converted[0], converted[1], scale


def _valid_costes_pixels(fi, si) -> bool:
    if fi.shape != si.shape or fi.ndim != 1:
        raise ValueError("Costes threshold helpers require equal-shape 1D pixel arrays")
    if fi.size < 3 or not (numpy.isfinite(fi).all() and numpy.isfinite(si).all()):
        _undefined_costes("fewer than three paired pixels or non-finite signal")
        return False
    if fi.min() < 0 or si.min() < 0 or fi.max() > 1 or si.max() > 1:
        raise ValueError("Costes expects normalized pixels in [0, 1]; check scale_max")
    non_zero = (fi > 0) | (si > 0)
    if numpy.count_nonzero(non_zero) < 3:
        _undefined_costes("fewer than three nonzero paired pixels")
        return False
    if numpy.ptp(fi[non_zero]) == 0 or numpy.ptp(si[non_zero]) == 0:
        _undefined_costes("constant channel in the regression subset")
        return False
    return True


def _costes_background_correlation(fi, si, selected) -> float:
    if numpy.count_nonzero(selected) < 3:
        return numpy.nan
    first, second = fi[selected], si[selected]
    if numpy.ptp(first) == 0 or numpy.ptp(second) == 0:
        return numpy.nan
    return float(scipy.stats.pearsonr(first, second)[0])


def _finish_costes_thresholds(fi, si, first_threshold, second_threshold):
    selected = (fi < first_threshold) | (si < second_threshold)
    if not numpy.isfinite(_costes_background_correlation(fi, si, selected)):
        return _undefined_costes("chosen background subset has undefined correlation")
    return float(first_threshold), float(second_threshold)


def _costes_pair(
    fi: NDArray[numpy.floating],
    si: NDArray[numpy.floating],
    scale: int,
    fast_costes: str,
) -> tuple[float, float]:
    search = bisection_costes if fast_costes == M_FASTER else linear_costes
    thr_fi, thr_si = search(None, None, fi, si, scale, fast_costes)
    if not (numpy.isfinite(thr_fi) and numpy.isfinite(thr_si)):
        return numpy.nan, numpy.nan
    fi_above = fi > thr_fi
    si_above = si > thr_si
    combined = fi_above & si_above
    tot_fi = fi[fi >= thr_fi].sum()
    tot_si = si[si >= thr_si].sum()
    if tot_fi <= 0 or tot_si <= 0:
        return _undefined_costes("zero above-threshold intensity denominator")
    return float(fi[combined].sum() / tot_fi), float(si[combined].sum() / tot_si)


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
    pixels_1: NDArray[numpy.number],
    pixels_2: NDArray[numpy.number],
    mask: NDArray[numpy.bool_],
    fast_costes: str = M_FASTER,
    thr: int = 15,
    *,
    scale_max: int | None = None,
) -> dict[str, float]:
    """Single boolean-mask Costes measurement; see get_correlation_costes.

    ``thr`` is retained for API compatibility and is not used by Costes.
    """
    first, second, scale = _prepare_costes_inputs(pixels_1, pixels_2, scale_max)
    if mask.shape != first.shape or mask.dtype != bool:
        raise ValueError(
            "Costes single-object mask must be boolean and match the images"
        )
    if fast_costes not in (M_FASTER, M_FAST, M_ACCURATE):
        raise ValueError("Unknown Costes threshold-search mode")
    C1, C2 = _costes_pair(first[mask], second[mask], scale, fast_costes)
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
# Costes' automated threshold (CellProfiler searches with explicit scale and
# undefined-subset guards). first_pixels/second_pixels are compatibility args.
# ---------------------------------------------------------------------------


def _costes_step_multiplier(correlation: float) -> int:
    """Fast linear search takes larger steps while correlation is far above zero."""
    for cutoff, multiplier in ((0.45, 10), (0.35, 5), (0.25, 2)):
        if correlation > cutoff:
            return multiplier
    return 1


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
    when Pearson R is much greater than 0. Inputs must be normalized 1D arrays.
    Undefined subsets return NaN thresholds with RuntimeWarning.
    """
    scale_max = _validate_costes_scale(scale_max)
    fi, si = numpy.asarray(fi, dtype=float), numpy.asarray(si, dtype=float)
    if not _valid_costes_pixels(fi, si):
        return numpy.nan, numpy.nan
    i_step = 1 / scale_max
    non_zero = (fi > 0) | (si > 0)
    fz = fi[non_zero]
    sz = si[non_zero]
    xvar = numpy.var(fz, axis=0, ddof=1)
    yvar = numpy.var(sz, axis=0, ddof=1)

    xmean = numpy.mean(fz, axis=0)
    ymean = numpy.mean(sz, axis=0)

    z = fz + sz
    zvar = numpy.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + numpy.sqrt(
        (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar)
    )
    if abs(covar) <= 8 * numpy.finfo(float).eps * max(xvar, yvar):
        return _undefined_costes("zero or ill-conditioned covariance")
    a = num / denom
    b = ymean - a * xmean
    if not (numpy.isfinite(a) and numpy.isfinite(b)):
        return _undefined_costes("non-finite regression")

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
        # Only run Pearson if the input has changed.
        if (positives := numpy.count_nonzero(combt)) != num_true:
            costReg = _costes_background_correlation(fi, si, combt)
            num_true = positives
        if not numpy.isfinite(costReg):
            return _undefined_costes(
                "background correlation became undefined during linear search"
            )
        if costReg <= 0:
            break
        # Accurate mode and thresholds below ten grid steps never skip candidates.
        multiplier = 1
        if fast_costes != M_ACCURATE and i >= i_step * 10:
            multiplier = _costes_step_multiplier(costReg)
        i -= i_step * multiplier
    return _finish_costes_thresholds(fi, si, thr_fi_c, thr_si_c)


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
    Inputs must be normalized 1D arrays. Undefined results return NaN thresholds
    with RuntimeWarning; constant candidates are treated as insufficient.
    """
    scale_max = _validate_costes_scale(scale_max)
    fi, si = numpy.asarray(fi, dtype=float), numpy.asarray(si, dtype=float)
    if not _valid_costes_pixels(fi, si):
        return numpy.nan, numpy.nan

    non_zero = (fi > 0) | (si > 0)
    fz = fi[non_zero]
    sz = si[non_zero]
    xvar = numpy.var(fz, axis=0, ddof=1)
    yvar = numpy.var(sz, axis=0, ddof=1)

    xmean = numpy.mean(fz, axis=0)
    ymean = numpy.mean(sz, axis=0)

    z = fz + sz
    zvar = numpy.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + numpy.sqrt(
        (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar)
    )
    if abs(covar) <= 8 * numpy.finfo(float).eps * max(xvar, yvar):
        return _undefined_costes("zero or ill-conditioned covariance")
    a = num / denom
    b = ymean - a * xmean
    if not (numpy.isfinite(a) and numpy.isfinite(b)):
        return _undefined_costes("non-finite regression")

    # Initialise variables
    left: float = 1
    right: float = scale_max
    mid: float = ((right - left) // (6 / 5)) + left
    lastmid: float = 0
    # None distinguishes an undefined search from a real threshold at zero.
    valid: float | None = None
    iterations = 0
    while lastmid != mid:
        iterations += 1
        if iterations > 256:
            return _undefined_costes("bisection did not converge")
        thr_fi_c = mid / scale_max
        thr_si_c = (a * thr_fi_c) + b
        combt = (fi < thr_fi_c) | (si < thr_si_c)
        costReg = _costes_background_correlation(fi, si, combt)
        if not numpy.isfinite(costReg) or costReg < 0:
            # Move above insufficient/constant candidates, never treat NaN as
            # an implicit convergence event (which formerly left stale bounds).
            left = mid - 1
        else:
            right = mid + 1
            valid = mid
        lastmid = mid
        divisor = 6 / 5 if right - left > 6 else 2
        mid = ((right - left) // divisor) + left

    if valid is None:
        return _undefined_costes("no finite nonnegative background correlation found")
    thr_fi_c = (valid - 1) / scale_max
    thr_si_c = (a * thr_fi_c) + b
    return _finish_costes_thresholds(fi, si, thr_fi_c, thr_si_c)


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

    Labels must be contiguous ``1..N`` (see :func:`cp_measure._sanitize.sanitize`).
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

    Labels must be contiguous ``1..N`` (see :func:`cp_measure._sanitize.sanitize`).
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


def _rwc_object_set(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    thr: float,
) -> dict[str, list[float]]:
    """CP-style dense ranks across foreground, with per-object reductions."""
    nobjects = int(masks.max(initial=0))
    if nobjects == 0:
        return {f"{F_RWC_FORMAT}_1": [], f"{F_RWC_FORMAT}_2": []}
    foreground = masks > 0
    labels = masks[foreground].astype(numpy.intp, copy=False)
    fi, si = pixels_1[foreground], pixels_2[foreground]
    if not (numpy.isfinite(fi).all() and numpy.isfinite(si).all()):
        raise ValueError("object_set ranking requires finite foreground intensities")
    _, rank1 = numpy.unique(fi, return_inverse=True)
    _, rank2 = numpy.unique(si, return_inverse=True)
    rank_count = max(rank1.max(), rank2.max()) + 1
    weights = (rank_count - numpy.abs(rank1 - rank2)) * 1.0 / rank_count

    indices = numpy.arange(1, nobjects + 1)
    # Array-valued maxima retain CP's float32 threshold arithmetic, including
    # boundary pixels. This is intentionally independent of the local API.
    threshold1 = (thr / 100) * scipy.ndimage.maximum(fi, labels, indices)
    threshold2 = (thr / 100) * scipy.ndimage.maximum(si, labels, indices)
    above1 = fi >= threshold1[labels - 1]
    above2 = si >= threshold2[labels - 1]
    joint = above1 & above2

    def reduce(
        values: NDArray[numpy.floating], selected: NDArray[numpy.bool_]
    ) -> NDArray[numpy.float64]:
        # bincount returns integer zeros for empty inputs even with weights.
        return numpy.bincount(labels[selected], weights=values, minlength=nobjects + 1)[
            1:
        ].astype(numpy.float64, copy=False)

    has_joint = numpy.bincount(labels[joint], minlength=nobjects + 1)[1:] > 0
    results: list[list[float]] = []
    for pixels, above in ((fi, above1), (si, above2)):
        numerator = reduce(pixels[joint] * weights[joint], joint)
        denominator = reduce(pixels[above], above)
        result = numpy.zeros(nobjects)
        # Preserve the existing API: absent/no-joint objects return zero;
        # an object with joint pixels but zero total signal returns NaN.
        with numpy.errstate(divide="ignore", invalid="ignore"):
            numpy.divide(numerator, denominator, out=result, where=has_joint)
        results.append(result.tolist())
    return {f"{F_RWC_FORMAT}_1": results[0], f"{F_RWC_FORMAT}_2": results[1]}


def get_correlation_rwc(
    pixels_1: NDArray[numpy.floating],
    pixels_2: NDArray[numpy.floating],
    masks: NDArray[numpy.integer],
    thr: int = 15,
    *,
    rank_scope: str = "object",
) -> dict[str, list[float]]:
    """Per-object rank-weighted colocalization coefficients between two channels.

    ``rank_scope="object"`` preserves the historical default: rank each object's
    pixels independently. ``rank_scope="object_set"`` uses CellProfiler 4.2.8's
    dense ranks across all foreground objects, then reduces per object. Other
    objects can therefore affect an object's result in object-set mode.
    Background is excluded; object-set mode requires finite foreground pixels.
    Existing empty/absent and zero-signal return conventions are retained.

    Labels must be contiguous ``1..N`` (see :func:`cp_measure._sanitize.sanitize`).
    """
    if rank_scope == "object_set":
        return _rwc_object_set(pixels_1, pixels_2, masks, thr)
    if rank_scope != "object":
        raise ValueError("rank_scope must be 'object' or 'object_set'")
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
    pixels_1: NDArray[numpy.number],
    pixels_2: NDArray[numpy.number],
    masks: NDArray[numpy.integer],
    fast_costes: str = M_FASTER,
    thr: int = 15,
    *,
    scale_max: int | None = None,
) -> dict[str, list[float]]:
    """Per-object Costes coefficients with an explicit acquisition grid.

    Float pixels must already be in [0, 1] and require ``scale_max`` (e.g.
    65535 for normalized 16-bit data). Integer pixels are divided by this
    scale internally. If both images have the same unsigned integer dtype,
    its storage maximum is inferred when ``scale_max`` is omitted. Supply
    the acquisition maximum for lower-bit-depth data in a larger container.
    Signed or mixed-dtype inputs always require an explicit common scale.

    Undefined regressions/background subsets or zero signal denominators
    return NaN and emit :class:`RuntimeWarning`, rather than plausible
    zero/one coefficients. ``thr`` is retained but unused. Input arrays are
    never modified. Labels must be contiguous ``1..N``; use the bulk API or
    :func:`cp_measure._sanitize.sanitize` for arbitrary IDs.
    """
    first, second, scale = _prepare_costes_inputs(pixels_1, pixels_2, scale_max)
    if masks.shape != first.shape or not numpy.issubdtype(masks.dtype, numpy.integer):
        raise ValueError("Costes labels must be an integer array matching the images")
    if fast_costes not in (M_FASTER, M_FAST, M_ACCURATE):
        raise ValueError("Unknown Costes threshold-search mode")
    c1: list[float] = []
    c2: list[float] = []
    for fi, si in _iter_label_pixels(first, second, masks):
        if fi is None or si is None or fi.size == 0:
            C1, C2 = _undefined_costes("missing or empty object label")
        else:
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

    Labels must be contiguous ``1..N`` (see :func:`cp_measure._sanitize.sanitize`).
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
