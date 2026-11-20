"""CellProfiler 4.2.8 Costes arithmetic (NumPy 1.x / SciPy 1.9).

Keep this reference path separate from the default per-object optimizations.
Regression reductions retain input precision; scalar expressions and Pearson
follow the historical promotion rules even when running on NumPy 2.
"""

import warnings

import numpy
import scipy.linalg
import scipy.ndimage
import scipy.stats


def _threshold_scalar(pixels, threshold):
    """NumPy 1.x array/scalar comparison promotion for normalized floats."""
    if numpy.isfinite(threshold) and abs(threshold) > numpy.finfo(pixels.dtype).max:
        return numpy.float64(threshold)
    return pixels.dtype.type(threshold)


def _pearson(first, second):
    """SciPy 1.9 Pearson coefficient arithmetic, without the unused p-value."""
    if first.size < 2:
        raise ValueError("Pearson requires at least two paired pixels")
    if numpy.ptp(first) == 0 or numpy.ptp(second) == 0:
        warnings.warn(
            "An input array is constant; the correlation coefficient is not defined.",
            scipy.stats.ConstantInputWarning,
            stacklevel=3,
        )
        return numpy.nan
    if first.size == 2:
        return float(
            numpy.sign(first[1] - first[0]) * numpy.sign(second[1] - second[0])
        )
    first = first.astype(numpy.float64)
    second = second.astype(numpy.float64)
    first_mean, second_mean = first.mean(), second.mean()
    first -= first_mean
    second -= second_mean
    first_norm, second_norm = scipy.linalg.norm(first), scipy.linalg.norm(second)
    if first_norm < 1e-13 * abs(first_mean) or second_norm < 1e-13 * abs(second_mean):
        warnings.warn(
            "An input array is nearly constant; the computed correlation coefficient may be inaccurate.",
            scipy.stats.NearConstantInputWarning,
            stacklevel=3,
        )
    coefficient = numpy.dot(first / first_norm, second / second_norm)
    return max(min(coefficient, 1.0), -1.0)


def _regression(first, second):
    selected = (first > 0) | (second > 0)
    first, second = first[selected], second[selected]
    first_var = numpy.var(first, ddof=1)
    second_var = numpy.var(second, ddof=1)
    first_mean = numpy.mean(first)
    second_mean = numpy.mean(second)
    sum_var = numpy.var(first + second, ddof=1)
    covariance = numpy.float64(0.5) * (sum_var - (first_var + second_var))
    difference = second_var - first_var
    numerator = difference + numpy.sqrt(
        difference * difference + 4 * (covariance * covariance)
    )
    slope = numerator / (2 * covariance)
    return slope, second_mean - slope * first_mean


def _thresholds(first, second, scale, mode):
    slope, intercept = _regression(first, second)

    def background(threshold):
        return (first < _threshold_scalar(first, threshold)) | (
            second < _threshold_scalar(second, slope * threshold + intercept)
        )

    if mode == "Faster":
        left, right = 1.0, float(scale)
        middle = ((right - left) // (6 / 5)) + left
        previous, valid = 0.0, 1.0
        for iteration in range(256):
            if previous == middle:
                threshold = (valid - 1) / scale
                return threshold, slope * threshold + intercept
            selected = background(middle / scale)
            if numpy.count_nonzero(selected) <= 2:
                left = middle - 1
            else:
                correlation = _pearson(first[selected], second[selected])
                if correlation < 0:
                    left = middle - 1
                elif correlation >= 0:
                    right, valid = middle + 1, middle
            previous = middle
            divisor = 6 / 5 if right - left > 6 else 2
            middle = ((right - left) // divisor) + left
        raise RuntimeError("CellProfiler-compatible Costes bisection did not converge")

    step = 1 / scale
    first_max, second_max = first.max(), second.max()
    candidate = step * ((numpy.float64(max(first_max, second_max)) // step) + 1)
    correlation = _pearson(first, second)
    threshold = candidate
    count = None
    while candidate > first_max and slope * candidate + intercept > second_max:
        candidate -= step
    while candidate > step:
        threshold = candidate
        selected = background(candidate)
        positives = numpy.count_nonzero(selected)
        if positives < 2:
            break
        if positives != count:
            correlation = _pearson(first[selected], second[selected])
            count = positives
        if correlation <= 0:
            break
        multiplier = 1
        if mode != "Accurate" and candidate >= step * 10:
            if correlation > 0.45:
                multiplier = 10
            elif correlation > 0.35:
                multiplier = 5
            elif correlation > 0.25:
                multiplier = 2
        candidate -= step * multiplier
    return threshold, slope * threshold + intercept


def image_costes(first, second, labels, scale, mode):
    """Whole-image thresholds, followed by CellProfiler's per-label reductions."""
    count = int(labels.max(initial=0))
    if count == 0:
        return numpy.empty(0), numpy.empty(0)
    valid = ~numpy.isnan(first) & ~numpy.isnan(second)
    if not numpy.any(valid):
        return numpy.full(count, numpy.nan), numpy.full(count, numpy.nan)
    first_threshold, second_threshold = _thresholds(
        first[valid], second[valid], scale, mode
    )
    first_threshold = _threshold_scalar(first, first_threshold)
    second_threshold = _threshold_scalar(second, second_threshold)
    combined = (first > first_threshold) & (second > second_threshold) & (labels > 0)
    if not numpy.any(combined):
        return numpy.zeros(count), numpy.zeros(count)
    indices = numpy.arange(1, count + 1)
    results = []
    for pixels, threshold in ((first, first_threshold), (second, second_threshold)):
        above = (pixels >= threshold) & (labels > 0)
        denominator = scipy.ndimage.sum(pixels[above], labels[above], indices)
        numerator = scipy.ndimage.sum(pixels[combined], labels[combined], indices)
        results.append(numerator / denominator)
    return tuple(results)
