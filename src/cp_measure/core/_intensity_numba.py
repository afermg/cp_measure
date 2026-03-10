"""
Numba-accelerated 2D intensity measurements.

Replaces the O(N*H*W) label-matrix allocation in the Python reference with
a two-pass label scan that runs in O(H*W) memory, then computes all 19
per-label feature arrays in compiled code.

This module is imported lazily — it is only loaded when numba is available
and the input is 2D.
"""

import numpy as np
from numba import njit

from cp_measure.core.measureobjectintensity import (
    C_LOCATION,
    INTEGRATED_INTENSITY,
    INTEGRATED_INTENSITY_EDGE,
    INTENSITY,
    LOC_CMI_X,
    LOC_CMI_Y,
    LOC_CMI_Z,
    LOC_MAX_X,
    LOC_MAX_Y,
    LOC_MAX_Z,
    LOWER_QUARTILE_INTENSITY,
    MAD_INTENSITY,
    MASS_DISPLACEMENT,
    MAX_INTENSITY,
    MAX_INTENSITY_EDGE,
    MEAN_INTENSITY,
    MEAN_INTENSITY_EDGE,
    MEDIAN_INTENSITY,
    MIN_INTENSITY,
    MIN_INTENSITY_EDGE,
    STD_INTENSITY,
    STD_INTENSITY_EDGE,
    UPPER_QUARTILE_INTENSITY,
)


@njit(cache=True)
def _is_edge_pixel_2d(masks, y, x, height, width):
    """Return True if any 4-connected in-bounds neighbor has a different label.

    Out-of-bounds neighbors do NOT count as different (inner boundary mode).
    Matches ``skimage.segmentation.find_boundaries(mode="inner")`` which uses
    ``connectivity=1`` (4-connected for 2D).
    """
    label = masks[y, x]
    if y > 0 and masks[y - 1, x] != label:
        return True
    if y + 1 < height and masks[y + 1, x] != label:
        return True
    if x > 0 and masks[y, x - 1] != label:
        return True
    if x + 1 < width and masks[y, x + 1] != label:
        return True
    return False


@njit(cache=True)
def _percentile_interpolate(sorted_vals, n, fraction):
    """CellProfiler custom percentile: qindex = n * fraction (NOT (n-1)*fraction).

    Linear interpolation between floor and ceil when possible; otherwise
    clamps to the last element.
    """
    qindex = n * fraction
    idx = int(np.floor(qindex))
    frac = qindex - idx
    if idx < n - 1:
        return sorted_vals[idx] * (1.0 - frac) + sorted_vals[idx + 1] * frac
    else:
        # Clamp to last valid index
        if idx >= n:
            idx = n - 1
        return sorted_vals[idx]


@njit(cache=True)
def _compute_intensity_features_2d(masks, pixels):
    """Compute all intensity features for a 2D labeled mask + image.

    Parameters
    ----------
    masks : int array (H, W) — label matrix, 0 = background
    pixels : float64 array (H, W) — intensity image

    Returns
    -------
    Tuple of 19 float64 arrays, each of length ``max_label``.
    Order: integrated_intensity, mean_intensity, std_intensity,
           min_intensity, max_intensity,
           integrated_intensity_edge, mean_intensity_edge, std_intensity_edge,
           min_intensity_edge, max_intensity_edge,
           mass_displacement,
           lower_quartile_intensity, median_intensity, mad_intensity,
           upper_quartile_intensity,
           cmi_x, cmi_y, max_x, max_y
    """
    height = masks.shape[0]
    width = masks.shape[1]

    # --- Find max label ---
    max_label = np.int64(0)
    for y in range(height):
        for x in range(width):
            v = np.int64(masks[y, x])
            if v > max_label:
                max_label = v

    if max_label == 0:
        e = np.zeros(0, dtype=np.float64)
        return (
            e,
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )

    # --- Pass 1: count pixels and edge pixels per label, cache edge flags ---
    counts = np.zeros(max_label, dtype=np.int64)
    edge_counts = np.zeros(max_label, dtype=np.int64)
    is_edge = np.zeros((height, width), dtype=np.bool_)
    for y in range(height):
        for x in range(width):
            label = np.int64(masks[y, x])
            if label > 0:
                li = label - 1
                counts[li] += 1
                if _is_edge_pixel_2d(masks, y, x, height, width):
                    edge_counts[li] += 1
                    is_edge[y, x] = True

    # --- Build offsets ---
    offsets = np.zeros(max_label + 1, dtype=np.int64)
    edge_offsets = np.zeros(max_label + 1, dtype=np.int64)
    for i in range(max_label):
        offsets[i + 1] = offsets[i] + counts[i]
        edge_offsets[i + 1] = edge_offsets[i] + edge_counts[i]

    total_pixels = offsets[max_label]
    total_edge = edge_offsets[max_label]

    # --- Allocate flat arrays ---
    vals = np.empty(total_pixels, dtype=np.float64)
    rows = np.empty(total_pixels, dtype=np.int32)
    cols = np.empty(total_pixels, dtype=np.int32)
    edge_vals = np.empty(total_edge, dtype=np.float64)

    # --- Pass 2: fill pixel data (row-major order), reuse cached edge flags ---
    pos = offsets[:-1].copy()
    epos = edge_offsets[:-1].copy()
    for y in range(height):
        for x in range(width):
            label = np.int64(masks[y, x])
            if label > 0:
                li = label - 1
                v = pixels[y, x]
                idx = pos[li]
                vals[idx] = v
                rows[idx] = y
                cols[idx] = x
                pos[li] += 1

                if is_edge[y, x]:
                    eidx = epos[li]
                    edge_vals[eidx] = v
                    epos[li] += 1

    # --- Allocate output arrays (zeros = sentinel for empty labels) ---
    integrated_intensity = np.zeros(max_label, dtype=np.float64)
    mean_intensity = np.zeros(max_label, dtype=np.float64)
    std_intensity = np.zeros(max_label, dtype=np.float64)
    min_intensity = np.zeros(max_label, dtype=np.float64)
    max_intensity = np.zeros(max_label, dtype=np.float64)
    integrated_intensity_edge = np.zeros(max_label, dtype=np.float64)
    mean_intensity_edge = np.zeros(max_label, dtype=np.float64)
    std_intensity_edge = np.zeros(max_label, dtype=np.float64)
    min_intensity_edge = np.zeros(max_label, dtype=np.float64)
    max_intensity_edge = np.zeros(max_label, dtype=np.float64)
    mass_displacement = np.zeros(max_label, dtype=np.float64)
    lower_quartile_intensity = np.zeros(max_label, dtype=np.float64)
    median_intensity = np.zeros(max_label, dtype=np.float64)
    mad_intensity = np.zeros(max_label, dtype=np.float64)
    upper_quartile_intensity = np.zeros(max_label, dtype=np.float64)
    cmi_x = np.zeros(max_label, dtype=np.float64)
    cmi_y = np.zeros(max_label, dtype=np.float64)
    max_x = np.zeros(max_label, dtype=np.float64)
    max_y = np.zeros(max_label, dtype=np.float64)

    # --- Per-label feature computation ---
    for li in range(max_label):
        n = counts[li]
        if n == 0:
            continue

        start = offsets[li]
        end = offsets[li + 1]

        # -- Basic stats: sum, mean, min, max, max_position --
        s = 0.0
        mn = vals[start]
        mx = vals[start]
        mx_pos = start  # absolute index of max
        for i in range(start, end):
            v = vals[i]
            s += v
            if v < mn:
                mn = v
            if v > mx:
                mx = v
                mx_pos = i

        integrated_intensity[li] = s
        mean_val = s / n
        mean_intensity[li] = mean_val
        min_intensity[li] = mn
        max_intensity[li] = mx
        max_x[li] = cols[mx_pos]
        max_y[li] = rows[mx_pos]

        # -- Population std (ddof=0) --
        var_sum = 0.0
        for i in range(start, end):
            d = vals[i] - mean_val
            var_sum += d * d
        std_intensity[li] = np.sqrt(var_sum / n)

        # -- Centroids: binary and intensity-weighted --
        sum_row = 0.0
        sum_col = 0.0
        sum_vr = 0.0
        sum_vc = 0.0
        for i in range(start, end):
            v = vals[i]
            r = float(rows[i])
            c = float(cols[i])
            sum_row += r
            sum_col += c
            sum_vr += v * r
            sum_vc += v * c

        cm_y_val = sum_row / n
        cm_x_val = sum_col / n

        if s != 0.0:
            cmi_y_val = sum_vr / s
            cmi_x_val = sum_vc / s
        else:
            cmi_y_val = np.nan
            cmi_x_val = np.nan

        cmi_y[li] = cmi_y_val
        cmi_x[li] = cmi_x_val

        # -- Mass displacement --
        dy = cm_y_val - cmi_y_val
        dx = cm_x_val - cmi_x_val
        mass_displacement[li] = np.sqrt(dy * dy + dx * dx)

        # -- Percentiles: sort values, then interpolate --
        label_vals = vals[start:end].copy()
        label_vals.sort()

        lower_quartile_intensity[li] = _percentile_interpolate(label_vals, n, 0.25)
        median_intensity[li] = _percentile_interpolate(label_vals, n, 0.5)
        upper_quartile_intensity[li] = _percentile_interpolate(label_vals, n, 0.75)

        # -- MAD: median of |x_i - median(x)| --
        med = median_intensity[li]
        mad_vals = np.empty(n, dtype=np.float64)
        for i in range(start, end):
            mad_vals[i - start] = abs(vals[i] - med)
        mad_vals.sort()
        mad_intensity[li] = _percentile_interpolate(mad_vals, n, 0.5)

        # -- Edge stats --
        en = edge_counts[li]
        if en > 0:
            estart = edge_offsets[li]
            eend = edge_offsets[li + 1]

            es = 0.0
            emn = edge_vals[estart]
            emx = edge_vals[estart]
            for i in range(estart, eend):
                v = edge_vals[i]
                es += v
                if v < emn:
                    emn = v
                if v > emx:
                    emx = v

            integrated_intensity_edge[li] = es
            emean = es / en
            mean_intensity_edge[li] = emean
            min_intensity_edge[li] = emn
            max_intensity_edge[li] = emx

            evar_sum = 0.0
            for i in range(estart, eend):
                d = edge_vals[i] - emean
                evar_sum += d * d
            std_intensity_edge[li] = np.sqrt(evar_sum / en)

    return (
        integrated_intensity,
        mean_intensity,
        std_intensity,
        min_intensity,
        max_intensity,
        integrated_intensity_edge,
        mean_intensity_edge,
        std_intensity_edge,
        min_intensity_edge,
        max_intensity_edge,
        mass_displacement,
        lower_quartile_intensity,
        median_intensity,
        mad_intensity,
        upper_quartile_intensity,
        cmi_x,
        cmi_y,
        max_x,
        max_y,
    )


# -- Build key strings from module constants (N4) --
def _key(category, feature):
    return f"{category}_{feature}"


def get_intensity_numba(masks, pixels):
    """Compute 2D intensity features using Numba-accelerated kernels.

    Parameters
    ----------
    masks : numpy.ndarray
        2D integer label matrix (0 = background).
    pixels : numpy.ndarray
        2D float intensity image, same shape as *masks*.

    Returns
    -------
    dict[str, numpy.ndarray]
        Feature name -> per-object array (length ``masks.max()``).
    """
    masks = np.ascontiguousarray(masks)
    if masks.dtype not in (np.int32, np.int64):
        masks = masks.astype(np.int64)
    pixels = np.ascontiguousarray(pixels, dtype=np.float64)

    (
        integrated_intensity,
        mean_intensity,
        std_intensity,
        min_intensity,
        max_intensity,
        integrated_intensity_edge,
        mean_intensity_edge,
        std_intensity_edge,
        min_intensity_edge,
        max_intensity_edge,
        mass_displacement,
        lower_quartile_intensity,
        median_intensity,
        mad_intensity,
        upper_quartile_intensity,
        cmi_x,
        cmi_y,
        max_x,
        max_y,
    ) = _compute_intensity_features_2d(masks, pixels)

    n = len(integrated_intensity)

    return {
        _key(INTENSITY, INTEGRATED_INTENSITY): integrated_intensity,
        _key(INTENSITY, MEAN_INTENSITY): mean_intensity,
        _key(INTENSITY, STD_INTENSITY): std_intensity,
        _key(INTENSITY, MIN_INTENSITY): min_intensity,
        _key(INTENSITY, MAX_INTENSITY): max_intensity,
        _key(INTENSITY, INTEGRATED_INTENSITY_EDGE): integrated_intensity_edge,
        _key(INTENSITY, MEAN_INTENSITY_EDGE): mean_intensity_edge,
        _key(INTENSITY, STD_INTENSITY_EDGE): std_intensity_edge,
        _key(INTENSITY, MIN_INTENSITY_EDGE): min_intensity_edge,
        _key(INTENSITY, MAX_INTENSITY_EDGE): max_intensity_edge,
        _key(INTENSITY, MASS_DISPLACEMENT): mass_displacement,
        _key(INTENSITY, LOWER_QUARTILE_INTENSITY): lower_quartile_intensity,
        _key(INTENSITY, MEDIAN_INTENSITY): median_intensity,
        _key(INTENSITY, MAD_INTENSITY): mad_intensity,
        _key(INTENSITY, UPPER_QUARTILE_INTENSITY): upper_quartile_intensity,
        _key(C_LOCATION, LOC_CMI_X): cmi_x,
        _key(C_LOCATION, LOC_CMI_Y): cmi_y,
        _key(C_LOCATION, LOC_CMI_Z): np.zeros(n, dtype=np.float64),
        _key(C_LOCATION, LOC_MAX_X): max_x,
        _key(C_LOCATION, LOC_MAX_Y): max_y,
        _key(C_LOCATION, LOC_MAX_Z): np.zeros(n, dtype=np.float64),
    }
