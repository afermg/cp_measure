"""Numba Haralick texture kernel (single-threaded, cached).

Replaces ``mahotas.features.haralick(crop, distance, ignore_zeros=True)`` — nearly
all of ``measuretexture.get_texture``'s cost — with one fused per-object kernel:
count pixel pairs into a grey-level co-occurrence matrix (GLCM) per direction,
drop the pairs touching background (``ignore_zeros``), reduce each GLCM to the 13
Haralick features. See :func:`cp_measure.core.measuretexture.get_texture` for what
those features mean.

The GLCM counts are integers and match ``mahotas`` exactly. The features are
float reductions of them and agree to ~1e-5 relative, with two deliberate
differences from ``mahotas/features/texture.py::haralick_features``
(``preserve_haralick_bug=False``, ``use_x_minus_y_variance=False``):

- Variance (and Correlation, through it) uses the centred ``sum((k - ux)**2 * px)``
  where mahotas uses the algebraically equal ``dot(px, k**2) - ux**2``. Against a
  high-precision reference the centred form is the accurate one; mahotas loses up
  to ~1e-6 relative to cancellation on low-contrast objects.
- ``HXY1`` and ``HXY2`` are collapsed to ``2 * HX``, exact for a symmetric GLCM,
  turning two O(fm1^2) loops into O(fm1). This is worth ~2.2x on the backend; see
  ``_haralick_13`` for the one regime where the residual would matter.

The GLCM is sized to ``crop.max() + 1``, not a fixed 256, because feature 9
(``px_minus_y.var()``) is taken over a length-``fm1`` array.

``img_as_ubyte`` / ``regionprops`` stay host-side (scipy/skimage). Serial; no
``prange``/``nogil``.
"""

import numpy as np
from numba import njit

# Direction deltas as (dz, dy, dx). 2D uses dz=0 (crop is (1, Y, X)); these mirror
# mahotas ``_2d_deltas`` [(0,1),(1,1),(1,0),(1,-1)] (as (dy,dx)) and ``_3d_deltas``.
DELTAS_2D = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 1, -1]], np.int64)
DELTAS_3D = np.array(
    [
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [1, -1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [1, -1, 1],
        [1, 0, -1],
        [0, 1, -1],
        [1, 1, -1],
        [1, -1, -1],
    ],
    np.int64,
)


@njit(cache=True, error_model="numpy")
def _entropy(a):
    """``-sum(a * log2(a))`` over positive entries (0*log2(1)=0), as mahotas."""
    s = 0.0
    for i in range(a.shape[0]):
        v = a[i]
        if v > 0.0:
            s -= v * np.log2(v)
    return s


@njit(cache=True, error_model="numpy")
def haralick_object(crop, offsets):
    """The 13 Haralick features per direction for one ``(Z, Y, X)`` object crop.

    ``crop`` is a contiguous ``(Z, Y, X)`` integer array of grey levels in
    ``0..gray_levels - 1`` (0 is background); a 2D crop arrives as ``Z == 1``.
    ``offsets`` is a contiguous ``(n_dir, 3)`` int64 array of ``distance * (dz, dy,
    dx)`` steps, one row per direction, each component signed and no larger than
    the crop.

    Returns ``(n_dir, 13)`` float64. If ANY direction's GLCM is empty after
    ``ignore_zeros`` (no non-background pairs), the whole object's block is NaN —
    matching mahotas raising ``ValueError`` and cp_measure catching it.
    """
    n_dir = offsets.shape[0]
    out = np.empty((n_dir, 13), np.float64)
    Z, Y, X = crop.shape
    fm1 = 0
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if crop[z, y, x] + 1 > fm1:
                    fm1 = crop[z, y, x] + 1

    cmat = np.empty((fm1, fm1), np.int64)
    for d in range(n_dir):
        dz, dy, dx = offsets[d, 0], offsets[d, 1], offsets[d, 2]
        cmat[:] = 0
        total = 0
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    z2 = z + dz
                    y2 = y + dy
                    x2 = x + dx
                    if 0 <= z2 < Z and 0 <= y2 < Y and 0 <= x2 < X:
                        a = crop[z, y, x]
                        b = crop[z2, y2, x2]
                        cmat[a, b] += 1
                        cmat[b, a] += 1  # symmetric=True
                        total += 2
        # ignore_zeros: T excludes pairs touching background (grey level 0), i.e.
        # row 0 + col 0 (symmetric, so 2*row0 - cmat[0,0]) -> O(fm1), no fm1^2 sum.
        row0 = 0
        for j in range(fm1):
            row0 += cmat[0, j]
        T = total - 2 * row0 + cmat[0, 0]
        if T == 0:  # mahotas raises -> cp_measure -> whole object NaN
            for dd in range(n_dir):
                for f in range(13):
                    out[dd, f] = np.nan
            return out

        _haralick_13(cmat, fm1, T, out, d)
    return out


@njit(cache=True, error_model="numpy")
def _haralick_13(cmat, fm1, T, out, d):
    """Fill ``out[d, :]`` with the 13 Haralick features of GLCM ``cmat`` (size fm1)."""
    px = np.zeros(fm1)  # marginal (== row & col marginal; GLCM is symmetric)
    px_plus_y = np.zeros(2 * fm1)  # P(i+j)
    px_minus_y = np.zeros(fm1)  # P(|i-j|)
    asm = 0.0
    entropy = 0.0
    ij_sum = 0.0  # sum_{i,j} i*j*p
    idm = 0.0
    # rows/cols 0 are the ignore_zeros background -> skip (start at 1)
    for i in range(1, fm1):
        for j in range(1, fm1):
            c = cmat[i, j]
            if c == 0:
                continue
            # c / T, never c * (1/T): the reciprocal is a ulp off for most T, which
            # leaves a uniform-intensity object with a tiny non-zero variance and so
            # slips past the sx == 0 guard below (Correlation ~1e15 instead of 1).
            p = c / T
            asm += p * p
            entropy -= p * np.log2(p)
            ij_sum += i * j * p
            idm += p / ((i - j) * (i - j) + 1)
            px[j] += p
            px_plus_y[i + j] += p
            px_minus_y[i - j if i >= j else j - i] += p

    ux = 0.0
    vx = 0.0
    for k in range(fm1):
        ux += k * px[k]
    for k in range(fm1):
        vx += (k - ux) * (k - ux) * px[k]
    sx = np.sqrt(vx)

    sum_avg = 0.0
    sum_var = 0.0
    for s in range(2 * fm1):
        sum_avg += s * px_plus_y[s]
    for s in range(2 * fm1):
        sum_var += s * s * px_plus_y[s]
    sum_var -= sum_avg * sum_avg

    contrast = 0.0
    for dd in range(fm1):
        contrast += dd * dd * px_minus_y[dd]

    # difference variance = var of the length-fm1 px_minus_y array (population)
    dm = 0.0
    for dd in range(fm1):
        dm += px_minus_y[dd]
    dm /= fm1
    diff_var = 0.0
    for dd in range(fm1):
        diff_var += (px_minus_y[dd] - dm) * (px_minus_y[dd] - dm)
    diff_var /= fm1

    # Info measures of correlation. For a SYMMETRIC GLCM both cross-entropies
    # collapse to the marginals: HXY1 = HXY2 = 2*HX (verified vs mahotas to ~1e-15).
    # (HXY1 = -sum p*log2(px*py) = -sum_i py[i]log2 px[i] - sum_j px[j]log2 py[j]
    # = 2*HX when px == py; likewise HXY2.)
    #
    # Feature 11 divides by HX, so the collapse is harmless there. Feature 12 feeds
    # HXY2 - entropy through sqrt(1 - exp(-2*diff)), which amplifies any residual as
    # diff goes to zero. Measured against mahotas on uniform, two-level, noise and
    # gradient crops, the worst InfoMeas2 error is 2.6e-13 -- the amplification only
    # bites on an exactly independent GLCM (p == outer(px, py) to the ulp), where
    # mahotas cancels to 0 and this gives ~1e-7. Integer counts over a ubyte image
    # do not produce that, and evaluating HXY2 over the outer product instead costs
    # 2.2x the runtime of the whole backend, so the collapse stays.
    hx = _entropy(px)
    hxy1 = 2.0 * hx
    hxy2 = 2.0 * hx

    out[d, 0] = asm
    out[d, 1] = contrast
    out[d, 2] = 1.0 if sx == 0.0 else (ij_sum - ux * ux) / (sx * sx)
    out[d, 3] = vx
    out[d, 4] = idm
    out[d, 5] = sum_avg
    out[d, 6] = sum_var
    out[d, 7] = _entropy(px_plus_y)
    out[d, 8] = entropy
    out[d, 9] = diff_var
    out[d, 10] = _entropy(px_minus_y)
    out[d, 11] = (entropy - hxy1) if hx == 0.0 else (entropy - hxy1) / hx
    diff = hxy2 - entropy
    arg = 1.0 - np.exp(-2.0 * diff)
    out[d, 12] = np.sqrt(arg if arg > 0.0 else 0.0)
