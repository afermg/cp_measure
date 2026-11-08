"""Exact 2D Euclidean distance transform (numba, single-threaded, cached).

``distance_transform_edt(mask)`` returns the same values as
``scipy.ndimage.distance_transform_edt`` for a boolean mask that contains at least
one background (False) pixel: for each pixel, the distance to the nearest
background pixel inside the array. Positions outside the array are not background.

The squared distances are computed exactly in integers with Meijster's two-phase
algorithm, and the result is their square root, which is how scipy forms its
distances too, so the values are bit-identical. A mask without any background
goes to scipy, whose result for that case measures to a point outside the array.
"""

import numpy as np
import scipy.ndimage
from numba import njit


@njit(cache=True, nogil=True)
def squared_edt_2d(mask):
    """Squared distance from each pixel to the nearest False pixel of ``mask``.

    Assumes at least one False pixel; columns without one hold ``height + width``
    in phase 1, larger than any real distance, so they never win in phase 2.
    """
    h, w = mask.shape
    far = h + w
    g = np.empty((h, w), np.int64)  # vertical distance to background per column
    for j in range(w):
        g[0, j] = far if mask[0, j] else 0
        for i in range(1, h):
            if not mask[i, j]:
                g[i, j] = 0
            elif g[i - 1, j] < far:
                g[i, j] = g[i - 1, j] + 1
            else:
                g[i, j] = far
        for i in range(h - 2, -1, -1):
            if g[i + 1, j] < g[i, j]:
                g[i, j] = g[i + 1, j] + 1
    out = np.empty((h, w), np.int64)
    s = np.empty(w, np.int64)  # columns of the lower envelope's parabolas
    t = np.empty(w, np.int64)  # first column each parabola is the minimum for
    for i in range(h):
        row = g[i]
        q = 0
        s[0] = 0
        t[0] = 0
        for u in range(1, w):
            gu = row[u] * row[u]
            while q >= 0:
                sq = s[q]
                if (t[q] - sq) * (t[q] - sq) + row[sq] * row[sq] <= (t[q] - u) * (
                    t[q] - u
                ) + gu:
                    break
                q -= 1
            if q < 0:
                q = 0
                s[0] = u
            else:
                sq = s[q]
                sep = (u * u - sq * sq + gu - row[sq] * row[sq]) // (2 * (u - sq)) + 1
                if sep < w:
                    q += 1
                    s[q] = u
                    t[q] = sep
        for u in range(w - 1, -1, -1):
            sq = s[q]
            out[i, u] = (u - sq) * (u - sq) + row[sq] * row[sq]
            if u == t[q]:
                q -= 1
    return out


def distance_transform_edt(mask):
    """``scipy.ndimage.distance_transform_edt`` for a 2D boolean ``mask``, exact."""
    mask = np.ascontiguousarray(mask, dtype=np.bool_)
    if mask.all():
        return scipy.ndimage.distance_transform_edt(mask)
    return np.sqrt(squared_edt_2d(mask))
