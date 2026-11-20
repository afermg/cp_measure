"""Numba kernel for one 2D image's intensity features.

Ports the 2D body of :func:`cp_measure.core.numba.measureobjectintensity._intensity_volume`
into one GIL-free njit function that writes per-object features into a
preallocated ``(B, n_feature, max_obj)`` array. The same ``_segment_numba`` helpers
are reused, so results are byte-identical to the serial path. Batches run it per
image on ``map_images`` threads; a ``prange`` driver aborted the process when
called from several Python threads under numba's default threading layer.

Column order matches ``_intensity_volume``'s ``measurement_names`` (16 base, then
5 edge). 3D volumes stay on the serial path.
"""

import numpy as np
from numba import njit

from cp_measure.primitives._segment_numba import (
    flatten_numba,
    inner_boundary,
    segment_moments,
    segment_quantiles,
    segment_resid_sumsq,
    segment_stats,
)

N_BASE = 16
N_EDGE = 5


@njit(nogil=True, cache=True)
def fill_intensity_2d(mask_zyx, pix_zyx, legacy, edge, out, b, nobj):
    """Fill ``out[b, :, :nobj]`` with one 2D image's intensity features.

    ``mask_zyx``/``pix_zyx`` are one ``(1, Y, X)`` volume. Writes nothing for an
    empty image (``nobj == 0``). The arithmetic mirrors ``_intensity_volume``'s
    2D path exactly (quartile MAD fraction ``1/2``).
    """
    if nobj == 0:
        return
    m = np.ascontiguousarray(mask_zyx)
    p = np.ascontiguousarray(pix_zyx)
    values, seg0, xc, yc, zc = flatten_numba(m, p)

    # Base features need at least one finite object pixel; with none, serial
    # leaves them zero (matched by out being zero-initialised). Edge features are
    # gated only on ``edge`` in the serial path, so they must still run below.
    if values.size > 0:
        (
            count,
            sumI,
            minI,
            maxI,
            max_x,
            max_y,
            max_z,
            sx,
            sy,
            sz,
            sxI,
            syI,
            szI,
        ) = segment_moments(values, seg0, xc, yc, zc, nobj)
        cnt = count.astype(np.float64)
        mean = sumI / cnt
        ss = segment_resid_sumsq(values, seg0, nobj, mean)
        std = np.sqrt(ss / cnt)
        cm_x = sx / cnt
        cm_y = sy / cnt
        cm_z = sz / cnt
        cmi_x = sxI / sumI
        cmi_y = syI / sumI
        cmi_z = szI / sumI
        massd = np.sqrt((cm_x - cmi_x) ** 2 + (cm_y - cmi_y) ** 2 + (cm_z - cmi_z) ** 2)
        lq, med, uq, mad = segment_quantiles(
            values, seg0, count, nobj, 1.0 / 2.0, legacy
        )

        out[b, 0, :nobj] = sumI
        out[b, 1, :nobj] = mean
        out[b, 2, :nobj] = std
        out[b, 3, :nobj] = minI
        out[b, 4, :nobj] = maxI
        out[b, 5, :nobj] = massd
        out[b, 6, :nobj] = lq
        out[b, 7, :nobj] = med
        out[b, 8, :nobj] = mad
        out[b, 9, :nobj] = uq
        out[b, 10, :nobj] = cmi_x
        out[b, 11, :nobj] = cmi_y
        out[b, 12, :nobj] = cmi_z
        out[b, 13, :nobj] = max_x
        out[b, 14, :nobj] = max_y
        out[b, 15, :nobj] = max_z
        # Objects without a finite pixel keep NumPy's zeros (not 0/0 or +-inf).
        for j in range(nobj):
            if count[j] == 0:
                for f in range(N_BASE):
                    out[b, f, j] = 0.0

    if not edge:
        return

    mask2d = m[0]
    pix2d = p[0]
    bnd = inner_boundary(np.ascontiguousarray(mask2d))
    H, W = mask2d.shape
    nb = 0
    for r in range(H):
        for c in range(W):
            if bnd[r, c] > 0 and np.isfinite(pix2d[r, c]):  # as NumPy
                nb += 1
    if nb == 0:
        return
    ev = np.empty(nb, np.float64)
    es = np.empty(nb, np.int64)
    idx = 0
    for r in range(H):
        for c in range(W):
            if bnd[r, c] > 0 and np.isfinite(pix2d[r, c]):
                ev[idx] = pix2d[r, c]
                es[idx] = bnd[r, c] - 1
                idx += 1
    ecount, esum, emin, emax = segment_stats(ev, es, nobj)
    ecnt = ecount.astype(np.float64)
    emean = esum / ecnt
    ess = segment_resid_sumsq(ev, es, nobj, emean)
    estd = np.sqrt(ess / ecnt)
    for j in range(nobj):
        if ecount[j] > 0:
            out[b, 16, j] = esum[j]
            out[b, 17, j] = emean[j]
            out[b, 18, j] = estd[j]
            out[b, 19, j] = emin[j]
            out[b, 20, j] = emax[j]
