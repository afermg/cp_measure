"""Quantile lever: numba per-segment sort vs one numpy lexsort-gather.

Both use the SAME CellProfiler convention (index = n*q interp, MAD at 1/ndim),
so values must match. Question: which is faster at scale?
PYTHONPATH=src python tasks/exp_quantiles.py
"""

import time

import numpy as np

from cp_measure.primitives import label_to_idx_lut
from cp_measure.primitives._segment_numba import (
    flatten_numba,
    segment_moments,
    segment_quantiles,
)


def quantiles_numpy_lexsort(values, seg0, counts, n, mad_frac):
    """Old CellProfiler algorithm on flat arrays: one lexsort + cumsum gather."""
    order = np.lexsort((values, seg0))
    areas = counts.astype(int)
    indices = np.cumsum(areas) - areas
    lq = np.zeros(n)
    med = np.zeros(n)
    uq = np.zeros(n)
    mad = np.zeros(n)
    for dest, frac in ((lq, 0.25), (med, 0.5), (uq, 0.75)):
        qi = indices.astype(float) + areas * frac
        qf = qi - np.floor(qi)
        qi = qi.astype(int)
        m = qi < indices + areas - 1
        dest[m] = values[order[qi[m]]] * (1 - qf[m]) + values[order[qi[m] + 1]] * qf[m]
        m2 = (~m) & (areas > 0)
        dest[m2] = values[order[qi[m2]]]
    madv = np.abs(values - med[seg0])
    order = np.lexsort((madv, seg0))
    qi = indices.astype(float) + areas * mad_frac
    qf = qi - np.floor(qi)
    qi = qi.astype(int)
    m = qi < indices + areas - 1
    mad[m] = madv[order[qi[m]]] * (1 - qf[m]) + madv[order[qi[m] + 1]] * qf[m]
    m2 = (~m) & (areas > 0)
    mad[m2] = madv[order[qi[m2]]]
    return lq, med, uq, mad


def make(H, W, nobj, seed=1):
    rng = np.random.default_rng(seed)
    mask = np.zeros((H, W), np.int32)
    g = int(np.ceil(np.sqrt(nobj)))
    ch, cw = H // g, W // g
    obj = max(min(ch, cw) // 2, 4)
    lab = 0
    for i in range(g):
        for j in range(g):
            if lab >= nobj:
                break
            mask[i * ch + 2 : i * ch + 2 + obj, j * cw + 2 : j * cw + 2 + obj] = lab + 1
            lab += 1
    return mask, rng.random((H, W))


def med(fn, repeats=7):
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2] * 1e3


def prep(px, nobj):
    m, p = make(px, px, nobj)
    masks = m[None]
    pix = p[None].astype(np.float64)
    lut, n = label_to_idx_lut(masks)
    v, seg, xc, yc, zc = flatten_numba(
        np.ascontiguousarray(masks), np.ascontiguousarray(pix), lut
    )
    count = segment_moments(v, seg, xc, yc, zc, n)[0]
    return v, seg, count, n


# warm
v, seg, count, n = prep(64, 4)
segment_quantiles(v, seg, count, n, 0.5)
quantiles_numpy_lexsort(v, seg, count, n, 0.5)

for px, nobj in [(1024, 64), (256, 16), (512, 200)]:
    v, seg, count, n = prep(px, nobj)
    a = segment_quantiles(v, seg, count, n, 0.5)
    b = quantiles_numpy_lexsort(v, seg, count, n, 0.5)
    dmax = max(np.max(np.abs(a[i] - b[i])) for i in range(4))
    t_nb = med(lambda: segment_quantiles(v, seg, count, n, 0.5))
    t_np = med(lambda: quantiles_numpy_lexsort(v, seg, count, n, 0.5))
    print(
        f"{px}^2/{nobj}obj  numba-sort {t_nb:7.2f} ms  |  numpy-lexsort {t_np:7.2f} ms"
        f"  ({t_nb / t_np:4.1f}x)  max|Δ|={dmax:.2e}"
    )
