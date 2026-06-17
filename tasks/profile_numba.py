"""Per-component breakdown of our numba get_intensity (edge on).

PYTHONPATH=src python tasks/profile_numba.py
"""

import time

import numpy as np
import skimage.segmentation

from cp_measure.primitives import label_to_idx_lut
from cp_measure.primitives._segment_numba import (
    flatten_numba,
    segment_moments,
    segment_quantiles,
    segment_resid_sumsq,
    segment_stats,
)


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


def warmup():
    m, p = make(64, 64, 4)
    masks = m[None]
    pix = p[None]
    lut, n = label_to_idx_lut(masks)
    mc = np.ascontiguousarray(masks)
    pc = np.ascontiguousarray(pix)
    v, seg, xc, yc, zc = flatten_numba(mc, pc, lut)
    cnt = segment_moments(v, seg, xc, yc, zc, n)[0]
    segment_resid_sumsq(v, seg, n, np.ones(n))
    segment_quantiles(v, seg, cnt, n, 0.5)
    segment_stats(v, seg, n)


def profile(px, nobj):
    m, p = make(px, px, nobj)
    masks = m[None]
    pix = p[None].astype(np.float64)
    fill = (m > 0).mean() * 100

    # chain once to obtain each stage's inputs
    lut, n = label_to_idx_lut(masks)
    mc = np.ascontiguousarray(masks)
    pc = np.ascontiguousarray(pix)
    v, seg, xc, yc, zc = flatten_numba(mc, pc, lut)
    mom = segment_moments(v, seg, xc, yc, zc, n)
    count, sumI = mom[0], mom[1]
    mean = sumI / count.astype(np.float64)
    bnd = skimage.segmentation.find_boundaries(masks, mode="inner")
    emask = bnd > 0
    ev = pix[emask].astype(np.float64)
    eseg = lut[masks[emask]]

    t = {}
    t["label_to_idx (find_objects)"] = med(lambda: label_to_idx_lut(masks))
    t["ascontiguous(masks+pix)"] = med(
        lambda: (np.ascontiguousarray(masks), np.ascontiguousarray(pix))
    )
    t["flatten_numba (2 full scans)"] = med(lambda: flatten_numba(mc, pc, lut))
    t["segment_moments"] = med(lambda: segment_moments(v, seg, xc, yc, zc, n))
    t["segment_resid_sumsq"] = med(lambda: segment_resid_sumsq(v, seg, n, mean))
    t["segment_quantiles"] = med(lambda: segment_quantiles(v, seg, count, n, 0.5))
    t["find_boundaries (skimage)"] = med(
        lambda: skimage.segmentation.find_boundaries(masks, mode="inner")
    )
    t["edge segment_stats"] = med(lambda: segment_stats(ev, eseg, n))

    total = sum(t.values())
    print(f"\n### {px}^2, {nobj} obj — fill={fill:.1f}%  (sum of components) ###")
    for k, ms in sorted(t.items(), key=lambda kv: -kv[1]):
        print(f"  {k:34s} {ms:8.2f} ms  ({100 * ms / total:4.1f}%)")
    print(f"  {'TOTAL (components)':34s} {total:8.2f} ms")


warmup()
print(f"numpy {np.__version__}")
profile(1024, 64)
profile(256, 16)
