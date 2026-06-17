"""End-to-end: full direct-scan intensity core vs current flat-path core.

Compares the WHOLE non-edge computation (moments + std + quartiles/MAD):
  - current: cp_measure.core.numba.get_intensity(..., edge_measurements=False)
             (flatten_labeled once, then 3 kernel passes over flat arrays)
  - direct : 3 image-scanning kernels, no host flattening, coords free

Both pay label_to_idx_lut (numpy.unique). edge excluded (common cost).
Run: PYTHONPATH=src python tasks/bench_directcore.py
"""

import time

import numpy as np
from numba import njit

from cp_measure.core.numba import get_intensity as nb_intensity
from cp_measure.core.measureobjectintensity import get_intensity as np_intensity
from cp_measure.primitives import label_to_idx_lut
from cp_measure.primitives._segment_numba import _interp


@njit(cache=True)
def moments_img(masks, pixels, lut, n):
    Z, Y, X = masks.shape
    count = np.zeros(n, np.int64)
    sumI = np.zeros(n, np.float64)
    minI = np.full(n, np.inf)
    maxI = np.full(n, -np.inf)
    mx = np.zeros(n)
    my = np.zeros(n)
    mz = np.zeros(n)
    sx = np.zeros(n)
    sy = np.zeros(n)
    sz = np.zeros(n)
    sxI = np.zeros(n)
    syI = np.zeros(n)
    szI = np.zeros(n)
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                L = masks[z, y, x]
                if L <= 0:
                    continue
                v = pixels[z, y, x]
                if not np.isfinite(v):
                    continue
                k = lut[L]
                count[k] += 1
                sumI[k] += v
                if v < minI[k]:
                    minI[k] = v
                if v >= maxI[k]:
                    maxI[k] = v
                    mx[k] = x
                    my[k] = y
                    mz[k] = z
                sx[k] += x
                sy[k] += y
                sz[k] += z
                sxI[k] += x * v
                syI[k] += y * v
                szI[k] += z * v
    return count, sumI, minI, maxI, mx, my, mz, sx, sy, sz, sxI, syI, szI


@njit(cache=True)
def resid_img(masks, pixels, lut, n, mean):
    Z, Y, X = masks.shape
    ss = np.zeros(n, np.float64)
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                L = masks[z, y, x]
                if L <= 0:
                    continue
                v = pixels[z, y, x]
                if not np.isfinite(v):
                    continue
                k = lut[L]
                d = v - mean[k]
                ss[k] += d * d
    return ss


@njit(cache=True)
def quant_img(masks, pixels, lut, counts, n, mad_frac):
    starts = np.zeros(n, np.int64)
    acc = 0
    for k in range(n):
        starts[k] = acc
        acc += counts[k]
    buf = np.empty(acc, np.float64)
    cursor = starts.copy()
    Z, Y, X = masks.shape
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                L = masks[z, y, x]
                if L <= 0:
                    continue
                v = pixels[z, y, x]
                if not np.isfinite(v):
                    continue
                k = lut[L]
                buf[cursor[k]] = v
                cursor[k] += 1
    lq = np.zeros(n)
    med = np.zeros(n)
    uq = np.zeros(n)
    mad = np.zeros(n)
    for k in range(n):
        cnt = counts[k]
        if cnt == 0:
            continue
        s = starts[k]
        seg = buf[s : s + cnt]
        seg.sort()
        lq[k] = _interp(seg, cnt, 0.25)
        med[k] = _interp(seg, cnt, 0.5)
        uq[k] = _interp(seg, cnt, 0.75)
        ad = np.abs(seg - med[k])
        ad.sort()
        mad[k] = _interp(ad, cnt, mad_frac)
    return lq, med, uq, mad


def direct_core(masks, pixels):
    orig_ndim = pixels.ndim
    if pixels.ndim == 2:
        pixels = pixels.reshape(1, *pixels.shape)
        masks = masks.reshape(1, *masks.shape)
    lut, n = label_to_idx_lut(masks)
    masks = np.ascontiguousarray(masks)
    pixels = np.ascontiguousarray(pixels, dtype=np.float64)
    (count, sumI, minI, maxI, mx, my, mz, sx, sy, sz, sxI, syI, szI) = moments_img(
        masks, pixels, lut, n
    )
    cnt = count.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sumI / cnt
        ss = resid_img(masks, pixels, lut, n, mean)
        std = np.sqrt(ss / cnt)
        cmi_x = sxI / sumI
        cmi_y = syI / sumI
        cmi_z = szI / sumI
        md = np.sqrt((sx / cnt - cmi_x) ** 2 + (sy / cnt - cmi_y) ** 2 + (sz / cnt - cmi_z) ** 2)
    lq, med, uq, mad = quant_img(masks, pixels, lut, count, n, 1.0 / orig_ndim)
    return (sumI, mean, std, minI, maxI, md, lq, med, uq, mad, cmi_x, cmi_y, cmi_z, mx, my, mz)


def make(H, W, nobj, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros((H, W), dtype=np.int32)
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


def med_ms(fn, repeats=7):
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2] * 1e3


def main():
    print(f"numpy {np.__version__}\n")
    wm, wp = make(64, 64, 4)
    direct_core(wm, wp)
    nb_intensity(wm, wp, edge_measurements=False)

    # correctness: direct core vs numpy reference (core only)
    m, p = make(256, 256, 16, seed=1)
    ref = np_intensity(m, p, edge_measurements=False)
    d = direct_core(m, p)
    assert np.allclose(d[0], ref["Intensity_IntegratedIntensity"])
    assert np.allclose(d[7], ref["Intensity_MedianIntensity"])
    assert np.allclose(d[2], ref["Intensity_StdIntensity"])
    print("sanity: direct core == numpy reference (sum/median/std)  ✓\n")

    for px, nobj in [(256, 16), (1024, 64)]:
        m, p = make(px, px, nobj, seed=1)
        t_cur = med_ms(lambda: nb_intensity(m, p, edge_measurements=False))
        t_dir = med_ms(lambda: direct_core(m, p))
        print(f"### {px}x{px}, {nobj} obj — FULL non-edge core ###")
        print(f"  current (flat path): {t_cur:7.2f} ms")
        print(f"  direct image-scan  : {t_dir:7.2f} ms   ({t_cur / t_dir:4.1f}x)")
        print()


if __name__ == "__main__":
    main()
