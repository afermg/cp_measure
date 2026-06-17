"""Does a numba flatten (replacing the numpy flatten) speed up get_intensity?

Keeps the flat-segment kernels UNCHANGED; only swaps how the flat arrays are
built:
  - numpy flatten_labeled : (masks>0)&isfinite mask + numpy.nonzero + 2 gathers
  - numba flatten         : one count scan + one fill scan, coords = loop indices

Measures (a) the flatten step alone and (b) the full non-edge core.
Run: PYTHONPATH=src python tasks/bench_numbaflat.py
"""

import time

import numpy as np
from numba import njit

from cp_measure.core.numba import get_intensity as nb_intensity
from cp_measure.core.measureobjectintensity import get_intensity as np_intensity
from cp_measure.primitives import flatten_labeled, label_to_idx_lut
from cp_measure.primitives._segment_numba import (
    segment_moments,
    segment_quantiles,
    segment_resid_sumsq,
)


@njit(cache=True)
def flatten_numba(masks, pixels, lut):
    """Build (values, seg0, xc, yc, zc) in two grid scans (count, then fill)."""
    Z, Y, X = masks.shape
    M = 0
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                L = masks[z, y, x]
                if L <= 0:
                    continue
                if np.isfinite(pixels[z, y, x]):
                    M += 1
    values = np.empty(M, np.float64)
    seg0 = np.empty(M, np.int64)
    xc = np.empty(M, np.float64)
    yc = np.empty(M, np.float64)
    zc = np.empty(M, np.float64)
    i = 0
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                L = masks[z, y, x]
                if L <= 0:
                    continue
                v = pixels[z, y, x]
                if not np.isfinite(v):
                    continue
                values[i] = v
                seg0[i] = lut[L]
                xc[i] = x
                yc[i] = y
                zc[i] = z
                i += 1
    return values, seg0, xc, yc, zc


def core_from_flat(v, seg0, xc, yc, zc, n, orig_ndim):
    (count, sumI, minI, maxI, mx, my, mz, sx, sy, sz, sxI, syI, szI) = segment_moments(
        v, seg0, xc, yc, zc, n
    )
    cnt = count.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sumI / cnt
        ss = segment_resid_sumsq(v, seg0, n, mean)
        std = np.sqrt(ss / cnt)
        cmi_x = sxI / sumI
        md = np.sqrt((sx / cnt - cmi_x) ** 2)
    lq, med, uq, mad = segment_quantiles(v, seg0, count, n, 1.0 / orig_ndim)
    return sumI, mean, std, minI, maxI, md, lq, med, uq, mad


def core_numbaflat(m, p):
    orig = p.ndim
    if p.ndim == 2:
        p = p.reshape(1, *p.shape)
        m = m.reshape(1, *m.shape)
    lut, n = label_to_idx_lut(m)
    m = np.ascontiguousarray(m)
    p = np.ascontiguousarray(p, dtype=np.float64)
    v, seg0, xc, yc, zc = flatten_numba(m, p, lut)
    return core_from_flat(v, seg0, xc, yc, zc, n, orig)


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
    lut, n = label_to_idx_lut(wm.reshape(1, 64, 64))
    flatten_numba(wm.reshape(1, 64, 64), wp.reshape(1, 64, 64).astype(np.float64), lut)
    core_numbaflat(wm, wp)
    nb_intensity(wm, wp, edge_measurements=False)

    # correctness
    m, p = make(256, 256, 16, seed=1)
    ref = np_intensity(m, p, edge_measurements=False)
    d = core_numbaflat(m, p)
    assert np.allclose(d[0], ref["Intensity_IntegratedIntensity"])
    assert np.allclose(d[7], ref["Intensity_MedianIntensity"])
    assert np.allclose(d[2], ref["Intensity_StdIntensity"])
    print("sanity: numba-flatten core == numpy reference (sum/median/std)  ✓\n")

    for px, nobj in [(256, 16), (1024, 64)]:
        m, p = make(px, px, nobj, seed=1)
        m3 = m.reshape(1, px, px)
        p3 = p.reshape(1, px, px).astype(np.float64)
        lut, n = label_to_idx_lut(m3)

        t_np_flat = med_ms(lambda: flatten_labeled(m3, p3, lut))
        t_nb_flat = med_ms(lambda: flatten_numba(m3, p3, lut))
        t_cur = med_ms(lambda: nb_intensity(m, p, edge_measurements=False))
        t_new = med_ms(lambda: core_numbaflat(m, p))
        print(f"### {px}x{px}, {nobj} obj ###")
        print(f"  flatten step : numpy {t_np_flat:7.2f} ms  vs  numba {t_nb_flat:7.2f} ms  ({t_np_flat / t_nb_flat:4.1f}x)")
        print(f"  FULL core    : current(numpy-flat) {t_cur:7.2f} ms  vs  numba-flat {t_new:7.2f} ms  ({t_cur / t_new:4.1f}x)")
        print()


if __name__ == "__main__":
    main()
