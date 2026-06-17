"""Does scanning the image directly in the kernel (no host flattening) help?

Isolates the moments stage:
  A. host flatten alone        (flatten_labeled)
  B. current: flatten + kernel (flatten_labeled + segment_moments over flat arrays)
  C. direct:  image-scan kernel (moments_from_image, no flat arrays, coords free)

Run: PYTHONPATH=src python tasks/bench_directscan.py
"""

import time

import numpy as np
from numba import njit

from cp_measure.primitives import flatten_labeled, label_to_idx_lut
from cp_measure.primitives._segment_numba import segment_moments


@njit(cache=True)
def moments_from_image(masks, pixels, lut, n):
    """Same accumulation as segment_moments, but scanning the 2D image directly.

    Coordinates are the loop indices (free); no values/seg0/xc/yc/zc arrays and
    no numpy.nonzero / boolean-mask gather on the host.
    """
    H, W = masks.shape
    count = np.zeros(n, np.int64)
    sumI = np.zeros(n, np.float64)
    minI = np.full(n, np.inf)
    maxI = np.full(n, -np.inf)
    mx = np.zeros(n, np.float64)
    my = np.zeros(n, np.float64)
    sx = np.zeros(n, np.float64)
    sy = np.zeros(n, np.float64)
    sxI = np.zeros(n, np.float64)
    syI = np.zeros(n, np.float64)
    for r in range(H):
        for c in range(W):
            L = masks[r, c]
            if L <= 0:
                continue
            v = pixels[r, c]
            if not np.isfinite(v):
                continue
            k = lut[L]
            count[k] += 1
            sumI[k] += v
            if v < minI[k]:
                minI[k] = v
            if v >= maxI[k]:
                maxI[k] = v
                mx[k] = c
                my[k] = r
            sx[k] += c
            sy[k] += r
            sxI[k] += c * v
            syI[k] += r * v
    return count, sumI, minI, maxI, mx, my, sx, sy, sxI, syI


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
            r, c = i * ch + 2, j * cw + 2
            mask[r : r + obj, c : c + obj] = lab + 1
            lab += 1
    return mask, rng.random((H, W))


def med(fn, repeats=7):
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2] * 1e3


def main():
    print(f"numpy {np.__version__}\n")
    # warm up JIT (both kernels)
    wm, wp = make(64, 64, 4)
    lut, n = label_to_idx_lut(wm)
    moments_from_image(wm, wp, lut, n)
    w3m, w3p = wm.reshape(1, 64, 64), wp.reshape(1, 64, 64)
    v, s, x, y, z = flatten_labeled(w3m, w3p, lut)
    segment_moments(v, s, x, y, z, n)

    # correctness sanity on a real case
    m, p = make(256, 256, 16, seed=1)
    lut, n = label_to_idx_lut(m)
    m3, p3 = m.reshape(1, 256, 256), p.reshape(1, 256, 256)
    vC = segment_moments(*flatten_labeled(m3, p3, lut), n)
    vD = moments_from_image(m, p, lut, n)
    assert np.allclose(vC[1], vD[1]) and np.array_equal(vC[0], vD[0])  # sum, count
    print("sanity: direct-scan moments == flat-path moments  ✓\n")

    for px, nobj in [(256, 16), (1024, 64)]:
        m, p = make(px, px, nobj, seed=1)
        lut, n = label_to_idx_lut(m)
        m3, p3 = m.reshape(1, px, px), p.reshape(1, px, px)

        t_flat_only = med(lambda: flatten_labeled(m3, p3, lut))
        t_current = med(lambda: segment_moments(*flatten_labeled(m3, p3, lut), n))
        t_direct = med(lambda: moments_from_image(m, p, lut, n))
        print(f"### {px}x{px}, {nobj} objects (moments stage only) ###")
        print(f"  A. host flatten alone      : {t_flat_only:7.2f} ms")
        print(f"  B. current (flatten+kernel): {t_current:7.2f} ms")
        print(f"  C. direct image-scan kernel: {t_direct:7.2f} ms   ({t_current / t_direct:4.1f}x vs B)")
        print()


if __name__ == "__main__":
    main()
