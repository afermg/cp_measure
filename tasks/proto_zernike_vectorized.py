"""Prototype: pure-numpy get_zernike that avoids centrosome's full (H,W,K) scatter
and the 60 per-channel scipy.ndimage.sum calls.

centrosome.zernike.zernike -> construct_zernike_polynomials builds a full (H,W,K) complex
array (~560MB at 1080^2, K~30) then score_zernike loops K channels x2 doing full-image
scipy.ndimage.sum. Both are foreground-sparse waste.

Vectorized: compute the Horner basis on masked (npix,) vectors only (centrosome already
does this internally), then per-label segment-sum via np.add.at over (npix,K). No full
array, no per-channel ndimage scans.

NOTE: segment-sum order != scipy.ndimage.sum order, so float result diverges at ~1e-12
(summation non-associativity) -- not bit-exact. Measure the divergence + speed.
"""

import time
import numpy as np
import centrosome.zernike as cz
import centrosome.cpmorphology as cm
from cp_measure.core.measureobjectsizeshape import get_zernike

DATA = "/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data"


def get_zernike_vectorized(masks, pixels, zernike_numbers=9):
    uniq = np.unique(masks); uniq = uniq[uniq > 0]
    n = len(uniq)
    indices = np.arange(1, n + 1, dtype=np.int32)
    zidx = cz.get_zernike_indexes(zernike_numbers + 1)
    K = zidx.shape[0]

    centers, radii = cz.minimum_enclosing_circle(masks, indices)
    radii = np.asarray(radii, float)

    rev = np.full(int(masks.max()) + 1, -1, int)
    rev[indices] = np.arange(n)
    mask = rev[masks] != -1
    ny, nx = masks.shape[:2]
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    ym = yy[mask]; xm = xx[mask]
    lm = masks[mask]
    ri = rev[lm]                      # per-pixel object index
    ym = (ym - centers[ri, 0]) / radii[ri]
    xm = (xm - centers[ri, 1]) / radii[ri]

    # Horner basis (centrosome's inner loop, masked vectors only)
    lut = cz.construct_zernike_lookuptable(zidx)
    r2 = xm * xm + ym * ym
    zc = ym + 1j * xm
    out_re = np.zeros((n, K)); out_im = np.zeros((n, K))
    z_pows = {}
    for idx in range(K):
        nn, mm = zidx[idx]
        s = np.zeros_like(xm)
        for k in range((nn - mm) // 2 + 1):
            s *= r2; s += lut[idx, k]
        s[r2 > 1] = 0
        if mm == 0:
            zf = s.astype(complex)
        else:
            if mm not in z_pows:
                z_pows[mm] = zc if mm == 1 else zc ** mm
            zf = s * z_pows[mm]
        # segment-sum per object (one add.at over all pixels)
        np.add.at(out_re[:, idx], ri, zf.real)
        np.add.at(out_im[:, idx], ri, zf.imag)

    areas = np.pi * radii * radii
    score = np.sqrt(out_re ** 2 + out_im ** 2) / areas[:, None]
    return {f"Zernike_{nn}_{mm}": score[:, i] for i, (nn, mm) in enumerate(zidx)}


def best(fn, reps=3):
    fn(); t = float("inf")
    for _ in range(reps):
        s = time.perf_counter(); fn(); t = min(t, time.perf_counter() - s)
    return t * 1e3


def main():
    for name in ("small", "large", "m2160"):
        d = np.load(f"{DATA}/{name}.npz")
        m = d["mask_int"]; px = d["pixels"].astype(float)
        ref = get_zernike(m, px)
        new = get_zernike_vectorized(m, px)
        keys = list(ref)
        maxdiff = max(np.abs(ref[k] - new[k]).max() for k in keys)
        maxrel = max((np.abs(ref[k] - new[k]) / (np.abs(ref[k]) + 1e-12)).max() for k in keys)
        t_ref = best(lambda: get_zernike(m, px))
        t_new = best(lambda: get_zernike_vectorized(m, px))
        print(f"[{name}] nobj={len(np.unique(m))-1}  ref={t_ref:7.1f}ms  vec={t_new:7.1f}ms  "
              f"speedup={t_ref/t_new:5.2f}x  maxabs={maxdiff:.2e}  maxrel={maxrel:.2e}")


if __name__ == "__main__":
    main()
