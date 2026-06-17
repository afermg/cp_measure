"""Grand comparison: main vs Alan #55 numpy vs our numba (single & batched).

PYTHONPATH=src python tasks/bench_grand.py
(reads Alan's #55 intensity from /tmp/alan_intensity.py)
"""

import importlib.util
import time

import numpy as np

from cp_measure.core.measureobjectintensity import get_intensity as main_intensity
from cp_measure.core.numba import get_intensity as nb_intensity
from cp_measure.primitives import label_to_idx_lut
from cp_measure.primitives._segment_numba import (
    flatten_numba,
    segment_moments,
    segment_quantiles,
    segment_resid_sumsq,
)

spec = importlib.util.spec_from_file_location("alan_intensity", "/tmp/alan_intensity.py")
alan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alan)


def batched_numba_core(masks_list, pixels_list):
    """One batched kernel call over a stack (core features, no edge)."""
    all_v, all_seg, all_x, all_y, all_z = [], [], [], [], []
    offset = 0
    for masks, pixels in zip(masks_list, pixels_list):
        m = masks.reshape(1, *masks.shape)
        p = pixels.reshape(1, *pixels.shape)
        lut, n = label_to_idx_lut(m)
        v, seg0, xc, yc, zc = flatten_numba(
            np.ascontiguousarray(m), np.ascontiguousarray(p), lut
        )
        all_v.append(v)
        all_seg.append(seg0 + offset)
        all_x.append(xc)
        all_y.append(yc)
        all_z.append(zc)
        offset += n
    v = np.concatenate(all_v)
    seg = np.concatenate(all_seg)
    xc, yc, zc = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_z)
    n = offset
    count, sumI, *_ = segment_moments(v, seg, xc, yc, zc, n)
    mean = sumI / count.astype(np.float64)
    segment_resid_sumsq(v, seg, n, mean)
    segment_quantiles(v, seg, count, n, 0.5)
    return n


def make(H, W, nobj, seed=0):
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


def main():
    print(f"numpy {np.__version__}\n")
    wm, wp = make(64, 64, 4)
    nb_intensity(wm, wp)
    batched_numba_core([wm], [wp])

    print("================= SINGLE IMAGE (edge on), ms =================")
    print(f"{'scene':22s}{'main':>10s}{'alan#55':>10s}{'numba':>10s}   numba vs alan")
    for px, nobj in [(256, 16), (512, 200), (1024, 64)]:
        m, p = make(px, px, nobj, seed=1)
        t_main = med(lambda: main_intensity(m, p), repeats=3)
        t_alan = med(lambda: alan.get_intensity(m, p))
        t_nb = med(lambda: nb_intensity(m, p))
        print(f"{f'{px}^2/{nobj}obj':22s}{t_main:10.1f}{t_alan:10.1f}{t_nb:10.1f}   {t_alan / t_nb:4.1f}x")

    print("\n================= BATCH (ms, total) =================")
    for px, nobj, B in [(256, 16, 32), (1024, 64, 8)]:
        masks = [make(px, px, nobj, seed=s)[0] for s in range(B)]
        pix = [make(px, px, nobj, seed=s)[1] for s in range(B)]
        a_loop = med(lambda: [alan.get_intensity(mm, pp) for mm, pp in zip(masks, pix)], repeats=3)
        nb_loop = med(lambda: [nb_intensity(mm, pp) for mm, pp in zip(masks, pix)], repeats=3)
        a_core = med(lambda: [alan.get_intensity(mm, pp, edge_measurements=False) for mm, pp in zip(masks, pix)], repeats=3)
        nb_batch = med(lambda: batched_numba_core(masks, pix), repeats=3)
        print(f"\n--- {B} x {px}^2, {nobj} obj ---")
        print(f"  alan#55  per-image (edge on)   : {a_loop:9.1f}")
        print(f"  numba    per-image (edge on)   : {nb_loop:9.1f}   ({a_loop / nb_loop:4.1f}x vs alan)")
        print(f"  alan#55  per-image core        : {a_core:9.1f}   (edge off)")
        print(f"  numba    ONE batched kernel    : {nb_batch:9.1f}   ({a_core / nb_batch:4.1f}x vs alan core)")


if __name__ == "__main__":
    main()
