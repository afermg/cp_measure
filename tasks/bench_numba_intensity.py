"""Speed: numba intensity backend vs main (numpy), single image and batched.

Run with the numba env + PYTHONPATH=src:
    PYTHONPATH=src python tasks/bench_numba_intensity.py
"""

import time

import numpy as np

from cp_measure.core.measureobjectintensity import get_intensity as np_intensity
from cp_measure.core.numba import get_intensity as nb_intensity
from cp_measure.primitives import label_to_idx_lut
from cp_measure.primitives._segment_numba import (
    flatten_numba,
    segment_moments,
    segment_quantiles,
    segment_resid_sumsq,
)


def make(H, W, nobj, seed=0):
    """Labeled mask (H,W) with ~nobj square objects on a grid + random pixels."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((H, W), dtype=np.int32)
    g = int(np.ceil(np.sqrt(nobj)))
    cell_h, cell_w = H // g, W // g
    obj = max(min(cell_h, cell_w) // 2, 4)
    lab = 0
    for i in range(g):
        for j in range(g):
            if lab >= nobj:
                break
            r, c = i * cell_h + 2, j * cell_w + 2
            mask[r : r + obj, c : c + obj] = lab + 1
            lab += 1
    return mask, rng.random((H, W))


def median_time(fn, repeats=5):
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def batched_numba_core(masks_list, pixels_list):
    """One batched kernel call over a stack of images (core features, no edge).

    Demonstrates the flat-segment design: per-image label offsets fold the whole
    batch into a single seg0, so segment_moments/quantiles run ONCE for all
    images. (Edge features stay per-image host work and are excluded here.)
    """
    all_v, all_seg, all_x, all_y, all_z, all_cnt = [], [], [], [], [], []
    offset = 0
    for masks, pixels in zip(masks_list, pixels_list):
        m = masks.reshape(1, *masks.shape)
        p = pixels.reshape(1, *pixels.shape)
        lut, n = label_to_idx_lut(m)
        v, seg0, xc, yc, zc = flatten_numba(
            np.ascontiguousarray(m), np.ascontiguousarray(p, dtype=np.float64), lut
        )
        all_v.append(v)
        all_seg.append(seg0 + offset)
        all_x.append(xc)
        all_y.append(yc)
        all_z.append(zc)
        all_cnt.append(n)
        offset += n
    v = np.concatenate(all_v)
    seg = np.concatenate(all_seg)
    xc = np.concatenate(all_x)
    yc = np.concatenate(all_y)
    zc = np.concatenate(all_z)
    n = offset
    count, sumI, *_ = segment_moments(v, seg, xc, yc, zc, n)
    mean = sumI / count.astype(np.float64)
    segment_resid_sumsq(v, seg, n, mean)
    segment_quantiles(v, seg, count, n, 0.5)
    return n


def bench_single(px, nobj):
    m, p = make(px, px, nobj, seed=1)
    t_np = median_time(lambda: np_intensity(m, p))
    t_nb = median_time(lambda: nb_intensity(m, p))
    print(f"--- SINGLE IMAGE  {px}x{px}, {nobj} objects (edge=on) ---")
    print(f"  numpy (main): {t_np * 1e3:9.1f} ms")
    print(f"  numba       : {t_nb * 1e3:9.1f} ms   ({t_np / t_nb:5.1f}x)")


def bench_batch(px, nobj, B):
    masks = [make(px, px, nobj, seed=s)[0] for s in range(B)]
    pix = [make(px, px, nobj, seed=s)[1] for s in range(B)]
    t_np_loop = median_time(
        lambda: [np_intensity(mm, pp) for mm, pp in zip(masks, pix)], repeats=3
    )
    t_nb_loop = median_time(
        lambda: [nb_intensity(mm, pp) for mm, pp in zip(masks, pix)], repeats=3
    )
    t_np_core = median_time(
        lambda: [np_intensity(mm, pp, edge_measurements=False) for mm, pp in zip(masks, pix)],
        repeats=3,
    )
    t_nb_batch = median_time(lambda: batched_numba_core(masks, pix), repeats=3)
    print(f"--- BATCH  {B} images x {px}x{px}, {nobj} obj each ---")
    print(f"  numpy (main), per-image loop : {t_np_loop * 1e3:9.1f} ms")
    print(f"  numba,        per-image loop : {t_nb_loop * 1e3:9.1f} ms   ({t_np_loop / t_nb_loop:5.1f}x)")
    print(f"  numpy (main), per-image core : {t_np_core * 1e3:9.1f} ms   (edge off)")
    print(f"  numba,  ONE batched kernel   : {t_nb_batch * 1e3:9.1f} ms   ({t_np_core / t_nb_batch:5.1f}x vs numpy core)")


def main():
    print(f"numpy {np.__version__}\n")

    # --- warm up the JIT (compile once; cache=True persists) ---
    wm, wp = make(64, 64, 4)
    nb_intensity(wm, wp)
    batched_numba_core([wm], [wp])

    # (px, nobj, batch_size) — object count scales with area; batch shrinks at
    # 1024 to keep numpy's per-object dense-allocation cost bounded.
    for px, nobj, B in [(256, 16, 32), (1024, 64, 8)]:
        print(f"############## {px}px ##############")
        bench_single(px, nobj)
        bench_batch(px, nobj, B)
        print()


if __name__ == "__main__":
    main()
