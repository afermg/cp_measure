"""Tiered benchmark of the MERGED numba stack (integration/all-numba) vs the numpy
baseline, on the real Cell-Painting tiers (tiny 256^2 / small 540^2 / large 1080^2).
Per-function speedup + the JOINT speedup (Sigma baseline / Sigma numba) — the latter
is what a full featurize() would see if all numba PRs were merged. Single-thread pinned."""

import time
import numpy as np

import cp_measure.core.measureobjectintensity as b_int
import cp_measure.core.measuregranularity as b_gran
import cp_measure.core.measureobjectsizeshape as b_ss
import cp_measure.core.measureobjectintensitydistribution as b_rad
import cp_measure.core.measurecolocalization as b_col
import cp_measure.core.measuretexture as b_tex
import cp_measure.core.numba as N

DATA = "/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data"

# key -> (numpy_fn, numba_fn_or_None, args_builder). args_builder(m, p, p2) -> (args, kwargs)
SC = lambda m, p, p2: ((m, p), {})           # single-channel (mask, pixels)
CO = lambda m, p, p2: ((p, p2, m), {})        # coloc (pixels_1, pixels_2, mask)
GF = lambda m, p, p2: ((m, p), {"subsample_size": 1.0, "image_sample_size": 1.0})

FUNCS = [
    ("intensity",            b_int.get_intensity,             N.get_intensity,            SC),
    ("sizeshape",            b_ss.get_sizeshape,              None,                       SC),  # not ported
    ("zernike",              b_ss.get_zernike,                N.get_zernike,              SC),
    ("feret",                b_ss.get_feret,                  N.get_feret,                SC),
    ("granularity",          b_gran.get_granularity,          N.get_granularity,          SC),
    ("texture",              b_tex.get_texture,               N.get_texture,              SC),
    ("radial_distribution*", b_rad.get_radial_distribution,   N.get_radial_distribution,  SC),  # *#22 divergence
    ("radial_zernikes",      b_rad.get_radial_zernikes,       N.get_radial_zernikes,      SC),
    ("coloc_pearson",        b_col.get_correlation_pearson,   N.get_correlation_pearson,  CO),
    ("coloc_manders",        b_col.get_correlation_manders_fold, N.get_correlation_manders_fold, CO),
    ("coloc_rwc",            b_col.get_correlation_rwc,       N.get_correlation_rwc,      CO),
    ("coloc_overlap",        b_col.get_correlation_overlap,   N.get_correlation_overlap,  CO),
    ("coloc_costes",         b_col.get_correlation_costes,    N.get_correlation_costes,   CO),
    ("granularity_fullres",  b_gran.get_granularity,          N.get_granularity,          GF),
]


def bench(fn, args, kwargs, reps):
    fn(*args, **kwargs)  # warmup / numba compile
    ts = []
    for _ in range(reps):
        s = time.perf_counter()
        fn(*args, **kwargs)
        ts.append(time.perf_counter() - s)
    return min(ts) * 1e3


for tier, reps in [("tiny", 15), ("small", 8), ("large", 4)]:
    d = np.load(f"{DATA}/{tier}.npz")
    m = d["mask_int"].astype(np.int32)
    p = d["pixels"].astype(np.float32)
    p2 = d["pixels_2"].astype(np.float32)
    nobj = int(m.max())
    print(f"\n================ {tier}  ({m.shape[0]}x{m.shape[1]}, {nobj} obj) ================")
    print(f"{'function':22s} {'numpy(ms)':>11s} {'numba(ms)':>11s} {'speedup':>9s}")
    sum_b = sum_n = 0.0  # joint over the DEFAULT set (excludes granularity_fullres)
    for key, fb, fn, build in FUNCS:
        args, kwargs = build(m, p, p2)
        try:
            tb = bench(fb, args, kwargs, reps)
        except Exception as e:
            print(f"{key:22s} baseline ERR: {type(e).__name__}: {e}")
            continue
        if fn is None:  # not ported -> numba == numpy
            tn = tb
            tag = "  (=numpy, not ported)"
        else:
            try:
                tn = bench(fn, args, kwargs, reps)
                tag = ""
            except Exception as e:
                print(f"{key:22s} numba ERR: {type(e).__name__}: {e}")
                continue
        print(f"{key:22s} {tb:11.2f} {tn:11.2f} {tb / tn:8.2f}x{tag}")
        if key != "granularity_fullres":
            sum_b += tb
            sum_n += tn
    print(f"{'-' * 56}")
    print(f"{'JOINT (default cfg)':22s} {sum_b:11.2f} {sum_n:11.2f} {sum_b / sum_n:8.2f}x"
          f"   <- full featurize speedup")
