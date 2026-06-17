"""Does spreading many independent images across CPU cores save time? Process M images
through the FULL numba feature set, sequentially vs ProcessPoolExecutor at K workers.
Per-worker warmup (import + numba compile/cache-load) runs BEFORE timing, so we measure
STEADY-STATE throughput — the regime a 10k-image run lives in, not one-off startup.

Kernels are single-threaded (no prange); parallelism lives only here in the batch layer."""

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

DATA = "/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data"
TIER = "small"  # 540^2 / 43 obj — representative single-cell field
M = 96          # images per timing run (replicated; each is independent work)

_G = {}


def _init():
    import cp_measure.core.numba as N

    d = np.load(f"{DATA}/{TIER}.npz")
    _G["m"] = d["mask_int"].astype(np.int32)
    _G["p"] = d["pixels"].astype(np.float32)
    _G["p2"] = d["pixels_2"].astype(np.float32)
    _G["fns_sc"] = [N.get_intensity, N.get_zernike, N.get_feret, N.get_granularity,
                    N.get_texture, N.get_radial_distribution, N.get_radial_zernikes]
    _G["fns_co"] = [N.get_correlation_pearson, N.get_correlation_manders_fold,
                    N.get_correlation_rwc, N.get_correlation_overlap, N.get_correlation_costes]
    _process(0)  # warmup: compile / load cached kernels in THIS worker before timing


def _process(_i):
    m, p, p2 = _G["m"], _G["p"], _G["p2"]
    for f in _G["fns_sc"]:
        f(m, p)
    for f in _G["fns_co"]:
        f(p, p2, m)
    return 1


def run(workers, reps=2):
    best = None
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        list(ex.map(_process, range(workers)))  # ensure all workers warmed
        for _ in range(reps):
            s = time.perf_counter()
            list(ex.map(_process, range(M), chunksize=max(1, M // (workers * 4))))
            dt = time.perf_counter() - s
            best = dt if best is None else min(best, dt)
    return best


if __name__ == "__main__":
    print(f"tier={TIER}, {M} images, full numba feature set per image, {len(os.sched_getaffinity(0))} cores allocated\n")
    print(f"{'workers':>8s} {'wall(s)':>9s} {'img/s':>8s} {'speedup':>8s} {'efficiency':>10s}")
    t1 = None
    for k in [1, 2, 4, 8, 16]:
        dt = run(k)
        if t1 is None:
            t1 = dt
        thr = M / dt
        sp = t1 / dt
        print(f"{k:8d} {dt:9.2f} {thr:8.1f} {sp:7.2f}x {100 * sp / k:9.0f}%")
