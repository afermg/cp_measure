"""For every numba function on the merged stack: (1) verify the to_bzyx batching API
(single image -> dict; list of B -> list of B dicts; 4D (B,Z,Y,X) -> list; batch-of-1
== single image), and (2) benchmark B separate single-image calls vs ONE batched call.

Our batching is `to_bzyx -> python loop -> unwrap` (serial, NOT a fused kernel), so the
expectation is batched ~= B singles. This measures whether the uniform API costs or saves
anything per call."""

import time
import numpy as np

import cp_measure.core.numba as N

DATA = "/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data"
d = np.load(f"{DATA}/tiny.npz")
m = d["mask_int"].astype(np.int32)
p = d["pixels"].astype(np.float32)
p2 = d["pixels_2"].astype(np.float32)
B = 8

# (name, fn, kind) — kind "sc" = (mask,pixels), "co" = (pixels_1,pixels_2,mask)
FUNCS = [
    ("intensity", N.get_intensity, "sc"),
    ("zernike", N.get_zernike, "sc"),
    ("feret", N.get_feret, "sc"),
    ("granularity", N.get_granularity, "sc"),
    ("texture", N.get_texture, "sc"),
    ("radial_distribution", N.get_radial_distribution, "sc"),
    ("radial_zernikes", N.get_radial_zernikes, "sc"),
    ("coloc_pearson", N.get_correlation_pearson, "co"),
    ("coloc_manders", N.get_correlation_manders_fold, "co"),
    ("coloc_rwc", N.get_correlation_rwc, "co"),
    ("coloc_overlap", N.get_correlation_overlap, "co"),
    ("coloc_costes", N.get_correlation_costes, "co"),
]


def single_args(kind):
    return (m, p) if kind == "sc" else (p, p2, m)


def list_args(kind):
    return ([m] * B, [p] * B) if kind == "sc" else ([p] * B, [p2] * B, [m] * B)


def stacked4d_args(kind):
    ms = np.stack([m[np.newaxis]] * B)  # (B,1,H,W)
    ps = np.stack([p[np.newaxis]] * B)
    p2s = np.stack([p2[np.newaxis]] * B)
    return (ms, ps) if kind == "sc" else (ps, p2s, ms)


def dicts_equal(a, b):
    if set(a) != set(b):
        return False
    return all(np.allclose(a[k], b[k], rtol=1e-6, atol=1e-8, equal_nan=True) for k in a)


def t(fn, args, reps=8):
    fn(*args)  # warmup
    ts = []
    for _ in range(reps):
        s = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - s)
    return min(ts) * 1e3


print(f"batch B={B}, tiny tier ({m.shape[0]}x{m.shape[1]}, {int(m.max())} obj)\n")
print(f"{'function':20s} {'single→dict':>11s} {'list→B':>7s} {'4D→B':>6s} {'b1==single':>10s} "
      f"{'B×single(ms)':>13s} {'batched(ms)':>12s} {'ratio':>6s}")
for name, fn, kind in FUNCS:
    sing = fn(*single_args(kind))
    lst = fn(*list_args(kind))
    st4 = fn(*stacked4d_args(kind))
    ok_single = isinstance(sing, dict)
    ok_list = isinstance(lst, list) and len(lst) == B and all(isinstance(x, dict) for x in lst)
    ok_4d = isinstance(st4, list) and len(st4) == B
    # batch-of-1 invariant
    b1 = fn(*(([m], [p]) if kind == "sc" else ([p], [p2], [m])))
    ok_inv = isinstance(b1, list) and len(b1) == 1 and dicts_equal(b1[0], sing)

    sa = single_args(kind)
    t_loop = t(lambda: [fn(*sa) for _ in range(B)], ())
    t_batch = t(fn, list_args(kind))
    print(f"{name:20s} {str(ok_single):>11s} {str(ok_list):>7s} {str(ok_4d):>6s} "
          f"{str(ok_inv):>10s} {t_loop:13.2f} {t_batch:12.2f} {t_loop / t_batch:5.2f}x")
