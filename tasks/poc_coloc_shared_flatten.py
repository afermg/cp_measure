"""POC: validate the ~3x coloc shared-flatten recommendation empirically.

The featurizer runs pearson + manders_fold + rwc as 3 separate wrappers on the
SAME (pixels_1, pixels_2, masks) per channel pair. Each wrapper re-does
labels_to_offsets + flatten_pairs_grouped + coloc_per_object, but coloc_per_object
ALREADY returns all 9 features. So one _run(compute_rwc=True) call provides
everything; the other two calls are pure redundancy. This times the redundancy.
"""

import time

import numpy as np

from cp_measure.core.numba.measurecolocalization import _run

rng = np.random.default_rng(0)
side, n = 1080, 144
yy, xx = np.mgrid[0:side, 0:side]
c = rng.integers(0, side, size=(n, 2))
best = np.full((side, side), np.inf)
lab = np.zeros((side, side), np.int32)
for i, (cy, cx) in enumerate(c, 1):
    d = (yy - cy) ** 2 + (xx - cx) ** 2
    m = d < best
    best[m] = d[m]
    lab[m] = i
m3 = lab[np.newaxis]  # (Z,Y,X)
p1 = rng.random((side, side))[np.newaxis]
p2 = rng.random((side, side))[np.newaxis]


def current_3_calls():
    # what the featurizer does today: pearson (rwc=F), manders (rwc=F), rwc (rwc=T)
    _run(m3, p1, p2, 0.15, False)
    _run(m3, p1, p2, 0.15, False)
    _run(m3, p1, p2, 0.15, True)


def shared_1_call():
    # one call gives all 9 outputs -> slice out pearson/manders/rwc/overlap
    return _run(m3, p1, p2, 0.15, True)


def t(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - s)
    return min(ts) * 1e3


a = t(current_3_calls)
b = t(shared_1_call)
print(f"current (3 separate _run calls): {a:7.2f} ms")
print(f"shared  (1 _run, fan-out 9-tuple): {b:7.2f} ms")
print(f"speedup on the coloc prep+kernel: {a / b:.2f}x")
print("(per channel pair; the featurizer pays this for every pair, so it compounds)")
