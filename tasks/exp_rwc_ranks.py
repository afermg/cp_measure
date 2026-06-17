"""Experiment: faster per-object dense ranks for RWC.

Current kernel does two per-object numba argsorts (~93ms here). Try instead ONE
numpy lexsort per channel (keyed segment-then-value) + an O(M) linear dense-rank
pass. Verify identical ranks, compare timing.
"""

import time

import numpy as np
from numba import njit

from cp_measure.core.numba._colocalization import _dense_rank
from cp_measure.primitives._segment_numba import flatten_pairs_grouped
from cp_measure.primitives.segment import label_to_idx_lut


def make_image(size=1080, grid=12, seed=0):
    rng = np.random.default_rng(seed)
    masks = np.zeros((size, size), np.int32)
    step = size // grid
    obj = step * 3 // 4
    label = 0
    for i in range(grid):
        for j in range(grid):
            label += 1
            r, c = i * step, j * step
            masks[r : r + obj, c : c + obj] = label
    return masks, rng.random((size, size)), rng.random((size, size))


@njit(cache=True)
def _ranks_per_object_kernel(g, offsets, n):
    """Current approach: argsort each block in numba."""
    out = np.empty(g.shape[0], np.int64)
    for k in range(n):
        lo, hi = offsets[k], offsets[k + 1]
        ranks, _ = _dense_rank(g[lo:hi])
        out[lo:hi] = ranks
    return out


@njit(cache=True)
def _dense_from_order(g, order, seg_ids):
    """O(M) linear dense-rank pass over a segment-then-value sorted order."""
    out = np.empty(g.shape[0], np.int64)
    prev_seg = -1
    r = 0
    prev_val = 0.0
    for idx in range(order.shape[0]):
        p = order[idx]
        s = seg_ids[p]
        v = g[p]
        if s != prev_seg:
            r = 0
            prev_seg = s
        elif v != prev_val:
            r += 1
        out[p] = r
        prev_val = v
    return out


def ranks_lexsort(g, seg_ids):
    order = np.lexsort((g, seg_ids))
    return _dense_from_order(g, order, seg_ids)


def t(fn, reps=7):
    best = float("inf")
    out = None
    for _ in range(reps):
        s = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - s)
    return best * 1e3, out


def main():
    masks, p1, p2 = make_image()
    mc = np.ascontiguousarray(masks)
    lut, n = label_to_idx_lut(mc)
    g1, g2, offs = flatten_pairs_grouped(
        mc[None], np.ascontiguousarray(p1)[None], np.ascontiguousarray(p2)[None], lut, n
    )
    counts = np.diff(offs)
    seg_ids = np.repeat(np.arange(n), counts)

    # warm up
    _ranks_per_object_kernel(g1, offs, n)
    ranks_lexsort(g1, seg_ids)

    ms_cur, r_cur = t(lambda: (_ranks_per_object_kernel(g1, offs, n), _ranks_per_object_kernel(g2, offs, n)))

    def lex_both():
        return ranks_lexsort(g1, seg_ids), ranks_lexsort(g2, seg_ids)

    ms_lex, r_lex = t(lex_both)

    ms_segids, _ = t(lambda: np.repeat(np.arange(n), np.diff(offs)))

    ok1 = np.array_equal(r_cur[0], r_lex[0])
    ok2 = np.array_equal(r_cur[1], r_lex[1])
    print(f"ranks identical: ch1={ok1} ch2={ok2}")
    print(f"current (2x numba argsort blocks):  {ms_cur:8.2f} ms")
    print(f"lexsort + linear pass (2 channels): {ms_lex:8.2f} ms")
    print(f"  (seg_ids build alone:             {ms_segids:8.2f} ms, built once for both)")


if __name__ == "__main__":
    main()
