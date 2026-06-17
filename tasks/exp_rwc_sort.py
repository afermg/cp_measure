"""Experiment: exact, faster per-object dense ranks for RWC.

Constraint: ranks must be bit-identical to the current argsort approach (which is
itself bit-identical to scipy rankdata 'dense'-1). Compare numba variants:
  A. current     : np.argsort + linear assign
  B. sort+ssorted: np.sort -> distinct -> np.searchsorted (no index moves)
  C. preallocated: A but reusing one scratch buffer (isolate alloc overhead)
"""

import time

import numpy as np
from numba import njit

from cp_measure.primitives._segment_numba import flatten_pairs_grouped
from cp_measure.primitives.segment import labels_to_offsets


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
def ranks_A(g, offsets, n):
    out = np.empty(g.shape[0], np.int64)
    for k in range(n):
        lo, hi = offsets[k], offsets[k + 1]
        vals = g[lo:hi]
        order = np.argsort(vals)
        r = 0
        out[lo + order[0]] = 0
        for j in range(1, order.shape[0]):
            if vals[order[j]] != vals[order[j - 1]]:
                r += 1
            out[lo + order[j]] = r
    return out


@njit(cache=True)
def ranks_B(g, offsets, n):
    out = np.empty(g.shape[0], np.int64)
    for k in range(n):
        lo, hi = offsets[k], offsets[k + 1]
        c = hi - lo
        vals = g[lo:hi]
        s = np.sort(vals)
        distinct = np.empty(c, np.float64)
        d = 0
        distinct[0] = s[0]
        for j in range(1, c):
            if s[j] != s[j - 1]:
                d += 1
                distinct[d] = s[j]
        distinct = distinct[: d + 1]
        for j in range(c):
            out[lo + j] = np.searchsorted(distinct, vals[j])
    return out


@njit(cache=True)
def ranks_C(g, offsets, n):
    out = np.empty(g.shape[0], np.int64)
    scratch = np.empty(g.shape[0], np.int64)  # reused order buffer
    for k in range(n):
        lo, hi = offsets[k], offsets[k + 1]
        vals = g[lo:hi]
        order = np.argsort(vals)
        scratch[: order.shape[0]] = order
        r = 0
        out[lo + scratch[0]] = 0
        for j in range(1, order.shape[0]):
            if vals[scratch[j]] != vals[scratch[j - 1]]:
                r += 1
            out[lo + scratch[j]] = r
    return out


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
    lut, n, offs = labels_to_offsets(mc)
    g1, g2 = flatten_pairs_grouped(
        mc[None], np.ascontiguousarray(p1)[None], np.ascontiguousarray(p2)[None], lut, offs
    )
    for f in (ranks_A, ranks_B, ranks_C):
        f(g1, offs, n)  # warm up
    rA = ranks_A(g1, offs, n)
    rB = ranks_B(g1, offs, n)
    rC = ranks_C(g1, offs, n)
    print(f"B==A: {np.array_equal(rA, rB)}   C==A: {np.array_equal(rA, rC)}")

    msA, _ = t(lambda: (ranks_A(g1, offs, n), ranks_A(g2, offs, n)))
    msB, _ = t(lambda: (ranks_B(g1, offs, n), ranks_B(g2, offs, n)))
    msC, _ = t(lambda: (ranks_C(g1, offs, n), ranks_C(g2, offs, n)))
    print(f"A argsort+linear        : {msA:7.2f} ms")
    print(f"B sort+searchsorted     : {msB:7.2f} ms")
    print(f"C argsort, prealloc      : {msC:7.2f} ms")


if __name__ == "__main__":
    main()
