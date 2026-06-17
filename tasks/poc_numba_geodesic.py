"""POC: is a numba geodesic distance transform faster than centrosome.propagate?

propagate (heap-Dijkstra C) is 80% of radial_distribution. Prototype a numba
chamfer geodesic (iterated forward/backward raster sweeps, O(passes*N), no heap)
from the centre seed within each per-object crop, and compare:
  - SPEED: numba geodesic total vs centrosome propagate total (summed over crops)
  - METRIC: how far the numba chamfer sits from centrosome's distances (it WILL
    differ — this quantifies the divergence a metric-change would introduce)
"""

import time

import centrosome.cpmorphology
import centrosome.propagate
import numpy as np
import scipy.ndimage
from numba import njit

SQ2 = np.sqrt(2.0)


@njit(cache=True)
def geodesic_chamfer(mask, si, sj):
    """Geodesic chamfer distance from (si,sj) within mask (1/sqrt2 weights).

    Iterated forward+backward raster sweeps to convergence — exact for the
    8-neighbour chamfer graph metric, restricted to the mask.
    """
    H, W = mask.shape
    INF = 1e18
    d = np.full((H, W), INF)
    d[si, sj] = 0.0
    changed = True
    while changed:
        changed = False
        for r in range(H):
            for c in range(W):
                if not mask[r, c]:
                    continue
                best = d[r, c]
                if r > 0 and mask[r - 1, c] and d[r - 1, c] + 1.0 < best:
                    best = d[r - 1, c] + 1.0
                if c > 0 and mask[r, c - 1] and d[r, c - 1] + 1.0 < best:
                    best = d[r, c - 1] + 1.0
                if r > 0 and c > 0 and mask[r - 1, c - 1] and d[r - 1, c - 1] + SQ2 < best:
                    best = d[r - 1, c - 1] + SQ2
                if r > 0 and c < W - 1 and mask[r - 1, c + 1] and d[r - 1, c + 1] + SQ2 < best:
                    best = d[r - 1, c + 1] + SQ2
                if best < d[r, c]:
                    d[r, c] = best
                    changed = True
        for r in range(H - 1, -1, -1):
            for c in range(W - 1, -1, -1):
                if not mask[r, c]:
                    continue
                best = d[r, c]
                if r < H - 1 and mask[r + 1, c] and d[r + 1, c] + 1.0 < best:
                    best = d[r + 1, c] + 1.0
                if c < W - 1 and mask[r, c + 1] and d[r, c + 1] + 1.0 < best:
                    best = d[r, c + 1] + 1.0
                if r < H - 1 and c < W - 1 and mask[r + 1, c + 1] and d[r + 1, c + 1] + SQ2 < best:
                    best = d[r + 1, c + 1] + SQ2
                if r < H - 1 and c > 0 and mask[r + 1, c - 1] and d[r + 1, c - 1] + SQ2 < best:
                    best = d[r + 1, c - 1] + SQ2
                if best < d[r, c]:
                    d[r, c] = best
                    changed = True
    return d


def make_crops(size=1080, grid=12, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.zeros((size, size), np.int32)
    step = size // grid
    obj = step * 3 // 4
    lab = 0
    for i in range(grid):
        for j in range(grid):
            lab += 1
            r, c = i * step, j * step
            labels[r : r + obj, c : c + obj] = lab
    crops = []
    for lab, sl in enumerate(scipy.ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        r, c = sl
        sub = labels[max(r.start - 1, 0) : r.stop + 1, max(c.start - 1, 0) : c.stop + 1]
        crops.append(np.pad((sub == lab).astype(np.int32), 1))
    return crops


def t(fn, reps=5):
    best = float("inf")
    out = None
    for _ in range(reps):
        s = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - s)
    return best * 1e3, out


def seed_of(m):
    d = scipy.ndimage.distance_transform_edt(m)
    i, j = centrosome.cpmorphology.maximum_position_of_labels(d, m, [1])
    return int(i[0]), int(j[0])


def main():
    crops = make_crops()
    seeds = [seed_of(m) for m in crops]
    mb = [m.astype(np.bool_) for m in crops]

    # warm up numba
    geodesic_chamfer(mb[0], seeds[0][0], seeds[0][1])

    def run_numba():
        for m, (si, sj) in zip(mb, seeds):
            geodesic_chamfer(m, si, sj)

    def run_centrosome():
        for m, (si, sj) in zip(crops, seeds):
            ctr = np.zeros(m.shape, int)
            ctr[si, sj] = 1
            centrosome.propagate.propagate(np.zeros(m.shape), ctr, m > 0, 1)

    ms_numba, _ = t(run_numba)
    ms_cent, _ = t(run_centrosome)

    # metric divergence on one object
    m, (si, sj) = crops[0], seeds[0]
    dn = geodesic_chamfer(mb[0], si, sj)
    ctr = np.zeros(m.shape, int)
    ctr[si, sj] = 1
    _, dc = centrosome.propagate.propagate(np.zeros(m.shape), ctr, m > 0, 1)
    mm = m > 0
    diff = np.abs(dn[mm] - dc[mm])

    print(f"{len(crops)} object crops (1080^2 grid)\n")
    print(f"centrosome propagate (per-crop sum)  {ms_cent:8.2f} ms")
    print(f"numba chamfer geodesic (per-crop sum){ms_numba:8.2f} ms   ({ms_cent / ms_numba:.2f}x faster)")
    print(f"\nmetric vs centrosome (obj0): maxdiff={diff.max():.4f} meandiff={diff.mean():.4f}")


if __name__ == "__main__":
    main()
