"""Prototype two PURE-NUMPY speedups for lanes untouched by PRs #69-73 (granularity, feret).

(1) GRANULARITY upsample: the granular-spectrum loop calls scipy.ndimage.map_coordinates
    on a FIXED resample grid every iteration (only `rec` changes). map_coordinates
    recomputes floor/frac/weights each call. Precompute them ONCE in pure numpy, then each
    iteration is a fancy-indexed weighted sum. Check accuracy vs map_coordinates + speed.

(2) FERET hull input: get_feret feeds the FULL object ijv to centrosome.convex_hull_ijv.
    hull(object) == hull(boundary), so feeding only boundary pixels (mask ^ erosion, pure
    numpy) shrinks the hull input a lot with bit-identical feret. Check point reduction +
    feret-diameter equality + speed.
"""

import time

import numpy as np
import scipy.ndimage
import skimage.morphology


# ---------- (1) granularity upsample ----------
def precompute_gather_2d(orig_shape, new_shape):
    """Pure-numpy equivalent of the fixed map_coordinates(order=1) upsample grid."""
    i = np.arange(orig_shape[0], dtype=float) * (new_shape[0] - 1) / (orig_shape[0] - 1)
    j = np.arange(orig_shape[1], dtype=float) * (new_shape[1] - 1) / (orig_shape[1] - 1)
    i0 = np.floor(i).astype(np.intp); fi = i - i0
    j0 = np.floor(j).astype(np.intp); fj = j - j0
    i1 = np.minimum(i0 + 1, new_shape[0] - 1)
    j1 = np.minimum(j0 + 1, new_shape[1] - 1)
    # weights as 2D outer combos, indices as broadcastable columns/rows
    return (i0[:, None], i1[:, None], fi[:, None],
            j0[None, :], j1[None, :], fj[None, :])


def gather_2d(rec, g):
    i0, i1, fi, j0, j1, fj = g
    v00 = rec[i0, j0]; v01 = rec[i0, j1]; v10 = rec[i1, j0]; v11 = rec[i1, j1]
    return (v00 * (1 - fj) + v01 * fj) * (1 - fi) + (v10 * (1 - fj) + v11 * fj) * fi


def bench_granularity(orig=1080, sub=0.25, ng=16, reps=3, seed=0):
    rng = np.random.default_rng(seed)
    new = int(orig * sub)
    orig_shape = np.array([orig, orig]); new_shape = np.array([new, new])
    recs = [rng.random((new, new)) for _ in range(ng)]  # stand-in for per-iter reconstruction

    # reference: map_coordinates as in main
    i, j = np.mgrid[0:orig, 0:orig].astype(float)
    i *= float(new - 1) / float(orig - 1); j *= float(new - 1) / float(orig - 1)

    def ref():
        return [scipy.ndimage.map_coordinates(r, (i, j), order=1) for r in recs]

    g = precompute_gather_2d(orig_shape, new_shape)

    def fast():
        return [gather_2d(r, g) for r in recs]

    a, b = ref(), fast()
    maxdiff = max(np.abs(x - y).max() for x, y in zip(a, b))

    def timeit(fn):
        fn()
        return min((lambda: (lambda t: (fn(), time.perf_counter() - t)[1])(time.perf_counter()))() for _ in range(reps))

    t_ref = timeit(ref) * 1e3
    t_fast = timeit(fast) * 1e3
    print(f"[granularity upsample] orig={orig}^2 sub={sub} new={new}^2 ng={ng}")
    print(f"  max|map_coordinates - numpy_gather| = {maxdiff:.2e}")
    print(f"  map_coordinates x{ng}: {t_ref:8.2f} ms   numpy_gather x{ng}: {t_fast:8.2f} ms   "
          f"speedup {t_ref / t_fast:.2f}x")


# ---------- (2) feret hull-from-boundary ----------
def bench_feret(size=1080, ngrid=12, reps=3, seed=0):
    import centrosome.cpmorphology
    from cp_measure.utils import masks_to_ijv

    # build non-touching square objects
    labels = np.zeros((size, size), np.int32)
    step = size // ngrid; obj = step * 3 // 4
    lab = 0
    for a in range(ngrid):
        for b in range(ngrid):
            lab += 1
            labels[a * step:a * step + obj, b * step:b * step + obj] = lab
    n = int(labels.max())
    indices = np.arange(1, n + 1)

    # boundary via pure-numpy erosion XOR (4-conn)
    fg = labels > 0
    ero = scipy.ndimage.binary_erosion(fg)
    boundary = fg & ~ero
    bmask = np.where(boundary, labels, 0)

    ijv_full = masks_to_ijv(labels)
    ijv_bnd = masks_to_ijv(bmask)
    print(f"\n[feret hull input] {size}^2  {n} objects")
    print(f"  full ijv points: {len(ijv_full):>9}   boundary ijv points: {len(ijv_bnd):>9}   "
          f"reduction {len(ijv_full) / len(ijv_bnd):.1f}x")

    def feret_from(ijv):
        ch, cc = centrosome.cpmorphology.convex_hull_ijv(ijv, indices)
        return centrosome.cpmorphology.feret_diameter(ch, cc, indices)

    full = feret_from(ijv_full)
    bnd = feret_from(ijv_bnd)
    eq = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(full, bnd))
    print(f"  feret_diameter identical (full vs boundary): {eq}")

    def timeit(fn):
        fn()
        best = float("inf")
        for _ in range(reps):
            t = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t)
        return best * 1e3

    # time hull+feret only (the masks_to_ijv + erosion are shared host prep)
    t_full = timeit(lambda: feret_from(ijv_full))
    t_bnd = timeit(lambda: feret_from(ijv_bnd))
    t_ero = timeit(lambda: scipy.ndimage.binary_erosion(fg))
    print(f"  convex_hull+feret  full: {t_full:7.2f} ms   boundary: {t_bnd:7.2f} ms   "
          f"(+{t_ero:.2f} ms erosion)   net speedup {t_full / (t_bnd + t_ero):.2f}x")


if __name__ == "__main__":
    bench_granularity()
    bench_feret()
