"""Is centrosome.propagate(zeros, seed, mask, 1) == Euclidean EDT-from-seed?

propagate (the 80% cost) computes geodesic distance from the centre seed within
the mask. If that equals a plain distance_transform_edt from the seed, we can
replace the C Dijkstra with a (numba-able) EDT — a huge win. Geodesic == Euclidean
only when straight seed->pixel lines stay inside the mask, so test convex AND
concave AND touching shapes and report the max divergence (and whether it shifts
the resulting distance bins).
"""

import centrosome.cpmorphology
import centrosome.propagate
import numpy as np
import scipy.ndimage


def propagate_d(mask):
    """d_from_center exactly as the baseline computes it for one object."""
    labels = mask.astype(np.int32)
    d_to_edge = scipy.ndimage.distance_transform_edt(mask)
    i, j = centrosome.cpmorphology.maximum_position_of_labels(d_to_edge, labels, [1])
    center = np.zeros(mask.shape, int)
    center[i, j] = 1
    _, d = centrosome.propagate.propagate(np.zeros(mask.shape), center, mask, 1)
    return d, (i, j), d_to_edge


def edt_from_seed(mask, ij):
    """Euclidean distance to the seed, restricted to the mask."""
    seed_complement = np.ones(mask.shape, bool)
    seed_complement[ij] = False
    d = scipy.ndimage.distance_transform_edt(seed_complement)
    return d


def report(name, mask):
    dprop, ij, _ = propagate_d(mask)
    dedt = edt_from_seed(mask, ij)
    m = mask > 0
    diff = np.abs(dprop[m] - dedt[m])
    # also: do the two give the same 4-bin assignment (scaled)?
    dte = scipy.ndimage.distance_transform_edt(mask)[m]
    nd_p = dprop[m] / (dprop[m] + dte + 0.001)
    nd_e = dedt[m] / (dedt[m] + dte + 0.001)
    bins_p = np.minimum((nd_p * 4).astype(int), 4)
    bins_e = np.minimum((nd_e * 4).astype(int), 4)
    bin_mismatch = np.sum(bins_p != bins_e)
    print(
        f"{name:<22} maxdiff={diff.max():8.4f}  meandiff={diff.mean():7.4f}  "
        f"bin-mismatch={bin_mismatch:5d}/{m.sum()} ({100 * bin_mismatch / m.sum():.1f}%)"
    )


def main():
    # convex square
    sq = np.zeros((60, 60), bool)
    sq[5:55, 5:55] = True
    report("convex square", sq)

    # concave L
    L = np.zeros((60, 60), bool)
    L[5:55, 5:25] = True
    L[35:55, 5:55] = True
    report("concave L", L)

    # concave U
    U = np.zeros((60, 60), bool)
    U[5:55, 5:55] = True
    U[5:40, 20:40] = False
    report("concave U", U)

    # thin crescent / ring-ish
    yy, xx = np.mgrid[0:80, 0:80]
    r = np.sqrt((yy - 40) ** 2 + (xx - 40) ** 2)
    ring = (r > 20) & (r < 35)
    report("ring", ring)

    # #22 example object 1 (irregular)
    obj = np.zeros((240, 240), bool)
    obj[50:100, 50:100] = True
    obj[80:120, 90:120] = True
    report("#22 object 1", obj)


if __name__ == "__main__":
    main()
