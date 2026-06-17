"""Where does the numba radial_distribution's ~104ms go? (Amdahl breakdown)

Times the components of the per-object loop: find_objects, per-crop scipy EDT,
per-crop maximum_position, the numba geodesic, the host per-pixel numpy
(nd/bin/wedge/where), concatenate, and the numba reduce.
"""

import time

import centrosome.cpmorphology
import numpy as np
import scipy.ndimage

from cp_measure.core.numba._radial import UNREACHED, geodesic_chamfer_fifo, radial_reduce


def make_image(size=1080, grid=12, seed=0):
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
    return labels, rng.random((size, size))


def main():
    labels, pixels = make_image()
    n = int(labels.max())
    bin_count = 4

    # warm up the numba kernels (cache load / JIT) before timing
    _m = np.pad(labels[scipy.ndimage.find_objects(labels)[0][0],
                       scipy.ndimage.find_objects(labels)[0][1]] == 1, 1)
    geodesic_chamfer_fifo(np.ascontiguousarray(_m), 1, 1)
    radial_reduce(
        np.ones(3), np.zeros(3, np.int64), np.zeros(3, np.int64),
        np.zeros(3, np.int64), 1, bin_count,
    )

    T = {k: 0.0 for k in ("find", "edt", "maxpos", "geo", "host", "reduce")}
    reps = 4
    for _ in range(reps):
        t0 = time.perf_counter()
        slices = scipy.ndimage.find_objects(labels)
        T["find"] += time.perf_counter() - t0
        vals, segs, bins, wedges = [], [], [], []
        for label in range(1, n + 1):
            sl = slices[label - 1]
            t0 = time.perf_counter()
            m = np.pad(labels[sl] == label, 1)
            pix = np.pad(pixels[sl].astype(np.float64), 1)
            T["host"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            d_to_edge = scipy.ndimage.distance_transform_edt(m)
            T["edt"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            ci_a, cj_a = centrosome.cpmorphology.maximum_position_of_labels(
                d_to_edge, m.astype(np.int32), indices=np.array([1])
            )
            ci, cj = int(ci_a[0]), int(cj_a[0])
            T["maxpos"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            d_from = geodesic_chamfer_fifo(np.ascontiguousarray(m), ci, cj)
            T["geo"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            good = m & (d_from < UNREACHED)
            nd = np.zeros(m.shape)
            nd[good] = d_from[good] / (d_from[good] + d_to_edge[good] + 0.001)
            bi = (nd * bin_count).astype(int)
            bi[bi > bin_count] = bin_count
            ii, jj = np.mgrid[0 : m.shape[0], 0 : m.shape[1]]
            wedge = (
                (ii > ci).astype(int)
                + (jj > cj).astype(int) * 2
                + (np.abs(ii - ci) > np.abs(jj - cj)).astype(int) * 4
            )
            gy, gx = np.where(good)
            vals.append(pix[gy, gx])
            segs.append(np.full(gy.size, label - 1, np.int64))
            bins.append(bi[gy, gx].astype(np.int64))
            wedges.append(wedge[gy, gx].astype(np.int64))
            T["host"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        radial_reduce(
            np.ascontiguousarray(np.concatenate(vals)),
            np.concatenate(segs),
            np.concatenate(bins),
            np.concatenate(wedges),
            n,
            bin_count,
        )
        T["reduce"] += time.perf_counter() - t0

    total = sum(T.values()) / reps * 1e3
    print(f"image 1080^2, {n} objects\n")
    for k in ("find", "edt", "maxpos", "geo", "host", "reduce"):
        print(f"{k:<10}{T[k] / reps * 1e3:8.2f} ms  ({T[k] / sum(T.values()) * 100:.0f}%)")
    print(f"{'TOTAL':<10}{total:8.2f} ms")


if __name__ == "__main__":
    main()
