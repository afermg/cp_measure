"""Component profile of the numba colocalization path.

Splits the per-call cost into to_bzyx / label_to_idx_lut (find_objects) /
flatten_pairs_grouped / coloc_per_object (rwc off vs on), to see what the ~12.7ms
floor and rwc's 107ms are actually made of.
"""

import time

import numpy as np

from cp_measure.core.numba._colocalization import coloc_per_object
from cp_measure.primitives._segment_numba import flatten_pairs_grouped
from cp_measure.primitives.segment import label_to_idx_lut
from cp_measure.primitives.shapes import to_bzyx


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
    masks_c = np.ascontiguousarray(masks)
    p1c = np.ascontiguousarray(p1)
    p2c = np.ascontiguousarray(p2)

    ms_bzyx, _ = t(lambda: to_bzyx(masks, p1))
    ms_lut, (lut, n) = t(lambda: label_to_idx_lut(masks_c))
    ms_flat, (g1, g2, offs) = t(
        lambda: flatten_pairs_grouped(masks_c[None], p1c[None], p2c[None], lut, n)
    )
    # warm up both kernel specialisations
    coloc_per_object(g1, g2, offs, n, 0.15, False)
    coloc_per_object(g1, g2, offs, n, 0.15, True)
    ms_kern_norwc, _ = t(lambda: coloc_per_object(g1, g2, offs, n, 0.15, False))
    ms_kern_rwc, _ = t(lambda: coloc_per_object(g1, g2, offs, n, 0.15, True))

    # cost of the two per-object argsorts alone
    def argsorts():
        for k in range(n):
            np.argsort(g1[offs[k] : offs[k + 1]])
            np.argsort(g2[offs[k] : offs[k + 1]])

    ms_argsort, _ = t(argsorts)

    M = int(offs[-1])
    print(f"image 1080^2, {n} objects, {M} masked px ({M / n:.0f} px/obj)\n")
    print(f"{'to_bzyx':<26}{ms_bzyx:>8.2f} ms")
    print(f"{'label_to_idx_lut':<26}{ms_lut:>8.2f} ms   (scipy find_objects)")
    print(f"{'flatten_pairs_grouped':<26}{ms_flat:>8.2f} ms   (2 grid scans)")
    print(f"{'kernel (rwc off)':<26}{ms_kern_norwc:>8.2f} ms")
    print(f"{'kernel (rwc on)':<26}{ms_kern_rwc:>8.2f} ms")
    print(f"{'  └ argsorts alone (numpy)':<26}{ms_argsort:>8.2f} ms")
    floor = ms_bzyx + ms_lut + ms_flat + ms_kern_norwc
    print(f"\nsort-free floor (sum)      {floor:>8.2f} ms")
    print(f"rwc total (sum)            {ms_bzyx + ms_lut + ms_flat + ms_kern_rwc:>8.2f} ms")


if __name__ == "__main__":
    main()
