"""Split the radial_distribution geometry cost to find the real numba target.

Times each centrosome/scipy primitive separately, whole-image vs per-object-crop,
so we know which piece dominates and how much the #22 per-object-crop restructure
saves (incl. eliminating the doubled color_labels).
"""

import time

import centrosome.cpmorphology
import centrosome.propagate
import numpy as np
import scipy.ndimage


def make_image(size=1080, grid=12, seed=0, touching=False):
    rng = np.random.default_rng(seed)
    labels = np.zeros((size, size), np.int32)
    step = size // grid
    obj = step if touching else step * 3 // 4
    lab = 0
    for i in range(grid):
        for j in range(grid):
            lab += 1
            r, c = i * step, j * step
            labels[r : r + obj, c : c + obj] = lab
    return labels, rng.random((size, size))


def t(fn, reps=5):
    best = float("inf")
    out = None
    for _ in range(reps):
        s = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - s)
    return best * 1e3, out


def main():
    labels, pixels = make_image()
    unique = np.unique(labels)
    unique = unique[unique > 0]
    n = len(unique)
    print(f"image 1080^2, {n} objects (non-touching)\n")

    # --- whole-image primitives ---
    ms_color, colors = t(lambda: centrosome.cpmorphology.color_labels(labels))
    ms_dte, d_to_edge = t(lambda: centrosome.cpmorphology.distance_to_edge(labels))
    ms_maxpos, _ = t(
        lambda: centrosome.cpmorphology.maximum_position_of_labels(
            d_to_edge, labels, unique
        )
    )

    center_labels = np.zeros(labels.shape, int)
    i, j = centrosome.cpmorphology.maximum_position_of_labels(d_to_edge, labels, unique)
    center_labels[i, j] = labels[i, j]

    def whole_propagate():
        ncolors = int(np.max(colors))
        for color in range(1, ncolors + 1):
            mask = colors == color
            centrosome.propagate.propagate(
                np.zeros(labels.shape), center_labels, mask, 1
            )

    ms_prop, _ = t(whole_propagate)

    print("WHOLE-IMAGE primitives:")
    print(f"  color_labels                {ms_color:8.2f} ms  (called 2x in baseline!)")
    print(f"  distance_to_edge (EDT)      {ms_dte:8.2f} ms  (= color_labels + scipy EDT)")
    print(f"  maximum_position_of_labels  {ms_maxpos:8.2f} ms")
    print(f"  propagate (C Dijkstra)      {ms_prop:8.2f} ms")

    # --- per-object-crop split: EDT vs propagate vs maxpos, summed over objects ---
    slices = scipy.ndimage.find_objects(labels)
    crops = []
    for lab, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        r, c = sl
        sub = labels[max(r.start - 1, 0) : r.stop + 1, max(c.start - 1, 0) : c.stop + 1]
        crops.append((np.pad((sub == lab).astype(np.int32), 1)))

    def crop_edt():
        for m in crops:
            scipy.ndimage.distance_transform_edt(m)

    def crop_prop():
        for m in crops:
            d = scipy.ndimage.distance_transform_edt(m)
            ii, jj = centrosome.cpmorphology.maximum_position_of_labels(d, m, [1])
            ctr = np.zeros(m.shape, int)
            ctr[ii, jj] = m[ii, jj]
            centrosome.propagate.propagate(np.zeros(m.shape), ctr, m > 0, 1)

    ms_crop_edt, _ = t(crop_edt)
    ms_crop_all, _ = t(crop_prop)
    print("\nPER-OBJECT-CROP (sum over objects, no color_labels needed):")
    print(f"  EDT only                    {ms_crop_edt:8.2f} ms")
    print(f"  EDT + maxpos + propagate    {ms_crop_all:8.2f} ms")


if __name__ == "__main__":
    main()
