"""Step 0 profile for the radial_distribution numba lane.

Two questions:
  (A) In the numpy baseline, how much time is the centrosome GEOMETRY (which we
      keep / import) vs the REDUCIBLE sparse-histogram + wedge-CV accumulation
      (the numba target)? This sets the achievable speedup ceiling.
  (B) Does the Issue-#22 per-object-crop approach (N small-array geometry calls)
      cost more or less than the one whole-image geometry pass?
"""

import time

import centrosome.cpmorphology
import centrosome.propagate
import numpy as np
import scipy.ndimage
import scipy.sparse


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


def t(fn, reps=5):
    best = float("inf")
    out = None
    for _ in range(reps):
        s = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - s)
    return best * 1e3, out


def whole_image_geometry(labels):
    """The baseline geometry block (everything before the sparse accumulation)."""
    unique = np.unique(labels)
    unique = unique[unique > 0]
    d_to_edge = centrosome.cpmorphology.distance_to_edge(labels)
    i, j = centrosome.cpmorphology.maximum_position_of_labels(d_to_edge, labels, unique)
    center_labels = np.zeros(labels.shape, int)
    center_labels[i, j] = labels[i, j]
    colors = centrosome.cpmorphology.color_labels(labels)
    ncolors = int(np.max(colors))
    d_from_center = np.zeros(labels.shape)
    cl = np.zeros(labels.shape, int)
    for color in range(1, ncolors + 1):
        mask = colors == color
        l_, d = centrosome.propagate.propagate(
            np.zeros(center_labels.shape), center_labels, mask, 1
        )
        d_from_center[mask] = d[mask]
        cl[mask] = l_[mask]
    return d_to_edge, d_from_center, cl, ncolors


def per_object_crop_geometry(labels):
    """The Issue-#22 fix: geometry per object on a cropped + 1px-padded sub-image."""
    slices = scipy.ndimage.find_objects(labels)
    n = 0
    for lab, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        n += 1
        r, c = sl
        sub = labels[
            max(r.start - 1, 0) : r.stop + 1, max(c.start - 1, 0) : c.stop + 1
        ]
        m = (sub == lab).astype(np.int32)
        m = np.pad(m, 1)  # ensure background border
        d_to_edge = centrosome.cpmorphology.distance_to_edge(m)
        i, j = centrosome.cpmorphology.maximum_position_of_labels(d_to_edge, m, [1])
        center = np.zeros(m.shape, int)
        center[i, j] = m[i, j]
        centrosome.propagate.propagate(np.zeros(m.shape), center, m > 0, 1)
    return n


def sparse_accumulation(labels, pixels, d_to_edge, d_from_center, cl, bin_count=4):
    """The reducible part: histograms + per-bin 8-wedge CV (numba target)."""
    unique = np.unique(labels)
    unique = unique[unique > 0]
    nobjects = len(unique)
    good_mask = cl > 0
    nd = np.zeros(labels.shape)
    total = d_from_center + d_to_edge
    nd[good_mask] = d_from_center[good_mask] / (total[good_mask] + 0.001)
    good_labels = labels[good_mask]
    bin_indexes = (nd * bin_count).astype(int)
    bin_indexes[bin_indexes > bin_count] = bin_count
    lab_bins = (good_labels - 1, bin_indexes[good_mask])
    hist = scipy.sparse.coo_matrix(
        (pixels[good_mask], lab_bins), (nobjects, bin_count + 1)
    ).toarray()
    num = scipy.sparse.coo_matrix(
        (np.ones(int(good_mask.sum())), lab_bins), (nobjects, bin_count + 1)
    ).toarray()
    # per-bin wedge CV (the inner loop) — abbreviated to the sparse builds
    for b in range(bin_count):
        bm = good_mask & (bin_indexes == b)
        scipy.sparse.coo_matrix(
            (pixels[bm], (labels[bm] - 1, np.zeros(int(bm.sum()), int))), (nobjects, 8)
        ).toarray()
    return hist, num


def main():
    labels, pixels = make_image()
    n = int(labels.max())
    print(f"image 1080^2, {n} objects\n")

    ms_geom, (d_to_edge, d_from_center, cl, ncolors) = t(
        lambda: whole_image_geometry(labels)
    )
    ms_acc, _ = t(
        lambda: sparse_accumulation(labels, pixels, d_to_edge, d_from_center, cl)
    )
    ms_crop, _ = t(lambda: per_object_crop_geometry(labels))

    print(f"whole-image geometry (KEEP/import)   {ms_geom:8.2f} ms   ncolors={ncolors}")
    print(f"sparse accumulation  (numba target)  {ms_acc:8.2f} ms")
    print(f"  -> reducible fraction of (geom+acc): {ms_acc / (ms_geom + ms_acc) * 100:.0f}%")
    print(f"per-object-crop geometry (#22 fix)   {ms_crop:8.2f} ms   ({ms_crop / ms_geom:.2f}x whole-image)")


if __name__ == "__main__":
    main()
