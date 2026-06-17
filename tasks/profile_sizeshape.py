"""Step-0 profile for the sizeshape numba lane.

Splits get_sizeshape's cost into:
  (A) skimage.regionprops_table  -- numerically-sensitive geometry (IMPORT boundary)
  (B) per-object EDT radius loop -- scipy EDT (import) + trivial max/mean/median reductions

Only (B)'s reductions are reimplementable; the EDT itself is exact-Euclidean (import,
like radial). This measures the Amdahl ceiling: if (A) dominates, a numba kernel for the
radius reductions buys almost nothing -- document the verdict.
"""

import time

import centrosome.cpmorphology
import numpy as np
import scipy.ndimage
import skimage.measure

from cp_measure.core.measureobjectsizeshape import get_sizeshape

rng = np.random.default_rng(0)


def make_image(side=1080, n=144):
    """Voronoi-ish blobby labels, like the texture/radial benchmarks."""
    masks = np.zeros((side, side), np.int32)
    centers = rng.integers(0, side, size=(n, 2))
    yy, xx = np.mgrid[0:side, 0:side]
    # assign each pixel to nearest center (cheap KD-free for the bench)
    lab = np.zeros((side, side), np.int32)
    best = np.full((side, side), np.inf)
    for i, (cy, cx) in enumerate(centers, start=1):
        d = (yy - cy) ** 2 + (xx - cx) ** 2
        m = d < best
        best[m] = d[m]
        lab[m] = i
    masks = lab
    pixels = rng.random((side, side)).astype(np.float32)
    return masks, pixels


def time_it(fn, *a, reps=3):
    fn(*a)  # warmup / cache
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn(*a)
        ts.append(time.perf_counter() - t)
    return min(ts)


masks, pixels = make_image()
nobjects = int((np.unique(masks) > 0).sum())
print(f"image {masks.shape}, {nobjects} objects\n")

# (0) whole function
t_full = time_it(get_sizeshape, masks, pixels)
print(f"get_sizeshape (full)        {t_full * 1e3:8.1f} ms")

# (A) regionprops_table only (same property list as the 2D advanced path)
desired = [
    "image", "area", "area_bbox", "area_convex", "equivalent_diameter_area",
    "bbox", "centroid", "euler_number", "extent", "axis_major_length",
    "axis_minor_length", "area_filled", "eccentricity", "orientation",
    "perimeter", "solidity", "perimeter_crofton", "inertia_tensor",
    "inertia_tensor_eigvals", "moments", "moments_hu", "moments_central",
    "moments_normalized",
]


def rprops():
    return skimage.measure.regionprops_table(masks, pixels, properties=desired)


t_rp = time_it(rprops)
print(f"  (A) regionprops_table     {t_rp * 1e3:8.1f} ms   ({100 * t_rp / t_full:4.1f}%)")

props = rprops()


# (B) per-object EDT radius loop (max/mean/median radius)
def radius_loop():
    n = nobjects
    max_r = np.zeros(n)
    mean_r = np.zeros(n)
    med_r = np.zeros(n)
    for index, mini in enumerate(props["image"]):
        mini = np.pad(mini, 1)
        dist = scipy.ndimage.distance_transform_edt(mini)
        max_r[index] = scipy.ndimage.maximum(dist, mini)
        mean_r[index] = scipy.ndimage.mean(dist, mini)
        med_r[index] = centrosome.cpmorphology.median_of_labels(
            dist, mini.astype("int"), [1]
        )
    return max_r, mean_r, med_r


t_rl = time_it(radius_loop)
print(f"  (B) EDT radius loop       {t_rl * 1e3:8.1f} ms   ({100 * t_rl / t_full:4.1f}%)")


# (B1) within the radius loop: EDT (import) vs reductions (reducible)
def edt_only():
    out = []
    for mini in props["image"]:
        mini = np.pad(mini, 1)
        out.append(scipy.ndimage.distance_transform_edt(mini))
    return out


t_edt = time_it(edt_only)
print(f"      (B1) EDT only         {t_edt * 1e3:8.1f} ms   ({100 * t_edt / t_full:4.1f}%)")
print(f"      (B2) reductions       {(t_rl - t_edt) * 1e3:8.1f} ms   ({100 * (t_rl - t_edt) / t_full:4.1f}%)")

print()
print("Amdahl ceiling if we numba ONLY the reducible reductions (B2):")
reducible = t_rl - t_edt
print(f"  best-case time  {(t_full - reducible) * 1e3:8.1f} ms  -> {t_full / (t_full - reducible):.2f}x")
print("Ceiling if we ALSO reimplement EDT (we should NOT -- exact-Euclidean import):")
print(f"  best-case time  {(t_full - t_rl) * 1e3:8.1f} ms  -> {t_full / (t_full - t_rl):.2f}x")
