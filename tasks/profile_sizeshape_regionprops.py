"""Sub-profile: which property GROUP dominates regionprops_table?

Workflow rule: before ruling a lane out as Amdahl-capped imported geometry,
check for a single reimplementable dominant primitive (cf. radial's geodesic,
texture's GLCM). Moments are mechanically reducible (polynomial pixel sums);
perimeter_crofton / convex-hull / euler / eccentricity are numerically-sensitive
geometry we IMPORT. This measures each group's share so the verdict is defensible.
"""

import time

import numpy as np
import skimage.measure

rng = np.random.default_rng(0)
side, n = 1080, 144
yy, xx = np.mgrid[0:side, 0:side]
centers = rng.integers(0, side, size=(n, 2))
lab = np.zeros((side, side), np.int32)
best = np.full((side, side), np.inf)
for i, (cy, cx) in enumerate(centers, start=1):
    d = (yy - cy) ** 2 + (xx - cx) ** 2
    m = d < best
    best[m] = d[m]
    lab[m] = i
masks = lab
pixels = rng.random((side, side)).astype(np.float32)


def t(props, reps=3):
    skimage.measure.regionprops_table(masks, pixels, properties=props)
    ts = []
    for _ in range(reps):
        s = time.perf_counter()
        skimage.measure.regionprops_table(masks, pixels, properties=props)
        ts.append(time.perf_counter() - s)
    return min(ts) * 1e3


BASE = ["area", "bbox", "centroid", "area_bbox", "equivalent_diameter_area",
        "axis_major_length", "axis_minor_length", "extent"]
GROUPS = {
    "base (area/bbox/axes)": BASE,
    "+ moments (spatial/central/norm/hu)": BASE + ["moments", "moments_central",
        "moments_normalized", "moments_hu", "inertia_tensor", "inertia_tensor_eigvals"],
    "+ eccentricity/orientation": BASE + ["eccentricity", "orientation"],
    "+ perimeter": BASE + ["perimeter"],
    "+ perimeter_crofton": BASE + ["perimeter_crofton"],
    "+ convex (area_convex/solidity)": BASE + ["area_convex", "solidity"],
    "+ euler_number": BASE + ["euler_number"],
    "+ area_filled": BASE + ["area_filled"],
}
base = t(BASE)
print(f"{'group':40s} {'ms':>8s} {'delta':>8s}")
for name, props in GROUPS.items():
    ms = t(props)
    print(f"{name:40s} {ms:8.1f} {ms - base:+8.1f}")
