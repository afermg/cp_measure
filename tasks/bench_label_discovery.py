"""numpy.unique vs scipy.ndimage.find_objects for label discovery (lut build).

Both yield the same ascending present-label set -> identical label_to_idx LUT.
PYTHONPATH=src python tasks/bench_label_discovery.py
"""

import time

import numpy as np
import scipy.ndimage as ndi


def unique_lut(masks):
    unique = np.unique(masks)
    labels = unique[unique > 0]
    n = int(labels.size)
    max_label = int(labels[-1]) if n else 0
    lut = np.full(max_label + 1, -1, np.int64)
    lut[labels] = np.arange(n, dtype=np.int64)
    return lut, n


def findobj_lut(masks):
    bboxes = ndi.find_objects(masks)
    labels = np.array([i + 1 for i, sl in enumerate(bboxes) if sl is not None], np.int64)
    n = int(labels.size)
    max_label = int(labels[-1]) if n else 0
    lut = np.full(max_label + 1, -1, np.int64)
    lut[labels] = np.arange(n, dtype=np.int64)
    return lut, n


def make2d(H, W, nobj, seed=0):
    mask = np.zeros((H, W), np.int32)
    g = int(np.ceil(np.sqrt(nobj)))
    ch, cw = H // g, W // g
    obj = max(min(ch, cw) // 2, 4)
    lab = 0
    for i in range(g):
        for j in range(g):
            if lab >= nobj:
                break
            mask[i * ch + 2 : i * ch + 2 + obj, j * cw + 2 : j * cw + 2 + obj] = lab + 1
            lab += 1
    return mask


def make3d_cubes(Z, Y, X, ncubes):
    mask = np.zeros((Z, Y, X), np.int32)
    for k in range(ncubes):
        y0 = 10 + k * 40
        mask[Z // 4 : 3 * Z // 4, y0 : y0 + 30, 10:40] = k + 1
    return mask


def med(fn, repeats=7):
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2] * 1e3


cases = [
    ("2D 1024x1024, 64 obj", make2d(1024, 1024, 64)),
    ("2D 512x512, 200 obj", make2d(512, 512, 200)),
    ("3D 32x240x240, 2 cubes", make3d_cubes(32, 240, 240, 2)),
]
for name, m in cases:
    # sanity: identical LUT
    la, na = unique_lut(m)
    lb, nb = findobj_lut(m)
    ok = na == nb and np.array_equal(la, lb)
    t_u = med(lambda: unique_lut(m))
    t_f = med(lambda: findobj_lut(m))
    print(f"{name:28s}  unique {t_u:7.2f} ms  find_objects {t_f:7.2f} ms  ({t_u / t_f:4.1f}x)  identical={ok}")
