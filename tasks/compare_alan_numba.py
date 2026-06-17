"""Compare Alan's #55 numpy get_intensity vs our numba backend, key-by-key.

PYTHONPATH=src python tasks/compare_alan_numba.py
"""

import importlib.util

import numpy as np

from cp_measure.core.numba import get_intensity as nb_intensity
from cp_measure.core.measureobjectintensity import get_intensity as old_intensity

spec = importlib.util.spec_from_file_location("alan_intensity", "/tmp/alan_intensity.py")
alan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alan)


def make(H, W, nobj, seed=1):
    rng = np.random.default_rng(seed)
    mask = np.zeros((H, W), dtype=np.int32)
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
    return mask, rng.random((H, W))


def cmp(a, b, label):
    print(f"\n=== {label} ===")
    keys = sorted(set(a) | set(b))
    for k in keys:
        if k not in a or k not in b:
            print(f"  {k:42s} MISSING in one")
            continue
        d = np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k])))
        flag = "  <-- DIFFERS" if d > 1e-6 else ""
        print(f"  {k:42s} max|Δ|={d:.3e}{flag}")


m, p = make(256, 256, 16)
print("2D, 256x256, 16 objects, continuous random pixels")
cmp(old_intensity(m, p), alan.get_intensity(m, p), "OLD numpy  vs  ALAN #55 numpy")
cmp(nb_intensity(m, p), alan.get_intensity(m, p), "OUR numba  vs  ALAN #55 numpy")
