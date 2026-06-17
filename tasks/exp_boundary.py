"""Boundary lever: numba inner-boundary kernel vs skimage.find_boundaries.

skimage mode='inner', connectivity=1: a foreground pixel is a boundary pixel if
any in-bounds 4-neighbour has a different label. Out-of-bounds neighbours are
ignored. We check the numba result matches skimage exactly, and time both.
PYTHONPATH=src python tasks/exp_boundary.py
"""

import time

import numpy as np
import skimage.segmentation
from numba import njit


@njit(cache=True)
def inner_boundary(masks):
    """Labeled inner boundary of a 2D label image (0 = not boundary)."""
    H, W = masks.shape
    out = np.zeros((H, W), masks.dtype)
    for r in range(H):
        for c in range(W):
            L = masks[r, c]
            if L <= 0:
                continue
            b = False
            if r > 0 and masks[r - 1, c] != L:
                b = True
            elif r < H - 1 and masks[r + 1, c] != L:
                b = True
            elif c > 0 and masks[r, c - 1] != L:
                b = True
            elif c < W - 1 and masks[r, c + 1] != L:
                b = True
            if b:
                out[r, c] = L
    return out


def make(H, W, nobj, seed=1):
    rng = np.random.default_rng(seed)
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


def med(fn, repeats=7):
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2] * 1e3


inner_boundary(make(64, 64, 4))  # warm

for px, nobj in [(1024, 64), (256, 16), (512, 200)]:
    m = make(px, px, nobj)
    nb = inner_boundary(m) > 0
    sk = skimage.segmentation.find_boundaries(m, mode="inner")
    match = np.array_equal(nb, sk)
    t_sk = med(lambda: skimage.segmentation.find_boundaries(m, mode="inner"))
    t_nb = med(lambda: inner_boundary(m))
    print(
        f"{px}^2/{nobj}obj  skimage {t_sk:7.2f} ms  |  numba {t_nb:7.2f} ms"
        f"  ({t_sk / t_nb:4.1f}x)  exact match={match}"
    )
