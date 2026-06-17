"""Step-0 profile for the texture numba lane.

(A) Split get_texture's cost: prep (img_as_ubyte + mask) / regionprops / the
    per-object mahotas.features.haralick loop (the numba target = GLCM + 13
    Haralick features, which we'd reimplement, not call mahotas).
(B) Sanity-check that the GLCM (co-occurrence matrix) is reproducible — it's an
    integer histogram, so a numba build should match mahotas's cooccurence exactly.
    (The 13 Haralick FORMULAS are the real bit-exactness risk; flagged separately.)
"""

import time

import mahotas.features
import mahotas.features.texture
import numpy as np
import skimage.exposure
import skimage.measure
import skimage.util


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
    pixels = rng.random((size, size))
    return labels, pixels


def t(fn, reps=4):
    best = float("inf")
    out = None
    for _ in range(reps):
        s = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - s)
    return best * 1e3, out


def main():
    labels, pixels = make_image()
    scale, gray_levels = 3, 256
    unique = np.unique(labels)
    unique = unique[unique > 0]
    n = len(unique)

    def prep():
        p = skimage.util.img_as_ubyte(pixels, force_copy=True)
        p[~labels.astype(bool)] = 0
        return p

    ms_prep, p = t(prep)
    ms_rp, props = t(lambda: skimage.measure.regionprops(labels, p))

    def haralick_loop():
        out = np.empty((4, 13, n))
        for idx, prop in enumerate(props):
            try:
                out[:, :, idx] = mahotas.features.haralick(
                    prop["intensity_image"], distance=scale, ignore_zeros=True
                )
            except ValueError:
                out[:, :, idx] = np.nan
        return out

    ms_har, _ = t(haralick_loop)

    total = ms_prep + ms_rp + ms_har
    print(f"image 1080^2, {n} objects, gray_levels={gray_levels}, scale={scale}\n")
    print(f"{'prep (ubyte+mask)':<26}{ms_prep:8.2f} ms  ({ms_prep / total * 100:.0f}%)")
    print(f"{'regionprops':<26}{ms_rp:8.2f} ms  ({ms_rp / total * 100:.0f}%)")
    print(f"{'haralick loop (TARGET)':<26}{ms_har:8.2f} ms  ({ms_har / total * 100:.0f}%)")
    print(f"{'TOTAL':<26}{total:8.2f} ms")

    # (B) GLCM reproducibility: numba/numpy build vs mahotas cooccurence (dir 0).
    obj0 = np.ascontiguousarray(props[0]["intensity_image"])
    coh = np.zeros((256, 256), np.int32)
    mahotas.features.texture.cooccurence(obj0, 0, coh, symmetric=False)
    # naive direction-0 GLCM (mahotas dir 0 = horizontal, distance 1), no ignore_zeros
    ref = np.zeros((256, 256), np.int64)
    a = obj0[:, :-1].ravel()
    b = obj0[:, 1:].ravel()
    np.add.at(ref, (a, b), 1)
    print(f"\nGLCM dir0 numpy-build == mahotas.cooccurence: {np.array_equal(coh, ref)}")
    print(f"  (nonzero cells: {np.count_nonzero(coh)}, total pairs: {int(coh.sum())})")


if __name__ == "__main__":
    main()
