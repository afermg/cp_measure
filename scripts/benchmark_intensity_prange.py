"""Batch-vs-serial wall-clock for the 2D intensity prange kernel.

Times a serial loop over single-image get_intensity calls against one batched
call (which runs the parallel prange kernel). Warms the JIT first. Reports the
speedup; no assertion — this is evidence, not a test.

Run: uv run --extra numba python scripts/benchmark_intensity_prange.py
"""

import time

import numpy as np

from cp_measure.core.numba.measureobjectintensity import get_intensity


def scene(size, nobj, rng):
    mask = np.zeros((size, size), np.int32)
    k = int(np.ceil(np.sqrt(nobj)))
    step = size // k
    lab = 1
    for i in range(k):
        for j in range(k):
            if lab > nobj:
                break
            mask[i * step : i * step + step - 1, j * step : j * step + step - 1] = lab
            lab += 1
    return mask, rng.random((size, size)).astype(np.float32)


def bench(fn, rounds=5):
    best = float("inf")
    for _ in range(rounds):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main():
    rng = np.random.default_rng(0)
    for B, size, nobj in [(16, 512, 64), (64, 256, 40)]:
        scenes = [scene(size, nobj, rng) for _ in range(B)]
        masks = [s[0] for s in scenes]
        pix = [s[1] for s in scenes]
        get_intensity(masks[0], pix[0])  # warm serial
        get_intensity(masks, pix)  # warm kernel
        serial = bench(lambda: [get_intensity(m, p) for m, p in zip(masks, pix)])
        batch = bench(lambda: get_intensity(masks, pix))
        print(
            f"B={B} size={size} nobj={nobj}:  "
            f"serial {serial * 1e3:7.1f} ms  batch {batch * 1e3:7.1f} ms  "
            f"speedup {serial / batch:.2f}x"
        )


if __name__ == "__main__":
    main()
