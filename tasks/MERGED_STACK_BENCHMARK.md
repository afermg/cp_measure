# Merged numba stack — tiered speedup vs numpy baseline (2026-06-04)

**Question:** if all numba PRs (#56–#65) were merged as-is, what is the total speedup
per function and jointly across a full `featurize`?

**Method.** Built integration branch `integration/all-numba` = `feat/bzyx-shape` (#59) +
all 6 PR tips merged (`#56 granularity`, `#57 intensity`, `#63 radial-dist` [carries #58
zernike], `#62 costes` [carries #60 coloc], `#64 texture`, `#65 feret`). The merge is only
trivial append-conflicts in `__init__.py` / `bulk._numba_registries` / `test_backend_correctness.py`.
Each function called directly (numpy baseline vs numba), warmup + min-of-N reps, single-thread
pinned (`OMP/OPENBLAS/MKL/NUMEXPR/NUMBA/VECLIB=1`). Real Cell-Painting tiers from
`cp_measure_3tier_bench/data/{tiny,small,large}.npz` (`mask_int`, `pixels`, `pixels_2`).
Bench script: `tasks/bench_tiered_merged.py`.

**JOINT = Σ(baseline ms) / Σ(numba ms)** over the default core feature set (the real
`featurize` payoff; Amdahl-weighted, NOT the mean of per-function factors). Excludes
`granularity_fullres` (default config uses `subsample_size=0.25`).

## Joint speedup (full default featurize)

| tier | image / objects | Σ numpy (ms) | Σ numba (ms) | **JOINT** | joint if granularity@fullres |
|------|-----------------|-------------:|-------------:|----------:|------------------------------:|
| tiny  | 256², 2 obj    |   447.1 |  70.7 | **6.3×**  | 6.8× |
| small | 540², 43 obj   |  1213.7 | 128.9 | **9.4×**  | 6.5× |
| large | 1080², 142 obj |  9831.9 | 430.7 | **22.8×** | 10.4× |

Joint rises with image size: per-object-heavy functions (intensity, coloc) dominate the
baseline at scale and get the largest speedups. At fullres granularity (~6× only, ~7 s
baseline) the large joint falls to ~10×.

## Per-function (numpy ms → numba ms → ×)

| function | tiny | small | large |
|----------|------|-------|-------|
| intensity            | 39.3→4.2 = **9.3×**   | 307→4.3 = **70.7×**  | 4256→16 = **260×**  |
| sizeshape (not ported)| 19.9→19.9 = 1.0×     | 58→58 = 1.0×         | 196→196 = 1.0×      |
| zernike              | 82→7.8 = 10.6×        | 169→14 = 12.1×       | 772→40 = 19.4×      |
| feret                | 4.8→0.9 = 5.4×        | 32→3.5 = 9.1×        | 361→7.6 = 47.3×     |
| granularity (default)| 59→11 = 5.3×          | 276→13 = 20.5×       | 1233→50 = 24.9×     |
| texture              | 11.9→2.1 = 5.8×       | 33→5.6 = 5.9×        | 108→21 = 5.1×       |
| radial_distribution* | 63→4.8 = 13.0×        | 73→7.4 = 9.9×        | 456→26 = 17.7×      |
| radial_zernikes      | 103→8.1 = 12.8×       | 76→14 = 5.3×         | 571→41 = 14.1×      |
| coloc_pearson        | 2.0→0.7 = 3.0×        | 17→0.9 = 19.9×       | 248→3.9 = 63.5×     |
| coloc_manders        | 11.8→0.6 = 19.3×      | 34→0.8 = 40.4×       | 377→3.9 = 95.7×     |
| coloc_rwc            | 20.3→9.1 = 2.2×       | 41→4.6 = 8.9×        | 399→17 = 23.1×      |
| coloc_overlap        | 14.8→0.6 = 24.2×      | 38→0.8 = 45.8×       | 384→4.0 = 97.0×     |
| coloc_costes         | 15.0→1.0 = 15.2×      | 61→1.1 = 56.5×       | 471→4.9 = 96.3×     |
| _granularity_fullres_| 231→31 = 7.4×         | 1405→245 = 5.7×      | 6974→1118 = 6.2×    |

Biggest baseline cost at large = **intensity (43% of total, 260×)**, so it drives the joint.
**texture (5×)** is the only weak lane (GLCM is genuinely expensive — its sparse follow-up is gated).
**sizeshape** is numpy in both (documented NO-GO, Amdahl 1.13×).

## Caveats (do not over-read the 22.8×)
- **vs current `main` baseline, NOT Alan's PR #55.** #55 rewrites numpy intensity/coloc/sizeshape
  (the biggest contributors) → it will SHRINK these factors. Re-baseline against #55 before quoting
  externally.
- **Config-dependent:** default (granularity subsample=0.25) → large 22.8×; fullres granularity → ~10×.
- **radial_distribution\*** intentionally diverges from baseline on multi-object fields (the #22
  per-object-crop fix); timing valid, output not bit-equal.
- **Sum-of-kernels**, not a real `featurize()` (which adds small dispatch overhead + the un-ported
  functions); end-to-end would be marginally lower.
- Higher than the old "fast 8.40× on large" note because this upstream re-port accelerates MORE lanes
  (all coloc via bincount, etc.) than the original `fast` RUN_SET credited.
- Synthetic-free real data, single 1080² field, single-thread, shared node (absolute ms vary run-to-run;
  ratios are stable).
