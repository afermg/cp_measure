# NumPy-only speedup scan — all feature functions (post-rebase)

**Date:** 2026-06-06. Branch `fix/radial-per-object-22` **rebased onto main** (now has #69–73).
**Constraint:** numpy / scipy / skimage / centrosome / mahotas only. No numba/jax/new deps.
**Profile:** `tasks/profile_all_functions.py`, real masks, large tier 1080²/142obj.

## Cost landscape (large, per call, after #69–73 + radial rebase)

| function | ms | % | status |
|----------|----:|---:|--------|
| granularity | 1330 | 36% | untouched — lever below |
| **zernike** | 793 | 21% | **PROTOTYPED 8.0× ⭐** |
| radial_distribution | 585 | 16% | our per-object PR (correctness) — leave |
| sizeshape | 364 | 10% | #70 done; little headroom |
| **radial_zernikes** | 236 | 6% | same lever as zernike |
| texture | 151 | 4% | sparse-GLCM, uncertain |
| intensity | 88 | | #72 done |
| costes 72 / rwc 42 / pearson 36 / manders 12 | | | #69 done |
| feret | 19 | | #71 + rebase done |
| **TOTAL** | **3730** | | |

## Opportunities (ranked by win × confidence)

### 1. zernike — **8.0× (782→98 ms), max diff 2.2e-16** ⭐ PROTOTYPED
`tasks/proto_zernike_vectorized.py`. `centrosome.zernike.zernike` →
`construct_zernike_polynomials` scatters the masked basis into a **full (H,W,K) complex
array (~560 MB at 1080²)**, then `score_zernike` does **~60 full-image `scipy.ndimage.sum`**
(K≈30 polynomials × real+imag). Both are foreground-sparse waste.
**Fix (pure numpy):** run centrosome's Horner basis on the masked `(npix,)` vectors only, then
per-object `np.add.at` segment-sum over `(npix,K)`. No full array, no per-channel ndimage scans.
Reuses `centrosome.construct_zernike_lookuptable` + `minimum_enclosing_circle` (existing dep).
Measured: small 6.9×, large 8.0×, m2160 13.0×. **Divergence is machine-eps (≤2.2e-16)** —
effectively bit-exact (per-object pixel counts small → summation order barely matters), unlike
the ~1e-12 I'd feared. Cleanest single-file win in the whole codebase.

### 2. radial_zernikes — same lever (236 ms), est. ~5–8×, not prototyped
`get_radial_zernikes` (in `measureobjectintensitydistribution.py`) uses the **same**
`centrosome.zernike.construct_zernike_polynomials` + `minimum_enclosing_circle` + `masks_to_ijv`
machinery, scoring via per-index `scipy.ndimage.sum_labels`. Identical vectorization applies
(masked basis + segment-sum). High confidence given #1; prototype to confirm + check the same
machine-eps divergence. Could **share the basis/geometry with #1 when both run** (one
`minimum_enclosing_circle`), an extra saving in the full pipeline.

### 3. granularity — **~1.35× (1330→~990 ms), ~1e-13** PROTOTYPED
`tasks/proto_numpy_levers.py`. The granular-spectrum loop calls `map_coordinates` on a **fixed**
upsample grid every iteration (ng=16); precompute floor/frac/neighbours once, gather as a
fancy-indexed separable weighted sum. 1.63× on the upsample step (the dominant chunk at
sub=0.25). Within granularity's `rtol=1e-6` (never bit-exact). Biggest *absolute* remaining
target; the modest ratio is because skimage `erosion`/`reconstruction` (on the small 270² image)
is the rest. Refinement: two 1-D interps to cut the four 1080² temporaries.

### 4. texture — sparse GLCM (151 ms), uncertain, prototype first
Port `cp_measure_jax`'s sparse co-occurrence build to numpy to replace the per-object
`mahotas.features.haralick` loop. numba #64 proved the symmetric-GLCM build is bit-exact vs
`mahotas.cooccurence`, so a numpy version can be faithful. BUT jax-agent estimate is only
~1.1–1.35× on CPU (mahotas is already C), and it's the highest-effort port (RLE decode + 13
Haralick formulas + degenerate/ignore_zeros edges). Prototype CPU speed before committing.

### 5. sizeshape — little headroom (364 ms)
#70 already dropped the per-object `scipy.ndimage` calls. Residual cost = `regionprops_table`
(skimage C, not numpy-addressable) + a per-object `distance_transform_edt` radius loop
(`measureobjectsizeshape.py:646`, inherently per-component EDT — hard to vectorize). The 3D path
(per-object `marching_cubes`) is geometry-bound. Low priority.

### 6. radial_distribution (585 ms) — leave
Already improved by our per-object PR #68 (≈1.65× vs the legacy whole-image path, and it's the
#22 correctness fix). The real speed path is the numba lane (#63, 21×). No further numpy lever
worth the risk.

## Bottom line

- **Do now (high confidence, near-bit-exact, one file each):** #1 zernike (8×) and #2
  radial_zernikes (same lever). Together ≈ **−870 ms / call** at machine-eps fidelity.
- **Next:** #3 granularity (~−340 ms, within 1e-13).
- **Investigate:** #4 texture (prototype CPU first).
- Combined realistic numpy win brings large-tier pipeline ≈ **3730 → ~2500 ms (~1.5×)**, with
  the Zernike pair the standout.
- Each is a clean single-concern PR matching the #69–73 style; zernike/radial_zernikes can share
  one geometry pass when both run.

## Verify
`.venv/bin/python -u tasks/proto_zernike_vectorized.py` (speed + diff vs centrosome get_zernike);
`.venv/bin/python -u tasks/profile_all_functions.py` (landscape). Real golden tests on the 3-tier
dataset; zernike/radial_zernikes need a tolerance or `legacy` decision (≤2.2e-16, well under any
reasonable rtol).
