# Numba Zernike backend (zernike + radial_zernikes) — implementation plan

Port the two Zernike features to a numba backend on the #49 dispatch seam, in ONE PR (they
share a single new primitive). **2D-only** (baseline returns `{}` for 3D). Batch-shaped via
`to_bzyx` like intensity/granularity. Builds on `to_bzyx` (#56) → stack on the granularity
branch, retarget to main after #56 merges.

## The two features (baselines)
- **`get_zernike`** (`core/measureobjectsizeshape.py:1013`) — *binary-shape* Zernike. Delegates
  wholesale to `centrosome.zernike.zernike(numbers, labels, indices)` (NO pixel weighting).
  Output: `Zernike_{n}_{m}` = **magnitude only**, normalised by **π·r²** (enclosing-circle area).
- **`get_radial_zernikes`** (`core/measureobjectintensitydistribution.py:308`) — *intensity-weighted*
  Zernike, done "in the open": `sum_labels(pixels·z.real)` / `sum_labels(pixels·z.imag)` per object.
  Output: `RadialDistribution_ZernikeMagnitude_{n}_{m}` (= `sqrt(vr²+vi²)/count`, count = pixel
  count per object) AND `_ZernikePhase_{n}_{m}` (= `arctan2(vr, vi)` — real-first, swapped order).

Both are the SAME reduction Σ w·V_nm, differing only in: weight (1 vs pixel), normalisation
denominator (π·r² vs pixel count), and output (magnitude only vs magnitude+phase).

## Cross-variant findings (3 agents, 2026-06-03)
- **centrosome.zernike is the canonical reference.** `construct_zernike_polynomials(x,y,idx)` builds
  `V_nm` per pixel: radial LUT/Horner gives `R_nm(r)/r^m`, then `z^m` with `z = y + i·x`
  (y=normalised ROW offset, x=normalised COL offset) supplies `r^m·e^{imθ}`; zeros where
  `x²+y² > 1` (STRICT). `minimum_enclosing_circle` gives per-object center+radius over the convex
  hull (integer pixel coords, no +0.5). `zernike()` = construct + per-label `scipy.ndimage.sum` of
  Re/Im + `sqrt(ΣRe²+ΣIm²)/(π·r²)`, magnitude only.
- **fast (numba):** keeps centrosome for indices/LUT/`construct_zernike_polynomials`/min-circle on
  the HOST; numba accelerates ONLY the per-object weighted-complex sum. Kernel family
  `_per_label[_zernike]_sum[_complex]` in `_intensity_distribution_numba.py` — a single M×K
  scatter-add into `(N,K)` Re/Im accumulators; one pass over all pixels × all (n,m). Shape Zernike =
  the `weight≡1` special case of the weighted kernel. Speedups: get_zernike **17–21×**,
  get_radial_zernikes **2.8–3.3×** (radial lower because the host `construct_*` dominates).
- **jax:** same idiom (`.at[poly*N+obj].add(...)` 2D-keyed scatter), evaluates the basis on-device
  from the centrosome LUT but keeps `get_zernike_indexes`/LUT/`minimum_enclosing_circle` host-side.
  Confirms the weighted-complex segment-sum as the unifying primitive; `arctan2(vr,vi)` +
  per-function normalisation preserved.
- **rust drifted 5–6%** precisely because it reimplemented the conventions. We DON'T — centrosome
  owns them, so we inherit them for free.

## Design decisions (LOCKED via 12-question round, 2026-06-03; +fused-kernel addendum)
> **ADDENDUM (2026-06-03): "do all in this PR" — FUSE basis eval + segment-sum into one numba
> kernel** (no `(M,K)` intermediate; maximal throughput). This RELAXES decision 5 (kernel is no
> longer a pure return-vr/vi primitive) and REVISES decision 1 (the fused kernel is zernike-specific
> — it embeds the polynomial eval — so it lives backend-local in `core/numba/_zernike.py`, NOT in
> `primitives/`). Consequence accepted: the zernike lane no longer grows the shared `segment_reduce`
> primitive; we chose throughput over that. bzyx (decision 8) is unaffected.
1. ~~Kernel home: `primitives/_segment_numba.py`~~ → **`core/numba/_zernike.py` (backend-local)** per
   the fused-kernel addendum (zernike-specific, not a generic primitive).
2. **One PR, both functions.** Share the kernel; radial passes `weights=pixels`, emits phase + a
   different denominator.
3. **Branch base: stack on the granularity branch (#56)** for `to_bzyx`; retarget to main after #56
   merges (same pattern as #57).
4. **Wrapper files mirror baseline module names:** `core/numba/measureobjectsizeshape.py`
   (`get_zernike`) + `core/numba/measureobjectintensitydistribution.py` (`get_radial_zernikes`).
5. ~~Kernel returns vr/vi only; host combines~~ → **FUSED kernel** (addendum): one numba pass per
   pixel evaluates all K polynomials from `C`/`m_arr` AND scatter-adds the weighted Re/Im into
   `(n,K)` `vr`/`vi` — no `(M,K)` `z` intermediate. Kernel still RETURNS `vr`/`vi`; host still does
   the per-function magnitude/phase + normalisation (so the fused kernel is shared by both zernikes).
6. ~~Complex `z[M,K]` kernel input~~ → moot under fusion: the kernel takes per-pixel `(xm, ym)` (or
   `r²` + `xm,ym`), the host-built coeff matrix `C[n_poly, max_nh]`, `m_arr[n_poly]`, `weights[M]`,
   `seg0[M]`, `n` — and computes `z` internally, never materialising it.
7. **One fused weighted kernel**; shape passes `weights≡1`.
8. **Per-image via the `to_bzyx` loop** — `(B,Z,Y,X)` convention (single=batch-of-1, single→dict,
   4D/list→list-of-dicts). **HARD REQUIREMENT — adhere to bzyx like intensity/granularity.** numba
   kernels stay single-threaded; batching/parallelism lives in the (future) batch layer.
9. **ACCELERATE the basis eval (non-default)** — do NOT keep centrosome's per-pixel
   `construct_zernike_polynomials` on the hot path. Instead (jax-style, on CPU): centrosome supplies
   `get_zernike_indexes` + `construct_zernike_lookuptable` (host, degree-only, cheap) → repack to a
   Horner coeff matrix `C`; we evaluate the basis vectorised (numpy: Vandermonde powers of `r²` `@ C`
   for the radial part, `z=ym+i·xm` power recurrence for the azimuthal part, zero where `r²>1`) →
   `z[M,K]`, fed to the pure segment-sum kernel. Raises radial's ceiling past the ~3× centrosome cap.
   **Consequence: WE now own the basis conventions** → pin + assert them (see below).
10. **Keep `minimum_enclosing_circle` on the host (centrosome)** — convex-hull geometry, not a
    segment-sum, and rust's 5-6% drift source. Feed it the same inputs as baseline.
11. **Correctness: rtol/atol + equal_nan vs the numpy baseline** (segment-sum reorders float adds vs
    `scipy.ndimage.sum`; our basis eval reorders vs centrosome's per-pixel loop).
12. **Match centrosome/baseline exactly on degenerate objects** (1-2 px, r→0 → NaN/inf): reproduce,
    do not guard.
13. **Reuse `label_to_idx_lut`** — zernike output is per-PRESENT-object (compacted, ascending label
    order), exactly what the lut gives. (Unlike granularity's dense `1..max`.)

### Conventions to PIN + ASSERT (now ours, since decision 9 evaluates the basis)
From the centrosome reference (verified): `z = ym + i·xm` where ym=normalised ROW offset, xm=norm COL
offset (so phase = `atan2(col, row)` — swapped); radial part = `R_nm(r)/r^m` via Horner over `r²` with
centrosome's LUT coeffs, `r^m·e^{imθ}` supplied by `z^m`; disk cutoff STRICT (`inside = r² <= 1`,
zero where `>1`); integer pixel coords, NO +0.5 offset; coords normalised `(ij - center)/r` with
centrosome's min-enclosing-circle. Output: shape `magnitude = sqrt(ΣRe²+ΣIm²)/(π·r²)` (magnitude
only); radial `magnitude = sqrt(vr²+vi²)/pixel_count` + `phase = arctan2(vr, vi)` (real-first).
Add a unit test of our basis eval vs `centrosome.zernike.construct_zernike_polynomials` (bit-tol).

## Files
- NEW kernel: `primitives/_segment_numba.py::segment_complex_weighted_sum` (or `core/numba/_zernike.py`
  per decision 7). NEW `core/numba/measureobjectsizeshape.py` (`get_zernike`) and
  `core/numba/measureobjectintensitydistribution.py` (`get_radial_zernikes`) — OR a single
  `core/numba/_zernike.py` exposing both wrappers.
- EDIT `core/numba/__init__.py` (re-export `get_zernike`, `get_radial_zernikes`),
  `bulk._numba_registries` (add `"zernike"`, `"radial_zernikes"`), `test_backend_correctness.py`.

## Kernel (FUSED, `core/numba/_zernike.py`)
```
@njit(cache=True)
zernike_moments(weights, xm, ym, seg0, C, n_half, m_arr, n):
    # weights[M] (1.0 for shape); xm/ym[M] normalised col/row offsets; seg0[M] 0-based object idx
    # C[K, max_nh] Horner coeffs (from centrosome LUT, reversed); n_half[K]; m_arr[K] azimuthal order
    # -> vr[n,K], vi[n,K]
    # per pixel i:
    #   r2 = xm²+ym²; if r2 > 1: skip (disk cutoff, STRICT >)
    #   for k: R = Horner(C[k, :n_half[k]] over r2)        # = R_nm(r)/r^m
    #          (zr,zi) = (ym+ i·xm)^{m_arr[k]}  via per-pixel power (cache z^m across k)
    #          w = weights[i]
    #          vr[seg0[i],k] += w*(R*zr); vi[seg0[i],k] += w*(R*zi)
```
No `(M,K)` materialisation. Host supplies `C`/`n_half`/`m_arr` (centrosome
`construct_zernike_lookuptable` repacked, degree-only — cheap), `xm`/`ym` (normalised coords from
`minimum_enclosing_circle`), `seg0` via `label_to_idx_lut`. Compute the `z^m` powers per pixel by
incrementing m in `m_arr` order (or precompute `z^0..z^max_m` per pixel in registers). Pin the
`z=ym+i·xm`, strict `r²>1` cutoff, and Horner conventions (assert vs centrosome — build step 2).

## Per-function wrapper logic (after to_bzyx, per (Z,Y,X) element; Z>1 → {})
- **radial_zernikes:** mirror baseline up to the sum — `get_zernike_indexes`,
  `minimum_enclosing_circle`, `masks_to_ijv`, normalise `yx=(ij-center)/r`,
  `construct_zernike_polynomials` → z; then `vr,vi = segment_complex_weighted_sum(pixels_at_ijv, z,
  lut[labels], n)`; `magnitude=sqrt(vr²+vi²)/count`, `phase=arctan2(vr,vi)`. Preserve the
  `len==0` fringe (zeros(0) per key) and the in-bounds ijv filter.
- **zernike (shape):** inline what `centrosome.zernike.zernike` does (like fast): min-circle →
  mgrid coord normalise + object mask → `construct_zernike_polynomials` → `vr,vi =
  segment_complex_weighted_sum(ones, z, lut[labels], n)`; `magnitude=sqrt(vr²+vi²)/(π·r²)`.

## Correctness harness
- numba == numpy baseline within tol (rtol≈1e-6, atol, equal_nan) for both functions, on a
  multi-object 2D scene; assert exact key sets (`Zernike_*`, `RadialDistribution_Zernike*`).
- Edge cases: empty mask, single object, 1-px object (degenerate min-circle r→0 → centrosome
  NaN/inf path — match baseline), label gaps (compacted output), all-pixels-in-bounds.
- Assert phase uses `arctan2(vr,vi)` order and the two different denominators give baseline values.
- 3D element → `{}`.
- Kernel unit test: `segment_complex_weighted_sum` vs a numpy `bincount`/`add.at` reference.

## Build order
1. **Host LUT→`C` repack helper** (`get_zernike_indexes` + `construct_zernike_lookuptable` →
   `C[K,max_nh]` reversed, `n_half[K]`, `m_arr[K]`). Unit-test the eval it implies (a numpy reference
   that computes `V_nm` from `C`/`m_arr`) vs `centrosome.zernike.construct_zernike_polynomials`
   (within tol) — LOCKS the conventions before the fused kernel.
2. **Fused `zernike_moments` kernel** in `core/numba/_zernike.py` + unit test: compare its `vr/vi`
   against the step-1 numpy reference basis × `add.at` segment-sum (within tol).
3. radial_zernikes wrapper: min-circle (centrosome) → ijv normalised coords → fused kernel
   (`weights=pixels`) → `magnitude=sqrt/pixel_count`, `phase=arctan2(vr,vi)`. == baseline within tol.
4. zernike wrapper: centrosome.zernike.zernike's flow (min-circle → mgrid coords + object mask) →
   fused kernel (`weights≡1`) → `magnitude=sqrt/(π·r²)`, magnitude only. == baseline within tol.
5. Batch path (`to_bzyx`, list-of-dicts, single→dict, 3D→`{}`) == per-image baseline.
6. Wire dispatch (`_numba_registries`: `zernike`, `radial_zernikes`) + `__init__` re-exports + tests;
   ruff; full suite.
7. Benchmark 1080²/144obj vs baseline; record for the per-feature table. Fusing (no `(M,K)`) should
   beat the old centrosome-basis ceiling for BOTH (radial well past ~3×; get_zernike past ~17-21×).

## Notes / risks
- **Fused kernel (this PR) moves convention ownership to us** — the step-1 numpy-eval-vs-centrosome
  unit test is the guard; do it FIRST so the conventions are locked before the kernel embeds them.
  No `(M,K)` `z` intermediate (that was the point of fusing). Cost: the kernel is zernike-specific
  (lives in `core/numba/_zernike.py`, not the shared primitive layer) — accepted tradeoff.
- **The fused kernel is shared by both zernikes** — it returns `vr/vi`; only the host post-step
  differs (shape: `/(π·r²)`, magnitude only; radial: `/pixel_count` + `arctan2(vr,vi)` phase).
- `minimum_enclosing_circle` stays centrosome (host) — do NOT reimplement (rust's drift source).
- Degenerate single-/two-pixel objects: centrosome min-circle has NaN/r=0 fallbacks; feed centrosome
  the same inputs so the degenerate values match baseline (decision 12).
- **bzyx is a hard requirement** (decision 8) — single image = batch-of-1 through the same loop;
  never a separate single-image path.
- Env/lint: `uv run pytest`; `uvx ruff@0.12.1 format --check . && uvx ruff@0.12.1 check .` before push.

## RESULTS (shipped 2026-06-03, PR #58, branch `feat/numba-zernike`)
All steps done; full suite **195 passed**, ruff clean. Stacked on #56.
New: `core/numba/_zernike.py` (`zernike_coeffs` host LUT repack, `_zernike_basis_numpy`
reference, fused `zernike_moments` kernel), `core/numba/measureobjectsizeshape.py`
(`get_zernike`), `core/numba/measureobjectintensitydistribution.py` (`get_radial_zernikes`);
wired in `_numba_registries` + `__init__`. Tests: `test_zernike_kernels.py` (basis bit-exact
vs centrosome + kernel vs numpy reference), `test_zernike_backend.py` (both backends vs baseline:
single/batch/3D→{}/single-object), + dispatch assertions in `test_backend_correctness.py`.

**Benchmark (1080², 144 objects, min of 3, JIT warmed):**
- **zernike**: numpy 632 ms → numba **22 ms = 28.7×**
- **radial_zernikes**: numpy 476 ms → numba **23 ms = 20.9×**

**Single-image optimisation passes (the fused kernel alone was not enough):**
1. Fused basis+sum kernel (no `(M,K)`): zernike already ~21×, but radial only **1.27×**.
2. Replaced `cp_measure.utils.masks_to_ijv` (per-label `np.where` over the full image,
   O(n_labels·HW) = **347 ms** on the bench) with a single `numpy.nonzero` pass → radial **16.4×**.
   The per-object sum is order-independent so raster vs per-label order matches baseline within tol.
3. Replaced `numpy.unique(labels)` (full-image sort, **8.9 ms**) + `searchsorted` with
   `label_to_idx_lut` (find_objects, **2.3 ms**) + `lut[l_]` in BOTH wrappers → +~7 ms each →
   zernike **28.7×**, radial **20.9×**. (Also switched shape from a full-image `mgrid`+mask to
   `nonzero`, ~2 ms.) Decision 13 honoured.
4. Hoisted degree-only work (`get_zernike_indexes`, `zernike_coeffs`) out of the per-image loop
   (computed once per batch, not B×).

**Single-image cost breakdown after optimisation (~22 ms):** `minimum_enclosing_circle` ~8 ms
(host centrosome, the dominant remainder — deliberately NOT reimplemented), fused kernel ~6 ms
(M=82944×K=30), `nonzero`+`label_to_idx_lut` ~5 ms, finalise. Kernel is near-optimal single-threaded.

**Batching is correct but gives NO compute speedup on CPU (by design):** work is linear in
pixels/objects; batch = B× single (serial per-image loop). The only batch speedup is parallelism
across images, which is an EXTERNAL shared batch layer's job — never inside these functions
(see [[no-parallelism-inside-functions]]). GPU/jax batching differs (amortises launch/transfer).

**Cross-function follow-up (noted, not done):** in a featurize run BOTH zernikes call
`minimum_enclosing_circle` on the same labels → ~8 ms computed twice. A shared/memoised geometry
pass (jax-style) would halve it, but it spans the two functions / featurize layer and the
id-based cache is fragile → separate PR.

## Is reimplementing the basis (vs importing centrosome) justified? — A/B (2026-06-03)
Question raised in review: we keep centrosome for the LUT coeffs + `minimum_enclosing_circle`, but
the fused kernel REIMPLEMENTS `construct_zernike_polynomials` (the per-pixel basis eval). Worth it,
or just import centrosome's basis and numba only the sum? Measured A/B on identical host setup
(nonzero, label_to_idx_lut, min_circle, coords), 1080²/144obj:

| approach | kernel time |
|---|--:|
| **A — fused kernel** (basis eval + weighted-complex sum, no (M,K)) | **6.3 ms** |
| **B — centrosome `construct_zernike_polynomials` + numba sum-only** | **61.8 ms** |
| &nbsp;&nbsp;└ of which centrosome `construct` alone | 25.7 ms |

**Reimplementing the basis is a ~10× kernel win, NOT marginal.** centrosome's `construct` is a
Python loop over the K polynomials that materialises an (M,K) complex array (25.7 ms), then the sum
re-traverses it (~30 ms more). Fusing evaluates + sums in one compiled pass with no (M,K) buffer →
6.3 ms. At the function level: radial ~21× (fused) vs ~6× (centrosome-construct); zernike ~28× vs
~9×. (Earlier worry that the fusion was marginal was WRONG — refuted by this A/B.)

**The reimplementation BOUNDARY is what makes it safe, and it's deliberate:**
- REIMPLEMENT (worth it): the per-pixel *evaluation recurrence* (Horner + z^m + cutoff) + the sum —
  mechanical, spec-fixed, cheap to verify (bit-exact test vs centrosome), expensive to run via centrosome.
- IMPORT (correctly NOT reimplemented): the LUT *coefficients* (numerically-sensitive factorials,
  but cheap/degree-only) and `minimum_enclosing_circle` (convex-hull geometry — high reimpl risk;
  it's now the 8 ms floor, vs the ~55 ms the fused kernel saved).
- This is why we are NOT the rust situation: rust drifted 5-6% reimplementing EVERYTHING incl.
  geometry/normalisation with NO bit-exact guard. We import the hard-to-verify / high-risk pieces
  (coeffs, geometry) and reimplement only the easy-to-verify, expensive-to-run evaluation, guarded
  by `test_basis_matches_centrosome`. If we'd reimplemented `minimum_enclosing_circle` too, the line
  would have been crossed — we deliberately did not.
