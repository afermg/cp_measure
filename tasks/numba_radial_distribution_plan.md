# Numba radial_distribution lane — implementation plan

Status: PLANNED 2026-06-04. Branch `feat/numba-radial-distribution`, **stacked on
#58** (`feat/numba-zernike`) — where the numba
`core/numba/measureobjectintensitydistribution.py` module already exists (it holds
`get_radial_zernikes`); this lane adds `get_radial_distribution` to it. Stack:
**#59 → #58 → radial_distribution**. Mirrors `core/measureobjectintensitydistribution.py`.

## Decisions (from the planning round)

1. **Fix Issue #22** (per-object results must not depend on other labels) by
   processing each object on its own **cropped single-label sub-image**. This is
   the correct per-object semantics, removes the need for `color_labels`, and is
   naturally batch-independent. **It therefore diverges from the current buggy
   baseline on multi-object images** — see Verification for how exactness is still
   anchored. This is the ONE lane that intentionally does not bit-match main; it is
   a documented bug FIX, not just acceleration (call this out in the PR).
2. **Stack on #58** (module already there; additive, no file-creation conflict).

## What the reference does (researched)

2D only (`labels.ndim == 3 → {}`). Per the function:
- **Geometry (centrosome, host-side — IMPORT, do not reimplement):**
  - `distance_to_edge(labels)` — EDT to the object edge.
  - `maximum_position_of_labels(d_to_edge, labels, indices)` — the per-object
    centre (point farthest from edge).
  - `color_labels(labels)` — graph-colour touching objects so `propagate` doesn't
    bleed across them. **Dropped under the #22 fix** (per-object crops separate
    touching objects for free).
  - `propagate.propagate(...)` per colour — geodesic distance `d_from_center` +
    nearest-centre label `cl`.
- **Reducible accumulation (numba — REIMPLEMENT):** `normalized_distance` → `bin_indexes`;
  then the repeated `scipy.sparse.coo_matrix(...).toarray()` builds: per-(object,bin)
  intensity `histogram` and pixel `number_at_distance` → `FracAtD`, `MeanFrac`; and
  per bin, the per-(object, 8-wedge) sums → `RadialCV = std/mean` across wedges.
  ~10 sparse-matrix constructions + a Python bin loop — the mechanical part.

This is the same import/reimplement boundary as zernike (import
`minimum_enclosing_circle`, reimplement the basis+sum) and coloc.

## Profile FIRST (step 0) — DONE 2026-06-04: VERDICT = do NOT build a numba kernel

`tasks/profile_radial.py`, 1080², 144 objects:

| block | time | note |
|---|---:|---|
| whole-image geometry (KEEP/import) | 910 ms | `distance_to_edge` + `propagate`; ncolors=1 |
| sparse accumulation (numba target) | 33 ms | the reducible part |
| **reducible fraction** | **~3%** | (≤~10% even with the full wedge-CV loop) |
| per-object-crop geometry (#22 fix) | 550 ms | **0.60× whole-image — FASTER** |

**Conclusion:** the geometry we (correctly) import is ~97% of runtime, so a numba
kernel for the reductions buys ~1.03× — not worth building. radial_distribution is
NOT a good numba lane (opposite of coloc, which was fully reducible). Step 0 did its
job: kill the lane before writing kernels.

**The real win the profile found** is orthogonal to numba: the per-object-crop
restructure is **1.65× faster** (550 vs 910 ms) AND fixes Issue #22 — pure
numpy/scipy, no kernel. That is a worthwhile *bug-fix + speedup* PR on the numpy
baseline, but it is NOT a numba accelerator lane and should be decided with afermg
(he may prefer the upstream scipy fix he already raised). The numba accelerator can
keep composing the (fixed) numpy `radial_distribution` as-is, like every other
un-numba'd feature.

## In-depth "more options" research (2026-06-04) — `propagate` is the wall

User asked to dig deeper before dropping. Split profile + experiments
(`tasks/profile_radial_split.py`, `exp_propagate_vs_edt.py`):

**Geometry breakdown (1080², 144 obj):** `propagate` **735 ms (80%)**, EDT 73,
`maximum_position` 40, `color_labels` 23 (called **2×** in the baseline — once
inside `distance_to_edge`, once explicitly), reductions 33.

So the whole game is `propagate` (centrosome `_propagate.so`, a heap-Dijkstra
geodesic; `image=zeros, weight=1`). Every numba angle:

- **Numba reductions:** 3% → ~1.03×. Pointless alone.
- **Replace `propagate` with Euclidean `distance_transform_edt`-from-seed:**
  REJECTED — diverges 1–3% of distance bins **even on a convex square**
  (maxdiff 2.24). `propagate` is a *chamfer-ish geodesic*, not Euclidean, so EDT is
  not equivalent. Concave shapes diverge far more (ring maxdiff 24).
- **Numba-reimplement `propagate` bit-exactly:** INFEASIBLE — compiled `.so`, no
  source; exact step-weights / tie-breaking unknowable, so can't match it. This is
  squarely the "import numerically-sensitive geometry" boundary.
- **Per-object crops (the #22 fix):** the real win — **~1.65×** (910→550 ms;
  `propagate` 735→~460 via small arrays) **+ fixes #22 + drops both `color_labels`
  calls**. Stays exact to the isolated-object reference.
- **Numba geodesic with our OWN metric (fast-marching / Dijkstra):** could be much
  faster, but **changes the distance metric vs centrosome → diverges everywhere**,
  not just multi-object. A "exact-Eikonal is more correct than centrosome's
  chamfer" decision — owner-gated, full re-validation, out of scope unless we
  deliberately redefine the method.

**Verdict:** faithful to the reference metric, the achievable improvement is the
per-object-crop restructure (~1.7× end-to-end + the #22 fix). It is a numpy/scipy
restructure, NOT a numba kernel lane (numba-ing the 3% reductions is a free add
while rewriting, not the point). Deeper speedup requires redefining the geodesic
metric — a separate decision for afermg.

## POC REVERSAL (2026-06-04) — numba geodesic IS the lane (`tasks/poc_numba_geodesic.py`)

The "metric change / can't reimplement" worry was WRONG. `propagate(image=zeros,
weight=1)` is exactly the **1/√2 chamfer shortest-path** within the mask. A numba
raster-sweep-to-convergence geodesic computes the identical metric (shortest path
is algorithm-independent — Dijkstra vs Bellman-Ford sweeps give the same minimum):

- **Speed:** numba chamfer 11.95 ms vs centrosome propagate 427.74 ms (per-crop
  sum, 144 obj) → **35.8×**, no heap (O(passes·N) raster sweeps).
- **Exactness:** `maxdiff = 0.000000` vs centrosome on convex square, concave L,
  concave U, ring, AND the #22 object. BIT-EXACT, not an approximation.

So the 80% bottleneck collapses bit-exactly. radial_distribution becomes a genuine
high-value numba lane after all.

**Revised design:** per-object crop (fixes #22) → scipy EDT for `d_to_edge` +
centre (exact Euclidean, ~22 ms, keep) → **numba chamfer geodesic for
`d_from_center`** (replaces propagate, 35× + bit-exact) → numba reductions. Project:
geometry ~910 ms → ~40-50 ms (EDT + chamfer + maxpos), i.e. a large end-to-end win
PLUS the #22 fix, all bit-exact vs the isolated-object reference.

**Geodesic algorithm — DECIDED: FIFO Bellman-Ford (SPFA)** (`exp_geodesic_worstcase.py`):
both raster-sweeps and FIFO are bit-exact vs centrosome on convex/concave-U/spiral;
FIFO is faster (0.08–0.14 ms vs 0.11–0.30 ms) and **bounded O(N)** regardless of
shape (no convergence-count risk). The "unbounded sweeps on a spiral" worry was
unfounded anyway — the EDT-argmax seed makes geodesics near-monotone, so even a
deliberate spiral converged in 4 sweeps — but FIFO is strictly better. Ring-buffer
queue (cap N+1) + in-queue flag, exactly the structure of the granularity
reconstruction kernel ([[cp-measure-upstream-numba-backends]] Vincent FIFO).

---

## BUILD-READY DESIGN (supersedes the sketch above)

### File layout (aligned with the stack)
- `core/numba/_radial.py` — two `@njit(cache=True, error_model="numpy")` kernels:
  - `geodesic_chamfer_fifo(mask, si, sj) -> d` — 1/√2 chamfer shortest-path from the
    centre seed within `mask`; ring-buffer FIFO Bellman-Ford. BIT-EXACT vs
    centrosome `propagate(zeros, seed, mask, 1)`.
  - `radial_reduce(values, seg0, bin_idx, wedge_idx, n, bin_count) -> (frac_at_d,
    mean_frac, radial_cv)` — one serial pass scatter-adds per-object
    `hist[bin]`, `num[bin]`, `wedge_sum[bin,8]`, `wedge_cnt[bin,8]`, then per
    object/bin computes the three features (formulas below). Shape `(n, bin_count+1)`.
- `core/numba/measureobjectintensitydistribution.py` — add `get_radial_distribution`
  next to `get_radial_zernikes`; thin host wrapper.

### Per-object host loop (the #22 fix — geometry on isolated crops)
For each label `L` (via `scipy.ndimage.find_objects`):
1. Crop `labels`/`pixels` to L's bbox; `m = (crop_labels == L)`; **`np.pad(m, 1)`**
   (background border → crop EDT/geodesic bit-identical to the isolated full image).
2. `d_to_edge = scipy.ndimage.distance_transform_edt(m)` — exact Euclidean, KEEP
   (do NOT chamfer this one).
3. `(ci, cj) = centrosome.cpmorphology.maximum_position_of_labels(d_to_edge, m, [1])`
   — KEEP (cheap, exact tie-break). `center` = that pixel.
4. `d_from_center = geodesic_chamfer_fifo(m, ci, cj)` — the numba kernel (replaces
   propagate). Pixels not reached (disconnected from centre) stay `inf` → excluded,
   matching the baseline's `cl > 0` good-mask.
5. Per object pixel (m & finite d_from_center): compute
   - `nd = d_from_center/(d_from_center + d_to_edge + 0.001)` (scaled) or
     `d_from_center/maximum_radius` (unscaled);
   - `bin = min(int(nd * bin_count), bin_count)`;
   - `wedge = (i>ci) + 2*(j>cj) + 4*(|i-ci|>|j-cj|)` (the 8 anisotropy wedges, crop-
     local coords — translation-invariant);
   - emit `(value=pixels[pixel], seg0=L_index, bin, wedge)`.

Concatenate across objects → flat arrays → ONE `radial_reduce` call (host-prep →
one kernel, like coloc/costes).

### Reduction formulas (exact vs reference)
Per object `o`, bins `0..bin_count` (`+1` overflow):
- `FracAtD[o,b] = hist[o,b] / Σ_b hist[o,b]`
- `fraction_at_bin[o,b] = num[o,b] / Σ_b num[o,b]`;
  `MeanFrac[o,b] = FracAtD[o,b] / (fraction_at_bin[o,b] + eps)`  (`eps = finfo(float).eps`)
- `RadialCV[o,b]`: over the 8 wedges with `wedge_cnt>0`, `means_w =
  wedge_sum/wedge_cnt`; `cv = std(means_w, ddof=0)/mean(means_w)`; `0` if no wedge
  populated (matches the `numpy.ma` masked std/mean + the `sum(~mask)==0 → 0` rule).
- Output keys: scaled → bins `0..bin_count-1` as `{MF}_{b+1}of{bin_count}`;
  unscaled → also bin `bin_count` as the `Overflow` feature. (`for bin in
  range(bin_count + (0 if scaled else 1))`, `bin==bin_count → Overflow`.) Reuse the
  `MF_*`/`OF_*` constants from the numpy module.

### Wrapper / bzyx / wiring (same as the rest of the stack)
`get_radial_distribution(masks, pixels, scaled=True, bin_count=4, maximum_radius=100)`
→ `to_bzyx`, per image `_radial_distribution_2d`, `Z>1 → {}` (mirror
`_radial_zernikes_2d`); single→dict, batch→list. Wire `__init__` `__all__` +
`_numba_registries` `"core"` (`radial_distribution`); NOT `_3D_FEATURES`. Extend the
dispatch test. Same trivial append-conflict pattern as #57/#58/#60/#62.

### Verification
- **Geodesic kernel:** `geodesic_chamfer_fifo` == centrosome `propagate(zeros,
  seed, mask, 1)` bit-exact across a shape battery (convex, concave L/U, ring,
  spiral, disconnected, 1–2 px) — the POC's check, widened.
- **End-to-end golden (handles #22):** `numba(multi_image)[key][k]` ==
  `numpy_baseline(object k ISOLATED)[key][0]`, `rtol=1e-6 atol=1e-8`, scaled
  True/False, bin_count ∈ {3,4,8}. (crop+pad ⇒ imported EDT/centre identical; numba
  geodesic ⇒ bit-exact propagate; reductions ⇒ match sparse accumulation.)
- **Independence test:** numba(multi)[k] == numba(object k alone) — the #22 property.
- **Reduction unit test:** scatter-add + CV vs a small numpy reference.
- 2D-only (`Z>1→{}`), empty image, single/batch (4D + ragged) bzyx paths.

### Projected performance (from POCs, 1080²/144 obj)
propagate 735 ms → numba geodesic ~12 ms; geometry ~910 ms → ~40–50 ms (EDT ~22 +
geodesic ~12 + maxpos + crop overhead). Large end-to-end win + the #22 fix; bench
to confirm before quoting.

**Recommendation:** BUILD the lane — per-object-crop + numba chamfer geodesic +
numba reductions. Big speedup, fixes #22, bit-exact to the isolated-object
reference. Still flag to afermg that we replaced centrosome `propagate` with an
equivalent numba kernel (bit-exact, so no behaviour change beyond the #22 fix).

---
(superseded earlier recommendation retained for history:)
ship per-object-crop #22 fix as numpy; pursue texture for numba.

## Per-object-crop design (the #22 fix)

Host loop over objects (the geometry is intrinsically per-object):
1. Crop `labels`/`pixels` to the object's bbox, **pad 1px of background** on all
   sides (so an object pixel on the bbox edge sees background at the correct
   distance — makes the crop's `distance_to_edge`/`propagate` bit-identical to the
   full isolated single-object image).
2. Run the centrosome geometry on the single-label crop → per-pixel
   `d_from_center`, `d_to_edge`, centre coords → `normalized_distance` → `bin_idx`;
   and the 8-wedge `radial_index` from (i,j) vs centre.
3. Emit the object's per-pixel `(value, bin_idx, wedge_idx)`.

Concatenate across objects into flat `(values, seg0, bin_idx, wedge_idx)` (the same
shape the segment kernels consume), then ONE numba kernel does all reductions —
matching the coloc/costes "host prep → one serial kernel" structure.

## Kernel (`core/numba/_radial.py`)

`radial_per_object(values, seg0, bin_idx, wedge_idx, n, bin_count, scaled)` →
per-object arrays for each feature. One serial object/pixel loop scatter-adds into
per-object buffers: `hist[bin]`, `num[bin]`, `wedge_sum[bin, 8]`, `wedge_cnt[bin, 8]`;
then per object/bin computes `FracAtD = hist/Σhist`, `MeanFrac =
(hist/Σhist)/((num/Σnum)+eps)`, and `RadialCV = std/mean` over the non-empty wedges
(matching `numpy.ma` masked std/mean; `RadialCV=0` when no wedge populated).
`error_model="numpy"` so empty-bin `0/0 → nan` matches the reference's
masked/`/sum` behaviour (as in `_costes.py`). Serial, no `prange`/`nogil`
([[no-parallelism-inside-functions]]). Reuse `labels_to_offsets`/`label_to_idx_lut`
for the per-object indexing where it fits.

## Wrapper (`measureobjectintensitydistribution.py`, alongside `get_radial_zernikes`)

`get_radial_distribution(masks, pixels, scaled=True, bin_count=4, maximum_radius=100)`
→ `to_bzyx(masks, pixels)` → per image `_radial_distribution_2d(...)`; `Z > 1 →
{}` (mirrors `_radial_zernikes_2d`). Reuse the `M_CATEGORY` / `MF_*` / `OF_*`
feature-name constants from the numpy module. Single image → dict; batch →
list-of-dicts via `unwrap`. Result keys identical to the baseline (`..._FracAtD_1of4`,
`..._MeanFrac_…`, `..._RadialCV_…`, + `…_Overflow` bins when `not scaled`).

## Verification

- **Golden anchor (handles the #22 fix):** for a MULTI-object image, the numba
  result for object k must equal the **numpy baseline run on object k's ISOLATED
  single-label mask** — i.e. `numba(multi)[key][k] == numpy(isolated_k)[key][0]`.
  This is exact: the crop+1px-pad makes the imported geometry bit-identical to the
  isolated-image geometry, and the reductions match the sparse accumulation. It
  validates the reductions AND the #22 fix together.
- **Independence test (the #22 property):** `numba(multi)` per-object == `numba`
  run on each object alone — the property Issue #22 asserts.
- **Single-object golden:** numba == numpy baseline directly (baseline is correct
  for one object), `rtol=1e-6 atol=1e-8`, `scaled` True/False (overflow bins),
  bin_count variations.
- **Kernel units** (`test_radial_kernels.py`): the scatter-add reductions + CV vs a
  small numpy reference.
- 2D-only (`Z>1 → {}`), empty-image, single/batch (4D + ragged list) bzyx paths.

## Wiring (same append pattern as the rest of the stack)

- `core/numba/__init__.py`: export `get_radial_distribution`, extend `__all__`.
- `bulk._numba_registries` `"core"` dict: `radial_distribution → numba`. NOT added
  to `_3D_FEATURES` (2D-only). Extend the dispatch test in
  `test_backend_correctness.py`. These touch the same files as #57/#58/#60/#62, so
  they resolve the **same trivial append conflicts** the stack already has.

## Alignment with the open PR stack (#57/#58/#60/#62)

- **Layout:** kernel in `core/numba/_radial.py`; wrapper in the existing
  `measureobjectintensitydistribution.py` — matches `_zernike.py`/`_costes.py`/
  `_colocalization.py` + their backend modules.
- **bzyx:** `to_bzyx`, single=batch-of-1, `unwrap`; 2D-only via `Z>1 → {}` exactly
  like `_radial_zernikes_2d`.
- **Primitives reuse:** `labels_to_offsets`/`label_to_idx_lut` + the flat-segment
  representation; no re-implementation of label discovery.
- **Boundary:** import the numerically-sensitive geometry (centrosome
  EDT/propagate/maximum_position), reimplement only the mechanical reductions —
  same call as zernike's `minimum_enclosing_circle`.
- **Kernel discipline:** single serial object loop, `error_model="numpy"`, no
  in-kernel parallelism; "host prep → one kernel" like coloc/costes.
- **Gating:** `HAS_NUMBA` via the existing dispatch; explicit registry composition,
  no error-driven fallback.
- **Process:** profile-first A/B, brief draft PR, plan + memory + todo updated.
- **THE ONE DIFFERENCE (deliberate, must be communicated):** every other lane is
  bit-exact vs main; this one intentionally diverges on multi-object images because
  it FIXES Issue #22. Flag prominently in the PR so reviewers know the divergence is
  the bug fix, not a regression. Reference Issue #22; afermg (owner) also raised it
  upstream to scipy — coordinate so we don't double-fix.

## Open / deferred

- Confirm with afermg whether the cp_measure-side per-object-crop fix is wanted now
  or whether the upstream scipy fix supersedes it (he flagged both in #22).
- Benchmark numbers after step 0.
