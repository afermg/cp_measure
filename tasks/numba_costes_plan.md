# Numba costes lane — implementation plan (PR B)

Status: PLANNED 2026-06-03. Stacked on **#60** (`feat/numba-coloc`), branch
`feat/numba-coloc-costes`. Finishes the colocalization module (the 5th + last
correlation feature). Mirrors `core/measurecolocalization.py::get_correlation_costes`.

## Scope (decided)

- Port **all three** `fast_costes` modes: `M_FASTER` → `bisection_costes` (the
  registry default + only dispatched path), `M_FAST` / `M_ACCURATE` →
  `linear_costes`. Full drop-in.
- **Exactness scope:** bit-exact vs the numpy reference on **float pixels** (the
  realistic cp_measure input). Integer-dtype input diverges *by design* (see
  §"Integer-dtype divergence"); documented, not reproduced.
- bzyx preserved, identical to #60.

## What the reference actually does (researched)

`get_correlation_costes_ind` (single object — `mask` is boolean, so `labels` are
all-ones and `lrange=[1]`, hence the `[0]` returns):

1. `extract_pixels` → `fi, si` = all object pixels of each channel.
2. **`calculate_threshold(...)` is DEAD** — its five outputs are unpacked then
   never referenced. costes builds its own thresholds. **Skip it entirely.**
3. `scale = infer_scale(pixels_1)` — dtype-keyed: uint8→255, uint16→65535,
   int32→2³²-1, uint32→2³², **else (incl. float64)→1**. Host-side.
4. `bisection_costes` (M_FASTER) or `linear_costes` (M_FAST/M_ACCURATE) → the
   image-wide Costes thresholds `(thr_fi_c, thr_si_c)`.
5. Apply thresholds to the object (note the `>` vs `>=` asymmetry):
   - `fi_above = fi > thr_fi_c`; `si_above = si > thr_si_c`;
     `combined = fi_above & si_above`.
   - `tot_fi = Σ fi[fi >= thr_fi_c]` (only if `any(fi_above)`); `tot_si` symmetric.
   - if `combined` non-empty: `C1 = Σ fi[combined] / tot_fi`, `C2 = Σ si[combined]
     / tot_si`; else `C1 = C2 = 0.0`.
   Output keys: `Correlation_Costes_1`, `Correlation_Costes_2`.

### Orthogonal regression `a, b` (both search fns, over `non_zero = (fi>0)|(si>0)`)

`xvar,yvar = var(ddof=1)`, `xmean,ymean = mean`, `z = fi[nz]+si[nz]`,
`zvar = var(z, ddof=1)`, `covar = 0.5(zvar-(xvar+yvar))`,
`a = ((yvar-xvar) + sqrt((yvar-xvar)² + 4covar²)) / (2 covar)`, `b = ymean - a·xmean`.

### bisection_costes (default)

`left=1, right=scale, mid=floor((right-left)/1.2)+left, lastmid=0, valid=1`.
Loop while `lastmid != mid`:
- `thr_fi_c = mid/scale`; `thr_si_c = a·thr_fi_c + b`;
  `combt = (fi < thr_fi_c) | (si < thr_si_c)`.
- if `count(combt) <= 2`: `left = mid-1`
- else `r = pearson(fi[combt], si[combt])`: if `r < 0` → `left = mid-1`;
  elif `r >= 0` → `right = mid+1; valid = mid`. (Reference's `except ValueError`
  branch → `left = mid-1`; only reachable for <2 samples, excluded by the count>2
  guard. A constant subset gives `r = NaN`; `NaN<0` and `NaN>=0` are both False, so
  neither bound moves and the loop exits next step — replicate by leaving bounds
  unchanged on NaN.)
- `lastmid = mid`; `mid = floor((right-left)/1.2)+left` if `right-left>6` else
  `floor((right-left)/2)+left`.
Final: `thr_fi_c = (valid-1)/scale`, `thr_si_c = a·thr_fi_c + b`.
(`//` is float floor-division; `6/5 = 1.2`.)

### linear_costes (M_FAST / M_ACCURATE)

`i_step = 1/scale`; same `a,b`; `img_max = max(fi.max(), si.max())`;
`i = i_step·(floor(img_max/i_step)+1)`; initial `r = pearson(fi, si)` (full);
`while i > fi_max and a·i+b > si_max: i -= i_step`; then `while i > i_step`:
- `thr_fi_c=i; thr_si_c=a·i+b; combt=(fi<thr_fi_c)|(si<thr_si_c)`;
- **num_true cache:** only recompute `r` when `count(combt) != num_true` (else
  reuse prior `r`) — must replicate, it changes which `r` is compared.
- `if r <= 0: break`; `elif M_ACCURATE or i < i_step·10: i -= i_step`;
  `elif r > 0.45: i -= 10·i_step`; `elif r > 0.35: i -= 5·i_step`;
  `elif r > 0.25: i -= 2·i_step`; `else: i -= i_step`.
- reference's `except ValueError: break` → reachable when a subset has <2 samples;
  replicate as: `count < 2` → break.

### pearson-on-subset (the exactness-critical primitive)

Match `scipy.stats.pearsonr`'s operation order to minimise branch-flip risk:
centre each vector by its mean, `normx = sqrt(Σxm²)`, `normy`, `r = Σ((xm/normx)·
(ym/normy))`, then **clamp to [-1, 1]**. (scipy normalises-then-dots and clamps.)
Bit-identical is not guaranteed — numpy uses pairwise summation, numba a serial
loop (~1e-15 apart) — but this is as close as it gets; see §Risk.

## Kernel + wiring

- New `core/numba/_costes.py`:
  - `_pearson_subset(fi, si, mask, ...)` or operate on a gathered subset — serial,
    scipy-order, clamped.
  - `costes_per_object(g1, g2, offsets, n, scale, mode)` → `(C1[n], C2[n])`.
    `mode`: 0=bisection, 1=linear-FAST, 2=linear-ACCURATE. Per object: regression
    `a,b` → search → C1/C2. Serial object loop, **no in-kernel parallelism**
    ([[no-parallelism-inside-functions]]). No sort (unlike rwc).
  - Reuses the #60 grouped layout: `labels_to_offsets` + `flatten_pairs_grouped`
    give `(g1, g2, offsets)`; costes reads each object's block. No new flatten.
- Append `get_correlation_costes` to `core/numba/measurecolocalization.py`:
  `to_bzyx` twice on the shared mask (reuse `unwrap`) → per image:
  `labels_to_offsets` + `flatten_pairs_grouped` + `infer_scale(pixels_1)` (host,
  imported from the numpy module) + `costes_per_object`. `fast_costes` str → mode
  code. **`thr` is accepted for signature parity but unused** (it only fed the dead
  `calculate_threshold`).
- Wire into `core/numba/__init__.py` (`__all__`) and `bulk._numba_registries`
  `"correlation"` dict (`costes` → numba). Extend the dispatch test in
  `test_backend_correctness.py` (costes now routes to the numba module).

## Verification (decoupled, so each layer is exact by construction)

1. **Control-flow exact:** `test_costes_kernels.py` — a pure-Python transcription
   of `bisection_costes` / `linear_costes` that uses the SAME Σ-pearson as the
   kernel, run with `scale_max = 255` on float `fi,si ∈ [0,1]` (exercises the REAL
   multi-iteration search). numba kernel must match it bit-for-bit. This isolates
   "is the control flow ported correctly" from "does pearson match scipy".
2. **pearson accuracy:** `_pearson_subset` vs `scipy.stats.pearsonr` on random
   subsets, `rtol`. (Establishes the approximation quality feeding the branches.)
3. **End-to-end golden:** `test_coloc_backend.py` extended — numba
   `get_correlation_costes` vs the numpy reference on **float64** pixels (scale=1),
   2D/3D, single + batch, all three `fast_costes` modes. `rtol=1e-6, atol=1e-8`.
   (scale=1 → short, stable search → robust to the iteration sensitivity.)

## Integer-dtype divergence (documented, NOT reproduced)

On uint8/uint16 input the reference (a) computes `z = fi + si` in that dtype →
**overflow** (uint8 200+100 wraps; uint16 too), corrupting `zvar→covar→a,b`, and
(b) the dtype-keyed `scale` (255/65535) mismatches the integer value range. The
float64 numba kernel does neither, so it diverges. Same stance as #60's overlap
`fi*si` overflow / float32 `lstsq` slope. Real float images are unaffected. So the
end-to-end golden uses float64 only; the real multi-iteration search is covered by
the kernel control-flow test (layer 1), not via integer dtype.

## Risk: iteration sensitivity (the one fragility)

costes branches on `pearson(subset)` against hard cutoffs (0, 0.25, 0.35, 0.45). A
~1e-15 difference between the kernel's serial-sum pearson and scipy's pairwise-sum
pearson, landing exactly on a cutoff, can flip one step → a ±1/scale threshold
shift → small C1/C2 drift. Mitigations: match scipy's formula order (above); the
float64 end-to-end path has scale=1 and a near-trivial search (low exposure); the
control-flow test uses the kernel's own pearson (no scipy mismatch). If a future
real-data case drifts, it is this, and it is inherent to porting a
branch-on-float-comparison iterative search — flag, don't pretend it away.

## Optimisation deep-dive (post-ship, `tasks/profile_costes.py` + `exp_costes_scale1.py`)

Profile (1080², 144 obj, float scale=1): prep 3.55 (bincount+flatten) / regression
1.34 / **bisection search 6.95** / final C1/C2 2.06 → kernel 10.36 ms. Speedup 41.8×.

- **Attempted scale==1 short-circuit — REJECTED (not exact).** Hypothesis: at
  scale=1 the [1,1] window makes the search a single trivial iteration whose
  pearson is dead, so the result is closed-form (bisection→(0,b)). A 200-seed check
  vs the reference *appeared* to confirm it — but that data was all positively
  correlated (`a>0`). On an anti-correlated object (`a<0`), iteration 1 sets
  `left=mid-1=0`, the next `mid` becomes 0 (not 1), the loop runs 2–3 more
  iterations and `valid` can reach 0 → `thr_fi_c=-1`, NOT 0. So the pearson IS
  load-bearing at scale=1 and there is no valid short-circuit. The end-to-end
  float64 golden caught it (3d/batch objects hit `a<0`); reverted. Lesson: a
  closed-form "the search is dead" claim must be verified across the FULL input
  domain (both signs of the slope), not just correlated data.
- **No cheaper exact pearson.** Dropping `_pearson_combt` from 3 passes
  (scipy-order normalise-then-dot) to 2 (accumulate then divide) changes rounding;
  the bisection branches on `sign(r)`, so a flip near `r≈0` shifts `valid` → the
  threshold → C drift. Not worth the exactness risk.
- The only real residual win is the cross-feature shared flatten (the deferred
  batch-layer concern in `tasks/numba_opt_followups.md`) — costes re-runs the
  ~3.1 ms prep like the other four. Not a per-function change.

Conclusion: costes is at its exact-computation floor; the iterative search is
genuinely data-dependent even on the float path. Left as shipped (41.8×).

## Open / deferred

- Benchmark vs numpy reference (`tasks/bench_coloc.py` already has the harness; add
  costes). The reference per-object `scipy.stats.pearsonr` in a Python loop should
  make this a large win; measure before quoting.
