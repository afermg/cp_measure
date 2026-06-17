# cp_measure — `legacy: bool` toggle candidate inventory

Goal: find every place where the code (numpy baseline and/or numba backends) has a **known behavior
divergence, numerical-convention choice, or documented quirk where two defensible answers exist** —
candidates for a `legacy`/new toggle like the one already shipped for intensity MAD
(`legacy_mad: bool = False`, commit `cb05ade` on `origin/speedups`).

Each candidate is classified:
- **(A)** genuine legacy-toggle candidate (two defensible answers; users may want either)
- **(B)** just-fix-it bug (one answer is wrong; no toggle, fix and document)
- **(C)** accept-and-document only (divergence is inputs-only / synthetic-only / within-tol; not worth a flag)

Sources cited inline: `tasks/lessons.md`, `tasks/primitive_existence_matrix.md`,
`tasks/backend_cross_pollination.md`, `tasks/todo.md`, the auto-memory
(`cp-measure-upstream-numba-backends.md`, `cp-measure-primitive-existence-matrix.md`), and the source on
the feature branches (`feat/numba-*`, `feat/intensity-bzyx`, `integration/all-numba`, `origin/speedups`,
`origin/main`).

Reference design (the existing toggle to mirror): `cb05ade legacy(intensity): Add flag supporting
original MAD implementation` (on `origin/speedups`), which added `legacy_mad: bool = False` to BOTH
`core/measureobjectintensity.py::get_intensity` and `core/numba/measureobjectintensity.py::get_intensity`.

---

## Verdict A — genuine legacy-toggle candidates (ranked by impact)

### A1. Intensity quartile interpolation convention — `n·q` (CellProfiler) vs `(n-1)·q` (numpy.percentile)  ★ highest impact
- **Features / keys:** `Intensity_LowerQuartileIntensity`, `Intensity_MedianIntensity`,
  `Intensity_UpperQuartileIntensity` (and, downstream, the MAD which subtracts the median). Every object,
  every channel — this is the most-used feature module (intensity is 53% of baseline runtime, per
  `cp-measure-benchmark-findings`).
- **Location:**
  - *Legacy / CellProfiler-baseline:* `origin/main:src/cp_measure/core/measureobjectintensity.py:269-290`
    — `indices = cumsum(areas) - areas`, then `qindex = indices + areas * fraction`. The **local** position
    within a segment is `areas * fraction = n·q`. Numba reproduces this exactly:
    `_segment_numba.py::_interp` (`origin/speedups:171`, `pos = n * frac`), called by `segment_quantiles`.
  - *New / "more-correct":* the rewritten numpy backend on `origin/speedups:src/cp_measure/core/
    measureobjectintensity.py:200` uses `numpy.percentile(vals, [0,25,50,75,100])`, which is the numpy
    `(n-1)·q` linear-interp convention.
- **Two behaviors, precisely:** CellProfiler/legacy puts the quantile at flat index `n·q` inside the
  sorted segment (matches the historical CellProfiler `MeasureObjectIntensity`); numpy.percentile puts it
  at `(n-1)·q`. They agree only at the exact endpoints; for interior quantiles they differ by up to one
  inter-sample gap (largest relative effect on small objects).
- **Determinism:** ALWAYS divergent (every object with >1 distinct interior value), on all real data —
  not an edge case. This is the loudest of the set.
- **Verdict: (A).** Both are defensible: CellProfiler-compatibility (reproduce historical pipelines /
  published feature values) vs the numpy/textbook convention. Note the **current state is an
  inconsistency bug between backends** (the rewritten numpy backend silently switched to `numpy.percentile`
  while the numba backend kept `n·q`), so this *also* needs reconciling — but the resolution is a toggle,
  not "pick one and force it." `primitive_existence_matrix.md §B` explicitly says to "pin the CellProfiler
  `index = n·q` linear-interp convention (all three backends use it)" — i.e. the legacy convention is the
  intended cross-backend contract, and the percentile rewrite drifted from it.
- **Flag scope:** function-level `legacy_quartiles: bool` (or fold into a single `legacy: bool`) on
  `get_intensity` in both `core/measureobjectintensity.py` and `core/numba/measureobjectintensity.py`.
  Default should match whatever the project decides is canonical — given the existing `legacy_mad=False`
  default (new=median MAD) the consistent default here is `legacy_quartiles=False` → numpy convention,
  with `True` restoring `n·q`.
- **Shares the intensity `legacy` flag?** YES — this belongs with the SAME flag family as `legacy_mad`
  (both are CellProfiler-vs-numpy quantile conventions in the same function). Strong candidate to merge
  `legacy_mad` + `legacy_quartiles` into one `legacy: bool` on `get_intensity`.

### A2. Intensity MAD definition — CellProfiler `(100/ndim)%` quantile vs textbook `median(|x−median|)`  ★ already shipped
- **Features / keys:** `Intensity_MADIntensity`.
- **Location (ALREADY IMPLEMENTED — this is the reference toggle):**
  - numpy: `origin/speedups:core/measureobjectintensity.py:206` — `legacy_mad` branches between
    `numpy.percentile(|vals−median|, 100/source_ndim)` (legacy) and `numpy.median(|vals−median|)` (new).
  - numba: `origin/speedups:core/numba/measureobjectintensity.py:131` — `mad_frac = 1/orig_ndim if
    legacy_mad else 0.5`, fed to `segment_quantiles`.
- **Two behaviors:** legacy = the original CellProfiler/cp_measure value, which (mis)used the
  `(100/ndim)%` quantile of the absolute deviations (33rd percentile in 3D, 50th in 2D) instead of the
  median; new = textbook `median(|x − median(x)|)` (which also matches the baseline's own docstring,
  `origin/main:...:58-60`).
- **Determinism:** in 2D the two AGREE (`1/2`-quantile == median); they diverge only in 3D (legacy → 1/3
  quantile). So the divergence is 3D-only, but always-on for 3D.
- **Verdict: (A) — DONE.** Listed for completeness; the toggle exists (`legacy_mad`, commit `cb05ade`).
- **Flag:** `legacy_mad`. A1 should join it under one intensity `legacy` flag.

### A3. radial_distribution Issue #22 — per-object-crop (independent) vs whole-image (leaky) baseline
- **Features / keys:** all `RadialDistribution_*` (`FracAtD`, `MeanFrac`, `RadialCV`) per radial bin.
- **Location:** `feat/numba-radial-distribution:src/cp_measure/core/numba/
  measureobjectintensitydistribution.py:9-17` + `tasks/numba_radial_distribution_plan.md`,
  `tasks/todo.md:350-358`. Shipped as PR #63.
- **Two behaviors:**
  - *Legacy / baseline:* `origin/main`'s `get_radial_distribution` runs the geometry (centrosome
    `propagate` / scipy EDT) on the **whole multi-object image**, so an object's radial profile depends on
    its NEIGHBORS (Issue #22 — owner-confirmed bug, root cause in centrosome/scipy EDT).
  - *New / more-correct:* PR #63 processes each object on its own cropped + 1px-padded single-label
    sub-image → per-object results are **independent of other labels**. The numba header explicitly states:
    "because of the #22 fix, this diverges from the current (buggy) numpy baseline on multi-object fields;
    it equals the baseline run on each object in ISOLATION."
- **Determinism:** divergent only on **multi-object fields where objects are close enough to influence
  each other's EDT/propagate**; single-isolated-object images are bit-identical.
- **Verdict: (A).** This is the cleanest legacy-toggle candidate in the codebase — `tasks/todo.md:354`
  literally calls it "the ONE lane that intentionally diverges from main ... it's a documented bug FIX."
  It is a bug fix, but reproducibility of *old pipelines / published numbers* is a real need, so exposing
  a `legacy` path that restores the whole-image (leaky) behavior is justified. (Also: afermg raised #22
  upstream to scipy; until that resolves, a toggle lets users match either the old cp_measure or a future
  fixed-scipy result.)
- **Flag scope:** its OWN flag — `legacy_radial: bool` (or `whole_image_geometry: bool`) on
  `get_radial_distribution`. Do NOT share the intensity `legacy` flag: different module, different axis of
  divergence (spatial leakage, not a quantile convention). A repo-wide umbrella `legacy: bool` that fans
  out to per-module flags is reasonable, but the underlying knob is independent.
- **Priority:** high — it's a deliberate, always-relevant-on-real-data divergence and the user flagged it
  as "a prime legacy-toggle candidate."

---

## Verdict B — just-fix-it bugs (one answer is wrong; no toggle, reconcile + document)

### B1. Colocalization integer-dtype overflow (uint8 `fi*si`; float32 `lstsq` slope)
- **Features / keys:** `Correlation_Overlap`, `Correlation_K`, `Correlation_Pearson` (slope), Manders/RWC
  unaffected (no pixel products).
- **Location:** `feat/numba-coloc:core/numba/measurecolocalization.py:17-25` (header), kernel upcasts to
  float64 at `:60-61`. Reference (`origin/main` coloc) does NOT upcast. Also `cp-measure-upstream-numba-
  backends.md` PR #60 note and `lessons.md` line 119.
- **Behaviors:** numpy reference computes `fi*si` in the input dtype — uint8 **overflows** (Overlap/K can
  exceed 1.0, impossible under Cauchy-Schwarz) and the Pearson slope's `lstsq` design matrix is uint8 →
  a **float32** result. The numba path upcasts to float64 → strictly more correct.
- **Determinism:** divergent ONLY on genuine integer-dtype input (uint8/uint16). Real float intensity
  images are bit-identical.
- **Verdict: (B) — bug, not a toggle.** `lessons.md:119-120` is explicit: "the float64 path is strictly
  MORE correct; matching an overflow/precision artifact is the wrong target." There is no defensible
  "legacy correctness" here — the legacy result is a numeric corruption (an impossible >1 correlation).
  Fix = upcast in the numpy reference too (or document that the numpy backend should be fed float). No
  user wants the overflowed value. Accept-and-document at minimum; a toggle would only preserve a bug.

### B2. Costes integer-dtype overflow (`z = fi + si` in scale-255 dtype)
- **Features / keys:** `Correlation_Costes` (the automated-threshold colocalization coefficients).
- **Location:** `feat/numba-coloc-costes:core/numba/_costes.py:13-16` (header). Reference overflows
  `z = fi + si` in the input dtype.
- **Behaviors:** identical class to B1 — reference computes the regression intermediate `z = fi + si` in
  uint8/uint16 (scale=255) and overflows, corrupting the regression; the float64 numba kernel does not.
  EXACT on float (scale=1).
- **Determinism:** integer-dtype only; float bit-exact.
- **Verdict: (B) — bug.** Same reasoning as B1; the overflow has no defensible interpretation. No toggle.

> B1/B2 note: if the project wants strict drop-in fidelity with old integer-dtype pipelines, these COULD
> be lumped under a repo-wide `legacy` umbrella, but doing so reproduces a numeric-overflow bug, so the
> recommendation is fix-the-reference + document, not a toggle. Flagged here so they aren't mistaken for
> convention choices like A1/A2.

---

## Verdict C — accept-and-document only (inputs-only / synthetic-only / within-tol; not toggle-worthy)

### C1. `maximum_position` labeled tie-break — scipy quicksort pick vs `>=`-last  (synthetic-only)
- **Features / keys:** `Location_MaxIntensity_X` / `_Y` (`/_Z`); also the radial centre tie-break.
- **Location:** numba `>=`-last rule at `_segment_numba.py:64,94` ("`>=` → keep LAST max in raster
  order"); documented in `lessons.md:31-34`, `backend_cross_pollination.md §5.2 / §7.2`,
  `primitive_existence_matrix.md` ("`>=`-last tie-break ... locked").
- **Behaviors:** on **exact-value intensity ties**, scipy's labeled `maximum_position` returns neither
  the first nor last pixel in raster order (quicksort-dependent, not stable across numpy versions); the
  numba/jax `>=`-last kernel deterministically keeps the last. The labeled scipy call even disagrees with
  the global scipy call.
- **Determinism:** diverges ONLY on exact ties → synthetic/saturated images (the `tiny` benchmark tier).
  `lessons.md:32`: "Real microscopy data has ~no exact ties (so a numba/jax `>=`-last kernel is
  bit-exact)."
- **Verdict: (C) — accept-and-document.** DECIDED `>=`-last everywhere (`backend_cross_pollination.md
  §7.2`: "No per-backend tie-break divergence"). The scipy behavior is non-reproducible (version-dependent
  quicksort), so there is no stable "legacy" value to toggle TO — a toggle would target a moving,
  irreproducible result. `lessons.md:33`: "Don't burn effort matching it; accept the synthetic-only diff."
  NOT a toggle candidate. (The user remembered this one specifically; the answer is: real-data bit-exact,
  documented synthetic-only diff, no flag.)

### C2. `to_bzyx` single-slice-volume `(1,Y,X)` MAD fraction (`2 if Z==1 else 3`)
- **Features / keys:** `Intensity_MADIntensity` only.
- **Location:** `feat/intensity-bzyx` / `feat/numba-intensity`; `lessons.md:98-100`, `todo.md:215`,
  `cp-measure-upstream-numba-backends.md` PR #57 note ("Known accepted divergence (Tim: keep as-is)").
- **Behaviors:** `to_bzyx` collapses both 2D `(H,W)` and a single-slice volume `(1,Y,X)` to the same
  canonical `(1,Y,X)`, erasing original ndim. Backends infer 2D as `Z==1`, so a degenerate `(1,Y,X)`
  *volume* gets `mad_frac = 1/2` (median MAD) where the baseline's `pixels.ndim==3` gave `1/3`.
- **Determinism:** diverges ONLY for a single-slice 3D volume `(1,Y,X)` — a degenerate shape; real 2D
  `(H,W)` and Z>1 volumes are bit-identical. Untested (conftest `DEPTH_3D=8` never exercises Z=1).
- **Verdict: (C) — accept-and-document.** `todo.md:230` records the explicit DECISION (Tim, 2026-06-03):
  "keep the inference as-is ... do NOT add an xfail, do NOT drop the note." The `Z==1` value (1/2 / median)
  is arguably *more* correct (matches the baseline's own docstring) and is now governed by the `legacy_mad`
  flag anyway (A2). The deeper fix is to carry per-element ndim out of `to_bzyx` (a separate low-priority
  cross-backend PR), NOT a user toggle. Note this is a *consequence* of A2's convention, not an independent
  knob.

### C3. Granularity bilinear-gather resampling (3e-14) and per-object mean summation order
- **Features / keys:** `Granularity_*` spectrum.
- **Location:** `feat/accelerator-numba-granularity:core/numba/_granularity.py:35`,
  `measuregranularity.py:12`; `lessons.md:90` (per-object mean bincount-vs-scipy), `:183-193` (bilinear
  gather); `cp-measure-upstream-numba-backends.md` PR #56.
- **Behaviors:** (a) the numba `bilinear_gather` replaces scipy `map_coordinates` (order-1) and matches it
  to ~3e-16/call, full-output delta ~3e-14; (b) per-object mean differs by floating summation order
  (bincount vs scipy.ndimage.mean). Both are ≪ the lane's existing `rtol=1e-6` contract — "granularity was
  never bit-exact" vs baseline.
- **Determinism:** always present but at machine-epsilon magnitude (within the documented tolerance).
- **Verdict: (C) — accept-and-document.** These are floating-point reassociation differences, not two
  defensible *answers*. No toggle; covered by the existing `rtol=1e-6` accuracy contract + regression test.

### C4. Zernike / radial_zernike phase + normalization convention (`atan2` arg order, all-pixels vs in-disk denom)
- **Features / keys:** `Zernike_*` (shape) and `RadialDistribution_ZernikeMagnitude/Phase_*`.
- **Location:** discussed in `primitive_existence_matrix.md §C` and `backend_cross_pollination.md §5.3`;
  the **rust** port drifts 5-6% here (`zernike.rs:235` swapped `atan2(re,im)`; `:206` normalizes by
  all-pixel count while accumulation skips `r²>1`). The shipped **numba** zernike
  (`feat/numba-zernike:core/numba/_zernike.py`) imports the centrosome LUT and reuses the baseline's exact
  convention; A/B and goldens show it matches baseline (28.7× / 20.9×, bit-exact-tested).
- **Behaviors:** there IS a real ambiguity (phase arg order; whether the magnitude denominator is the
  all-pixel count or the in-disk count) — but the numba backend deliberately matches the numpy baseline's
  exact convention; only the (non-shipping) rust idea diverged.
- **Determinism:** N/A for the shipped backends — they agree with baseline by construction. The 5-6% drift
  is a rust-only artifact of getting the convention wrong.
- **Verdict: (C) — accept-and-document / already-resolved.** NOT a toggle: there is one baseline
  convention and the shipped backends honor it. The lesson (`primitive_existence_matrix.md:217`) is to
  PIN the baseline's convention and assert all backends match — a correctness guard, not a user-facing
  legacy/new choice. (If the baseline convention itself is ever judged "wrong" and a corrected variant is
  added, THEN it would graduate to an A-style toggle. Today it's a single locked convention.)

---

## Cross-cutting recommendation on flag design

- **One intensity `legacy: bool`** covering A1 (quartile `n·q`) + A2 (MAD `1/ndim`-quantile) — both are
  the same CellProfiler-vs-numpy quantile-convention axis in `get_intensity`, present in both the numpy and
  numba backends. A2 already exists as `legacy_mad`; A1 should be added and (ideally) the two unified.
  C2 is a side-effect of A2's convention, not a separate flag.
- **radial_distribution gets its OWN flag** (A3) — different module, different divergence axis (spatial
  EDT leakage / Issue #22), independent default.
- **B1/B2 (coloc/costes integer overflow) are NOT toggles** — they reproduce numeric corruption; fix the
  reference (upcast to float64) and document the integer-dtype caveat. Only put them behind a repo-wide
  `legacy` umbrella if exact bit-for-bit replay of *buggy* integer pipelines is an explicit requirement.
- **C1 (max-position tie-break)** has no stable legacy target (scipy is version-dependent) → no toggle.
- A repo-wide `legacy: bool` that fans out to the per-module knobs (intensity quantiles, radial geometry)
  is a clean public API, as long as each underlying convention stays independently addressable.

## Already-resolved / NOT toggle-worthy (explicit)
- C1 max-position tie-break: real-data bit-exact, synthetic-only diff, scipy target non-reproducible.
- C3 granularity 3e-14 / summation order: within `rtol=1e-6`, never-bit-exact by design.
- C4 zernike convention: shipped numba matches baseline; only the non-shipping rust idea drifted.
- C2 `(1,Y,X)` MAD: degenerate-shape-only, decided "keep as-is," governed by A2's flag.
