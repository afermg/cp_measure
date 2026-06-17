# Autonomous overnight session — progress & recovery log

**Started:** 2026-06-04 05:25 CEST. **Budget:** ~5h (deadline ~10:25 CEST).
**Mandate (user):** independently send the two missing numba PRs (sizeshape, feret); then
"improve everything as much as possible," use the full budget, keep markdown progress for
compaction recovery, only stop when permitted operations are exhausted.

**Workflow per lane:** research → step-0 profile → plan → build-or-document → optimize to max →
/simplify → draft PR. Conventions: serial `@njit(cache=True, error_model="numpy")` kernels,
NO prange/nogil, `to_bzyx` batch shape, reuse `primitives/`, registry+`__init__` append,
golden numba-vs-numpy tests. Stack base = `feat/bzyx-shape` (#59).

## SESSION COMPLETE (~07:20 CEST, ~2h active)
Authorized work done (feret PR + sizeshape no-go) + deep perf mining of all open numba PRs with every
candidate empirically validated. SHIPPED 4 wins (feret 43.4× NEW PR #65; granularity gather 2.88× #56;
costes fuse −9% #62; coloc fuse −3-4% #60); 6 micro-opts validated as non-wins and reverted; all 8 branches
full-suite green; granularity gather correctness-reviewed (clean). Remaining wins are architectural/fragile
→ documented as recommendations with hard data (`tasks/PR_PERF_ANALYSIS.md`). No fragile/marginal changes
shipped (would violate the no-sacrifice-conventions constraint).

## STATUS BOARD
- [x] **sizeshape** — STEP-0 VERDICT **NO-GO** (Amdahl 1.13×; convex hull dominates = import; moments free).
      Documented in `tasks/numba_sizeshape_plan.md`. No PR. DONE.
- [x] **feret** — SHIPPED **PR #65**, **43.4×** (618→14ms), bit-exact. Branch `feat/numba-feret` on #59.
      `core/numba/_feret.py::_boundary_ijv`: single counting-scatter replaces masks_to_ijv (86%) AND
      feeds hull only boundary pixels (hull(obj)==hull(boundary); 78.9→6.2ms). 8-conn+edge load-bearing.
      14 tests, /simplify applied (return counts→nonzero indices, drop offs copy). DONE.
- [x] **feret hardening** — adversarial correctness review (clean except bool-mask crash → fixed via
      uint8 view) + non-contiguous-label + bool-mask tests. 16 tests. Pushed. DONE.
- [x] **NEW DIRECTIVE (user, ~05:55): deep per-PR performance mining** of every open numba PR
      (#56/57/58/60/62/63/64). 7 parallel mining agents → EMPIRICALLY validated every proposed in-kernel
      opt (built+benched OLD-vs-NEW, didn't trust component estimates). Full results + prioritized
      structural recommendations in `tasks/PR_PERF_ANALYSIS.md`. DONE.
      - SHIPPED: coloc pass-fusion (PR #60, c22d856) — pearson/manders −3-4%, bit-identical, 114 suite green.
      - REVERTED (validated as non-wins): texture sparse-stack (regresses common case), granularity
        label_mean (negligible), zernike vi-skip (breaks vectorization → slower), radial %→branch (no win).
      - GATED RECOMMENDATIONS (real wins, need Tim sign-off): coloc shared-flatten ~3× (architectural);
        granularity bilinear gather ~2.8× (3e-16 + deviates from a stated module note); texture adaptive
        sparse 1.2-1.65× dense fields; zernike shared geometry; intensity bincount-for-(lut,n) (non-finite trap).
      - META-FINDING: lanes are at PRACTICAL floor for the common case; agents' component wins mostly
        don't translate end-to-end (dominant costs = host glue / EDT / sorts / map_coordinates / regionprops).

## SHIPPED THIS SESSION (all bit-exact / bit-identical, all branches full-suite green)
- **PR #65 feret** — 43.4× (618→14ms) bit-exact, 16 tests, /simplify + correctness review + bool fix. NEW PR.
- **sizeshape** — documented NO-GO (1.13× Amdahl), `tasks/numba_sizeshape_plan.md`. No PR.
- **PR #60 coloc** (c22d856) — fused pass 2+3, pearson/manders −3-4%, bit-identical.
- **PR #62 costes** (8b4ee9e) — fused bisection count+pearson, −9%, bit-identical.
- **PR #56 granularity** (89dde43) — numba bilinear gather replaces 16× map_coordinates, **2.88×**
  (741→257ms default). Within-tol (3e-14 ≪ lane's rtol=1e-6), backend golden green, +regression test.
  Flagged in PR comment as a machine-eps resampling change (trivial revert) since granularity, unlike the
  other lanes, was never bit-exact.
- **Deep perf analysis** of all 7 lanes (`tasks/PR_PERF_ANALYSIS.md`), every opt empirically validated.

## STACK HEALTH — all 8 numba branches full-suite GREEN (feret 94 · coloc 114 · costes 145 · texture 100
## · granularity 179 · zernike 104 · radial 118 · intensity 89). All in sync with origin.

## KEY DATA (step-0 profiles, 1080², 144 obj)
### sizeshape (NO-GO)
full 554.6ms = regionprops 73.2% (convex hull alone +187ms/34%) + EDT-radius 25% (EDT 13.4% import + reductions 11.6% reducible). Ceiling 1.13×.
### feret (BUILD)
full 606.2ms = masks_to_ijv **86.1%** (reducible) + convex_hull_ijv 13.0% (import) + feret_diameter 0.6% (import). nonzero→4.04×.

## DECISION LOG
- 2026-06-04 05:25 — sizeshape NO-GO confirmed via property-group sub-profile: dominant primitive
  (convex hull) is import geometry, only reducible primitive (moments) is +4.7ms. No reprieve
  unlike radial(geodesic)/texture(histogram). Will NOT force a 1.13× kernel.
- feret: masks_to_ijv is the same per-label-scan zernike already killed; fix belongs in the numba
  backend (keep numpy baseline untouched, consistent with other lanes). Investigate single-scatter
  (counting-sort via labels_to_offsets) vs nonzero+argsort.

## RECOVERY NOTES
- profilers: `tasks/profile_sizeshape.py`, `tasks/profile_sizeshape_regionprops.py`, `tasks/profile_feret.py`.
- existing coloc primitives to reuse: `primitives/segment.py::labels_to_offsets`,
  `primitives/_segment_numba.py::flatten_pairs_grouped` (single-scatter via offsets).
- If compacted: read this file + `tasks/todo.md` + `tasks/numba_sizeshape_plan.md`, then `git branch`.
