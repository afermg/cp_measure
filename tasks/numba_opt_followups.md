# Numba optimisation follow-ups (cross-cutting, deferred)

Two reusable wins surfaced during the colocalization deep-dive (PR #60). Both are
out of any single feature-PR's scope but are concrete and worth doing.

## 1. `bincount` label prep for the merged numba `intensity` backend

`primitives/segment.labels_to_offsets` (added in PR #60) derives `(lut, n,
offsets)` from a single `np.bincount(masks.ravel())`, replacing
`scipy.ndimage.find_objects` **and** a separate count scan. In coloc this cut the
shared prep ~2× (6.2 → 3.1 ms on 1080²/144 obj) and ~doubled the sort-free
features.

The merged numba `intensity` backend (PR #54) still uses `label_to_idx_lut`
(find_objects) + `flatten_numba`'s own count pass. The same one-`bincount` trick
should apply: derive offsets once, make the flatten single-scatter. **Caveat:**
`intensity` needs per-pixel coordinates (xc/yc/zc) that coloc does not, so the
flatten kernel differs — this is a port of the *idea* (bincount → offsets →
single scatter), not a drop-in reuse of `flatten_pairs_grouped`. Measure first;
intensity's flatten also drops non-finite pixels, which changes the count, so
offsets-from-bincount would need a finite mask or a different count source.

## 2. Shared flatten across the 4 (soon 5) colocalization features

The featurizer calls `pearson`, `manders_fold`, `rwc`, `overlap` (and soon
`costes`) **on the same `(pixels_1, pixels_2, masks)` image**. Each call
independently redoes:
- `to_bzyx` normalisation,
- `labels_to_offsets` (the `bincount`),
- `flatten_pairs_grouped` (the scatter), and
- a full `coloc_per_object` run — which *already computes all nine outputs* and
  uses only 2–3 of them per call.

So on an N-feature sweep the ~3.1 ms prep + the fused kernel are paid N× when one
flatten + one kernel run would suffice. This is the **shared-flatten / batch-layer
concern**, deliberately deferred — it is NOT a per-function change and must not
become in-kernel parallelism ([[no-parallelism-inside-functions]]). The clean
shapes for it:
- a single combined entry (`get_colocalization(pixels_1, pixels_2, masks)`) that
  flattens once, runs the fused kernel once, and returns the union dict; the four
  registry functions become thin selectors over a memoised/shared result, OR
- an external batch/orchestration layer that owns the flatten and feeds the
  kernels — the same layer that will own cross-image parallelism.

Either way the fused kernel is already the right primitive (it computes
everything in one pass set); only the call wiring needs to stop re-flattening.

## Status / pointers

- Coloc lane: PR #60 (`feat/numba-coloc`), plan `tasks/numba_colocalization_plan.md`.
- rwc remains sort-bound (per-object argsort, intrinsic to a rank metric);
  separately investigated — see the deep-dive section of the coloc plan.
