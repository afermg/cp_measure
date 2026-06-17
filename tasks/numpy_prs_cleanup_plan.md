# Plan v2 — slim + fix the numpy perf PRs (#74/#75/#76/#77)

> v1 proposed a foundation PR with a shared `foreground_scatter`/`segment_sum` seam. A review
> refuted that: the seam fits none of its three callers (each needs a different per-object
> frame — bbox-min / circle radii / fractional grid), the sequencing was impossible (F would own
> code that lives in #74/#77), and it serialized 4-5 PRs behind a new one. **v2 drops the
> foundation PR.** The only genuinely shared piece is "label → index + per-object bbox origin",
> which is an *additive, backward-compatible* extension of the existing `label_to_idx_lut`.
> Everything else is an in-place fix on the PR that already owns the code.

## What's real (verified)
- **Granularity boundary bug — REAL, ~1.3% of configs.** `map_coordinates(mode='constant')`
  returns `0.0` for a source coord that floats just above `new-1` (e.g. `63.00000000000001`),
  while the fused operator clamps to `rec[new-1]`. 249/19224 realistic (size, subsample) combos
  overshoot (e.g. 158×0.4→63, 80×0.2→16); in those, every edge-touching object diverges.
- **Eigenvalue-clip bug — REAL, ~4% of thin objects.** `inertia_2d` lacks skimage's
  `clip(eigvals, 0, None)`, so oblique thin objects give `NaN` axis_minor / ecc>1 (+ a warning).
  The numba `_moments.py` lacks the clip **too**.

---

## #77 `perf/sizeshape-moments-scatter` — the main slim (+ the shared bit)

### a) Extend `label_to_idx_lut` (additive, in `primitives/segment.py`)
`find_objects` is already called inside it; today the slices are discarded. Add an opt-in:
```python
def label_to_idx_lut(masks, *, return_bbox=False):
    ...                       # unchanged default path -> (lut, n)
    if return_bbox:
        origins = numpy.array([(sl[0].start, sl[1].start) for sl in bboxes if sl is not None])
        return lut, n, origins   # per-object (rmin, cmin), ascending-label order
```
Backward-compatible (default unchanged), so #74 — the other `label_to_idx_lut` caller — is
untouched. This is the **only** shared change, and it eliminates the `1<<31` sentinel +
`minimum.at` in `_moments.py`. **[findings 3, 9, 13]**

### b) `primitives/_moments.py` — fix + converge with the numba copy
- `spatial_moments_2d(labels, *, advanced=False)`:
  - Build `seg` + per-object `(rmin, cmin)` from `label_to_idx_lut(return_bbox=True)`; local
    coords `local_r = rows - rmin[seg]` (drop `unique`/`searchsorted`/`minimum.at`/sentinel).
  - **Keep** the `n == 0` early return (empty `(0,4,4)`/`(0,7)`). **[empty-mask gap]**
  - Powers from `_ORDER` (`[r**k for k in range(_ORDER)]`). **[finding 14]**
  - Compute `normalized`/`hu` **only when `advanced`** (else `None`); `raw`/`central` always.
    **[finding 8]**
  - Docstring: state the real large-object divergence (summation-order, grows with object size;
    worst seen ~3e-12 rel on a 1000² object) — not "bit-exact", not "~1e-14". **[findings 10, 14]**
- `inertia_2d(central)`: **`numpy.clip(eig, 0, None)`** on both eigenvalues before returning.
  **[finding 2 — the bug]**
- **Adopt** the numba branch's `moment_feature_dict` + `derive_normalized_hu` verbatim (plus the
  clip) so the numpy and numba `_moments.py` converge on one assembly. NOTE: this is a real 3-way
  reconcile (#77's 166 lines vs #78's 210), **not** mechanical — do it as a deliberate port, and
  **apply the same clip to #78's copy**. **[findings 4, 6, 11(numba)]**
  - `moment_feature_dict` must emit keys in the **grouped** order get_sizeshape currently uses
    (all Spatial, then Central, then Normalized, then Hu, then Inertia) — if the numba version
    interleaves by `(p,q)`, use a grouped variant — so featurize column order is unchanged. Add a
    regression test pinning the AreaShape column order. **[finding 7]**
- Keep bare tuples (no NamedTuple) — a NamedTuple collides with `moment_feature_dict`'s
  positional 6-element inertia contract. Document the tuple shapes instead. **[finding 15 dropped]**

### c) `get_sizeshape` — rewire (2D), preserve 3D
- 2D advanced: `inertia_2d(central)` **once**; pass its result to both the axis/ecc/orientation
  derivation *and* the tensor features. **[finding 5]**
- Build the 6-tuple `moment_feature_dict` needs explicitly:
  `inertia6 = (it00, it_off, it_off, it11, eig_major, eig_minor)` (off-diagonal duplicated), then
  `results |= moment_feature_dict(raw, central, normalized, hu, inertia6)` — the **5-arg**
  signature, called **only under `calculate_advanced`** (where normalized/hu are non-None).
  **[findings 1, 8 — the v1 `moment_feature_dict(moments, inertia)` call was wrong]**
- axis/ecc/orientation (always emitted in 2D) come from the single `inertia_2d`; so `inertia_2d`
  runs unconditionally in 2D, `moment_feature_dict`/normalized/hu only under advanced.
- **3D unchanged**: keep the regionprops path for 3D axis lengths AND the 3D advanced moment
  block (3-index moments + 3×3 inertia). `moment_feature_dict` is 2D-only; do **not** route 3D
  through it. **[finding 6/3D gap]**
- Drop the standalone `nobjects = (unique(masks)>0).sum()`; reuse `n` from `label_to_idx_lut`.
  **[finding 9]**

---

## #74 `perf/zernike-vectorize` — minimal
- Wrap `_zernike_scores`'s `/radii` division in `numpy.errstate(invalid="ignore", divide="ignore")`
  so single-pixel (radius-0) objects don't warn. **[finding 11]**
- **Leave `_zernike_scores` in `utils.py`** (the move to `primitives/` is cosmetic, low value, and
  would force import churn on #75 — defer it). **[finding 7 deferred]**

## #75 `perf/radial-zernike-vectorize` — don't regress
- Do **NOT** add a hard `pixels.shape != labels.shape` raise — the original code *supported*
  smaller pixels via the `ijv` in-bounds clip; a raise is stricter than both old and current.
  **Decision:** keep the current co-shaped contract (normal usage), OR restore the in-bounds clip
  if smaller-pixels support is actually needed. Recommend: keep co-shaped, add a clear assertion
  message only. **[finding 12 corrected — v1 mislabeled a regression as a fix]**

## #76 `perf/granularity-fused` — the boundary bug, semantics (A)
- Fix `_make_fused_upsample_mean` to **reproduce `map_coordinates(mode='constant')`**: drop
  (zero) the operator contribution for any foreground pixel whose source coord floats outside
  `[0, new-1]`; keep `counts` over **all** foreground (denominator unchanged). This stays
  **bit-exact with the old path**, so the existing `test_granularity.py` golden (verbatim
  `map_coordinates` reference) still passes. **[finding 1, semantics A]**
- Add a golden test on an **overshooting** size (e.g. 158→63) with an edge object, vs
  `_reference_unfused`.
- **Do NOT** adopt v1's option (B) (keep the clamped edge value): it changes outputs and breaks
  the existing golden. Whether the old map_coordinates zeroing is itself worth fixing is a
  **separate, explicitly-labeled bugfix**, not part of this perf PR. **[A/B contradiction resolved]**
- 3D path already uses `map_coordinates` (same zeroing) → consistent, no change. Keep #76
  **independent of #77** (don't reach for the extended `label_to_idx_lut`).

---

## Sequencing (no foundation PR)
All four stay on their current bases; each gets in-place fixes. #75 stays stacked on #74. The
`label_to_idx_lut` extension is additive, so it ships inside #77 without touching #74. No new PR,
no serialization, no cross-stack #78 dependency.

**#78 (numba):** apply the eigenvalue clip to its `_moments.py` too (same bug), and let the numpy
`moment_feature_dict` port match it — the actual git convergence of the two `_moments.py` happens
whenever the stacks merge (a known, separate coordination, not forced here).

## Perf guardrails (unchanged — do NOT regress while slimming)
- Keep `_moment_matrix`'s 16 buffered `bincount`s (not one `numpy.add.at`).
- Keep the two scatter passes (raw + centred); don't derive central via binomial recurrence.
- Keep the granularity operator CSR over the full grid; don't compact columns.

## Coverage (15 findings)
1→#76(A) · 2→#77 clip (+#78) · 3→#77 label_to_idx_lut bbox · 4→#77 moment_feature_dict(5-arg) ·
5→#77 single inertia · 6→#77 adopt+port (deliberate, not mechanical) · 7→column-order test
(placement deferred) · 8→#77 advanced gate · 9→#77 drop unique · 10→#77 docstring ·
11→#74 errstate (+#78 clip) · 12→#75 keep clip / no hard-raise · 13→#77 bbox (no sentinel) ·
14→#77 `_ORDER` powers · 15→dropped (NamedTuple conflicts with moment_feature_dict).

## Decisions (resolved)
1. **#75 smaller-pixels → co-shaped (Option A).** Require `pixels.shape == labels.shape`; raise a
   clear error otherwise. Do not carry forward the old in-bounds clip.
2. **Column order → match PyPI 0.1.19 exactly.** Verified: 0.1.19 and the current branch already
   emit the identical *grouped* order
   `Spatial(3×4) · Central(3×4) · Normalized(4×4) · Hu(0..6) · InertiaTensor(2×2) · Eigenvalues(0,1)`.
   `moment_feature_dict` must reproduce this exact order (note Spatial/Central are 3×4 but
   Normalized is 4×4 — not a single uniform loop). Add a regression test that pins get_sizeshape's
   full key list against the 0.1.19 order so any future drift fails CI.
