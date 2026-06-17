# Numba colocalization lane — implementation plan

Status: PR A SHIPPED 2026-06-03 (#60, OPEN draft on #59). PR B (costes) NOT STARTED.
Backend module mirrors `core/measurecolocalization.py`.

UPDATE (PR A as built): everything below shipped as planned. One deviation worth
noting — golden tests use integer-VALUED float64 arrays (not integer dtypes) for
rank-tie coverage, because feeding a real uint8/uint16 image to the numpy
reference overflows `fi*si` (Overlap/K can exceed 1) and runs the slope `lstsq` in
float32. Our float64 path is strictly more accurate; that divergence is documented
in the backend module + test, not asserted away. 31 tests, full suite 111 green.

## Scope (decided)

- **PR A** (`feat/numba-coloc`, base `feat/bzyx-shape` / #59 — sibling to #56/#57/#58):
  `pearson`, `manders_fold`, `rwc`, **+ `overlap`** (the 5th, unregistered, ride-along).
  All four come out of **one fused per-object kernel** + a grouped flatten.
- **PR B** (`feat/numba-coloc-costes`, **stacked on PR A**): `costes`, with the full
  iterative threshold search (bisection + linear) reimplemented as a numba kernel.

Split rationale: costes is the outlier — closed-form orthogonal-regression coeffs
*plus* a control-flow-heavy iterative search calling `scipy.stats.pearsonr` on
shrinking subsets. The other three (+overlap) are pure fused reductions sharing
one moment+rank kernel. Same discipline as intensity #54→#57 and the zernike split.

## bzyx is mandatory — same contract as #57/#58

Every entrypoint normalises through `primitives/shapes.py::to_bzyx`, exactly like
`get_intensity` (#57), `get_zernike` / `get_radial_zernikes` (#58). NON-NEGOTIABLE:
this lane must not invent its own reshape path.

The wrinkle: colocalization takes a **triple** `(pixels_1, pixels_2, masks)`, but
`to_bzyx(masks, pixels)` is a pair normaliser. Resolution — call it twice against
the shared mask and reuse the single `unwrap`:

```python
def get_correlation_pearson(pixels_1, pixels_2, masks, **kw):
    m_list, p1_list, unwrap = to_bzyx(masks, pixels_1)
    _,      p2_list, _      = to_bzyx(masks, pixels_2)   # same mask → same batch/ndim guards
    results = [_pearson_image(m, p1, p2, **kw)
               for m, p1, p2 in zip(m_list, p1_list, p2_list)]
    return unwrap(results)
```

- Single image → `unwrap` returns the lone per-image dict (matches today's numpy
  `get_correlation_*`, which returns `{feature: [per-object...]}`). Golden test
  compares this dict directly.
- 4D / list batch → list-of-dicts, same convention intensity uses.
- `to_bzyx`'s second call re-runs the same batch/ndim guards on `pixels_2`; if the
  two channels disagree in structure that's a real error and should raise. (If the
  double-guard ever looks wasteful, that's the deferred `to_bzyx`-carries-ndim
  follow-up's job — do NOT special-case it here.)

**2D/3D is a non-issue for coloc.** Unlike intensity (centroids) and zernike
(geometry), every colocalization feature is a function of the per-object *value
vectors only* — no pixel coordinates. So after `to_bzyx` promotes each image to
`(Z,Y,X)`, the flatten step emits flat value vectors and the kernels never branch
on ndim. The `(1,Y,X)`-vs-`(H,W)` divergence that dogs intensity's MAD term
**cannot occur here** — note this explicitly as a clean property of the lane.

## Why the numpy baseline is slow (the win)

`apply_correlation_fun` loops `labels_to_binmasks(masks)` — which materialises an
`(N, H, W)` boolean stack — and each `_ind` call re-indexes the *whole* image
(`pixels_1[mask]`) and funnels a single-object reduction through
`scipy.ndimage.sum/maximum` with `labels` all-ones and `lrange=[1]` (hence the `[0]`
everywhere). That's O(N · H · W) plus per-object scipy dispatch overhead.

Replace with: **one** `to_bzyx`-normalised flatten → values grouped by segment →
**one** fused per-object kernel. Expect a large speedup of the radial-zernikes kind
(baseline is dominated by per-object full-image passes). Numbers TBD by benchmark
(`tasks/bench_*` harness); do not quote a factor in the PR until measured.

## Math, reduced to per-object value vectors `fi`, `si`

All thresholded features use `thr=15` (% of per-object channel max).

- **pearson** (no threshold): two-pass centred (mirror intensity's
  mean-then-`segment_resid_sumsq`), NOT raw moments — avoids cancellation and
  matches `numpy.corrcoef`'s centred algorithm.
  `corr = Σ(fi-mfi)(si-msi) / sqrt(Σ(fi-mfi)²·Σ(si-msi)²)`;
  `slope = Σ(fi-mfi)(si-msi) / Σ(fi-mfi)²` (the lstsq `A·fi+B=si` reduces to this).
- **threshold** (shared by manders/overlap/rwc): per-object `tff=0.15·max(fi)`,
  `tss=0.15·max(si)`; `combined = (fi≥tff)&(si≥tss)`;
  `tot_fi_thr = Σ fi[fi≥tff]`, `tot_si_thr = Σ si[si≥tss]` (note: each over its own
  single-channel threshold, NOT `combined`).
- **manders**: `M1 = Σ fi[combined] / tot_fi_thr`, `M2 = Σ si[combined] / tot_si_thr`.
- **overlap**: over `combined` — `fpsq=Σfi²`, `spsq=Σsi²`, `cross=Σfi·si`;
  `overlap = cross/√(fpsq·spsq)`, `K1=cross/fpsq`, `K2=cross/spsq`.
- **rwc**: needs per-object **dense ranks**. Sort `fi`, assign a rank that increments
  only when the value changes (ties share a rank → reproduces the baseline's
  `lexsort` + unique-diff + `cumsum`, which is order-independent). Same for `si`.
  `R = max(maxrank_fi, maxrank_si) + 1`; `w = (R - |rank_fi - rank_si|)/R`;
  `RWC1 = Σ (fi·w)[combined] / tot_fi_thr`, `RWC2 = Σ (si·w)[combined] / tot_si_thr`.

## Kernels (PR A)

1. `primitives`: a **grouped pair-flatten** — emit `(g1, g2, offsets)` where `g1/g2`
   are the two channels' values reordered so each object's pixels are the contiguous
   block `[offsets[k] : offsets[k+1]]`. Counting-sort placement (O(M), seg ids dense
   `0..n-1` from `label_to_idx_lut`), single-threaded. Reuses `label_to_idx_lut`;
   add alongside the existing `flatten_numba` in `_segment_numba.py` (do NOT
   re-implement label discovery).
2. `core/numba/_colocalization.py`: **one fused** `@njit(cache=True)` per-object
   kernel `coloc_per_object(g1, g2, offsets, n, thr_frac)` that, per object, walks
   its block to get max/means, sorts for the two dense-rank arrays, then a final
   pass accumulating every sum above. Returns arrays for corr, slope, M1, M2,
   overlap, K1, K2, RWC1, RWC2. Single thread per object; objects iterated in a
   plain `for` — **no parallelism inside the kernel** ([[no-parallelism-inside-functions]]).
3. `core/numba/measurecolocalization.py`: the four `get_correlation_*` thin wrappers,
   each `to_bzyx`-normalised as above, slicing the relevant keys out of the fused
   result and emitting the **identical feature-name dict** (reuse the `F_*` constants
   imported from the numpy module, like intensity reuses its constants).

## costes kernel (PR B, stacked on PR A)

- Closed-form `a, b`: variances over `non_zero=(fi>0)|(si>0)` with `ddof=1`,
  `covar=0.5(zvar-(xvar+yvar))`, `a=((yvar-xvar)+√((yvar-xvar)²+4covar²))/(2covar)`,
  `b=ymean-a·xmean`. Reimplement (cheap, mechanical).
- `infer_scale(pixels_1)` (255/65535/2³²/1 by dtype) reproduced host-side; pass
  `scale_max` into the kernel.
- Iterative search reimplemented in numba, **bit-reproducing** the control flow:
  - default `M_FASTER` → bisection: `mid=floor((right-left)/1.2)+left`, window>6 uses
    the sixth-rule else midpoint; `valid` tracks last `R≥0` candidate.
  - `M_FAST`/`M_ACCURATE` → linear descend with the 10×/5×/2×/1× fast-step heuristics.
  - pearson-on-subset computed with our own Σ-based formula (identical to
    `scipy.stats.pearsonr`'s `r`; we ignore its p-value). The baseline's `ValueError`
    branch (constant input / <3 samples) maps to a zero-variance / count guard →
    `left = mid - 1`. Reproduce exactly.
- Apply costes thresholds → `C1/C2` like manders but with `>` (strict) and the
  costes `tot_*_thr_c` single-channel sums.

## Verification

- Golden test vs numpy (`test_coloc_backend.py`), key-by-key, `rtol=1e-6 atol=1e-8`,
  matching `test_backend_correctness.py` style. Cover 2D **and** 3D, single image
  **and** batch (4D + ragged list) — the bzyx paths must all be exercised.
- **Discrete-intensity cases (uint8 + uint16) are mandatory**, not just continuous
  random floats: they exercise rwc tie/dense-rank handling and the costes
  integer-step bisection (with float pixels `scale=1` so the search is trivial and
  the real control flow never runs).
- Kernel unit tests (`test_coloc_kernels.py`): grouped-flatten correctness, dense
  ranks vs `scipy.stats.rankdata(method='dense')-1`, pearson-on-subset vs
  `scipy.stats.pearsonr`.
- Known baseline fragility to document, not reproduce as a crash: `overlap` reads
  `overlap[0]` even when `combined_thresh` is empty (NameError in numpy). Our kernel
  returns `0.0` for the all-below-threshold object; note the divergence is only on
  an input where the baseline itself errors.

## Wiring

- `core/numba/__init__.py`: export the four (PR A) then `get_correlation_costes`
  (PR B); extend `__all__`.
- `bulk.py::_numba_registries`: add the numba `pearson/manders_fold/rwc` (and
  `costes` in PR B) into the `"correlation"` dict (today it just passes `_CORRELATION`
  through). **`overlap` asymmetry**: it is not in `_CORRELATION`, so wiring the numba
  `overlap` would give numba a feature numpy's registry lacks. Decision: expose it as
  importable from `core/numba` and add it to the numba `"correlation"` registry, and
  flag to Alan that the *numpy* `_CORRELATION` should gain `overlap` for symmetry
  (one-line, his call). Do not silently diverge the registries without the note.
- Extend `test_backend_correctness.py`'s dispatch test to assert the coloc keys land
  on the numba module under `set_accelerator("numba")`.

## Results (A/B, `tasks/bench_coloc.py`, 1080², 144 objects, float pixels, JIT warmed)

| feature       | numpy ms | numba ms | speedup | (pre-bincount) |
|---------------|---------:|---------:|--------:|---------------:|
| pearson       |   232.5  |    7.4   | 31.6×   | 18.7×          |
| manders_fold  |   454.0  |    7.3   | 62.0×   | 36.4×          |
| overlap       |   495.8  |    7.3   | 67.7×   | 39.8×          |
| rwc           |   560.4  |  100.1   |  5.6×   |  5.3×          |

## Optimisation deep-dive (post-ship, `tasks/profile_coloc.py` + `exp_*.py`)

Profile (1080², 144 obj): find_objects 3.5 / flatten 2.6 / kernel-rwc-off 4.2 →
~10 ms floor; rwc kernel 97 ms (argsorts ≈93 ms).

- **rwc ranking — DEAD END (confirmed, 4 exact strategies tested).** The two
  per-object dense-rank `argsort`s are rwc's whole cost and intrinsic to a rank
  metric. All ranks bit-identical to the current code; timings (2 channels,
  `tasks/exp_rwc_*.py`):
  - current `argsort` + linear dense-rank: **93.8 ms** (winner)
  - global `np.lexsort` (segment-then-value) + O(M) linear: 220 ms (2.3× worse)
  - `np.sort` + distinct + `searchsorted`: 168 ms (1.8× worse — searchsorted's
    log factor + the distinct build outweigh dropping argsort's index moves)
  - `argsort` with a reused scratch buffer: 94.2 ms (no change → per-object
    allocation overhead is negligible; the sort itself is the cost)
  144 small sorts (log≈12) beat one 646k keyed sort (log≈19, ×2 keys). numba's
  in-kernel quicksort (93 ms) is a touch slower than numpy introsort (77 ms), but
  moving the sort host-side breaks the one-kernel design for ~19%. The only thing
  faster would be a non-comparison (radix/counting) sort — not exact-safe on
  arbitrary float64. Left as-is; rwc is at its floor.
- **Shared label prep — 2× win, APPLIED.** `find_objects` + the flatten's count
  scan were 3 full-image passes. `labels_to_offsets` (one `np.bincount`) yields
  lut+n+offsets, and `flatten_pairs_grouped` became a single scatter scan: 6.2→3.1
  ms prep, bit-identical. ~doubled the three sort-free features.

## Open / deferred

- rwc per-object argsort is the residual cost (sort-bound). Low priority.
- Aggregate win not in PR-A scope: the featurizer calls all 4 functions on the
  SAME image, so the ~3.1 ms prep + a full fused kernel run is paid 4× (each call
  computes all 9 outputs, uses 2-3). A shared-flatten / combined-entry layer would
  fix it — this is the deferred batch-layer concern, not a per-function change.
- The same `labels_to_offsets` bincount trick could speed the numba `intensity`
  prep (separate merged backend) — follow-up.
- If Alan's PR #55 reworks numpy coloc, re-baseline. (#55 currently touches
  intensity/coloc/sizeshape numpy speed.)
