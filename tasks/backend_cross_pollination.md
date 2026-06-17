# cp_measure — cross-pollination & backend-ecosystem investigation

**Goal of the refactor:** turn base `cp_measure` into a backend ecosystem. A user installs one of
`{none, numba, jax, numba+jax}`; at runtime the library auto-routes each feature function to the
fastest implementation available. Implementations must be **modular yet parallel** (each backend a
drop-in for the same function), and the numba pieces must be separable so a **jax-only install still
works via a pure-numpy fallback**.

This document is the cross-pollination findings from a deep read of all five sibling repos
(baseline / fast-numba / jax / speed-npvec / rust). **Rust is mined for ideas only — no Rust code
will ship.** The batching rewrite is deferred to a follow-up.

Sources: deep per-repo analyses, 2026-05-30. File:line refs are into each repo's `src/` tree.

---

## 0. TL;DR — the five things that matter

1. **numba is already optional in both `fast` and `jax`** (guarded `import numba` + working numpy
   fallbacks). The refactor's "modularise numba" goal is ~80% done structurally. **But we replace ALL
   the existing `try/except` import guards with clean capability detection** (`importlib.util.find_spec`,
   §1a) — see §7.4: no try/except anywhere in the ecosystem. The work is *consolidating* the duplicated
   fallbacks behind one flag-driven resolver, not inventing fallbacks.

2. **The real hidden dependency is NOT numba — it's `scipy` + `scikit-image` + `centrosome` + `mahotas`.**
   Every "jax" function still calls these on the host (find_objects, distance_transform_edt,
   regionprops, marching_cubes, morphology, minimum_enclosing_circle, convex_hull_ijv, zernike LUTs,
   haralick). A truly minimal jax-only install is **not** achievable today. Decide deliberately which
   of these stay as mandatory base deps (recommendation: all four — reimplementing them is out of scope).

3. **One abstraction unifies almost everything: segment-reduce over labels.**
   `segment_reduce(values, labels, n, op)` + `segment_quantile`. numpy→`scipy.ndimage.*(v,lbl,index)`,
   numba→bincount/scatter kernel, jax→`jax.ops.segment_*`. Intensity, both Zernikes, all
   colocalization, radial_distribution histograms, and sizeshape moments are all instances of it. The
   `speed` repo independently re-derived the numpy version; jax independently re-derived the jax
   version; numba has four redundant copies of the kernel. **Extract this once and most per-function
   divergence disappears.**

4. **Rust's transferable lesson = "scatter once, accumulate running sums in one reduce loop, size
   buffers to the data not the theoretical max."** CORRECTION (2026-05-30 mining,
   `primitive_existence_matrix.md`): rust intensity (827×) is NOT a single fused pass — it's **two-phase**
   (scatter pixels to per-label buckets, then reduce); the four cross-sums (Σ, Σx, Σy, Σx·I, Σy·I) live in
   the *reduce* loop (`intensity.rs:150-163`), and std is a separate 2nd pass. Rust texture (0.21×) =
   dense 256² GLCM scanned ~6× per object. Port the scatter-then-reduce + cross-sums pattern to numba;
   texture is a cautionary tale, not a template.

5. **Two correctness hazards to fix *during* the refactor, not after:** (a) the
   `try/except Exception: pass` silent numpy fallback in intensity (present in both fast and jax) —
   it once hid the float64-index GPU bug and will hide the next one; (b) the per-object `max_position`
   tie-break loop, the one piece that resists vectorisation and forces a per-object Python loop in
   every backend.

---

## 1. Backend-ecosystem architecture

### 1a. Dispatch seam
All four repos expose features through the **same registry shape**: `bulk.py` has `get_core_measurements()`,
`get_core_measurements_3d()`, `get_correlation_measurements()`, `get_multimask_measurements()`, each
returning `{name: callable}` with lazy imports inside the function body. The featurizer pulls these dicts
and calls `func(mask, pixels, **params)` (single) or `func(pixels_1=, pixels_2=, masks=, **params)`
(colocalization). **This flat name→callable map is the natural dispatch seam** — swap which module each
name resolves to based on installed backends. No featurizer change needed.

Today there is **no central dispatch**: the `jax` repo re-queries `jax.default_backend()` *inside each
function*, and `fast` checks per-module `_HAS_*` flags. Recommendation: **one capability resolver**
(`detect_backends() -> {numba: bool, jax: bool, jax_gpu: bool}`) consulted once, feeding a single
routing table (§2). Replace the three different flag names (`_HAS_NUMBA`, `_HAS_SIZESHAPE_NUMBA`,
`_HAVE_NUMBA_INTENSITY`) with one.

**NO try/except anywhere — clean early checks only (§7.4).** Capability detection uses
`importlib.util.find_spec("numba")` / `find_spec("jax")` (availability without importing or catching),
plus `jax.default_backend()` for GPU presence. These run **once at import**, set the boolean flags, and
every dispatch decision keys off those flags. The resolved backend path is then called **directly, with
no guard** — if it errors, it raises (loudly, by design). This eliminates the silent-fallback bug class
(§5.1) structurally: there is no "attempt accelerated, swallow error, fall back" path to hide a bug. A
backend is either present (flag true → used) or absent (flag false → never attempted).

### 1b. The contract every backend must satisfy (from baseline)
- Single-image: `func(masks, pixels, **params)`; colocalization: `func(pixels_1=, pixels_2=, masks=, **params)`.
- Return: `dict[str, np.ndarray]`, **every value a 1-D array length = n_labels**, index-aligned to
  sorted positive labels. The featurizer `column_stack`s these — wrong length or order breaks it.
- Exact dict **keys** must match (string-formatted with scale/bin/n/m/gray_levels). A backend that
  changes a key name is a silent schema break.
- 3D: `{intensity, sizeshape, texture, granularity}` support 3D; `{radial_distribution, radial_zernikes,
  zernike, feret}` return `{}` on 3D.

### 1c. Layered design (recommended)
```
featurize()                          # unchanged orchestration
  └─ registry (name → resolved backend callable)   # the dispatch table, §2
       └─ per-function backend impls               # numpy | numba | jax variants
            └─ SHARED PRIMITIVE LAYER               # §3 — the real cross-pollination payoff
                 segment_reduce / segment_quantile
                 boundary_ijv / label_to_idx
                 convex_hull (once per object)
                 host helpers: find_objects, EDT, regionprops, centrosome, mahotas
```
The primitive layer is where numba/numpy/jax actually differ; everything above is shared.

---

## 2. Per-function routing table & cross-pollination

Legend: **none** = pure numpy (no extra deps) · **nb** = numba · **jax** = jax (CPU-XLA or GPU).
"Best backend" from the corrected benchmark (large 1080²/142obj). ⚠ = correctness/perf trap (§5).

| function | best backend(s) | core idea to standardise | cross-pollination notes |
|---|---|---|---|
| **intensity** | jax-GPU (337×) / nb (355×) | segment_reduce + segment_quantile; single-pass moments | Rust's **four cross-sums in one reduce loop** (Σ, Σx, Σy, Σx·I, Σy·I → centroid + mass-disp; rust is two-phase scatter-then-reduce, not one fused pass — `intensity.rs:150-163`) is the ideal nb kernel. `speed` proves the numpy segment-reduce. jax uses 2× lexsort for quartile/MAD. ⚠ silent fallback; ⚠ max_position loop. |
| **sizeshape** | nb (modest) / none | regionprops_table (whole-image) + EDT radius loop | jax repo computes **moments as `bincount(weights)`** (no per-object loop) — port to the numpy backend. numba `_numba_sizeshape` (perim/crofton/fill/convex) already has a complete numpy fallback shipped in jax repo. 3D is 100% skimage everywhere — leave it. |
| **texture** | jax-GPU (14×) | sparse GLCM (sort+RLE) NOT dense | jax repo's `_haralick_sparse`/`_build_all_glcms_2d` is **pure-numpy and better than baseline** — make it the `none` backend. mahotas stays the fallback. ⚠ **do NOT copy rust's dense-256 GLCM** (that's the 0.21× bug). bbox-crop per object is worth keeping. |
| **granularity** | jax-GPU (64×) / nb (11×) | morphology image→image ops (loosely coupled) | numba: Vincent/Robinson 3-scan reconstruction + Van Herk/Gil-Werman disk erosion (O(r) not O(r²)). jax: `lax.reduce_window` + `lax.while_loop` reconstruction. Both swap cleanly behind a `reconstruction()`/`disk_erode()` primitive. skimage is the `none` fallback. |
| **granularity_fullres** | jax-GPU (82×) | (same fn, `subsample_size=1.0`) | NOT a separate function — it's `get_granularity` with no subsampling. Dominant baseline cost (~41% on large). jax-cpu is a **trap** here (~12s fixed XLA cost). |
| **feret** | nb / rust-idea | convex hull once, share with zernike | Rust: integer-exact (i64) monotone-chain hull (`geometry.rs:29-67`). CORRECTION: rust feret width is **O(n²) brute force** (all-pairs + per-edge perp dist, `ferret.rs:47-88`), NOT rotating calipers; and rust **recomputes the hull 3×** (feret + both zernikes), so "compute hull once and feed feret + both zernikes" is an *opportunity we'd create*, not what rust does. numba `_boundary_ijv` already extracts boundary pixels (shared primitive). centrosome does the geometry in baseline. |
| **zernike (shape)** | jax-GPU (147×) / nb (13×) | weighted segment_sum over (n,m) | numba has **4 redundant kernels** (2 live, 1 dead — see `primitive_existence_matrix.md §C`); jax has `_zernike_gpu` scatter-add. **Rust's LUT+Horner-on-r²+iterative-complex-powers fused kernel is the best formulation** (this fused claim IS accurate for rust zernike) — one kernel covers shape + radial via a `weight` arg. Maps 1:1 to jax segment_sum. centrosome LUT stays a dep. |
| **radial_zernikes** | jax-GPU (25×) / nb | same Zernike kernel, weighted by pixel value | Same machinery as shape zernike. ⚠ **rust 5-6% off** — likely `atan2(real,imag)` arg order and all-pixels-vs-in-disk normalization denominator; verify the *baseline's* convention before porting. |
| **radial_distribution** | jax-GPU (11×) / nb (2.6×) | BFS propagation + bincount histograms | jax `_propagate_jax` (while_loop relaxation, 8 fixed shifts) replaces centrosome's sequential propagate and **ports to a numpy/numba iterative relaxation**. Histograms = `np.add.at` segment scatter (already vectorised in numba repo). ⚠ jax `_radial_histograms` defined but **never wired up** (dead). |
| **coloc pearson** | jax-GPU (40×) / nb (8×) | single-pass 5 moments → r + slope | Rust: closed-form `r` and least-squares slope from `Σx,Σy,Σx²,Σy²,Σxy` in one loop — avoids `corrcoef`+`lstsq` allocations. Pure numpy in jax repo already (bincount). |
| **coloc manders** | jax-GPU (62×) / nb (36×) | threshold + segment sums | numpy `maximum.at`+`bincount` core, jax `.at[].max/.add`. Clean segment-reduce instance. |
| **coloc rwc** | jax-GPU (61×) / nb (16×) | dense-rank + weighted Manders | jax does **rank fully on device** (lexsort + cumsum + `associative_scan(max)`); numpy/numba path is a per-label rank loop (slow). The on-device rank is a nice idea but GPU-only. |
| **coloc costes** | nb (18×) / none | iterative threshold search — **keep on CPU** | Data-dependent bisection/linear loop; **not jit-able / not a vmap target** (jax repo has no jax path here, by design). Rust's "recompute Pearson only when below-threshold count changes" memoization is a cheap win. |

**Functions with NO acceleration anywhere worth it:** texture-3D (regionprops-bound), sizeshape-3D
(marching_cubes-bound), costes (sequential). Route these to `none`/skimage always.

---

## 3. The shared primitive layer (the core deliverable)

Extracting these is where the four parallel codebases collapse into one. Each primitive has a
numpy/none impl (mandatory), and optional numba/jax impls dispatched by capability.

### 3a. `segment_reduce(values, labels_0indexed, n, op)` + `segment_quantile(...)`
The single most reused pattern. Instances: intensity (sum/mean/min/max/quartile/MAD), zernike &
radial_zernikes (weighted complex sum), manders/rwc (sums), radial_distribution (histograms),
sizeshape (moments).

| op | none (numpy) | numba | jax |
|---|---|---|---|
| sum | `scipy.ndimage.sum(v,lbl,idx)` or `bincount(lbl,weights=v)` | scatter-add in `prange` | `segment_sum` |
| mean | sum/count | sum/count | sum/count |
| min/max | `ndimage.minimum/maximum` | scatter-reduce | `segment_min/max` |
| quantile/MAD | `lexsort((v,lbl))` + `cumsum(area)-area` gather | per-label buffer + sort/partition | `argsort`+`cumsum` segment gather |

This **collapses numba's 4 near-identical Zernike kernels into 1** and gives jax a free port (segment_sum).
The `speed` repo's intensity rewrite is essentially the numpy reference impl of this — lift it wholesale.

### 3b. `boundary_ijv(masks) -> (bi, bj, blabel)`
Labeled inner-boundary pixels as coordinate triples. numba repo has the kernel (two-pass count-then-fill);
the numpy fallback (boolean-roll-OR + nonzero) is **already written twice** in the numba repo and once in
jax. Used by feret + radial_zernikes (+ intensity edge stats conceptually). Extract once.

### 3c. `label_to_idx` dense compaction
`bincount(masks.ravel())` → `flatnonzero` → dense `label→0..N-1` lookup. Every backend needs it; it's the
contract feeding segment_reduce. Trivial, but currently re-inlined per function.

### 3d. `convex_hull(coords)` once per object
Integer-exact monotone chain (rust `geometry.rs:29-67`). CORRECTION: neither rust nor baseline computes
the hull once — rust **recomputes it 3×** (feret + both Zernikes), baseline recomputes via centrosome per
feature. So a shared `object_hull` cache (compute once, feed feret + both Zernikes) is the redundancy-cut
*we'd introduce*. (numpy/centrosome is fine for `none`; this is an organisational win, not a backend split.)

### 3e. Host helpers (backend-agnostic, stay numpy/scipy)
`find_objects`→slices, `distance_transform_edt`, `regionprops_table`, `marching_cubes`, morphology
footprints (disk/ball), centrosome (MEC, zernike LUT, feret), mahotas haralick. **These are shared by
ALL backends including jax.** They define the irreducible scipy/skimage/centrosome/mahotas dependency floor.

---

## 4. numba modularisation plan (the jax-only-install requirement)

**Status:** mostly already done. Concrete steps:

1. **Move numba to an optional extra** (`[project.optional-dependencies] numba = ["numba>=..."]`). The
   jax repo already does exactly this; the fast repo still has it as a hard dep (only change needed there).
2. **Replace all import guards with one capability check (NO try/except, §7.4).** A single resolver:
   `HAS_NUMBA = importlib.util.find_spec("numba") is not None`, run once at import, exposing one flag
   (drop the three names `_HAS_NUMBA`/`_HAS_SIZESHAPE_NUMBA`/`_HAVE_NUMBA_INTENSITY`). Dispatch reads the
   flag; the chosen path is called directly and unguarded. If a backend is flagged present but its kernel
   raises, that's a real bug and must surface — not be caught.
3. **Every numba-accelerated function already has a numpy fallback** — verify and *consolidate the
   duplicates* into the §3 primitive layer rather than per-function inline `else:` branches:
   - intensity: fallback = the full reference algorithm still inlined (lift to segment_reduce).
   - sizeshape: jax repo's numpy branch (`_sizeshape_2d` perim/crofton/fill/convex) is complete — make it
     the shared `none` impl; numba becomes the accelerator of the same primitives.
   - zernike/radial_zernikes: explicit numpy `else:` (argsort+searchsorted+reduceat) → replace with
     `segment_reduce`.
   - feret: inline numpy `else:` (boolean-OR boundary) → `boundary_ijv` none-impl.
   - granularity: skimage `else:` branches already there.
4. **Where jax currently calls numba** (the user's specific concern): only `cp_measure_jax/core/
   _numba_sizeshape.py`, imported solely by `measureobjectsizeshape.py` behind `_HAS_NUMBA`. A jax-only
   install **already** falls to the numpy branch — nothing breaks today. After step 3 it falls to the
   shared `none` primitive instead of a repo-local duplicate.

**Net:** `none` and `jax` installs never touch numba; `numba`/`numba+jax` installs get the accelerated
primitives. Single code path per primitive, backend chosen by the resolver.

---

## 5. Hazards & correctness traps (fix during refactor)

1. **Silent `try/except Exception: pass` fallback in intensity** (both fast `measureobjectintensity.py:213-216`
   and jax `:216-219`). Hid the float64-index GPU bug for an entire benchmark round. **Fix — DECIDED (§7.4):
   remove ALL try/except.** Backend is chosen up front by flag; the path runs unguarded and raises on error.
   This deletes the bug class rather than mitigating it (no log/debug-flag half-measure needed).
2. **`max_position` tie-break per-object loop.** scipy's labeled `maximum_position` tie-break is a
   non-reproducible quirk; numba/jax use `>=`-last, the `speed` repo keeps a per-object loop to match.
   This is the one feature that resists vectorisation in every backend. **DECIDED: `>=`-last everywhere**
   (§7.2) — bit-exact on real data, synthetic-only diff documented. Backends agree by construction.
3. **radial_zernikes ~5-6% off (rust).** Likely `atan2(real,imag)` arg order + all-pixels-vs-in-disk
   normalization denominator. **Action:** pin down the *baseline's* exact phase convention and
   normalization, encode it in the shared Zernike primitive, and assert all backends match it. (Rust
   itself is not shipping, but the bug reveals an ambiguity the numba/jax ports could also hit.)
4. **Dead jax kernels:** `_perimeter_2d` and `_radial_histograms` are defined but never called (numpy
   bincount runs instead). Either wire them up or delete — they imply a GPU path that isn't active.
5. **dtype/x64 fragility:** jax intensity relies on `int64` indices (global x64 flag). If x64 is ever off,
   indexing throws → silent fallback. Make the int dtype explicit and assert, don't rely on the global.

**Trap to NOT chase:** the benchmark's "feret 0.04× / radial_zernikes 0.33× regression" in the `speed`
repo is a **measurement artifact** — the feret/radial_zernikes files differ from baseline by typing/cosmetics
only (functionally identical). Don't hunt for a regression that isn't in the code. CORRECTION (2026-05-30,
`primitive_existence_matrix.md`): the earlier "byte-identical except intensity" framing was wrong — ALL 6
speed core files differ, and `measuregranularity.py` is a REAL functional change (subsampling re-enabled),
so speed's granularity number is not apples-to-apples with baseline.

---

## 6. What each repo uniquely contributes (the cross-pollination summary)

- **baseline** → the contract (dict keys, 1-D-per-object alignment, 2D/3D rules). The `none` backend.
- **speed (np-vec)** → proves the **numpy `segment_reduce`** via the intensity rewrite (drop the dense
  `(N,Y,X)` allocation). Its intensity rewrite is the template for the `none` intensity backend.
  CORRECTION (2026-05-30): it has TWO real changes, not one — granularity also re-enables subsampling
  (so don't reuse its granularity timing); the other 4 core files are typing-only.
- **fast (numba)** → modular `@njit` kernels (single-pass moments, segment-sort quartiles, Van Herk disk,
  Vincent/Robinson reconstruction, two-pass boundary_ijv) + the already-working optional-import structure.
  All single-threaded (no prange) — which is the design we keep (§7.3); parallelism lives in the batch layer.
- **jax** → on-device segment ops / lexsort / reduce_window / while_loop, the **sparse-GLCM Haralick**
  (better than baseline, pure-numpy, liftable to `none`), bincount-formulated moments, on-device rank.
  Most "jax" CPU paths are actually numpy — informative for the `none` backend.
- **rust (ideas only)** → the **scatter-once + running-sums-in-one-reduce-loop + size-to-data** discipline:
  four-cross-sums centroid (intensity 827×, two-phase not single-pass), LUT+Horner Zernike (genuinely fused),
  5-moment Pearson, integer-exact monotone-chain hull (recomputed 3× in rust — share it in our port; rust
  feret is O(n²) brute force, not rotating calipers). Counter-example: dense-256 GLCM (texture 0.21×) = what
  not to do. Also: rust has **no internal
  threads** — its batch speedup is caller-side GIL-release. Same model we adopt (§7.3): kernels stay serial,
  parallelism lives in the batch layer over images, not inside the kernel.

---

## 7. Resolved decisions (2026-05-30)

1. **Dependency floor — KEEP.** scipy + scikit-image + centrosome + mahotas stay mandatory base deps.
   Reimplementing minimum_enclosing_circle / haralick / marching_cubes is out of scope. *Caveat:* core
   elements MAY be rewritten later **if they prove to be bottlenecks** — but the dependencies themselves
   are fine to require.
2. **Canonical tie-break — `>=`-last EVERYWHERE.** Compatibility is required (must match CellProfiler/
   baseline on real data). All backends use `>=`-last for `max_position`; document the synthetic-only diff.
   No per-backend tie-break divergence.
3. **numba kernels stay SINGLE-THREADED (no `prange`/`nogil`).** Parallelism comes from the **batch
   layer**, not intra-image. This matches the `fast` repo's existing all-serial `@njit(cache=True)` design
   — no change needed there. The batch layer (deferred rewrite) owns all parallelism: across images on
   CPU (process/thread pool), batched kernels on GPU.
4. **NO `try/except` anywhere — clean early capability checks + flags.** Backend availability is detected
   once at import via `importlib.util.find_spec` (no attempt-import-and-catch), plus `jax.default_backend()`
   for GPU. Flags drive dispatch; the resolved path runs **directly, unguarded** and raises on error. This
   removes every existing import-guard and the silent intensity fallback (§5.1). Rule: a backend is present
   (flag → used) or absent (flag → never attempted); there is no error-driven fallback. Errors surface.

### Still open (lower priority)
- **Resolver granularity:** route per-function via a static best-backend table (from the corrected
  benchmark, §2) — simplest, recommended. vs measure-once-and-cache at import. Defaulting to static unless
  you say otherwise.

---

## 8. Next step (deferred per your note)
Append the **batching rewrite** — unified `featurize(masks, pixels, batch_size=...)` where single image =
batch of 1, built on the §3 primitive layer (segment_reduce already generalises to a per-image-keyed
segment dimension). That's the lever for the GPU's idle ~46% and is tracked separately.
