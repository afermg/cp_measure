# NumPy-path speedup investigation (default backend, no extra deps)

**Date:** 2026-06-06
**Scope:** Speed up the *default* numpy/scipy/skimage/centrosome/mahotas path in
`src/cp_measure/core/*.py` — NOT the optional `core/numba/` backends, NOT jax. What's
already done, what's in flight, and what algorithmic tricks from the numba/jax variants
are portable to pure numpy.

---

## TL;DR

1. **The numpy-speedup effort is active and just landed.** PR #55 (Alan's omnibus numpy
   speedup) was **closed 2026-06-05** and split into five single-concern PRs **#69–#73,
   all MERGED to main on 2026-06-05.** They cover **intensity, colocalization, sizeshape,
   and `masks_to_ijv`.** Our branch `fix/radial-per-object-22` is based on older main
   (`69d6c97`) and **predates all of them** — rebase before doing more numpy work.
2. **No open numpy-perf PRs remain** — every other open PR (#56–#65) is a numba backend;
   #59 is the bzyx helper; #68 is our radial fix; #38 is output formats.
3. **The agents' "ready to port" intensity/coloc/ijv items are now ALREADY DONE** in main
   via #69/#71/#72. Don't re-propose them.
4. **Remaining pure-numpy levers (untouched by #69–73), prototyped today:**
   - **Feret hull-from-boundary — 3.52×, bit-exact** (TOP: easy, high-confidence).
   - **Granularity precompute-once upsample — 1.63× on the upsample step** (~1e-13 accuracy).
   - Texture sparse-GLCM (jax port) — not prototyped, uncertain CPU value, higher effort.
   - Zernike shared geometry — small, architectural.

---

## A. What already landed (do NOT re-propose)

PR #55 "Intensity/sizeshape/colocalisation Speedups" (author **afermg**) — **CLOSED**
2026-06-05, *not merged as-is*. Maintainer's closing note: *"Closing in favor of five
smaller, single-concern PRs."* All five merged the same day, base `main`:

| PR | Merged | What | Reported win |
|----|--------|------|--------------|
| **#69** `perf(coloc)` | 2026-06-05 | consolidate per-label flat-segment reductions (bincount/`maximum.at`) | up to ~35× (manders 23×, overlap 27×) |
| **#70** `perf(sizeshape)` | 2026-06-05 | drop per-object `scipy.ndimage` ops → direct numpy on boolean sub-arrays | ~3× feret |
| **#71** `perf(utils)` | 2026-06-05 | rewrite `masks_to_ijv` as one `nonzero` + stable argsort (was O(L·pixels)) | feeds the feret/zernike/overlap paths |
| **#72** `perf(intensity)` | 2026-06-05 | bbox-based rewrite of the numpy intensity backend (drops O(N·H·W) `label_matrices`) | ~74× (2D, 200 obj) / ~10.7× (3D) |
| **#73** `feat(intensity)` | 2026-06-05 | `legacy` flag for percentile/MAD convention (stacked on #72) | (correctness toggle, ties to our legacy theme) |

Commits on the closed `origin/speedups` branch (`025b7fa` intensity, `b1dbc80`
sizeshape/ijv, `7e53777` coloc) are the same work; superseded by the merged #69–73.

> **Correction to sub-agent reports:** one agent claimed these "landed in main" and another
> described the baseline as still per-label scipy loops. Both read inconsistent states. Ground
> truth: the speedups merged to **main on 2026-06-05** but are **NOT in our branch HEAD**
> (we branched off `69d6c97`, before the merges). The `cp_measure_speed` sibling variant only
> ever rewrote intensity (other core files byte-identical to baseline) and is now superseded.

## B. Lanes UNtouched by #69–73 (the real remaining numpy work)

Verified on current `origin/main`:
- **granularity** (`measuregranularity.py`) — still 10× `scipy.ndimage.map_coordinates`;
  the granular-spectrum loop upsamples every iteration. Granularity is the single largest
  baseline cost on the `large` tier (~41%). **Untouched.**
- **texture** (`measuretexture.py:215`) — still `for prop in regionprops: mahotas.haralick`
  per object. **Untouched.**
- **feret** (`measureobjectsizeshape.py:1036`) — `convex_hull_ijv(masks_to_ijv(masks), …)`
  feeds the **full** object ijv to the hull. #71 sped `masks_to_ijv` but **not the hull
  input size.** **Untouched.**
- **zernike / radial_zernikes** — recompute `minimum_enclosing_circle` + pixel passes
  independently when both run. **Untouched.**

---

## C. Prototyped levers (pure numpy, evidence today)

Script: `tasks/proto_numpy_levers.py` (run on the repo `.venv`). Both target untouched lanes.

### C1. Feret: hull from boundary pixels — **3.52×, BIT-EXACT**  ⭐ recommended first

`hull(object) == hull(boundary)` (interior pixels are never hull vertices). Extract the
boundary in pure numpy (`fg ^ binary_erosion(fg)`), feed only those ijv rows to
`centrosome.convex_hull_ijv`. Measured (1080², 144 obj):

```
full ijv points:   646416   boundary ijv points: 38016   reduction 17.0x
feret_diameter identical (full vs boundary): True
convex_hull+feret  full: 63.25 ms   boundary: 4.61 ms   (+13.35 ms erosion)   net 3.52x
```

- **Bit-identical** min/max feret → no accuracy/contract risk.
- Pure numpy + scipy (`binary_erosion`), no new deps, no numba.
- Erosion cost (13ms) is the only overhead and is dwarfed by the hull saving.
- Stacks on top of #70/#71 (independent of the `masks_to_ijv` rewrite).
- Refinement: extract boundary ijv directly from the erosion mask instead of a 2nd
  `masks_to_ijv`, and 8-conn boundary matches centrosome's hull staircase (the numba
  feret lane #65 already proved the boundary-only equivalence with 8-conn + edge pixels).

### C2. Granularity: precompute upsample grid once — **1.63× upsample, ~1e-13 accuracy**

The granular-spectrum loop calls `map_coordinates(rec, (i,j), order=1)` with a **fixed**
`(i,j)` grid every iteration; map_coordinates recomputes floor/frac/weights each call and
upsamples 270²→1080² sixteen times. Precompute (floor, frac, clamped neighbours) **once**,
then each iteration is a fancy-indexed separable weighted sum. Measured (orig 1080², sub
0.25, ng 16):

```
max|map_coordinates - numpy_gather| = 9.55e-14
map_coordinates x16: 641.62 ms   numpy_gather x16: 394.77 ms   speedup 1.63x
```

- `9.55e-14` ≪ granularity's existing `rtol=1e-6` (the lane was never bit-exact), so it
  respects the accuracy contract — same as the gated numba gather (PR #56 shipped a 3e-14
  gather under the same reasoning).
- Pure numpy. (The existing `tasks/poc_granularity_gather.py` got 2.88× but used a **numba**
  kernel; this is the dependency-free version.)
- End-to-end granularity win ≈ 1.3–1.4× (upsample is the dominant chunk at sub=0.25; erosion
  + skimage `reconstruction` run on the small 270² image).
- **Refinement worth trying:** do the gather as two successive 1-D interpolations (rows then
  cols) to cut the four 1080² temporaries → likely closes more of the gap to the numba 2.88×.

### Not prototyped (flagged for follow-up)

- **Texture sparse-GLCM** (lifted from `cp_measure_jax/.../measuretexture.py`): build the
  co-occurrence matrix sparsely across all objects in one pass instead of per-object dense
  `mahotas.haralick`. The numba texture lane (#64) proved the symmetric-GLCM build is
  bit-exact vs `mahotas.cooccurence`, so a numpy sparse build can be made faithful. BUT the
  jax sub-agent estimated only **~1.1–1.35× on CPU** (the win is mostly GPU vectorization),
  and it's the highest-effort port (RLE decode + 13 Haralick formulas + degenerate-GLCM/
  ignore_zeros edge cases). **Prototype before committing** — uncertain it beats mahotas's C
  per-object path on CPU.
- **Zernike shared geometry**: when `get_zernike` + `get_radial_zernikes` both run, compute
  `minimum_enclosing_circle` + the pixel pass once. Small, needs a shared call site.

---

## D. Recommended order

1. **Feret hull-from-boundary** (#C1) — bit-exact, 3.5×, ~15 LOC, one file. Highest
   confidence, zero contract risk. Mirrors numba #65's host-side trick in pure numpy.
2. **Granularity precompute-once gather** (#C2) — 1.6× upsample / ~1.35× end-to-end,
   within-tol, one file. Try the separable 1-D refinement for more.
3. *(investigate)* **Texture sparse-GLCM** — prototype CPU speed vs mahotas first; only
   pursue if it clears ~1.3×.
4. *(small)* **Zernike shared geometry** when both zernike fns run.

**Before any of the above: rebase `fix/radial-per-object-22` (and future numpy work) onto
current main** so you build on #69–73 rather than re-deriving them. Each lever above is one
file, independent, and PR-able single-concern (matching Alan's #69–73 style).

## E. Verification recipe

- Env: repo `.venv` (`/ictstr01/.../cp_measure/.venv/bin/python`); has cp_measure editable +
  centrosome/scipy/skimage/mahotas. (`uv` is not on PATH; use the venv directly.)
- Re-run prototypes: `.venv/bin/python tasks/proto_numpy_levers.py`.
- For a real implementation, golden-test against current main's `get_feret` / `get_granularity`
  on the 3-tier dataset (tiny/small/large) — feret must be **bit-identical**, granularity
  within `rtol=1e-6`.
