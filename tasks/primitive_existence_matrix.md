# cp_measure — primitive existence matrix (which primitive exists in which language)

Deep source-mining of all 5 sibling repos (2026-05-30), one agent per repo, to answer:
for each shared §3 primitive, **does a native implementation exist in `none`(numpy) / numba / jax,
and in what form?** Companion to `backend_cross_pollination.md` (the architecture) — this doc is the
empirical "what already exists where" with file:line anchors.

Repos:
- baseline `cp_measure` (numpy/scipy/skimage = the `none` backend) — `src/cp_measure/core/`
- `cp_measure_fast` (numba) — `src/cp_measure/core/_*_numba.py`
- `cp_measure_jax` (jax) — `src/cp_measure_jax/core/`
- `cp_measure_speed` (np-vec) — `cp_measure/core/`
- `cp_measure_rust` (PyO3/Rust, **ideas only, no rust ships**) — `src/*.rs`

---

## 0. Headline — the matrix is bimodal

Only **two** primitives genuinely exist (and need to exist) in three languages: `segment_reduce`
and `segment_quantile`. Everything else is **numpy-only in every backend including jax** — those are
shared host helpers that just need *deduplication into one impl*, not a per-language port.

- jax contributes **zero** live primitive kernels *below* segment_reduce (`_perimeter_2d` and
  `_radial_histograms` are both dead code).
- numba contributes exactly **one** extra primitive kernel below segment_reduce: `_boundary_ijv`.

⇒ The backend-divergent surface = **2 dispatched primitives + ~4 shared numpy helpers + 1 optional
numba boundary kernel.** PR2.5 (the primitive layer) is far smaller than "port everything to 3 langs".

Legend: ✓ = real native impl · host = runs as numpy on host even in that backend · dead = defined,
never called · idea = rust mining-only · — = doesn't exist

| Primitive | none / numpy | numba | jax | rust (idea) | needs N langs? |
|---|---|---|---|---|---|
| **A. segment_reduce** (sum/mean/min/max/std, centroid moments) | ✓ `scipy.ndimage.*(index=)` + `bincount`/`add.at` | ✓ `_moments` single-pass + scatter | ✓ `jax.ops.segment_*` + `.at[].add` (device) | idea: 2-phase scatter→reduce, 5 cross-sums in reduce loop | **YES — 3** |
| **B. segment_quantile / MAD** | ✓ `lexsort`+`cumsum(area)` gather+interp | ✓ CSR offsets + **per-segment sort** + `_interp` | ✓ 2× `lexsort` + int64 `starts/last` + `_q` (device) | idea: per-obj buffer + full sort ×2 | **YES — 3** (3 different sort strategies) |
| **C. Zernike complex segment_sum** (specialization of A) | ✓ `argsort`+`searchsorted`+`reduceat` (dup ×2) | ✓ **4 kernels, 2 live, 1 dead** | ✓ `.at[keys].add` (device) | idea: LUT+Horner-on-r²+iterative z^m, one kernel shape+radial | folds into A |
| **D. boundary_ijv** | host: `find_boundaries`/boolean-roll-OR | ✓ `_boundary_ijv` 2-pass count-then-fill | **dead** (`_perimeter_2d`) | — | numpy + optional numba |
| **E. label_to_idx** | ✓ `bincount`+`flatnonzero` / `unique[>0]` (dup ~8 sites) | host (numpy) | host (numpy) | (implicit) | **numpy-only everywhere** |
| **F. convex_hull / MEC** | ✓ `centrosome.convex_hull_ijv`, skimage `area_convex` | host | host | idea: int-exact monotone chain | **numpy-only everywhere** |
| **G. host helpers** (EDT, regionprops, marching_cubes, mahotas, centrosome LUT) | ✓ scipy/skimage/centrosome/mahotas | host | host | — | **numpy-only everywhere** |
| *bonus:* sparse-GLCM Haralick | ✓ `_haralick_sparse` (**pure numpy, lives in jax repo**) | — | device GLCM build only | idea: NOT dense-256 | numpy `none` + optional jax build |

---

## A. segment_reduce (per-label sum/mean/min/max/std + centroid moments)

The one true multi-language primitive — three genuinely different native forms.

- **none / numpy (baseline):** `scipy.ndimage.{sum,mean,minimum,maximum,maximum_position,sum_labels}`
  with `index=lindexes`, wrapped in `fix` (=`utils._ensure_np_array`). std = `sqrt(ndimage.mean((v -
  mean[llabels-1])**2, ...))`. Scatter target index = `label-1`.
  - `measureobjectintensity.py:196-243` (count/sum/mean/std/min/max/argmax/centroids),
    `:313-337` (edge-pixel reductions).
  - coloc flattened-pixel sums `measurecolocalization.py:171-187, 224-228, 533-569`.
  - granularity per-label means `measuregranularity.py:249, 299`.
  - radial-dist sparse-COO (label × radial-bin) scatter-add `measureobjectintensitydistribution.py:220-238`.
- **numba (fast):** `_moments(masks,pixels,label_to_idx,nobjects)` `_intensity_numba.py:19-53` —
  `@njit(cache=True)`, serial, single raster pass into 9 per-label arrays: count(i64), sumI, minI(+inf),
  maxI(-inf), argflat(i64,-1), and four cross-sums sumx, sumy, sumxI(=Σc·v), sumyI(=Σr·v). max uses `>=`
  (keep LAST → matches scipy tie-break). std = second pass `_resid_sumsq` `:56-72`. numpy fallbacks via
  `numpy.bincount`/`numpy.add.at` in coloc `:701-705,778-781,840-843`, granularity `:282-285`,
  radial-dist `:288-296`.
- **jax:** `_intensity_core_jax` `measureobjectintensity.py:51-106`, `@jax.jit`. `jax.ops.segment_sum`
  (count/integrated/std-resid/centroids), `segment_min`/`segment_max` (incl. int64 arange for argmax).
  Scatter variant `.at[idx].add/.max` in coloc `:56-66,221-255`, zernike `measureobjectsizeshape.py:673-674`,
  texture GLCM build. **bincount-formulated moments are pure numpy on host** (`measureobjectsizeshape.py:288-320`).
- **rust idea:** `intensity.rs` — **two-phase scatter-then-reduce** (NOT single fused pass). Phase 1
  scatters pixels to `per_label: Vec<Vec<PixelInfo>>`; Phase 2 (`:150-163`) accumulates 5 running sums
  in one loop → centroid, intensity-centroid, mass-displacement closed-form. Buffers sized to actual
  pixel count (push), not bbox/theoretical max. Std is a separate 2nd pass.

**Verdict:** dispatch this. numpy `segment_reduce` = lift speed's intensity rewrite; numba = `_moments`;
jax = `jax.ops.segment_*`. Transferable rust discipline: scatter once, accumulate cross-sums in one
reduce loop, size buffers to data.

---

## B. segment_quantile / MAD (per-label median, quartiles, MAD)

Genuinely 3-language, and the three **disagree on sort strategy** — important.

- **none / numpy (baseline & speed — the cleanest reference):** ONE global `lexsort((v,lbl))` +
  `indices = cumsum(areas) - areas` (exclusive prefix = per-label start offset) + interpolated-rank
  gather; **no per-object loop**. speed `measureobjectintensity.py:230-264`; baseline `:263-303`.
  MAD reuses the same machinery on `|v - median[llabels-1]|`. ⚠ MAD fraction is `areas/pixels.ndim`
  (=0.5 in 2D, but 1/3 in 3D — a baseline quirk carried verbatim).
- **numba (fast):** `_quantiles` `_intensity_numba.py:89-133` — **avoids global sort**. Builds CSR
  segment offsets (prefix-sum of count), scatters pixels into one flat buffer via a cursor, then per
  label slices `buf[s:s+n]`, `seg.sort()` in place, and `_interp(seg,n,frac)` `:75-86`. MAD = abs-dev
  buffer + sort + `_interp(.,.,1/ndim)`.
- **jax:** inside `_intensity_core_jax` `measureobjectintensity.py:79-102`. **int64** `starts =
  cumsum(cnt_i) - cnt_i`, `last = starts + cnt_i - 1` (THE float64-index-bug fix — `:80-82`). `_q(svals,
  frac)` `:84-90` lerp gather with int64 clipped indices. Quartiles via `lexsort((vf,seg))` `:92`; MAD via
  a 2nd `lexsort((madv,seg))` `:100`. Runs on device. numpy fallback (argsort+cumsum) `:285-316`.
- **rust idea:** per-object buffer + `sort_unstable` ×2 (values, then abs-devs); `interpolated_percentile`
  `intensity.rs:40-53` uses `index = n·q` (NOT `(n-1)·q`).

**Pin in the shared primitive:** the CellProfiler `index = n·q` linear-interp convention (all three use
it) and the 3D MAD-fraction quirk. Three sort strategies are all valid; pick per backend (global lexsort
for numpy/jax, per-segment sort for numba).

---

## C. Zernike complex segment_sum (a weighted specialization of A)

- **numpy:** `argsort`+`searchsorted`+`numpy.add.reduceat`, **duplicated** in
  `measureobjectintensitydistribution.py:388-406` and `measureobjectsizeshape.py:1065-1078`.
- **numba: 4 redundant kernels confirmed**, all in `_intensity_distribution_numba.py`, same
  segment-reduce form, differ only by input layout = {weighted, unweighted} × {split re/im, complex}:
  `_per_label_zernike_sum` `:5-23`, `_per_label_zernike_sum_complex` `:26-45` (**live**, called
  `measureobjectintensitydistribution.py:382`), `_per_label_complex_sum` `:48-64` (**live**, called
  `measureobjectsizeshape.py:1059`), `_per_label_complex_sum_complex` `:67-84` (**dead**). #1 has no
  live caller either. → collapse to one `segment_reduce(weight)` + complex/split adapter.
- **jax:** `.at[keys.ravel()].add(...)` device segment_sum, `measureobjectsizeshape.py:671-674`.
- **rust idea (the best basis formulation):** `zernike.rs` — radial LUT (`build_radial_lut:37-60`,
  coeffs ordered for Horner on r²), iterative complex powers `z^m = z^{m-1}·z` (`compute_z_powers:68-76`,
  no per-pixel cos/sin/atan2), fused accumulate (`accumulate_zernike:81-101`) with `weight=1.0` for
  shape / `weight=pixel` for radial → **one kernel, both features**.

⚠ **Two correctness hazards (the ~5-6% radial-zernike drift), confirmed in rust:**
(1) `atan2(re, im)` arg order **swapped** vs conventional `atan2(im, re)` — `zernike.rs:235` (axis
convention `z = y + i·x` at `:73` interacts; verify both together). (2) magnitude normalized by
**all-pixel count** (`coords.len()`, `:206`) while accumulation **skips r²>1 pixels** (`:218`) — basis
mismatch. Shape Zernike divides by disk area `π r²` instead. **Action:** pin the baseline's exact phase
arg order + normalization denominator (all-pixels vs in-disk) and assert all backends match.

---

## D. boundary_ijv (labeled boundary pixels as (i,j,label))

- **none / numpy:** `skimage.segmentation.find_boundaries(mode="inner")` → boolean mask
  (baseline `measureobjectintensity.py:174`); boolean-roll-OR + `nonzero` fallback in fast
  `measureobjectsizeshape.py:1102-1115`; jax host `_utils.py:25-36`. Generic full-object ijv (not
  boundary) = `utils.masks_to_ijv` (loop + `np.where`).
- **numba:** `_boundary_ijv(masks)` `_sizeshape_numba.py:5-48` — `@njit(cache=True)`, serial, classic
  **two-pass count-then-fill**; preserves input int dtype in `bl`. The only non-numpy boundary impl.
- **jax:** `_perimeter_2d` `measureobjectsizeshape.py:130-143` — **DEAD** (zero callers; its only dep
  `_seg` is also dead).

**Verdict:** numpy host + optional numba kernel. No jax version. Dedup the boolean-roll copies into the
numpy `none` impl.

---

## E. label_to_idx (compact arbitrary labels → contiguous 0..N-1)

**numpy-only in every backend**, re-inlined ~8× with 3 idioms:
- baseline: `numpy.unique(masks)[>0]` + implicit `label-1` arithmetic everywhere; centrosome path
  renumbers to 1..N (`measureobjectsizeshape.py:1017-1019`).
- fast: `bincount(masks.ravel())`+`flatnonzero` + LUT scatter `measureobjectintensity.py:136-143`;
  variants with `flatnonzero[1:]`, reverse-index+`-1` sentinel `measureobjectsizeshape.py:1040-1041`;
  coloc remap LUT ×4 `:626-641,671-673,719-721,796-798`.
- jax: numpy LUT scatter into `(max+1,)`, int64 in intensity `measureobjectintensity.py:154-159`, int32
  elsewhere (`measureobjectsizeshape.py:262-272`, texture `:499-505`, etc).

**Verdict:** ONE shared numpy helper. Not a per-language primitive — it's pure duplication.

---

## F. convex_hull / MEC

**numpy/centrosome-only everywhere.** skimage `area_convex`/`solidity` via `regionprops_table`;
explicit `centrosome.cpmorphology.convex_hull_ijv` only for Feret (`measureobjectsizeshape.py:1038`).
Zernike uses `minimum_enclosing_circle` (`measureobjectintensitydistribution.py:319`), not a hull.
- ⚠ rust **recomputes its hull 3×** (feret `ferret.rs:17`, shape-zernike `zernike.rs:136`, radial
  `zernike.rs:204`) — "compute once and share across feret+both Zernikes" is an *opportunity we'd
  create*, not something that already exists. Rust feret is **O(n²) brute force** (`ferret.rs:47-88`),
  NOT rotating calipers. Hull algo = int-exact Andrew monotone chain (`geometry.rs:29-67`).

**Verdict:** shared numpy/centrosome. Organisational win = compute hull once per object, feed
feret+both Zernikes. No numba/jax version exists or is needed now.

---

## G. host helpers — numpy/scipy/skimage/centrosome/mahotas in ALL backends

`find_objects`, `distance_transform_edt`, `regionprops_table`, `marching_cubes`, morphology footprints,
centrosome (MEC, zernike LUT, feret geometry), mahotas haralick. These are the irreducible dependency
floor (locked: KEEP mandatory, `backend_cross_pollination.md §7.1`). jax calls all of these on host too.

**bonus — sparse-GLCM Haralick:** `_haralick_sparse` `cp_measure_jax/.../measuretexture.py:129-240` is
**pure numpy** (lives in the jax repo but uses only `numpy.bincount`) and beats baseline → it becomes the
`none` texture backend. The device GLCM build (`.at[codes].add(1)`, `:435`) is the only real jax texture
path. Do NOT copy rust's dense-256 GLCM (`texture.rs`, swept ~6× per object = the 0.21× slowdown).

---

## Corrections to previously-recorded findings (mined, not assumed)

1. **`speed` variant did NOT change only intensity.** All 6 core files differ. **`measuregranularity.py`
   has a real functional change** — re-enables `subsample_size<1` resampling (`map_coordinates`) that
   baseline had commented out. intensity = the real rewrite. The other 4 (coloc, sizeshape, texture,
   intensitydistribution) are typing/cosmetic only. ⚠ **benchmark-validity:** speed's granularity is NOT
   apples-to-apples with baseline — revisit the speed granularity number. (Supersedes the lessons.md
   "speed changed only intensity, md5-verified byte-identical" claim — those 4 files differ by typing,
   functionally identical, but granularity is functionally different.)
2. **Rust intensity is two-phase scatter→reduce, not a single fused pass**; cross-sums live in the
   reduce loop (`intensity.rs:150-163`); std is a separate 2nd pass.
3. **Rust hull computed 3× per object, not once-and-shared**; rust feret is O(n²) brute force, not
   rotating calipers.

---

## Design conclusion → PR2.5 scope

Build per-language implementations of exactly **two** primitives — `segment_reduce` and
`segment_quantile` (Zernike-complex-sum = weighted variant of the first). Everything else
(`label_to_idx`, boundary host-extraction, `convex_hull`, EDT/regionprops/mahotas/centrosome) is
numpy-shared across all backends and needs only **deduplication into one helper each**, plus the single
optional numba `_boundary_ijv` kernel and the numpy sparse-GLCM lifted from the jax repo. jax adds
nothing below segment_reduce; numba adds exactly one thing.

Conventions to encode in the shared primitives before porting:
- `>=`-last tie-break for max_position everywhere (locked, §7.2).
- `index = n·q` linear-interp quantile + the 3D MAD-fraction quirk.
- baseline's exact Zernike `atan2` arg order + normalization denominator (all-pixels vs in-disk).
