# Numba texture lane — implementation plan

Status: PLANNED 2026-06-04. Branch `feat/numba-texture`, **base #59**
(`feat/bzyx-shape`), sibling to #56/#57/#58/#60 — independent of the other lanes.
Mirrors `core/measuretexture.py::get_texture`. Step-0 profile: `tasks/profile_texture.py`.

## Why this lane (step-0 verdict)

`mahotas.features.haralick` is **~99%** of `get_texture` (1224/1232 ms, 144 obj) and
is fully reimplementable; the GLCM co-occurrence build is **bit-exact** vs
`mahotas.features.texture.cooccurence` (integer histogram, algorithm-independent).
Ideal reducible profile; texture is the slowest feature → biggest absolute win.
Keep `skimage.util.img_as_ubyte` + `skimage.measure.regionprops` host-side (~1%).

**No Issue-#22 analogue:** texture is already per-object (regionprops crops with
background 0 + `ignore_zeros`), so the numba result matches the numpy baseline
**directly** — the golden compares them feature-by-feature (no isolated-object trick).

## Scope (decided)

- **2D + 3D in one PR.** Unified kernel over `(dz, dy, dx)` offsets: 2D = 4
  directions (dz=0), 3D = 13 directions. `n_directions = 13 if pixels.ndim > 2 else 4`.
- Defaults from the reference call `haralick(crop, distance=scale, ignore_zeros=True)`:
  `symmetric=True`, `preserve_haralick_bug=False`, `compute_14th_feature=False`,
  `use_x_minus_y_variance=False`, `return_mean=False`. Only the 13 features.

## Reference internals (researched — the exact spec to reproduce)

### Host prep (KEEP, scipy/skimage, ~1%)
`pixels = img_as_ubyte(pixels, force_copy=True)`; `pixels[~masks.bool] = 0`; if
`gray_levels != 256`: `rescale_intensity(...)→uint8`; `props =
regionprops(masks, pixels)`; per object `crop = prop.intensity_image` (uint8, bg 0).

### Direction deltas (× `distance`)
- 2D `_2d_deltas`: `[(0,1),(1,1),(1,0),(1,-1)]`
- 3D `_3d_deltas`: `[(1,0,0),(1,1,0),(0,1,0),(1,-1,0),(0,0,1),(1,0,1),(0,1,1),
  (1,1,1),(1,-1,1),(1,0,-1),(0,1,-1),(1,1,-1),(1,-1,-1)]`
- offset = `distance * delta`; the GLCM counts pairs `(crop[p], crop[p+offset])`
  for all in-bounds `p`. **`symmetric=True`** → also count the reverse, i.e. the
  matrix is `C + Cᵀ`.

### GLCM (per direction)
- size `fm1 = crop.max() + 1` (NOT a fixed 256 — `maxv = len(cmat)` feeds the `k`
  ranges and especially `px_minus_y.var()` over a length-`fm1` array). Allocate
  `(fm1, fm1)` int.
- `ignore_zeros=True` is applied in `haralick_features`, AFTER building: `cmat[0,:]=0;
  cmat[:,0]=0` (drop all pairs touching background 0).
- `T = cmat.sum()`. **If `T == 0` → mahotas raises ValueError on the first such
  direction → `haralick()` raises → cp_measure `except ValueError` sets the WHOLE
  object's 4×13 (or 13×13) block to NaN.** Replicate: if ANY direction's GLCM is
  empty, the object's entire feature block is NaN.

### The 13 Haralick features (per direction), `p = cmat/T`
Let `maxv=fm1`, `k=arange(maxv)`, `k2=k²`, `tk=arange(2maxv)`, `tk2=tk²`;
`px=p.sum(axis=0)`, `py=p.sum(axis=1)`; `ux=px·k`, `uy=py·k`,
`vx=px·k²-ux²`, `vy=py·k²-uy²`, `sx=√vx`, `sy=√vy`.
`px_plus_y[s]=Σ_{i+j=s} p[i,j]` (len 2maxv); `px_minus_y[d]=Σ_{|i-j|=d} p[i,j]` (len maxv).
`_entropy(a) = -Σ a·log2(a')` where `a'=a` with zeros replaced by 1 (so 0·log2(1)=0).

0. **AngularSecondMoment** = `Σ p²` (`pravel·pravel`)
1. **Contrast** = `k2 · px_minus_y`
2. **Correlation** = if `sx==0 or sy==0` → `1.0`; else `(Σ_{ij} i·j·p[i,j] − ux·uy)/(sx·sy)`
3. **Variance** = `vx`
4. **InverseDifferenceMoment** = `Σ p[i,j]/((i−j)²+1)`
5. **SumAverage** = `tk · px_plus_y`
6. **SumVariance** = `tk2 · px_plus_y − feats[5]²`  (bug=False branch; uses SumAverage, not SumEntropy)
7. **SumEntropy** = `_entropy(px_plus_y)`
8. **Entropy** = `_entropy(pravel)`
9. **DifferenceVariance** = `px_minus_y.var()`  (numpy population var over the length-`maxv` array; use_x_minus_y_variance=False)
10. **DifferenceEntropy** = `_entropy(px_minus_y)`
11. **InfoMeas1**: `HX=_entropy(px)`, `HY=_entropy(py)`, `crosspxpy=outer(px,py)` with
    zeros→1, `HXY1=−Σ pravel·log2(crosspxpy)`; = `(feats[8]−HXY1)/max(HX,HY)` (or
    `feats[8]−HXY1` if `max(HX,HY)==0`).
12. **InfoMeas2**: `HXY2=_entropy(crosspxpy)`; = `√(max(0, 1 − exp(−2·(HXY2 − feats[8]))))`

Feature names (`core/measuretexture.F_HARALICK`, in index order): AngularSecondMoment,
Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance,
SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, InfoMeas1, InfoMeas2.

### Output keys
`"{feature}_{scale}_{direction:02d}_{gray_levels}"` per (direction, feature), each a
length-nobjects array. e.g. `AngularSecondMoment_3_00_256`. Reuse `F_HARALICK`.

## Kernel (`core/numba/_texture.py`)

`haralick_object(crop, offsets, distance) -> (n_dir, 13) float64` —
`@njit(cache=True, error_model="numpy")`, serial.
- `crop` as `(Z, Y, X)` (2D = `(1, Y, X)`); `offsets` = int `(n_dir, 3)` array of
  `distance*delta` `(dz,dy,dx)`. One kernel for both 2D and 3D.
- Per direction: find `fm1` (max over crop +1; reuse one cmat buffer sized to the
  object's global max+1, allocated once per object), build symmetric GLCM, zero
  row/col 0, `T=sum`; if `T==0` mark the whole object NaN and return. Else compute
  the 13 features above into row `dir`.
- Helpers (all njit, on the GLCM): `_entropy`, `_compute_plus_minus` (the i+j / |i−j|
  marginals), the px/py marginals + moments. Match mahotas's op order closely
  (dot-products) to stay within rtol; serial-sum vs BLAS diff is ~1e-12.

## Wrapper (`core/numba/measuretexture.py`)

`get_texture(masks, pixels, scale=3, gray_levels=256)`:
- `to_bzyx(masks, pixels)`; per image `_texture_image`. (texture is NOT 2D-only —
  3D volumes are valid, 13 directions.)
- `_texture_image`: do the host prep (img_as_ubyte + mask-zero + optional rescale +
  regionprops) exactly like the reference; choose `offsets` (4 vs 13) by ndim; per
  object call `haralick_object`; assemble the `{feat}_{scale}_{dir:02d}_{gray}` dict.
- Single image → dict; batch → list of dicts (unwrap). Reuse `F_HARALICK`.

## Wiring (same append pattern as the stack)
`core/numba/__init__.py` `__all__` + `_numba_registries` `"core"` → `texture`.
`texture` IS in `_3D_FEATURES` (stays). Extend the dispatch test in
`test_backend_correctness.py`.

## Verification
- **Kernel**: `haralick_object` GLCM vs `mahotas.features.texture.cooccurence`
  (symmetric, per direction, distance) — bit-exact (integer); the 13 features per
  direction vs `mahotas.features.haralick(crop, distance, ignore_zeros=True)` —
  `rtol=1e-6 atol=1e-8`. Cover: a normal object, a 3D crop (13 dir), the empty-GLCM
  (tiny object at distance 3) → all-NaN, a constant crop (sx/sy==0 → Correlation 1),
  `gray_levels != 256`.
- **Backend golden**: numba `get_texture` vs numpy `get_texture` key-by-key
  (`equal_nan=True`), 2D + 3D, single + batch, default + non-default scale/gray_levels.
- Edge: object whose `intensity_image` all-zero after masking; 1–2 px objects.

## Risks / notes
- The bit-exactness work is the 13 formulas (this spec) + the edge cases
  (T==0→NaN object, sx/sy==0→1, max(HX,HY)==0, InfoMeas2 max(0,·), `_entropy`
  zeros→1, crosspxpy zeros→1, `px_minus_y.var()` length=`fm1`). All deterministic.
- numpy/BLAS vs numba serial-sum: ~1e-12, within rtol; if a feature drifts it's the
  dot-product order — match mahotas's order.
- `gray_levels` rescale + `img_as_ubyte` stay host-side (scipy/skimage exact) — do
  not reimplement.
- Benchmark after build (expect a large win; mahotas is C but per-object Python +
  4×cooccurence + haralick_features overhead dominates the 1.2 s).
