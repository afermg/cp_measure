# Phase 1 — sizeshape moments kernel (numpy + numba)

## Goal
Replace skimage `regionprops_table`'s **moment machinery** (the per-region `einsum`-based
`moments`/`moments_central`/`moments_normalized`/`moments_hu`/`inertia_tensor`/
`inertia_tensor_eigvals` + the moment-derived geometry) with a single label-scatter pass.

**Measured target (large tile, 1080²/142 obj):** the moment machinery is ~120 ms of the
352 ms `get_sizeshape` (≈40 ms of it is `einsum_path` *recomputed per region*). cProfile
breakdown + prototypes in `tasks/cprofile_ss2.py`, `tasks/proto_moments.py`,
`tasks/proto_central.py`.

**Phase 1 alone:** sizeshape ≈ 352 → ~250 ms (~1.4×). It is the bit-exact, low-risk
foundation; the big jump to ~3× needs Phase 2 (convex hull). It does NOT touch convex hull,
perimeter, EDT, area, bbox, euler — those stay on skimage/scipy this phase.

## Features owned by Phase 1
These are the `get_sizeshape` outputs that skimage derives from moments (so they must ALL be
taken over together — leaving e.g. `axis_major_length` in `regionprops` makes skimage recompute
the moments internally and the einsum cost returns):

- Raw spatial: `F_SPATIAL_MOMENT_p_q` (16) ← `moments`
- Central:     `F_CENTRAL_MOMENT_p_q` (16) ← `moments_central`
- Normalized:  `F_NORMALIZED_MOMENT_p_q`   ← `moments_normalized`
- Hu (7):      `F_HU_MOMENT_k`             ← `moments_hu`
- `F_INERTIA_TENSOR*` + eigvals            ← `inertia_tensor`, `inertia_tensor_eigvals`
- `F_MAJOR_AXIS_LENGTH`, `F_MINOR_AXIS_LENGTH`, `F_ECCENTRICITY`, `F_ORIENTATION`
- (keep in regionprops: `centroid`, `equivalent_diameter_area`, `area`, `area_bbox`,
  `area_convex`, `solidity`, `perimeter`, `perimeter_crofton`, `euler_number`, `extent`,
  `area_filled`, `image` — none are moment-derived / einsum-heavy)

## Shared math (identical for both lanes) → `cp_measure/primitives/_moments.py`
Inputs per object: raw spatial moments `M[p,q]` (local bbox coords, p,q≤3) **and** central
moments `mu[p,q]` (both computed by the lane-specific accumulator). Pure-numpy vectorised
derivation across all objects (cheap — NOT the bottleneck):

1. `area = M[0,0]`; local centroid `(M10/M00, M01/M00)`; global centroid `+ bbox_min`.
2. `nu[p,q] = mu[p,q] / mu[0,0]**(1+(p+q)/2)` for `p+q>=2`, else skimage's value (0/NaN) — match
   `skimage.measure.moments_normalized` exactly.
3. `hu` (7) — standard invariants from `nu`; replicate `skimage.measure.moments_hu` term-for-term
   (sign of hu[6] is convention-sensitive).
4. `inertia = [[mu02/mu00, -mu11/mu00], [-mu11/mu00, mu20/mu00]]`; eigvals via 2×2 closed form,
   descending (matches `inertia_tensor_eigvals`).
5. `axis_major/minor = 4*sqrt(eigval_max/min)`; `eccentricity = sqrt(1 - lmin/lmax)`.
6. `orientation` — replicate skimage's `regionprops` formula exactly (atan2 of `-2*mu11`,
   `mu02-mu20`, with its half-angle/sign convention); cp_measure then scales by `180/pi`.

**Strategy to guarantee a match:** read each formula out of the installed skimage
(`skimage/measure/_regionprops.py` + `_moments.py`) and port it vectorised. Where cheap, cross-
check against `skimage.measure.inertia_tensor_eigvals` / `moments_hu` / `moments_normalized`
helpers in the golden test.

## Accumulator — two label-scatter passes over foreground pixels
Central moments need the centroid first, so two passes (both bit-exact; binomial-from-raw is
NOT used — it loses ~1e-4 to cancellation, see proto_moments.py):
- **Pass A:** per-object `M[p,q]` in local coords `(row-rmin, col-cmin)`; also `bbox_min`.
- **Pass B:** per-object `mu[p,q]` in centered coords `(row-cr_global, col-cc_global)`.

### numpy lane (default backend)
- File: `src/cp_measure/core/measureobjectsizeshape.py`.
- Strip the 10 moment-derived names from `desired_properties` (2D path; keep the rest).
- Accumulator = `numpy.bincount`-scatter of the 16+16 moment terms (vectorised, as in
  `proto_moments.py`/`proto_central.py`): `ijv = nonzero(mask)`, `obj = searchsorted(ul, lab)`,
  `bbox_min` via `numpy.minimum.at`.
- Call `_moments.derive(...)`, splice results into the existing result dict.
- 3D path: same (skimage 3D moment props are also einsum-heavy); reuse the shared derivation with
  the 3D moment index set, OR scope Phase 1 to 2D and leave 3D on regionprops (decide by
  measuring 3D cost; 2D-first is the safe default).

### numba lane (accelerator)
- New `src/cp_measure/core/numba/_sizeshape.py`: `@njit` kernel `moments_object(masks_2d)` doing
  Pass A + Pass B in one compiled function (loop foreground pixels twice), returning `(M, mu,
  bbox_min, counts)` per label. No per-region Python, no einsum.
- New wrapper `get_sizeshape` (in `core/numba/measureobjectsizeshape.py` or `_sizeshape.py`):
  `to_bzyx` (2D-only; 3D → numpy baseline like the other numba lanes), call the kernel, call the
  **shared** `_moments.derive(...)`, and call the numpy backend for the NON-moment props
  (area/bbox/convex/perimeter/euler/EDT-radii) — i.e. Phase 1 numba = numba moments + numpy rest.
- Register `"sizeshape": get_sizeshape` in `_numba_registries()["core"]` + export in
  `core/numba/__init__.py`. (NOTE: integration `bulk.py` `_numba_registries` is merge-mangled —
  fix that too, or land on a clean `feat/numba-sizeshape` off `#59`.)
- Stack: `#59` (bzyx) → `feat/numba-sizeshape`.

## Exactness & tests
- Raw spatial moments: **bit-exact** (0.00, verified).
- Central / normalized / hu / inertia / axis / ecc / orientation: **~1e-13 relative** (moments
  reach ~1e8 magnitude; not machine-zero but far under any feature tolerance). Decide whether the
  numpy default needs a `legacy` toggle — likely NOT (existing tests assert shape/non-triviality
  for moments, like zernike), but confirm there's no strict numerical golden in
  `test_core_measurements`.
- Golden test (both lanes): every moment-derived feature vs `skimage.regionprops_table` within
  `rtol=1e-9` (raw moments `atol=0`); edge cases — single-pixel object (μ=0, degenerate inertia →
  orientation/eccentricity NaN parity), non-contiguous labels, edge-touching, 1-row/1-col object
  (rank-deficient inertia). numba lane additionally: numba == numpy backend; 3D → `{}`/baseline.

## Risks / open questions
- **orientation & hu sign conventions** are the fiddly part — must port skimage's exact branch.
- **Degenerate objects** (single pixel, collinear): skimage returns specific NaN/0; replicate.
- numba `inertia_tensor_eigvals` ordering + the `equivalent_diameter`/`centroid` kept-in-
  regionprops boundary must stay consistent with what the derivation assumes.
- Phase 1's ~1.4× is modest alone — confirm it's worth shipping before Phase 2, or bundle the
  numba sizeshape lane as "Phase 1+2" so the accelerator entry lands with a real (~3×) win.

## Expected impact
- sizeshape ≈ 352 → ~250 ms (Phase 1); → ~100–130 ms with Phase 2 (convex hull).
- numba pipeline 718 → ~620 ms (P1) → ~490 ms (P1+P2); numba-vs-numpy 2.74× → ~3.2× → ~4×.
- The numpy-lane moments scatter is independently a default-backend win (~1.4× on sizeshape).
