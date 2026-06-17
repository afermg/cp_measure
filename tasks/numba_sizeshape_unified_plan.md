# Unified numba sizeshape lane — implementation plan

## Premise
sizeshape was a numba NO-GO under the old analysis. This deep dive overturned that: post-#77
(`get_sizeshape` ~220 ms) the remaining cost is **reimplementable bit-exact**, and the key is a
**single shared foreground-pixel pass** feeding every primitive instead of separate lanes.

Measured cost landscape (large tile, 1080² / 142 obj):

| primitive | numpy now | numba target | bit-exact status |
|-----------|----------:|-------------:|------------------|
| shared prep (label→offsets) | ~9 ms (repeated per primitive) | ~2–3 ms (once) | — |
| moments + inertia (#77) | ~33 ms | **~2 ms kernel** | ✅ proven 0.00 (`proto_numba_moments_kernel.py`) |
| convex hull (area_convex/solidity) | ~115 ms | ~45 ms hybrid / ~18 ms full | ✅ hybrid proven 142/142 (`proto_hybrid_hull.py`) |
| perimeter + perimeter_crofton | ~45 ms | ~10 ms | ✅ proven 142/142 in numpy (`proto_perimeter.py`) |
| euler_number | ~21 ms | ~5 ms | ✅ deterministic 2×2 patterns |
| EDT radius loop | ~25 ms | keep scipy | n/a (Euclidean — don't reimplement) |
| **total** | **~220 ms** | **~85–110 ms (~2–2.6×)** | all bit-exact¹ |

¹ convex hull bit-exact via the monotone-chain + `grid_points_in_poly` hybrid; the full raster
port is the only non-bit-exact-by-default variant (deferred / optional).

## Architecture: one pass, four primitives
`core/numba/_sizeshape.py` wrapper (`to_bzyx`, 2D-only; 3D → numpy baseline) does:

1. **Shared prep** — from the labelled image, build the foreground pixel list grouped by object
   (rows, cols, per-object offsets) via the stack's `primitives/segment.labels_to_offsets`
   (one `bincount`, no `nonzero`/`searchsorted`). This is the seam every primitive reads.
2. **Moment kernel** (`@njit`) — two passes over the grouped pixels: pass A accumulates the 16 raw
   moments in each object's local bbox frame + tracks bbox-min inline; pass B accumulates the 16
   central moments (centred coords, after centroid). Returns `(raw, central)` per object.
3. **Convex-hull kernel** (`@njit`) — per object: candidate boundary pixels → ±0.5 diamond
   offsets → monotone-chain hull (combinatorially exact). Then **skimage's
   `grid_points_in_poly`** rasterises each hull (kept as-is — proven bit-exact, NOT ported).
4. **Perimeter/euler kernel** (`@njit`) — one label-aware neighbour-pattern pass: per foreground
   pixel, compare label to its 8 neighbours (same-label shifts), accumulate the perimeter
   3×3-pattern histogram + the euler 2×2-pattern counts per object; apply skimage's exact LUT
   weights host-side.
5. **EDT radii** — keep `scipy.ndimage.distance_transform_edt` per object (Euclidean; the one
   genuine import boundary), reductions already direct numpy (#70).
6. **Derivation** (host numpy) — reuse #77's `_moments.py`: normalized + Hu from central, inertia
   tensor + eigvals from central. Assemble the full result dict under the existing `F_*` names.

## Shared-derivation refactor (prerequisite, in #77 or a small follow-up)
Split `primitives/_moments.py` so the **accumulation** (numpy `bincount` scatter) and the
**derivation** (normalized/Hu/inertia algebra) are separate:
- keep `spatial_moments_2d` (numpy) → `(raw, central, normalized, hu)`
- extract `derive_normalized_hu(central) -> (normalized, hu)` and keep `inertia_2d(central)`
- the numba lane calls the numba kernel for `(raw, central)`, then the **same** `derive_*` host
  functions → one source of truth for the math, two accumulation backends.

## Base / stacking
Needs both: the numba infra (#59 `to_bzyx`, `primitives/segment` offsets, `bulk._numba_registries`
dispatch, `_detect.HAS_NUMBA`) **and** #77's `_moments.py` derivation. So base the lane on a tree
with #77 merged **and** the numba stack — i.e. land #77 to main, then build
`feat/numba-sizeshape` on the rebased numba stack. Register `"sizeshape"` in
`_numba_registries()["core"]` + export in `core/numba/__init__.py`. (Fix the merge-mangled
`bulk._numba_registries` while there, or land on a clean branch off the bzyx base.)

## Build order (by proven-ness × value; each independently testable)
1. **Moments + inertia kernel** — proven bit-exact, the foundation + shared prep. Establishes the
   `_sizeshape.py` wrapper and the labels_to_offsets seam. (~33 → ~2–4 ms.)
2. **Convex hull hybrid** — the big win (~115 → ~45 ms), proven bit-exact (142/142). **Gated on the
   one unknown:** whether the numba monotone-chain hull beats scipy QHull's per-region cost (the
   python proxy is 17× *slower* — only a numba prototype settles it). PROTOTYPE FIRST.
3. **Perimeter + euler** — deterministic, bit-exact, numba beats per-region C (whole-image numpy
   was slower — numba's compiled single pass wins). (~66 → ~15 ms.)

## Testing
Golden vs the numpy backend AND `skimage.regionprops_table`:
- moments/inertia: bit-exact raw, ~1e-13 central/normalized/Hu/inertia (reuse #77's goldens).
- convex hull: `area_convex`/`solidity` exact vs skimage (142/142 target); the hybrid is proven.
- perimeter/crofton/euler: exact vs skimage per-object.
- numba == numpy backend on the 3-tier data; edge cases: single-pixel (degenerate inertia/hull),
  non-contiguous labels, edge-touching, empty, 3D → numpy baseline.
- Conventions match the stack: `to_bzyx` 2D-only, serial `njit`, registry/`__init__`/dispatch
  append, golden + kernel tests.

## Risks / open questions
- **numba monotone-chain hull vs scipy QHull speed** — THE gating unknown for the convex win
  (Phase 2). Prototype before committing; if it doesn't beat QHull, the convex lane's value drops
  to the per-region-Python savings only.
- **Full `grid_points_in_poly` port** (convex 115 → ~18 ms, ~6×) — would remove the 32 ms raster
  floor but reintroduces edge-classification bit-exactness risk (a naive crossing test is 2/142).
  Keep as an optional Phase 4, NOT default.
- **3D** — left on the numpy baseline (3D moments = 4×4×4, 3D hull = 3×3 eigensolve; out of scope).
- **EDT** stays scipy — do not reimplement Euclidean EDT.

## Projected impact
sizeshape ~220 → **~85–110 ms (~2–2.6×)**, bit-exact. In the numba-merged pipeline this removes
sizeshape as the dominant remaining component (it was ~49% of the numba pipeline at the old
350 ms; #77 already cut it to 220, this lane cuts it to ~100).
