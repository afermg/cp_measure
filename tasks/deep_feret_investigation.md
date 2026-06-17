# DEEP investigation: speeding up `get_feret` (numpy + existing deps only)

**Date:** 2026-06-06
**Constraint:** numpy / scipy / skimage / centrosome only. No numba, no jax, no new deps.
**Script:** `tasks/deep_feret_investigation.py` (real Cell-Painting masks: tiny 256²/2obj,
small 540²/43obj, large 1080²/142obj, m2160 2160²/568obj; + a dense `touching8` synthetic).

## Headline

1. **The big win is branch-specific and already exists on main.** Our branch
   `fix/radial-per-object-22` still ships the **pre-#71 `masks_to_ijv`** (per-label
   `np.where` + in-loop `np.concatenate`, O(nobj·pixels)). On our branch it is **~97% of
   `get_feret`** (large: 355 of 365 ms; m2160: 6195 of 6219 ms). Main's #71 rewrite
   (`nonzero` + stable argsort) already fixed this: **large 365→37 ms (~10×), m2160
   6219→108 ms (~58×).** ⇒ The single highest-impact action is to **rebase onto main / adopt
   #71** — no new code.

2. **On current main, `get_feret` is already near its floor under this constraint.** Every
   additional numpy lever I tried either loses on real masks or isn't numpy-addressable.
   My earlier `proto_numpy_levers.py` "hull-from-boundary 3.52×" was a **synthetic-density
   artifact** — it does NOT generalise to real masks (see below). I'm retracting it as a
   general recommendation.

## Component breakdown — `get_feret` on current main (#71 baseline)

| case | masks_to_ijv | convex_hull_ijv | feret_diameter | total |
|------|---|---|---|---|
| small 540²/43 | 2.99 | 3.61 | 4.75 | 12.1 ms |
| large 1080²/142 | 12.48 | 14.36 | 7.99 | 37.3 ms |
| m2160 2160²/568 | 89.82 | 53.08 | 14.30 | 108.3 ms |

Cost is **balanced across three parts** (~33% / ~38% / ~21% on large). No dominant term to attack.

- `masks_to_ijv` — already #71-optimised (`nonzero` + stable argsort).
- `convex_hull_ijv(fast=True)` — dispatches to a **compiled C extension** (`_convex_hull`).
  Not numpy-addressable; the only lever is feeding it fewer points.
- `feret_diameter` — pure-python but **already vectorised across all objects**: the
  `while len(chull_idx)>0` loop iterates over rotating-calipers *steps* (shrinking the active
  hull set), not per object. numpy + `lexsort` + `scind.min/max`. Near its floor.

## Levers tested — all bit-exact, all net losses on real masks

Speedup vs main `get_feret` (>1 = faster). `A` = `centrosome.convex_hull(masks,idx)` (reduces
via `outline()` internally); `B` = explicit `outline`→ijv→hull; `Bn` = B without argsort:

| case | A | B | Bn | bit-exact? |
|------|---|---|----|------------|
| tiny 256²/2 | **3.40×** | 3.25× | 3.27× | yes |
| small 540²/43 | 0.89× | 0.83× | 0.84× | yes |
| large 1080²/142 | 0.85× | 0.78× | 0.80× | yes |
| m2160 2160²/568 | 0.69× | 0.63× | 0.65× | yes |
| touching8 (dense) | **3.97×** | 3.64× | 3.76× | yes |

**Why boundary-reduction loses on real masks:** `outline(masks)` (or `binary_erosion`) is a
whole-image op costing ~10 ms (large) / ~48 ms (m2160) — *as much as the hull it saves*,
because real cells are small/sparse (few interior pixels: large has only 124 k fg px). It
only wins when interior ≫ boundary, i.e. **large dense/convex objects** (tiny, touching8).
My earlier 3.52× used a dense 144-square grid (646 k pts) — pathologically favourable.

**Skip-the-argsort lever** (feed unsorted ijv to the C hull, idx from `unique(masks)`):
bit-exact but **0.66–0.83× (slower)** on large/m2160 — the C `convex_hull_ijv` runs faster on
label-contiguous input, so the argsort pays for itself. Reject.

**8-extreme-points approximation** (bbox + diagonal extrema per object instead of the hull):
rejected on principle — drops true hull vertices → wrong min-feret, not bit-exact.

## Recommendation

1. **Rebase `fix/radial-per-object-22` onto current main** (or cherry-pick #71). That is the
   entire ~10–58× win for this branch; the slow `masks_to_ijv` is the only real feret problem,
   and it's already solved upstream.
2. **Do NOT pursue hull-from-boundary as a general feret optimisation.** On main with real
   masks it regresses (0.63–0.89×). (Supersedes the `proto_numpy_levers.py` C1 finding.)
3. **Optional, data-dependent:** only if a workload is known to have large/dense/convex objects
   (interior ≫ perimeter), an *adaptive* guard — `if fg_px / boundary_px > k: use centrosome.
   convex_hull` — captures the 3–4× there without regressing sparse masks. Not worth it for
   typical Cell-Painting data. Flag, don't build.
4. `feret_diameter` is already vectorised; no pure-numpy win without an algorithm change (risky
   for bit-exactness). Leave it.

**Net:** beyond adopting #71, there is no robust numpy-only speedup for feret on realistic data.
The earlier-reported feret lever was a synthetic artifact and should not be implemented.

## Verification

`.venv/bin/python -u tasks/deep_feret_investigation.py` (the script inlines main's #71
`masks_to_ijv` so the baseline reflects main, not our stale branch). All candidates assert
`feret_diameter` bit-identical to the main path on every case.
