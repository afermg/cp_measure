# Numba sizeshape lane — STEP-0 VERDICT: NO-GO (Amdahl-capped, no reimplementable hot primitive)

**Decision (2026-06-04): do NOT ship a numba sizeshape backend.** Leave it on numpy.

## Step-0 profile (1080², 144 objects, `tasks/profile_sizeshape.py`)

| component | ms | % of fn | boundary |
|---|---:|---:|---|
| `get_sizeshape` (full) | 554.6 | 100% | |
| (A) `regionprops_table` | 406.1 | 73.2% | IMPORT (geometry bundle) |
| (B) EDT radius loop | 138.4 | 25.0% | |
| — (B1) scipy EDT | 74.2 | 13.4% | IMPORT (exact-Euclidean) |
| — (B2) max/mean/median reductions | 64.2 | 11.6% | **reducible** |

**Amdahl ceiling, numba-ing only the reducible reductions (B2): 1.13×.**
(Even illegally reimplementing the EDT too: 1.33×.)

## Is there a reimplementable dominant primitive? (the radial/texture check) — NO

`regionprops_table` property-group decomposition (`tasks/profile_sizeshape_regionprops.py`,
delta over a base of area/bbox/centroid/axes):

| group | Δ ms | reducible? |
|---|---:|---|
| **convex hull (area_convex + solidity)** | **+187.2** | NO — computational geometry (import) |
| perimeter | +43.7 | NO — numerically-tuned |
| area_filled | +38.7 | NO |
| perimeter_crofton | +28.4 | NO — Crofton formula (import) |
| euler_number | +25.7 | NO |
| **moments (spatial/central/norm/hu/inertia)** | **+4.7** | YES — but negligible |

The single dominant primitive is the **convex hull** (~46% of regionprops) — exactly the
import boundary (cp_measure already imports `centrosome.convex_hull_ijv` for feret). The one
mechanically-reducible group (moments = polynomial pixel sums) is essentially **free** (+4.7 ms).

Contrast with the lanes that justified a kernel:
- radial: dominant primitive = `propagate` = 1/√2 chamfer **shortest-path** → algorithm-independent → bit-exact reimplementable (35× on the bottleneck).
- texture: dominant primitive = GLCM = **integer histogram** → algorithm-independent → bit-exact (16.8×).
- sizeshape: dominant primitive = **convex hull** → numerically-sensitive geometry → IMPORT. The reducible primitive (moments) is negligible. No reprieve.

## Conclusion
A numba kernel would, at maximum, fuse the per-object EDT-radius reductions (B2) — replacing
3 scipy calls × 144 objects with one njit call — for a whole-function ceiling of **1.13×**.
Against the 16–68× bar set by the other lanes, that does not justify a new kernel + test file +
maintenance burden. Verdict: **NO-GO**, documented. sizeshape stays on the numpy backend.

(If sizeshape ever becomes a measured bottleneck, the highest-value move is NOT numba — it is
swapping skimage's per-object convex hull for centrosome's batched `convex_hull_ijv` (one call
for all objects), but that risks hull-area divergence vs skimage and is out of scope for the
numba port. Recorded as a future idea, not actioned.)
