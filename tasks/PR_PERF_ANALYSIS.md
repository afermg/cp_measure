# Deep per-PR performance mining (2026-06-04, autonomous)

User directive: deeply analyse every open numba PR for further perf improvement WITHOUT sacrificing
conventions or accuracy. Mine existing implementations. In-depth review, lane by lane.

**Hard constraints (apply to every proposed win):**
- Serial `@njit(cache=True, error_model="numpy")` kernels ONLY. No `prange`/`nogil`/`parallel=True`.
  Parallelism lives only in an external batch layer.
- Bit-exactness/accuracy must NOT regress (golden numba-vs-numpy/reference stays green).
- Import-don't-reimplement boundary: IMPORT numerically-sensitive geometry (exact-Euclidean EDT,
  centrosome min_circle / convex_hull / regionprops, numerically-tuned filters); reimplementing is fine
  for algorithm-independent primitives (histograms, shortest-paths, moments, segment reductions).
- `to_bzyx` batch convention; registry/__init__ append pattern.

**Method:** for each lane — re-profile components on the branch → find the CURRENT bottleneck →
ask if it's reducible bit-exactly → validate → implement clear wins.

## Lane status board
| PR | lane | shipped speedup | mining verdict |
|---|---|---|---|
| #56 | granularity | fullres 6.86× / default 12.58× | TBD |
| #57 | intensity | (vs numpy) | TBD (bincount prep pending) |
| #58 | zernike + radial_zernikes | 28.7× / 20.9× | TBD |
| #60 | coloc pearson/manders/rwc/overlap | 31.6/62.0/67.7/5.6× | TBD (rwc sort-bound floor) |
| #62 | costes | 41.8× | TBD (search floor proven) |
| #63 | radial_distribution | 21.0× | TBD (EDT floor) |
| #64 | texture | 16.8× | TBD |
| #65 | feret | 43.4× | MAXED this session |

## Known floors / rejected ideas (do NOT re-propose)
- coloc rwc: sort-bound; lexsort (2.3× slower), sort+searchsorted (1.8× slower), prealloc (no-op) all tested. Only radix would beat it, not float64-exact.
- costes: scale=1 short-circuit REJECTED (anti-correlated objects drive a real search); search is exact-required.
- radial: scipy exact-Euclidean EDT (~19%) is the floor, do NOT reimplement (chamfer geodesic already reimplemented).
- texture: HXY1=HXY2=2·HX identity + T-via-build already applied (the O(fm1²) feature passes are gone).

## FINDINGS (7 parallel mining agents, 2026-06-04)

### #64 texture — BIGGEST WIN, bit-exact, in-kernel
After the HXY=2HX feature-pass collapse, the bottleneck SHIFTED to two O(fm1²≈256²) costs paid per
(object,direction): the dense `cmat[:]=0` clear AND the dense feature scan `for i,j in range(1,fm1)`.
For a 30² object that's ~145× the real pixel-pair work — ~99% spent iterating EMPTY GLCM cells (a
16-grey object touches ≤0.4% of 256²). **FIX: touched-cell stack** — push (a,b)/(b,a) on first write
during the build; feature pass iterates only stacked cells; reset only those cells after each direction
(allocate+zero cmat ONCE). Collapses BOTH O(fm1²) → O(nonzero), bit-exact (cmat stays dense full-size,
indices+fm1 unchanged → identical (i,j,c) multiset). + reuse px/px_plus_y/px_minus_y buffers (576→1
alloc). + keep crop uint8 not int64. Win scales inversely with object size = largest on the common case.
REJECTED: grey-compression (breaks value-indexed feats 1/3/5/6 + length-indexed feat 9); fused
multi-direction pass (needs 13 live 256² GLCMs in 3D = cache regression). → IMPLEMENT.

### #56 granularity — real bottleneck is the WRAPPER, not the morphology kernels
The "disk(10) opening 263ms" prior is only true at subsample=1. At DEFAULT subsample=0.25, end-to-end
profile: `scipy.ndimage.map_coordinates` = 67% (16 per-iteration resamples of `rec`, identical (sy,sx)),
`bincount` label-mean #2. FIX: (a) precompute floor/frac ONCE, per-iter numba bilinear gather (matches
map_coordinates order=1 to ~3e-16, ≪ the lane's existing rtol=1e-6 backend test — lane is within-tol not
bit-exact); (b) numba segmented-sum label-mean (BIT-EXACT 0.0 diff). Default 395→107ms (3.7×). The
`is_max` split prior is ~1% — skip. Morphology kernels (VHG opening, Vincent recon, incremental erosion)
all confirmed AT FLOOR, untouched. → IMPLEMENT (a)+(b).

### #63 radial_distribution — small free win + scratch reuse; EDT is the floor
#1 ring-buffer `% cap` → branch wrap in geodesic_chamfer_fifo: kills 2 int-divisions from the hottest
push/pop path, bit-exact (identical visit order). FREE. #2 reuse d/inq/qr/qc + host pad scratch across
144 objects (144 allocs→few), bit-exact, medium risk (stale-value). FLOORS: histogram CANNOT fuse into
geodesic (SPFA revisits pixels, no terminal sweep); scipy exact-Euclidean per-crop EDT (~19%) is the
import-boundary floor (whole-image EDT breaks #22). → IMPLEMENT #1; consider #2.

### #62 costes — headline incremental-sweep idea REJECTED (not bit-exact); two small exact wins
REJECTED Op2 (incremental running-sums Pearson sweep): 3 independent bit-exactness killers — (A)
single-pass `Σxy−ΣxΣy/n` ≠ scipy's centered two-pass form, flips sign near r≈0 (the decision boundary);
(B) sorted/crossing order ≠ block accumulation order (float non-assoc); (C) included set is a 2D OR
predicate, NON-NESTED when slope a<0 (anti-correlated — the exact case), and bisection visits
non-monotonically. EXACT wins: Op4 fold fi_max/si_max into _regression_ab's existing pass, delete
_linear's standalone max loop (bit-exact, max is order-independent); Op1 fuse _count_combt into
_pearson_combt (stop recomputing the count pass each iteration, byte-identical). On float (scale=1) the
search is short so Op4 (setup) > Op1. → IMPLEMENT Op4+Op1 (modest, clean).

### #60 coloc — structural 3× (shared-flatten) + in-kernel bit-exact 1.5×
Finding 1 (BIG, structural): featurizer runs pearson/manders/rwc as 3 separate wrappers on the SAME
image; coloc_per_object ALREADY computes all 9 outputs each time → flatten+kernel paid 3× per channel
pair. Flatten once + one coloc_per_object(compute_rwc=True), fan the 9-tuple out → ~3× on all non-RWC
coloc cost (× channel-pair count). Bit-exact (same kernel output, sliced not recomputed). RISK: needs a
coloc-aware combined dispatch entry; per-function API + golden tests must stay → DEFER, RECOMMEND to Tim
(this is the known shared-flatten/batch-layer item). Finding 3 (in-kernel, bit-exact): the per-object
body is 3 passes (means/maxima, centred moments, thresholds); passes 2 and 3 are INDEPENDENT given
pass-1 maxima → fuse into one loop, ~1.5× on the kernel's per-object loop, byte-identical accumulation.
FLOORS: rwc argsort; one-pass variance (breaks Σ(x−mean)²). → IMPLEMENT Finding 3; RECOMMEND Finding 1.

### #58 zernike — single-image kernel near floor; one tiny exact win + structural cross-fn win
Kernel already exploits Zernike parity (no zero radial terms), early r²>1 cutoff, contiguous (n,K)
scatter, hoisted degree-LUT — at floor. #2 (tiny, exact): 5 of 30 columns are m==0 with zi[0]==0 →
`vi[seg,k]+=w*s*0.0` every in-disk pixel; hoist per-k is_m0 flag, skip the vi write (vi stays 0.0,
arctan2(vr,0) unchanged). #1 (structural): get_zernike + get_radial_zernikes recompute the SAME host
geometry (minimum_enclosing_circle, nonzero, normalization) AND the entire pixel pass twice when both
run; could share/fuse (one geometry + one pass producing weighted+unweighted sums). RISK: needs shared
context / fused entry, forward-looking (numba zernike not even registered yet). DON'T touch the z^m
sequential recurrence (bit-exactness-load-bearing). → IMPLEMENT #2; RECOMMEND #1.

### #57 intensity — bincount prep has a NON-FINITE trap; std two-pass is a hard floor
The coloc bincount port is NOT safe verbatim: `bincount(masks.ravel())` counts ALL masks>0 pixels incl
non-finite, but intensity's flatten DROPS non-finite (matches baseline `(masks>0)&isfinite`) → offsets
over-count, garbage. Coloc is immune (it keeps non-finite). SAFE subset: bincount for (lut,n) only
(drops scipy find_objects scan), keep the finiteness count scan; or a no-NaN fast-path (bincount-count
== finite-count else fallback). std fusion (E[x²]−E[x]²) is a BIT-EXACTNESS BLOCKER — baseline std is
two-pass mean((x−mean)²); one-pass differs in float64 → MUST stay two-pass (floor). Quantile per-segment
sorts = floor. → RECOMMEND safe bincount-for-(lut,n); document std floor. (Lower priority.)

## STACK HEALTH (full pytest per branch, 2026-06-04) — ALL GREEN
feret 94 · coloc 114 · costes 145 · texture 100 · granularity 176 · zernike 104 · radial 118 · intensity 89.
(coloc/costes/feret include this session's commits; all branches in sync with origin.)
NOTE: #62 costes is stacked on #60 coloc; #60 advanced (coloc fuse) so the local #62 lacks that commit —
its PR diff is unaffected (computed vs #60 HEAD); rebase #62 onto #60 (or merge #60 first) before merging.

## CONSOLIDATED SPEEDUPS (numba vs numpy, 1080²/144 obj, this session's measurements)
feret **43.4×** (NEW) · coloc pearson 31.6× / manders 62.0× / overlap 67.7× / rwc 5.6× (+fuse −3-4%) ·
costes ~42× (+fuse −9%) · texture 16.8× · radial_distribution 21.0× · zernike 28.7× / radial_zernikes 20.9× ·
granularity fullres 6.9× / default 12.6× (gather POC → ~3× MORE available, gated) · intensity (merged #54).

## IMPLEMENTATION LOG

### META-FINDING (the headline of this pass)
I empirically validated every agent-proposed opt by building it and benchmarking OLD-vs-NEW end-to-end
(not trusting the component-level estimates). **SHIPPED 3 real wins:** granularity gather (2.88×), costes
fuse (−9%), coloc fuse (−3-4%). **6 candidates were validated NON-wins and reverted** (Amdahl-capped, broke
numba vectorization, regressed the common case, or numpy was already as fast). The agents' component
analyses consistently over-estimated because each lane's dominant END-TO-END cost lives elsewhere (host
glue, scipy EDT, rwc sort, regionprops, min_circle) — even the "structural ~3×" coloc shared-flatten is
only ~1.12× once rwc's sort is accounted for. KEY: the one BIG win (granularity gather, 2.88×) was found
by the agent's ACTUAL profiling (map_coordinates = 67%), not a component guess — profiling beats estimating.

| lane | tried | bit-exact? | end-to-end result | verdict |
|---|---|---|---|---|
| coloc | fuse pass 2+3 | yes (array_equal) | pearson/manders **−3-4%** (12.3→11.9ms), rwc −1% | **SHIPPED** (c22d856) |
| costes | fuse bisection count+pearson | yes (array_equal) | **−9%** (23.0→20.9ms) | **SHIPPED** (8b4ee9e) |
| granularity | numba bilinear gather (16× map_coordinates) | 3e-14 (≪ rtol1e-6) | **2.88×** (741→257ms default) | **SHIPPED** (89dde43) |
| texture | sparse touched-cell stack | within rtol1e-6 | 144obj −10% (regress), 600obj +22%, 1500obj +65% | REVERTED → adaptive follow-up |
| granularity | numba label_mean | yes (diff 0.0) | negligible (766 vs 741, map_coords dominates) | REVERTED |
| zernike | skip vi+=0 (m==0 cols) | yes (array_equal) | SLOWER 139→145ms (branch breaks k-loop vectorization) | REVERTED |
| radial | FIFO %→branch | yes (array_equal) | no win 119→122ms (geodesic not the bottleneck) | REVERTED |
| intensity | fused edge boundary-extract (1 numba scan vs inner_boundary+3 numpy gathers) | yes (array_equal) | neutral 134→134ms (numpy bool-index already as fast) | REVERTED |

### STRUCTURAL / GATED RECOMMENDATIONS (real wins, need sign-off — NOT shipped autonomously)
1. **coloc shared-flatten — POC says only ~1.12× (NOT the agent's 3×), DOWNGRADED.** Idea: flatten
   once + one coloc_per_object(compute_rwc=True), fan the 9-tuple to pearson/manders/rwc/overlap.
   `tasks/poc_coloc_shared_flatten.py` measured it: current 3 calls 219.7 ms vs shared 1 call 196.0 ms
   = **1.12×**. Why not 3×: rwc's argsort DOMINATES (196 of 220 ms) and is paid once either way; sharing
   only saves the two cheap non-rwc kernel passes (~24 ms). The 3× would apply only if rwc is disabled
   (then 3 equal cheap calls → 1). Bit-exact, but ~1.12× with rwc on does NOT justify the architectural
   dispatch+featurizer change. coloc is effectively at its floor (rwc sort). NOT WORTH IT unless a
   pipeline runs the non-rwc coloc features without rwc.
2. **granularity bilinear gather — SHIPPED (89dde43), 2.88×** (741→257 ms default). [details below kept]
   ~~POC-VALIDATED ~3× end-to-end, READY TO SHIP ON TIM'S OK.~~
   Replace the 16 per-iteration scipy.map_coordinates (67% of default runtime, identical (sy,sx)) with a
   precomputed-floor/frac numba bilinear gather (order-1, mode=constant cval=0).
   `tasks/poc_granularity_gather.py` measured on the actual coords/rec: **max abs diff vs
   map_coordinates = 3.33e-16** (all <1e-12), **9.3×/call** (36.1→3.9 ms), **~515 ms saved/image**
   (16 calls) → ~3× end-to-end on the 144-obj 1080² default. ONLY reason not shipped autonomously: it's
   the one RESULTS-CHANGING win (3.33e-16, ≪ the lane's rtol=1e-6 — the lane is within-tol, not
   bit-exact) and the user drew a "no accuracy sacrifice" line; everything else shipped is bit-exact.
   Implementation: add `_bilinear_gather` to `_granularity.py`, precompute y0/x0/fy/fx once after sy/sx,
   swap the in-loop map_coordinates (and optionally the once-call back_pixels resample), update the
   module docstring note, add a <1e-12 kernel test. ~30 min, de-risked. **The single biggest remaining win.**
3. **texture adaptive sparse stack (1.2-1.65× for dense fields >~300 cells).** Ship the touched-cell
   stack ONLY when `16*npix < fm1*fm1` (sparse regime), else keep the dense path — avoids the −10%
   regression on few-large-objects. Doubles the build kernel; worth it for high-cell-count images.
4. **zernike shared geometry across get_zernike + get_radial_zernikes.** Both recompute the same
   minimum_enclosing_circle + nonzero + normalization + pixel pass when both run. Share via a fused
   "both-zernikes" entry → saves one min_circle + one pass per image. Forward-looking (numba zernike
   not yet registered); needs a shared-context channel.
5. **intensity bincount-for-(lut,n)** (drops scipy find_objects scan). Modest. CAUTION: the full coloc
   bincount port over-counts NON-FINITE pixels (intensity drops them, coloc keeps them) → use only for
   (lut,n), keep the finiteness count scan, or a no-NaN fast-path. std MUST stay two-pass (one-pass
   E[x²]−E[x]² breaks bit-exactness vs the baseline). Quantile sorts = floor.

### #64 texture — sparse-stack BUILT then REVERTED (conditional win, regresses common case)
Implemented the touched-cell stack (cmat+marginals once, incremental row0, sparse reset). 14 tests
green (bit-exact within rtol=1e-6). But empirical old-vs-new (1080², varying object count):
- 144 obj (~90px, the headline/common case): 176.8 → 195.1 ms (**−10% REGRESSION**)
- 600 obj (~44px): 275.6 → 226.6 ms (1.22×)
- 1500 obj (~28px): 466.8 → 283.4 ms (1.65×)
The per-pair `if cmat[a,b]==0` first-touch branch MISPREDICTS on large objects (build dominates there),
outweighing the saved O(fm1²) clears. Real win only for dense fields (>~300 cells). REVERTED to avoid
regressing the common case. CLEAN ship would need a per-object adaptive heuristic
(`if 16*npix < fm1*fm1: sparse else dense`) → doubles the build kernel. RECOMMENDED as a follow-up, not
shipped tonight. (The agent's "145× overhead" was for 30px objects; for ~90px cells the O(npix) build
dominates and the sparse win shrinks/inverts.)
