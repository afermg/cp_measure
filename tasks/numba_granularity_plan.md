# Numba granularity backend + (B,Z,Y,X) batching — implementation plan

Port `get_granularity` to a numba backend on the #49/#54 dispatch seam, AND introduce a
canonical `(B, Z, Y, X)` batch shape across the optimised functions. **2D-only numba
morphology; 3D (Z>1) and non-numba paths dispatch to the numpy baseline.** Bit-compatible
with baseline in BOTH fullres (`subsample_size=1.0`) and default subsampled (`0.25`) modes.

## Decisions locked (2026-06-03, 12-question round)
1. **Batch scope:** numba CPU image-loop only. Kernels stay single-threaded (locked);
   no GPU/jax work here.
2. **Batch layer:** inside the function (not a separate featurize layer), via the canonical
   shape below.
3. **Parallelism:** serial loop over batch now; multiprocessing is a later, orthogonal step.
4. **Canonical input shape: `(B, Z, Y, X)` for ALL optimised functions.** Everything is a
   special case:
   - 2D `(H,W)` ndarray → `(1,1,H,W)` (single → returns a dict)
   - 3D `(Z,Y,X)` ndarray → `(1,Z,Y,X)` — **one volume, NOT a batch** (single → dict).
     Preserves the current intensity 3D-volume semantics; backward-compatible.
   - 4D `(B,Z,Y,X)` ndarray → batch (→ returns a list).
   - **list** of 2D/3D arrays → batch, supports **ragged** (different H,W) (→ list).
   - ⇒ to pass a batch of 2D images as an array use `(B,1,H,W)`; a bare `(N,H,W)` is read as
     one volume (the price of backward compat). Document this.
5. **Output:** list of per-image dicts when batched; single dict when a lone image/volume is
   passed. Internally one loop; only input-normalize + output-unwrap differ.
6. **Labels:** independent per image (`1..k` each); results grouped per image. No global
   offsetting required from the caller.
7. **One code path:** single = batch-of-1 (kills the single-vs-batch divergence bug class).
8. **(B,Z,Y,X) rollout:** this PR ships the shared normalize helper + granularity on it.
   **Intensity retrofit is SPLIT into an immediate follow-up PR** (same end state, smaller
   diffs). The helper is built batch-correct here so the follow-up is a mechanical migration;
   intensity output must stay bit-identical (golden test) in that follow-up.
9. **Disk morphology:** VHG row-decomposed immediately (O(r·HW)).
10. **Correctness target:** bit-exact (`==`) for the deterministic morphology kernels vs
    skimage; `rtol/atol` only for the order-1 interpolated per-object-mean readback.
11. **Test/bench env:** provision `uv` in this repo (uv.lock exists; uv not yet installed on
    box — step 0).

## Why granularity is its own (independent) lane
Hot work is **morphology**, not a segment-reduce. It does NOT extend the shared segment
primitive — it consumes a per-object mean via `numpy.bincount` (host). Touches none of
`primitives/_segment_numba.py`; runs parallel to the zernike/radial lanes. (The shared
`(B,Z,Y,X)` helper is the one new cross-cutting piece — see below.)

## Baseline algorithm (recap)
subsample (map_coordinates) → bg-subtract via greyscale open (disk erosion+dilation,
radius=`element_size`=10) → spectrum loop `ng=16`: erode disk(1) → `skimage…reconstruction`
→ upsample `rec` to orig scale → `scipy.ndimage.mean`/obj → `gss=(cur-new)*100/start`.
**Output indexed densely by label `1..max(mask)`** (length `max_label`, 0 for absent) — NOT
compacted to present objects. So use dense `bincount`, NOT `label_to_idx_lut`.

## Cross-variant findings (recap)
- `fast` (numba, 2D): VHG disk min/max; 5-tap disk(1); Vincent/Robinson hybrid raster+FIFO
  reconstruction (exact-equiv to skimage, reused int32 scratch); bincount means; **point-query
  map_coordinates** readback at in-mask pixels (`_needs_resize`; flat-gather if fullres);
  **cascaded mask** (rec_g ≤ rec_{g-1}). Bit-exact; ~4.2×→5.8× large, ~1.3× default.
- `jax`: same algebra, confirms readback rearrangement.
- `speed`: byte-identical to baseline (prior "re-enables subsampling" note WRONG).
- Nobody numba-ifies `map_coordinates`; resampling stays scipy.

## Non-fullres (subsampled) — only place subsampling enters the kernels
Resampling stays scipy; kernels are scale-agnostic. The one branch is the readback:
- fullres (`subsample_size ≥ 1`): `rec` at orig scale → flat-gather `rec.ravel()[pos]`.
- subsampled (`< 1`, default): `rec` at subsampled scale → `map_coordinates(rec,[sy,sx],order=1)`
  at original label-pixel coords. Both → per-object means at original-label resolution.

## Files
- NEW `src/cp_measure/core/numba/_granularity.py` — numba morphology kernels.
- NEW `src/cp_measure/core/numba/measuregranularity.py` — `get_granularity` (numba 2D path,
  else numpy baseline), batch-shaped.
- NEW shared helper, e.g. `src/cp_measure/primitives/shapes.py` —
  `to_bzyx(masks, pixels) -> (masks4d, pixels4d, unwrap)` where `unwrap(list_of_dicts)`
  returns a single dict if the input was a lone image/volume, else the list.
- EDIT `src/cp_measure/core/numba/measureobjectintensity.py` — consume `(B,Z,Y,X)` via the
  helper; loop B; return dict (single) / list (batch). Output bit-identical to current.
- EDIT `src/cp_measure/core/numba/__init__.py` — re-export `get_granularity`.
- EDIT `src/cp_measure/bulk.py::_numba_registries` — add `"granularity"`.
- EDIT `test/test_backend_correctness.py` — granularity + batch + intensity-golden cases.

## Kernels (`_granularity.py`, all `@njit(cache=True)`, single-threaded, 2D)
1. `disk_erosion_2d(img, radius)` / `disk_dilation_2d(img, radius)` — **VHG row-decomposed**:
   per disk-row half-width `dx(dy)=floor(sqrt(r²-dy²))`, 1-D van-Herk/Gil-Werman sliding min/max
   along that row, combine across the `2r+1` shifted rows; `nearest`-clamp border (match
   skimage). Exact for the true disk.
2. `erosion_4conn_2d(img)` / `dilation_4conn_2d(img)` — 5-tap min/max (disk(1)); OOB neighbours
   ignored (= clamp), matches skimage disk(1). `dilation_4conn_2d` used inside reconstruction.
3. `reconstruction_by_dilation_2d(seed, mask, queue)` — Vincent/Robinson hybrid, 4-conn:
   3 raster pairs (`out=min(mask,max(out,causal-nbrs))`), last bwd pass seeds a FIFO, then
   FIFO dilate-under-mask until drained. `queue`: caller int32 scratch ≥ `12*H*W`, coords
   `(row<<16)|col`, reused across all 16 iterations (assert image ≤ 65535²). Exact-equiv to
   `skimage.morphology.reconstruction(seed, mask, disk(1))`.

## Per-image wrapper logic (after normalize to (B,Z,Y,X), loop b in 0..B-1)
For each `(Z,Y,X)` element: if `Z>1` → numpy baseline (3D). Else take the `(H,W)` plane and run:
```
1. subsample image+mask (scipy map_coordinates, as baseline)
2. bg subtract: subsample → disk_dilation_2d(disk_erosion_2d(back, r), r) → upsample → pixels-=back; clip(0)
3. precompute ONCE per image:
     max_label; counts = bincount(mask.ravel(), minlength=max_label+1)[1:]   # dense 1..max
     in-obj flat pos; if subsampled: scaled (sy,sx) coords + needs_resize
     queue = empty(12*H*W, int32)         # H,W of the working (subsampled) image
     current_mean = bincount-mean of ORIG pixels over orig_mask (dense 1..max)
     start_mean = maximum(current_mean, eps); recon_mask = pixels
4. for id in 1..ng:
     ero = erosion_4conn_2d(ero)
     rec = ero if ero.max()==0 else reconstruction_by_dilation_2d(ero, recon_mask, queue)
     recon_mask = rec                                          # cascaded mask
     rec_valid = map_coordinates(rec,[sy,sx],order=1) if needs_resize else rec.ravel()[pos]
     new_mean = bincount(labels_in_obj, weights=rec_valid, minlength=max_label+1)[1:]/counts
     results[f"Granularity_{id}"] = (current_mean-new_mean)*100/start_mean; current_mean=new_mean
```
Collect each image's `results` dict → `unwrap([...])`.

## Correctness harness (`test_backend_correctness.py`)
- Per-kernel vs skimage, **bit-exact (`==`)**: `disk_*` vs `erosion/dilation(disk(r))`;
  `*_4conn` vs `disk(1)`; `reconstruction_by_dilation_2d` vs `…reconstruction`.
- Wrapper numba == baseline (within tol on the interpolated readback): 2D fullres (1.0) and
  2D default (0.25); edges: empty mask, single object, 1-px object, label gaps (assert dense
  1..max incl. zeros), all-zero after bg-subtract.
- **Batch:** list/4D of images == looping the single-image baseline; per-image dicts; ragged
  sizes; single input still returns a dict.
- **Intensity golden test:** capture current intensity output on fixtures BEFORE the retrofit,
  assert bit-identical AFTER (the retrofit must not change results); plus intensity batch ==
  per-image loop.
- 3D `(Z>1)` input routes to numpy baseline.

## Build order
0. **Provision uv** in the repo (uv.lock present); confirm `uv run pytest` works with numba.
1. Shared `to_bzyx` helper + unit tests (all special cases: 2D/3D/4D/list/ragged → shapes;
   unwrap dict-vs-list).
2. `_granularity.py` kernels + per-kernel bit-exact skimage tests (VHG disk; 4-conn; recon).
3. Granularity wrapper, **fullres single-image** path == baseline.
4. Add subsampled readback (point-query) == baseline default.
5. Cascaded mask + triple-raster recon; re-assert correctness.
6. Batch path (loop B, list-of-dicts, ragged) == per-image baseline.
7. Wire dispatch (`_numba_registries`) + `__init__` re-export + all backend-correctness tests.
8. Benchmark 1080²/142obj fullres + default vs baseline; record speedup (Alan's per-feature
   table). Expect ~4–6× fullres, ~1.3× default.

### Follow-up PR (immediately after this one)
- **Intensity retrofit to `(B,Z,Y,X)`**: golden test first (freeze current intensity output on
  fixtures) → migrate intensity to `to_bzyx` + B-loop → assert golden bit-identical + batch
  works. Mechanical because the helper already lands here.

## Conventions to pin (assert in tests)
- disk border = `nearest`; 4-conn border = ignore-OOB (match skimage).
- reconstruction deterministic (no tie-break ambiguity).
- granularity output dense `1..max_label`, 0 for absent (NOT compacted objects).
- readback `map_coordinates` order=1.
- 3D ndarray = one volume (B=1), never a batch; batch-of-2D needs `(B,1,H,W)` or a list.

## RESULTS (implemented 2026-06-03, branch `feat/accelerator-numba-granularity`)
All 8 steps done; full suite **177 passed**. New tests: `test_primitives_shapes.py` (11),
`test_granularity_kernels.py` (76, bit-exact vs skimage), `test_granularity_backend.py` (12,
vs baseline within tol), `test_backend_correctness.py` (+granularity dispatch).
**Reconstruction: Vincent (1993) FIFO-hybrid** (1 fwd + 1 bwd raster, then a ring-buffer
FIFO with a per-pixel in-queue flag → ≤N live entries, N+1 buffer cannot overflow). O(N),
bit-exact vs skimage. Replaced an initial raster-until-convergence version after profiling
showed reconstruction = 83% of fullres time with up to 247 raster passes per spectrum step.
Also dropped the baseline's dead subsampled-mask resample (never read).
Benchmark (1080², 144 objects, min of 3, JIT warmed):
- **fullres** (subsample=1.0): numpy 9000 ms → numba **1369 ms = 6.57×** (raster-converge was 4.01×)
- **default** (subsample=0.25): numpy 1176 ms → numba **94 ms = 12.47×**
  (default beats the ~1.3× estimate: eliminating the 16 full-res upsamples + 16
  `scipy.ndimage.mean` unique+argsort dominates the tiny 270² morphology.)
- fullres split after FIFO: bg-opening disk(10) ~263 ms (now the largest single chunk),
  erosion ~24 ms, reconstruction ~1000 ms, readback ~80 ms.

## Risks / notes
- **Intensity retrofit touches merged, tested code** — the golden test (step 7) is the guard;
  keep the diff mechanical (normalize + loop), output un-changed for single inputs.
- uv not installed on the box — step 0 may need `pip install uv` / curl installer + network.
- VHG disk must be validated bit-exact vs skimage at several radii before trusting it.
