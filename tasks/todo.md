# cp_measure — TODO

## Featurizer `return_as` (PR #38 — OPEN, has merge conflicts vs upstream main as of 2026-05-30)

- [x] Implement `return_as` parameter on `featurize()` supporting `"tuple"`, `"pandas"`, `"pyarrow"`, `"anndata"`
- [x] Create `_converters.py` with lazy-import converters and shared helpers (`_lazy_import`, `_unpack_rows`, `_meta_entry`)
- [x] Add optional dependencies in `pyproject.toml` (pandas, pyarrow, anndata, all)
- [x] Add `@overload` signatures for type-safe return types
- [x] Write 31 tests in `test/test_return_as.py` — all passing
- [x] Code review + simplify pass (extracted helpers, removed parameter sprawl, fixed no-op ternary, `copy=False` on float32 cast)
- [x] Push to `timtreis/feat/return-as-formats`, opened draft PR afermg/cp_measure#38
- [ ] Verify AnnData slot mapping works well with downstream scanpy/squidpy workflows
- [ ] Verify PyArrow schema metadata survives Parquet round-trip (`pq.write_table` → `pq.read_table`)
- [ ] Consider whether `obs_names` uniqueness should be enforced when multiple images are concatenated
- [x] ~~The featurizer branch (`feat/multi-mask-featurizer`) hasn't been merged to main yet~~ — resolved 2026-05-29: PR #32 MERGED to upstream main

## General

- [x] ~~`feat/multi-mask-featurizer` branch (PR #32) needs to be merged into main~~ — resolved 2026-05-29: PR #32 MERGED (`featurizer.py` now on upstream main)
- [x] ~~`bulk.py` has unused `get_core_measurements_3d()` — featurizer reimplements its logic via `_warn_and_filter_2d_only`~~ — resolved 2026-06-03: featurizer.py:217 now calls `get_core_measurements_3d()` (PR #50); `_3D_FEATURES` lives in bulk.py:45

## Cross-repo performance benchmark (3-tier, 5 implementations)

Work lives in `/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/`
(materialize_data.py, impl_bench.py, bench.sbatch, submit_all.sh, collect.py, REPORT.md;
batched: bench_batch.sbatch, submit_batch.sh, collect_batch.py, REPORT_BATCH.md).
Implementations: baseline (=cp_measure_fast/reference), fast (numba), speed (np-vec),
jax (cpu/a100mig/h100), rust. Data tiers in `data/{tiny,small,large}.npz` from cp_measure_fast/data.

- [x] Built 3-tier SLURM benchmark; ran 5 impls × 3 tiers (jax on cpu+A100mig+H100) + batched sweep
- [x] Single-thread pinned re-run (OMP/OPENBLAS/MKL/NUMBA/VECLIB=1; jax-cpu XLA single-thread). Results in REPORT.md
- [x] Inventory of all 5 implementations + actual backend per function: `tasks/implementation_inventory.md`
- [x] Built rust into the fast pixi env via maturin (editable cp312 wheel)
- [x] **fast (numba) intensity port** — new `cp_measure_fast/src/cp_measure/core/_intensity_numba.py`
      (single-pass moments, segment-sort quartiles/MAD, numba inner-boundary edge kernel);
      dispatched from `measureobjectintensity.py`. large 152ms→18.3ms (~372× vs baseline), bit-exact on real data.
- [x] **jax intensity port** — `cp_measure_jax/.../measureobjectintensity.py` `_get_intensity_jax_2d` +
      `_intensity_core_jax`/`_edge_core_jax` (segment ops + lexsort). Fixed float-index bug (see below).
- [ ] **CONFIRM jax GPU intensity timing** — the float64-index bug made `get_intensity` silently fall back to
      numpy (GPU never used; "47ms" was numpy-on-CPU). Bug fixed (int64 `starts`/`last` in `_q`). GPU re-test
      jobs submitted (H100 37201581, A100mig 37201582) — read `logs/cpm_jaxgpu_*.out` to confirm real GPU speed.
- [ ] **Re-run the full benchmark** with the new fast+jax intensity AND with `granularity_fullres`
      (subsample_size=1.0) — main defaults to subsample_size=0.25 so default-granularity is cheap; full-res is the
      heavyweight.
- [ ] **CRITICAL — impl_bench RUN_SET is wrong; reported fast/jax global speedups are UNDER-counted.**
      Verified by diffing each variant's core vs baseline + running all 12 functions:
      * `fast` accelerates ~ALL functions (every core module differs from reference): intensity, granularity,
        zernike, feret, radial_distribution, radial_zernikes, AND all colocalization (reference loops per-object via
        `apply_correlation_fun`; fast uses `bincount` — measured manders 40×, rwc 14×, costes 17× on large).
        RUN_SET only credited {granularity, zernike, radial_zernikes, feret} → fast global (1.3×) is far too low.
      * `jax` also accelerates `colocalization_pearson` (8 vs 274ms) and `costes` (19 vs 477ms) — both excluded.
      * `speed` RUN_SET={intensity} is INCOMPLETE — 2026-05-30 mining (`primitive_existence_matrix.md`)
        found `measuregranularity.py` is ALSO functionally changed (subsampling re-enabled); add granularity
        to speed's RUN_SET and treat its old granularity number as not-apples-to-apples with baseline.
      * ALL 12 functions in fast & jax are CORRECT vs baseline (Δ ≤ ~1e-6) — no crash/correctness bugs.
      FIX: drop the static RUN_SET — run all 12 functions for every variant and mark `=base` EMPIRICALLY
      (measured output ≈ baseline within tol), not from the inventory. This also prevents un-run functions from
      hiding bugs (which is exactly how the jax-intensity fallback went unnoticed).
- [ ] **Make jax `get_intensity` not silently swallow errors** — the broad `try/except` in the dispatcher hid the
      index bug. Consider logging on fallback or a debug flag so a broken GPU path is visible.
- [ ] Consider further jax-intensity optimisation if GPU still not winning (single-image is orchestration-bound:
      ~8ms GPU compute vs many host syncs / 21 device→host pulls). Batching amortises — see REPORT_BATCH.md.
- [ ] **rust texture regression** — rust `get_texture` is SLOWER than baseline (0.3× on small/large); investigate.
- [ ] `neighbors`/`overlap` (multimask, two-mask) were excluded from the benchmark — add if needed.
- [ ] Best-of-breed dispatch (route each function to fastest backend) → ~57× global ceiling on large; build the table.

## CORRECTED benchmark + CPU-vs-GPU economics (2026-05-30)

Corrected full benchmark DONE (all 13 functions incl. `granularity_fullres`, jax intensity called via direct
path, empirical correctness check). Results in `cp_measure_3tier_bench/REPORT.md`; collaborator summary in
`RESULTS_SUMMARY.md`. **Corrected global speedups (large 1080²): fast 8.40×, speed 1.24×, rust 1.82×,
jax-cpu 1.63×, jax-A100mig 33.87×, jax-H100 66.74×.** (Old wrong numbers were fast 1.3×, jax 1.64×, rust 4.0×.)
- [x] Smell hunt for hidden errors: confirmed the under-counting was pervasive (fast accelerates ~all functions,
      jax also pearson/costes) but NO correctness bugs except **rust `radial_zernikes` ~5-6% off baseline**
      (rust deprioritised). speed scoping later CORRECTED (granularity also changed — see
      `primitive_existence_matrix.md`). All in `tasks/lessons.md`.
- [x] Tile-size throughput study (`REPORT_TILE.md`, collect_throughput.py): mosaicked tiles 540/1080/2160/4320².
      jax-H100 throughput PEAKS at ~1-2k tiles (3.14 MP/s) and DEGRADES at 4k (2.44, +OOM on 20GB A100-MIG).
      Recommend ~1k² tiles. Output 3 in REPORT.md annotated that s/MP is NOT pixel-linear (mix of
      per-pixel/per-object/fixed costs) — don't rescale across tile sizes by megapixels.
- [x] CPU-cores vs GPU head-to-head (parallel_cpu_bench.py uses ProcessPoolExecutor; gpu_batch_bench.py).
      16-vCPU node fast-numba = **2.51 MP/s** (eff falls to 52% at 16 — it's 8 physical cores + 8 hyperthreads,
      NOT a bug/memory-wall; confirmed via diag_scaling.py topology: 4-8 physical cores + SMT siblings).
      1 H100 naive sequential = 2.64 MP/s; **packed (multi-process) plateaus at ~4.9 MP/s**.
- [x] GPU saturation (gpu_share.sbatch, with nvidia-smi util sampling): multi-process packing is a
      **memory-bound dead-end** — N=8 uses 69/80GB, throughput plateaus ~4.9 MP/s at only **54% mean GPU util**
      (N=4 = 11% util!). The H100 is half-idle even packed; host-side serial work (find_boundaries/bincount/sort)
      starves the SMs.
- [x] **CONSOLIDATED WRITEUP (2026-05-30):** all scattered reports merged into single
      `cp_measure_3tier_bench/BENCHMARK.md` (cross-language deliverable: headline table, Amdahl
      breakdown, per-function detail, tile throughput, batching+rust-t8 threading finding, CPU-vs-GPU
      cost, correctness, per-language verdict). RESULTS_SUMMARY.md banner-marked superseded.
- [x] AWS cost (subagent research, us-east-1, 2026-05-30): 16-vCPU c7i.4xlarge $0.71/hr; **single H100 now
      exists = p5.4xlarge $6.88/hr** (p5.48xlarge $55.04/8). Cost per 100k² image: CPU **$0.79** vs packed-H100
      **$4.39** → on AWS the CPU node is ~5.6× cheaper per result; GPU is 1.7× faster wall-clock. Budget GPU
      clouds (Lambda ~$2.49, RunPod ~$1.99/hr) narrow it to ~$1.3-1.6.

## Backend-ecosystem refactor (none / numba / jax / numba+jax) — cross-pollination

Full investigation: `tasks/backend_cross_pollination.md` (single source of truth for the refactor).
Goal: base cp_measure auto-routes each feature to the fastest installed backend; modular yet parallel;
numba separable so jax-only installs use a numpy fallback.

- [x] **Cross-pollination investigation DONE (2026-05-30)** — deep read of all 5 repos (baseline/fast-numba/
      jax/speed-npvec/rust) via 5 parallel subagents. Findings → `tasks/backend_cross_pollination.md`:
      per-function routing table, shared-primitive layer, numba modularisation plan, hazards.
- [x] **Key finding: the unifying abstraction is `segment_reduce(values, labels, n, op)` + `segment_quantile`.**
      Intensity, both Zernikes, all coloc, radial_distribution histograms, sizeshape moments are all instances.
      All 3 repos independently re-derived it (speed=numpy, jax=jax, numba=4 redundant copies). Extract once →
      per-function backend divergence collapses. Other shared primitives: `boundary_ijv`, `label_to_idx`,
      `convex_hull` (compute once, share across feret+both zernikes).
- [x] **Key finding: numba is already optional** in fast+jax (guarded imports + numpy fallbacks). The real hard
      dep is scipy/skimage/centrosome/mahotas (host-side in EVERY backend incl jax) — minimal jax-only install
      not achievable, and that's fine.
- [x] **Key finding: rust = inspiration only, no rust ships.** Lesson = "scatter once, accumulate running
      sums in one reduce loop, size buffers to data" (intensity 827×; rust is two-phase, NOT a single fused
      pass — corrected in `primitive_existence_matrix.md`); counter-example dense-256 GLCM (texture 0.21×).
      jax sparse-GLCM Haralick is pure-numpy & beats baseline → make it the `none` texture backend.
- [x] **DECIDED: do NOT port remaining 4 funcs (granularity/_fullres/radial_distribution/sizeshape) to rust** —
      zero new algorithmic knowledge (morphology/geometry where optimal algos already in numba/jax/skimage).
      Use writing the numba kernel as the forcing-function instead. radial_distribution + sizeshape are the two
      funcs NOT yet numba-accelerated in `fast` → that's where any unexploited fused-pass win lives.

### LOCKED DECISIONS (2026-05-30) — see backend_cross_pollination.md §7
- [x] Dependency floor: KEEP scipy/skimage/centrosome/mahotas mandatory (may rewrite core elements IF they
      bottleneck, but deps stay).
- [x] Tie-break: `>=`-last EVERYWHERE (compatibility required; bit-exact on real data, synthetic-only diff documented).
- [x] numba kernels stay SINGLE-THREADED (no prange/nogil); ALL parallelism lives in the batch layer.
- [x] **NO try/except anywhere.** Capability detected once at import via `importlib.util.find_spec` + flags;
      dispatch reads flags; resolved path runs DIRECTLY/unguarded and raises on error (deletes the
      silent-fallback bug class). Removes all existing import guards + the intensity try/except.

### Build-on-Alan's-PR-#49 + roadmap (2026-05-30/31 session)
- [x] **Analysed Alan's PR #49** (`afermg/cp_measure#49`, `feat/accelerator-backend`, OPEN, base=main): adds
      `set_accelerator()` global + `_dispatch()` in bulk.py; backends unwired (raise NotImplementedError).
      It nails OUR §1a dispatch seam (the `get_*_measurements` dicts) — Alan independently confirmed it.
      Orthogonal to the §3 primitive layer. **DECISION: build ON TOP of #49, don't compete.**
- [x] **Verified #49 semantics**: function-local `from cp_measure import _ACCELERATOR` DOES pick up later
      `set_accelerator()` rebind (runtime-switchable, good). Unwired backends raise loudly (matches §7.4).
- [x] ~~**PR #49 — one-line fix to suggest:** `_dispatch` falls through to implicit `return None`...~~
      — resolved 2026-06-03 on `feat/accelerator-numba-intensity`: `_dispatch` (bulk.py:85-88) now ends with a
      trailing `raise ValueError(...)` (exhaustive); `find_spec` detection also landed (`_detect.py` HAS_NUMBA).
- [ ] **Gaps in #49 vs our plan (raise as discussion, don't block merge):** (a) it's a coarse GLOBAL
      single-backend switch, not per-function best-of-breed routing — a global "jax" routes costes +
      granularity_fullres to jax too (both are traps); the ~57× ceiling needs per-function routing.
      (b) no `find_spec` capability detection (selection isn't validated against what's installed).
      (c) `"faster"` backend naming is ambiguous + conflicts with our plan to fold `speed` into `none`.
- [ ] **Refined PR roadmap (proposed back to Alan)** — two reorderings vs Tim's Slack sketch:
      **(1) batch-shape the kernels from the FIRST function PR** (single image = batch-of-1) so we don't
      build single-image numba/jax kernels then rewrite them all for batching (the double-build trap).
      **(2) extract the §3 primitive layer BEFORE the per-function rollout** or each backend PR re-derives
      segment_reduce. Sequence: PR1 scaffold(#49) → PR2 routing+find_spec+correctness-harness+decide
      set_accelerator shape (str vs per-func map) → PR2.5 primitive layer (batch-shaped) → PR3.. numba
      (1/func, batched; lead with radial_distribution+sizeshape — the un-numba'd funcs — and intensity) →
      PR4.. jax (value-gated behind batching) → PR5 `profile_accelerators(sample)` writes routing table.
- [ ] **Reconsider §7 resolver decision:** Tim's PR5 `profile_accelerators(<data sample>)` (dynamic,
      per-function) is arguably BETTER than the static best-backend table, because throughput is NOT
      pixel-linear (best backend flips with tile size/object count/hardware; jax-cpu granularity_fullres
      trap). Synthesis: ship static defaults, let profiling override. Flip §7 recommendation toward PR5.

### Primitive existence matrix (deep-dive deliverable, 2026-05-31)
- [x] **Mined all 5 repos (5 parallel agents) for which primitive exists in which language** →
      `tasks/primitive_existence_matrix.md`. **Bimodal finding:** only `segment_reduce` + `segment_quantile`
      genuinely need 3-language impls (Zernike-complex-sum = weighted segment_reduce). `label_to_idx`,
      boundary host-extraction, `convex_hull`, host helpers are numpy-only in EVERY backend incl jax →
      dedup into one helper each, not per-language. jax adds ZERO live primitives below segment_reduce
      (`_perimeter_2d`/`_radial_histograms` dead); numba adds exactly ONE (`_boundary_ijv`). ⇒ PR2.5 is
      small: 2 dispatched primitives + ~4 shared numpy helpers + 1 numba kernel + numpy sparse-GLCM.
- [x] **Fixed contradicted notes** (lessons.md, backend_cross_pollination.md, todo.md, implementation_inventory.md
      + 2 memory files): speed≠intensity-only (granularity also functionally changed → its benchmark number
      not apples-to-apples); rust intensity is two-phase NOT single-pass; rust hull computed 3× not once;
      rust feret O(n²) not rotating-calipers; numba has 4 Zernike kernels not 2.
- [ ] **Benchmark-validity follow-up:** add `granularity` to `speed`'s RUN_SET and re-measure (speed's
      `measuregranularity.py` re-enables subsampling baseline had off — old `speed 1.24×` is suspect).

### Open follow-ups from the investigation
- [ ] **Correctness conventions to pin before porting Zernike:** baseline's exact `atan2` arg order + radial_zernikes
      normalization denominator (all-pixels vs in-disk). Rust drifts 5-6% here; encode the baseline convention in the
      shared Zernike primitive and assert all backends match.
- [ ] **Wire up or delete dead jax kernels** `_perimeter_2d` and `_radial_histograms` (defined, never called;
      numpy bincount runs instead).
- [ ] **Resolver granularity** (still open, low priority): static best-backend table (recommended) vs measure-once-at-import.
- [ ] **Extract the shared-primitive layer first** (§3): `segment_reduce`/`segment_quantile`, `boundary_ijv`,
      `label_to_idx`, `convex_hull`. Consolidate numba's 4 Zernike kernels → 1; lift speed's intensity rewrite as the
      numpy `segment_reduce`; lift jax sparse-GLCM as numpy texture.

## PR #54 — numba intensity (MERGED 2026-06-03)
- [x] Addressed Alan's review on `afermg/cp_measure#54`: reworded the absent-numba
      `RuntimeError` ("you can install it via…"); inlined `label_to_idx_lut` import from
      `primitives.segment` and dropped the `primitives/__init__` re-export (kept the
      `core/numba/__init__` re-export as the backend API surface — replied with rationale).
      Commit `69c7e48`. PR #54 merged to main (`4ca0a35`).

## Numba granularity backend (PR #56 — OPEN draft, branch `feat/accelerator-numba-granularity`)
Detailed plan + results in `tasks/numba_granularity_plan.md`. Commits: `b845f1d` (backend),
`af73608` (/simplify cleanup), `8d84546` (ruff format). CI GREEN (lint + mypy 3.10-3.14 + tests).

- [x] Built numba granularity backend on the #49/#54 dispatch seam. New files:
      `core/numba/_granularity.py` (VHG row-decomposed disk erosion/dilation, 5-tap disk(1),
      **Vincent 1993 FIFO-hybrid reconstruction**), `core/numba/measuregranularity.py` (wrapper:
      numba 2D path, 3D→numpy baseline, cascaded mask, point-query readback + dense bincount),
      `primitives/shapes.py` (`to_bzyx` canonical (B,Z,Y,X) batch normaliser). Wired in
      `bulk._numba_registries` + `core/numba/__init__`.
- [x] Tests: `test_granularity_kernels.py` (76, bit-exact vs skimage), `test_granularity_backend.py`
      (12, vs baseline within tol), `test_primitives_shapes.py` (11), +granularity dispatch in
      `test_backend_correctness.py`. Full suite 177 passed.
- [x] Benchmark (1080², 144 obj): **fullres 6.86×, default 12.58×** vs numpy baseline.
- [x] /simplify pass: np.full init, modulo→running-counter in VHG loop, removed redundant
      per-iteration `ero.max()` scan (recon handles all-zero cheaply), to_bzyx docstring note.
- [x] Provisioned local test env: `pixi global install uv` → `uv sync --extra numba --extra test`
      → `uv run pytest`. (uv at `/home/icb/tim.treis/.pixi-home/bin/uv`.)
- [x] **FOLLOW-UP PR #57 (OPEN draft, `feat/intensity-bzyx`, STACKED on #56): migrate intensity
      onto `to_bzyx`.** `get_intensity` is now a thin `to_bzyx` wrapper; per-volume work in
      `_intensity_volume`. Single→dict, 4D/list→list-of-dicts. Golden guard = numba-vs-numpy test
      stays green (2D+3D). mad_frac = `2 if Z==1 else 3` (= old `pixels.ndim` except a degenerate
      single-slice volume, documented). 2D-mask-on-stack broadcast passes through unchanged. +batch
      tests; 180 passed, ruff clean. **Retarget PR base to `main` once #56 merges.**
- [x] **/simplify cleanup on #57** — removed the now-dead `masked_image = pixels` alias (refactor left
      it as an unconditional alias; replaced its 2 uses with `pixels`). Committed `5d08a15`, pushed to
      `feat/intensity-bzyx`. Ruff clean, tests pass. (The 4-agent /simplify review otherwise found the
      diff clean.)
- [ ] **Coordinate with PR #55 (Alan's numpy speedups)**: #55 rewrites the default numpy
      intensity/coloc/sizeshape. When it lands: (a) re-baseline the numba intensity speedup vs the
      NEW numpy, (b) verify `test_backend_correctness.py` intensity stays green — #55 changes the
      intensity result-dict key construction (tuple-list → f-string), a silent-break hazard for the
      numba-vs-numpy assertion.
- [ ] Optional further granularity opt (diminishing): bg-opening disk(10) is now the largest single
      fullres chunk (~263 ms); split `_disk_reduce` into ero/dil kernels to drop the `is_max` branch.
- [ ] **Follow-up (cross-backend, low priority): carry per-element original ndim out of `to_bzyx`**
      instead of inferring `2 if Z==1 else 3` in each backend. **DECISION 2026-06-03: keep the
      inference as-is** (Tim: "keep it like this" — do NOT add an xfail, do NOT drop the note).
      Affects ONLY the `Intensity_MADIntensity` feature (the `1/ndim` quantile), and ONLY for input
      shape `(1,Y,X)` — a single-slice 3D *volume*: baseline `pixels.ndim==3` → MAD at 1/3-quantile;
      our `Z==1` → 1/2 (the standard median MAD, which actually matches the baseline's own docstring).
      Normal 2D `(H,W)` and Z>1 volumes are bit-identical. Untested (conftest DEPTH_3D=8, so the
      golden test never hits Z=1 → silent). Root cause: `to_bzyx` erases (H,W) vs (1,Y,X). The deeper
      fix (return per-element ndim) would also fix granularity's identical `Z==1` inference, but it
      changes `to_bzyx`'s pinned pure-normaliser signature → its own cross-backend PR, NOT a feature PR.
- [ ] **Note (deferred, rule-of-three): extract a `batched(masks, pixels, per_volume_fn)` helper**
      into `primitives/shapes.py` once a THIRD backend repeats the `to_bzyx → loop → unwrap` envelope.
      Now at 4 functions across 3 modules; trigger = once intensity (#57) is also on to_bzyx so all
      consumers are uniform. Still deferred (the differing arg counts / per-batch coeffs hoist make a
      generic helper non-trivial). Leave the explicit loop in each backend for now.

## Numba zernike + radial_zernikes (PR #58 — OPEN draft, branch `feat/numba-zernike`, base #59)
Plan + A/B justification in `tasks/numba_zernike_plan.md`. Commits: backend + /simplify dedup.
- [x] **Built numba zernike + radial_zernikes in ONE PR**, sharing a FUSED kernel
      `core/numba/_zernike.py::zernike_moments` (per pixel: Horner basis eval over r² with centrosome
      LUT coeffs + `z=ym+i·xm` azimuthal recurrence + strict `r²>1` cutoff, AND weighted-complex
      scatter-sum into `(n,K)` vr/vi — no `(M,K)` intermediate). Wrappers
      `core/numba/measureobjectsizeshape.py` (`get_zernike`, weight=1, /π·r², magnitude only) +
      `measureobjectintensitydistribution.py` (`get_radial_zernikes`, weight=pixels, /pixel-count,
      +phase=arctan2(vr,vi)). Shared `zernike_moments_per_object` helper (label_to_idx_lut + nonzero
      + min_circle + coords). Centrosome kept HOST-side for LUT + `minimum_enclosing_circle`.
      2D-only (3D→{}); batch via to_bzyx. **zernike 28.7×, radial 20.9×** (1080²/144obj).
- [x] **A/B proved the basis reimplementation is worth it**: fused kernel 6.3ms vs centrosome
      `construct_zernike_polynomials`+numba-sum 61.8ms (~10× kernel win; centrosome's construct alone
      is 25.7ms). Boundary: reimplement the mechanical/test-guardable EVALUATION; IMPORT the
      numerically-sensitive coeffs + the geometry (min_circle) → NOT the rust drift situation.
- [x] **Big single-image opts** (the fused kernel alone left radial at 1.27×): replaced
      `masks_to_ijv` (347ms per-label scan) with one `nonzero`; replaced `np.unique` (8.9ms) with
      `label_to_idx_lut` (2.3ms); hoisted degree-only LUT out of the per-image loop.
- [x] code-review (high effort, 7 angles): core math sound/bit-exact vs centrosome; only finding =
      the `(1,Y,X)` divergence (same root cause as the to_bzyx ndim follow-up above), low severity,
      direct-call-only. 1-px-object NaN is baseline-consistent (not flagged).
- [x] **radial_distribution numba lane — SHIPPED PR #63 (2026-06-04), 21.0×.** `_radial.py`
      (`geodesic_chamfer_fifo` bit-exact vs propagate + fused `radial_object` per-crop kernel) +
      `get_radial_distribution` (to_bzyx, 2D-only, per-object crop+1px pad = #22 fix). scipy EDT kept.
      1004→48ms (9.5× → 21× after folding host glue + maxpos into the kernel; geodesic was 35× but
      Amdahl-capped at ~9% of the function). Golden numba(multi)[k]==numpy(isolated k) on unique-centre
      objects + #22 independence; suite 118 green. Stack #59 → #58 → #63. Flag to afermg: propagate
      replaced by bit-exact numba kernel (no behaviour change beyond #22).
- [x] **texture numba lane — SHIPPED PR #64 (2026-06-04), 16.8× (4.8×→16.8× after kernel opt).**
      OPT: collapsed the two per-direction O(fm1²≈256²) loops bit-exact — symmetric GLCM ⇒ HXY1==HXY2==2·HX
      (cross-entropies collapse to marginals); T computed during the GLCM build not a separate fm1² sum.
      1984→118ms. `_texture.py::haralick_object`
      (symmetric GLCM bit-exact vs mahotas.cooccurence + ignore_zeros + 13 Haralick formulas, 2D=4dir/
      3D=13dir, GLCM sized crop.max()+1) + `measuretexture.py` wrapper (to_bzyx, host img_as_ubyte/
      regionprops kept, reuse F_HARALICK). No #22 analogue (already per-object → golden vs numpy direct).
      Step-0: haralick=99% reducible, GLCM bit-exact. 1938→402ms; 4.8× (fm1²≈256² feature passes vs C
      mahotas — could optimise the HXY loops if pushed). Suite 100 green. Stack #59 → #64. Plan
      `tasks/numba_texture_plan.md`. Remaining core lanes: sizeshape, feret (geometry tail — profile first).

## Merged-stack tiered speedup (2026-06-04) — full report `tasks/MERGED_STACK_BENCHMARK.md`
- [x] Built integration branch `integration/all-numba` (#59 + all 6 PR tips; merge = trivial append
      conflicts only) and benchmarked every function numpy-vs-numba on real tiers (tiny/small/large),
      single-thread pinned. **JOINT featurize speedup: tiny 6.3× / small 9.4× / large 22.8×** (default
      cfg; ~10× large if granularity@fullres). Driven by intensity (260× large, 43% of baseline) + coloc
      (63-97×). texture is the laggard (5×); sizeshape =1.0× (NO-GO). Bench `tasks/bench_tiered_merged.py`.
- [ ] **Re-baseline vs Alan's PR #55** before quoting externally — #55 rewrites numpy intensity/coloc/
      sizeshape (the top contributors), will shrink these factors.

## Deep per-PR perf mining (2026-06-04, autonomous) — full report `tasks/PR_PERF_ANALYSIS.md`
- [x] Mined every open numba PR (7 agents) + EMPIRICALLY validated each proposed opt end-to-end.
      META: lanes at practical floor for the common case; component wins mostly don't translate.
- [x] **SHIPPED: coloc pass-fusion** (PR #60, c22d856) — pearson/manders −3-4%, bit-identical, 114 suite green.
- [x] Validated-and-reverted non-wins: texture sparse-stack (regresses common case), granularity label_mean
      (negligible), zernike vi-skip (breaks vectorization), radial %→branch (no win). Don't re-try these.
- [ ] **RECOMMENDED (real wins, need sign-off — gated on conventions/accuracy):**
  - [~] **coloc shared-flatten — POC says only ~1.12× (NOT 3×), likely NOT worth it.** rwc argsort
        dominates coloc and is paid once anyway (`tasks/poc_coloc_shared_flatten.py`); flatten-once saves
        only the 2 cheap non-rwc passes. Worth it ONLY if a pipeline runs non-rwc coloc without rwc.
  - [x] **granularity bilinear gather — SHIPPED PR #56 (89dde43), 2.88×** (741→257ms default). Within-tol
        (3e-14 ≪ lane's rtol=1e-6), backend golden green, +regression test. Flagged in PR comment as a
        machine-eps resampling change (trivial revert) — granularity was never bit-exact.
  - [x] **granularity reconstruction: triple-raster + int32-packed FIFO — SHIPPED PR #56 (c8b9414).**
        From the 2026-06-04 cross-repo investigation (cp_measure_fast findings #18/#34): 3 fwd+bwd raster
        pairs before FIFO seeding (cuts seeds ~50-95%) + `(i<<16)|j` int32 coords (shift/mask decode, no
        IDIV/MOD; half cache). **Bit-IDENTICAL** (result independent of raster-pass count; kept the
        overflow-proof N+1 ring + dedup flag — measured faster than fast's no-dedup 12·N variant). ~1.5×
        reconstruction; end-to-end ~3% default / ~8% fullres (recon is ~20% / ~81% post-gather — re-profiled,
        Amdahl-honest). Non-seeding passes factored into `_geodesic_raster_fwd/_bwd`. +long-geodesic test,
        180 suite green. Scratch: `tasks/profile_granularity_recon.py`, `tasks/ab_granularity_recon{,2}.py`.
        CROSS-REPO META: all 8 lanes checked vs fast/jax/speed/rust — this was the ONLY unharvested win;
        every other lane at/above reference (sizeshape NO-GO independently confirmed: fast's
        `_sizeshape_numba.py` is feret-only, numba'd zero moments).
  - [ ] **texture adaptive sparse stack** — ship the touched-cell stack only when `16*npix < fm1²`
        (dense fields, 1.2-1.65×), keep dense path otherwise (avoids the −10% common-case regression).
  - [ ] **zernike shared geometry** across get_zernike + get_radial_zernikes (one min_circle/pass when both run).
  - [ ] **intensity bincount-for-(lut,n)** only (drops find_objects scan); full coloc bincount port unsafe
        (over-counts non-finite pixels intensity drops); std must stay two-pass.

## Numba feret + sizeshape lanes — the geometry tail (2026-06-04, autonomous session)
- [x] **feret — SHIPPED PR #65** (`feat/numba-feret`, base #59, sibling). Step-0 (`tasks/profile_feret.py`):
      bottleneck is plumbing not geometry — `utils.masks_to_ijv` per-label scan = 86.1%, hull 13%, feret
      0.6%. `core/numba/_feret.py::_boundary_ijv` = one numba pass doing TWO bit-exact reductions:
      (1) counting-scatter replaces masks_to_ijv (same rows/order); (2) feeds hull only boundary pixels
      (hull(obj)==hull(boundary); 78.9→6.2ms, ~17× fewer points). **43.4× (618→14ms), bit-exact.** 8-conn+
      edge load-bearing. Kernel+wrapper in `_feret.py` (dodge #58 measureobjectsizeshape.py collision).
      Empty-mask: numpy baseline also raises → test asserts both raise. 14 tests, /simplify applied
      (return counts→nonzero indices; drop offs copy). Stack #59 → #65.
- [x] **sizeshape — STEP-0 NO-GO, documented, NOT ported** (`tasks/numba_sizeshape_plan.md`,
      `tasks/profile_sizeshape*.py`). regionprops 73% (hot primitive = convex hull +187ms/34% = import
      geometry; reducible moments +4.7ms negligible) + EDT-radius 25% (EDT 13% import + reductions 12%).
      Amdahl ceiling 1.13×. No reimplementable hot primitive (unlike radial/texture) → don't force a
      low-value kernel. ALL 9 core lanes now resolved (8 numba shipped + sizeshape no-go).
- [ ] **(future, non-numba, low pri)** if sizeshape ever a bottleneck: batched centrosome
      `convex_hull_ijv` for all objects vs skimage per-object hull — risks hull-area divergence, out of
      scope for the numba port. Recorded not actioned.
- [ ] ~~**radial_distribution numba lane — REVIVED BY POC (2026-06-04), BUILD IT.**~~ (DONE — see above) Step-0 showed geometry
      = 97% (propagate 80%), so first verdict was "kill it." BUT the POC (`tasks/poc_numba_geodesic.py`)
      found propagate(zeros,w=1) IS the 1/√2 chamfer SHORTEST-PATH, and a numba raster-sweep geodesic
      matches it BIT-EXACTLY (maxdiff 0.0 on convex/concave L/U/ring/#22) at 35.8× (12 vs 428ms). Shortest-
      path is algorithm-independent → not a metric change, not boundary-violating. REVISED LANE: per-object
      crop (fixes #22) + scipy EDT for d_to_edge (keep exact Euclidean) + numba chamfer geodesic for
      d_from_center + numba reductions → ~910ms geometry to ~40-50ms + #22 fix, bit-exact to isolated-object
      ref. Stack on #58. Flag to afermg (propagate→bit-exact numba kernel). Full plan + POC reversal in
      `tasks/numba_radial_distribution_plan.md`.
- [ ] ~~**radial_distribution lane — PLANNED**~~ (superseded by the step-0 verdict above):
      branch `feat/numba-radial-distribution` STACKED ON #58** (the numba
      `measureobjectintensitydistribution.py` module lives there alongside radial_zernikes). Stack:
      #59 → #58 → radial. Decisions: (1) FIX Issue #22 via per-object crop+1px-pad (correct per-object
      semantics; removes color_labels; the ONE lane that intentionally diverges from main on multi-object
      — it's a documented bug FIX, flag in PR + coordinate with afermg who raised #22 upstream to scipy);
      (2) import centrosome geometry (distance_to_edge/propagate/maximum_position), reimplement only the
      sparse-histogram/wedge-CV reductions in `core/numba/_radial.py::radial_per_object` (error_model
      numpy). PROFILE FIRST (step 0): geometry may dominate (we keep it), so measure the ceiling before
      building. Golden anchor: numba(multi)[k] == numpy(isolated single-object k); + independence test.
      Aligns with #57/#58/#60/#62 conventions (to_bzyx 2D-only, _radial.py kernel, primitives reuse,
      registry/__init__ append pattern, serial kernel) EXCEPT the deliberate #22 divergence.

## Numba colocalization lane (full plan: `tasks/numba_colocalization_plan.md`)
- [x] **PR A #60 `feat/numba-coloc` (OPEN draft, base #59, sibling to #56/#57/#58) — SHIPPED 2026-06-03:**
      `pearson`/`manders_fold`/`rwc` + ride-along `overlap` via one grouped pair-flatten
      (`flatten_pairs_grouped`) + one fused `coloc_per_object` njit kernel. bzyx via `to_bzyx`-called-
      -twice (shared mask, reuse `unwrap`). Value-vector-only → no 2D/3D branch, no `(1,Y,X)` divergence.
      31 tests added (golden 2D/3D + single/batch + cont/tie regimes; kernel units); full suite 111 green,
      lint clean. KEY FINDING: integer-dtype pixels make the numpy reference overflow uint8 `fi*si`
      (overlap>1) and run the slope `lstsq` in float32 — our float64 path is correct but diverges, so
      golden tests use integer-VALUED floats for rank ties (documented in module + test).
- [ ] ~~**PR A original plan**~~ (superseded by the SHIPPED entry above):  `pearson` + `manders_fold` +
      `rwc` + ride-along `overlap`. All four from ONE fused `coloc_per_object` njit kernel +
      a grouped pair-flatten primitive (counting-sort, reuses `label_to_idx_lut`). bzyx-normalised
      like #57/#58 — `to_bzyx` called twice (shared mask, reuse the one `unwrap`) since coloc is a
      `(pixels_1,pixels_2,masks)` TRIPLE. Coloc is value-vector-only → NO 2D/3D branch and the
      `(1,Y,X)` MAD-divergence cannot occur (clean property). Golden test 2D/3D + single/batch +
      uint8/uint16 (rwc dense ranks). Wire into `__init__`/`_numba_registries`; flag the `overlap`
      registry asymmetry to Alan (numpy `_CORRELATION` lacks it).
- [ ] **PR B `feat/numba-coloc-costes` (STACKED on #60) — PLANNED, full plan `tasks/numba_costes_plan.md`:**
      `costes` — closed-form a,b + full numba iterative search, ALL 3 modes (bisection M_FASTER + linear
      M_FAST/M_ACCURATE). New `core/numba/_costes.py::costes_per_object`, reuses #60's `labels_to_offsets`
      + `flatten_pairs_grouped` (no sort). Skip the DEAD `calculate_threshold`. scipy-order clamped
      pearson-on-subset. `thr` accepted but UNUSED (only fed the dead call). RESEARCH FINDINGS: (1) costes'
      real multi-iteration search needs integer dtype (scale 255/65535) but the reference overflows
      `z=fi+si` there → EXACT on float64 only (scale=1, trivial search), integer-dtype divergence
      documented like #60's overlap. (2) Verify decoupled: kernel control-flow vs a Python transcription
      at scale=255 (exact by construction), pearson-vs-scipy (rtol), end-to-end golden float64 2D/3D/batch/
      3-modes. (3) Iteration sensitivity (branches on pearson vs 0/0.25/0.35/0.45 cutoffs) is the one
      inherent fragility — match scipy's formula order to minimise branch-flip risk.
- [ ] Baseline win: numpy `apply_correlation_fun` materialises an `(N,H,W)` bool stack and re-indexes
      the whole image per object via `scipy.ndimage` (labels all-ones, lrange=[1], hence `[0]`).
      Expect a large speedup; benchmark before quoting a factor.
- [x] **OPT deep-dive (2026-06-03):** `bincount` floor win (`labels_to_offsets` + single-scatter flatten)
      → pearson 31.6×/manders 62.0×/overlap 67.7×/rwc 5.6×. Lives IN #60 (briefly split to #61 then
      recombined — perf belongs with the feature; keep opt coloc-only, do NOT touch intensity). rwc
      confirmed sort-bound: 4 exact rank strategies tested, current argsort wins (lexsort 2.3× /
      sort+searchsorted 1.8× / prealloc no-op). Stack: #59 → #60.
- [ ] **CROSS-CUTTING opt follow-ups (valuable, deferred) — see `tasks/numba_opt_followups.md`:**
      (1) port the `bincount`/single-scatter prep to the merged numba `intensity` backend (#54);
      (2) shared-flatten across the 4–5 coloc features (featurizer re-flattens + re-runs the fused
      kernel per feature on the same image) — the deferred shared-flatten/batch-layer concern, NOT a
      per-function change and NOT in-kernel parallelism.

## PR-stack restructure: extracted `to_bzyx` into its own PR #59 (2026-06-03)
- [x] **Extracted `primitives/shapes.py` (`to_bzyx`) + its test into PR #59** (base `main`) so the
      shared contract reviews first and unblocks the fan-out. Rebased #56/#57/#58 onto #59 (siblings).
      Final stack: **#59 (to_bzyx, base main) → #56 granularity / #57 intensity / #58 zernike (base #59)**.
      Each PR's diff is now its own content only. Runbook: `tasks/pr_extraction_plan.md`.
- [x] **/simplify on #59**: XOR-flattened the both-must-be-X guards, dropped the dead `unwrap`
      length-guard (kept the closure), parametrized the 4 raises tests. Amended #59 + re-rebased the
      3 features (conflict-free via `--onto`).
- [ ] **Merge order:** #59 FIRST → then #56/#57/#58 in any order (each resolves a trivial one-line
      append conflict in `_numba_registries`/`__init__`/`test_backend_correctness` as it lands;
      GitHub auto-retargets feature bases to main after #59 merges). Retarget #57/#58 to main once #56
      merges if not auto-done.
- [ ] **Local `backup/{56,57,58}` branches** point at pre-rebase commits — safe to delete once the
      PRs are confirmed good on GitHub.

## Batching rewrite (deferred — append after cross-pollination)

- [ ] **BUILD: unified `featurize(masks, pixels, batch_size=...)` with batching as the SINGLE code path**
      (single image = batch of 1 → eliminates single/batch divergence, the bug class that bit us twice).
      GPU backend: batched jax kernels (pad object dim to max, per-image segment offsets, one global lexsort with
      image key, move find_boundaries/bincount on-device). CPU backend: loop/multiprocess per-image. Output:
      single dict for 2D in, long-format table (image_id+object_id) for batch → plugs into existing `return_as`.
      **Rationale:** in-process batching is the only way to reclaim the GPU's idle ~46% (one VRAM footprint, fused
      work) — multi-process packing is exhausted at ~5 MP/s/54% util/VRAM-bound. Could ~2× GPU throughput →
      improve BOTH wall-clock AND $/result (possibly flipping the cost verdict vs CPU).
- [ ] **NEXT STEP agreed: prototype batching on `intensity` first** (jax segment-op version already exists in
      cp_measure_jax). Make get_intensity accept (H,W) or (N,H,W) through one batched path; assert
      batch-of-1 == current single-image result; measure batched throughput vs the multi-process 4.9 MP/s ceiling.
      Then roll out function-by-function behind the unified featurize. Caveats: texture GLCM is VRAM-hungry
      (per-function batch caps); N=1 carries minor batching overhead (price of one reliable path).

## `legacy` convention toggles (2026-06-05 session) — two NEW PRs against main
Both reproduce old/CellProfiler behaviour behind `legacy=True`, with the corrected behaviour as the
new default. Plan: `tasks/legacy_toggles_plan.md`; candidate mining: `tasks/legacy_toggle_candidates.md`.

- [x] **PR #67 `feat/intensity-legacy` → main (OPEN): intensity quartile/MAD convention toggle.**
      `legacy: bool = False` on numpy + numba `get_intensity`. False = `numpy.percentile` `(n-1)*q`
      quartiles + textbook-median MAD (NEW DEFAULT); True = CellProfiler `n*q` + `(1/ndim)`-quantile MAD
      (today's main, byte-for-byte). numpy parameterizes the existing cumsum-offset block via
      `span = areas if legacy else (areas-1)`; numba via `_interp(..., legacy)`. Affects ONLY the 4
      keys `Intensity_{Lower,Upper}Quartile/Median/MAD`. Wired prominently through bulk
      (`_LEGACY_FEATURES=("intensity",)`, `_apply_legacy`, `get_core_measurements(legacy=)`) + featurizer
      (`make_featurizer_config(legacy=)` → `config["legacy"]` → featurize). Golden test parametrized over
      both values + value anchor on `[0,1,2,3]`. 2× /simplify. **Supersedes Alan's `legacy_mad`.**
- [x] **PR #68 `fix/radial-per-object-22` → main (OPEN, closes #22): per-object radial in numpy baseline.**
      `get_radial_distribution` now measures each object on its own cropped + 1px-padded `== label` mask
      (reusing the SAME centrosome algorithm, extracted as private `_radial_distribution_image`), so each
      object's result equals it computed in isolation. `legacy=True` = the CellProfiler/centrosome
      whole-image result (the #22 leak is rooted upstream in scipy, per Alan's own issue comments).
      Tests: #22 independence (bit-exact), leak-exists, odd-centre new==legacy. 1× /simplify + docstring.
- [ ] **MERGE SEQUENCING:** #67 first (independent, against main) → **Alan rebases #55 on #67** (drops his
      `legacy_mad`, keeps the intensity perf restructure + coloc/sizeshape; #67 fixes #55's RED macOS+ubuntu
      CI — the quartile `n*q` vs `(n-1)*q` mismatch between numba `_interp` and #55's `numpy.percentile`).
      Then #68. The perf win in #55's intensity is the per-object-crop restructure, NOT `numpy.percentile`.
- [ ] **AFTER #67 + #68 both land:** add `"radial_distribution"` to `bulk._LEGACY_FEATURES` so the single
      `legacy` umbrella covers radial too (one `make_featurizer_config(legacy=True)` = old behaviour everywhere).
      (Can't be done in #68 — `_LEGACY_FEATURES` lives in #67, not on main yet.)
- [ ] **Radial #22 status:** #63 (numba lane) fixes #22 NUMBA-only and is an unmerged draft; #68 fixes the
      NUMPY default. No speedup PR touches radial (verified). Once both land, backends agree on the new default.
- [ ] **CHANGELOG/README breaking-change note** for both toggles (intensity 4 quartile keys + radial
      multi-object results change by default; `legacy=True` restores). No CHANGELOG file exists yet.
- [ ] **Deferred / not-toggle-worthy** (from `tasks/legacy_toggle_candidates.md`): scipy `maximum_position`
      tie-break (no stable legacy value → accept+document); integer-dtype coloc/costes overflow (bugs, fix
      don't toggle); `(1,Y,X)` MAD, granularity within-tol, zernike atan2 (resolved/within-contract).

## Granularity reconstruction opt (2026-06-05) — shipped to PR #56
- [x] **Triple-raster + int32-packed FIFO** in `_granularity.py::reconstruction_by_dilation_2d` (commit
      `c8b9414` on `feat/accelerator-numba-granularity`). 3 fwd+bwd raster pairs before FIFO seeding + `(i<<16)|j`
      int32 coords (shift/mask decode). Kept the overflow-proof N+1 ring + dedup flag (measured FASTER than
      fast's no-dedup 12·N variant). **Bit-identical.** Clean X-vs-baseline (1080²/144obj, pinned):
      fullres 8.26×→10.97× (numba 1.33× faster); default ~6× (recon only ~20% there post-gather, so ~unchanged).
      From cross-repo mining (fast findings #18/#34) — the ONLY unharvested in-kernel win across all 8 lanes.

## Cross-cutting investigations (2026-06-05) — reports in tasks/
- [x] **Recovered the autonomous overnight session** from `.jsonl` logs (node died mid-handoff at 07:22; all
      work was committed/pushed first — nothing lost). Reconciled UTC-vs-CEST timestamp confusion.
- [x] **Cross-repo legacy-toggle mining** (`tasks/legacy_toggle_candidates.md`) — only intensity quartile +
      radial #22 are genuine toggles; the rest are bugs/accept-and-document.
- [x] **Merged-stack tiered benchmark** (`tasks/MERGED_STACK_BENCHMARK.md`): built `integration/all-numba`
      (all 6 PR tips merge with only trivial append-conflicts) and benched every function numpy-vs-numba on the
      real tiers. **JOINT featurize speedup: tiny 6.3× / small 9.4× / large 22.8×** (intensity 260× drives it;
      texture 5× laggard; sizeshape 1.0× NO-GO). vs CURRENT main, not #55 → re-baseline before quoting externally.
- [x] **Batching conformance + scaling** (`tasks/bench_batching_conformance.py`, `bench_multiprocess_images.py`):
      all 12 numba funcs honour the `to_bzyx` batch API (single→dict, list/4D→list, batch-of-1==single). But the
      in-process batch loop is **1.0× (API uniformity only, serial loop)**. Real throughput win = process-pool over
      images: **8.86× at 16 workers** (124.6 img/s), efficiency 100%→55% (10 physical cores + HT). 10k images:
      ~12min(1 core) → ~1.3min(16) → shard across SLURM nodes for more.

## Default-numpy perf lane + numba sizeshape (2026-06-06 session)

Recovered a dead-node session (zernike numpy work survived in the working tree) and pushed the
default-backend numpy speedups to completion, then built the numba sizeshape lane.

### numpy default-backend PRs (all off `main`, CI green unless noted)
- [x] **PR #74** `perf(sizeshape): vectorize get_zernike` — masked-basis Horner + `numpy.bincount`
  segment-sum replacing centrosome's full `(H,W,K)` scatter. **8.6× at typical density** (3.7×
  dense → 44× sparse; speedup is foreground-fraction dependent). Bit-exact (0.0, machine-eps).
  Shared helper `_zernike_scores` in `utils.py`. Switched `range(1,n+1)` → actual unique labels.
  Body softened from "8×" to the measured range. /simplify'd (mgrid→nonzero, add.at→bincount).
- [x] **PR #75** `perf(radial): vectorize get_radial_zernikes` (~2×) — reuses `_zernike_scores`
  (weight=pixels). **Stacked on #74** (base `perf/zernike-vectorize`); CI runs after #74 merges &
  retarget to main. Also FIXES a latent `IndexError` crash on non-contiguous labels (`ij[label-1]`).
- [x] **PR #76** `perf(granularity): fuse per-step upsample+mean into one sparse operator` (~2–3×)
  — `_make_fused_upsample_mean` precomputes the bilinear-restore + per-object-mean as one sparse
  mat-vec (the per-iteration `map_coordinates`+`ndimage.mean` were 55% of the loop). ~1e-12 (within
  the lane's rtol=1e-6). 2D only; golden test = verbatim copy of the prior impl. /simplify'd.
- [x] **PR #77** `perf(sizeshape): scatter-based spatial moments + inertia` (1.6×, 352→220 ms) —
  `primitives/_moments.py::spatial_moments_2d` (label-scatter) replaces regionprops' per-region
  einsum for moments/central/normalized/Hu; inertia derived from the central moments. raw bit-exact,
  rest ~1e-13. 2D only. /simplify'd (incl. mypy shape-narrowing fix CI caught).
- [x] **PR #65** `fix(numba/feret): accept pixels=None` — the numba feret backend crashed on its own
  `(mask, None)` dispatch convention (`to_bzyx` rejects None). Bit-exact; regression test added.

### numba sizeshape lane — **PR #78** (built in `cp_measure_wt_integration`, branch `feat/numba-sizeshape`, base `integration/all-numba`)
- [x] Phase 1 moment kernel (`core/numba/_sizeshape.py::_moment_kernel`) — fused 2-pass (raw+bbox,
  central), bit-exact, 3.5×.
- [x] Phase 2 convex hull (`convex_area_2d`) — numba monotone-chain over boundary pixels +
  **skimage's `grid_points_in_poly` kept** (no pnpoly port). Bit-exact 142/142, ~2.8×.
- [x] Phase 3 perimeter / perimeter_crofton / euler — label-aware neighbour-pattern kernels
  (border 3×3 for perimeter; shared 2×2 XF histogram for crofton+euler). Bit-exact, ~5.6×.
- [x] Phase 4 `get_sizeshape` wrapper + dispatch (option B). **305→100 ms (~3×)**, all 78 features
  match numpy. **Option B**: axes/eccentricity/orientation derived from kernel central moments
  (`primitives/_moments.axes_eccentricity_orientation`) → regionprops call is fully moment-free.
  Rewrote `bulk._numba_registries` (integration merge had left it un-parseable) to compose all
  numba backends + register `sizeshape`; `set_accelerator("numba")` now routes it. 24 tests. /simplify'd.

### Decisions
- [x] numpy default-backend perf lane declared **done** — 4 big levers landed (#74/#75/#76/#77) + the
  earlier #69–73. Remaining sizeshape headroom (convex hull) is numba-only (numpy can't match
  skimage's rasterized `area_convex` fast & bit-exact). texture: not worth in numpy (mahotas is C).
- [x] Head-to-head measured: **numba-merged ≈ 2.7× over numpy-merged** (large tile); numba's real
  wins are where numpy can't reimplement (radial_distribution 13×, costes 16×, texture 6×); zernike/
  radial_zernikes only ~1.3× (numpy PRs closed the gap); granularity numba still ~5× (it reimplements
  the morphology #76 left on skimage). sizeshape was the numba pipeline's #1 component → now ~3× via #78.

### Open follow-ups
- [ ] Merge order: merge **#74** first → then retarget **#75** to `main` (CI runs then).
- [ ] **Productionize #78**: rebase `feat/numba-sizeshape` onto `main` once #77 + the numba stack
  (#59 bzyx, #60 coloc's `labels_to_offsets`, rest) land. Then: drop the `primitives/_moments.py`
  COPY (it's #77's file — this branch predates #77), keep `axes_eccentricity_orientation` (fold into
  #77), the registry edit becomes a clean one-line add. CI only runs once base = main.
- [ ] **#78 perf follow-up**: the 4 sizeshape primitives each call `labels_to_offsets`/`nonzero`
  independently (4× over the full raster) — compute the shared prep once in `_sizeshape_2d` and thread
  it in (~5–10 ms / ~7%). Deferred (touches 4 public signatures + tests).
- [ ] **#77 / numpy could adopt option B** (`axes_eccentricity_orientation`) to kill its own residual
  order-2 moment einsum (~15–20 ms) — free numpy win, the helper already exists.
- [ ] `integration/all-numba`'s `bulk._numba_registries` is still un-parseable (the fix lives only on
  `feat/numba-sizeshape`) — fix at the integration/merge level too.
- [ ] Housekeeping: `git stash drop stash@{0}` (its files were restored into `tasks/` this session);
  the scheduled `/handoff` cron (`249e8cad`, ~10:09 CEST) is session-only.

---

## Session 2026-06-07 — #78 polish, #77 option-B fix, fused coloc, 3-way bench, unify plan

### Completed this session
- [x] **#78 shared-prep follow-up DONE** (was open) — `feat/numba-sizeshape` commit `a4974d0`: added
  `_Prep` NamedTuple + `_foreground_prep`; 4 sizeshape primitives take optional `prep`, computed once
  in `_sizeshape_2d`. Removed 3 redundant `labels_to_offsets` (~9ms/~5%). /simplify shrank `_Prep` to
  lut/n/offsets (moved the full-raster nonzero into `spatial_moments_2d`, its only consumer).
- [x] **#78 /code-review (xhigh): ZERO correctness bugs.** All candidates refuted/test-covered
  (pixels unused by sizeshape — golden passes random pixels; object order = ascending labels in both
  labels_to_offsets & regionprops; orientation*180/pi both; 3D arg order ok; div-by-zero preexisting).
- [x] **#78 review-followup DONE** — commit `04bfeda`: extracted `moment_feature_dict` to
  `primitives/_moments.py` (single source for 53 moment/inertia keys; numba calls it, numpy adopts at
  #77 rebase); clarified the unused `pixels`/`_pixels_zyx` in the wrapper; added end-to-end empty-mask
  test. 25 tests pass.
- [x] **#79 folded into #78** — the shared-prep PR was briefly stacked draft #79; folded (it only
  edits a #78 file so can't merge to main alone). GitHub auto-closed #79 as MERGED; branch deleted.
- [x] **#77 OPTION B FIXED** (was open) — `perf/sizeshape-moments-scatter` commit `f7b32be`: the
  original #77 stripped explicit moment props but KEPT axis/ecc/orientation in regionprops, which
  re-trigger `moments_central` einsum → scatter was pure overhead, 2D sizeshape REGRESSED 198→216ms.
  Fix: added `axes_eccentricity_orientation` to `_moments.py`, derive all four from scatter central
  moments → regionprops now moment-free. sizeshape large 216→167ms (now 1.2× FASTER than pypi). 3D
  keeps axis on regionprops. `test_axes_match_*` added. 12 tests pass.
- [x] **Fused numba coloc SHIPPED** — `feat/numba-coloc-costes` commit `b0fe736`:
  `get_correlation_all(features=None|subset)` runs flatten+`coloc_per_object` ONCE for all coloc
  features (the 5 public fns each re-ran the whole kernel → ~5× waste). Stateless; the 5 fns are now
  gated wrappers delegating to it; registry KEPT per-group keys (single-entry broke featurize w/
  KeyError 'pearson'). large all-coloc 38ms vs 103ms (5-sep) vs 140ms (numpy) = 2.7×/3.7×.

### Decisions / findings
- [x] **3-way "everything merges" benchmark** (pypi 0.1.19 pre-opt / all-numpy-merged / all-numba-
  merged incl sizeshape). LARGE joint: numpy 6.2× / numba 26.2× vs pypi, numba 4.2× vs numpy.
  Harness + results in `cp_measure_wt_integration/tasks/` (bench_3env.py, combine_3env.py,
  res_{pypi,numpy,numba}.json, bench_3way_results.txt). Method: 3 processes, IDENTICAL deps
  (numpy 1.26.4), only cp_measure code differs.
- [x] **Numpy coloc fusion = NO WIN (1.02×)** — `cp_measure_wt_integration/tasks/proto_numpy_coloc_fused.py`.
  Fusion is backend-specific: numba's fused kernel computes all features in one cheap pass (re-running
  was the cost); numpy's #69 already made prep cheap (find_objects bbox) and the per-feature work
  (corrcoef/lstsq/lexsort/costes ~80ms) is distinct & unshareable. Leave numpy coloc as-is.

### Open follow-ups (NEW / updated)
- [ ] **Retarget #75** to `main` after #74 merges (unchanged).
- [ ] **Land #77 to main** (user's chosen sequencing), THEN rebase the numba stack + #78.
- [ ] **PR AUDIT findings:** #59 (bzyx foundation) and #38 (return_as) CONFLICT with current main
  (#59 forked at `ba4b1e6`, before #69-73 merged to main) → the whole numba stack needs rebasing onto
  current main before it can merge. No merge-mangling in any PR head. (#77 ineffectiveness = FIXED above.)
- [ ] **Productionize #78** (updated): at #77 rebase, drop the `_moments.py` copy (use #77's, re-add
  `moment_feature_dict` + `axes_eccentricity_orientation` there); wire numpy `get_sizeshape` to call
  `moment_feature_dict`; optionally extract a shared EDT-radius helper. Ship plan deferred items in
  `cp_measure_wt_integration/tasks/`.
- [ ] **Fused coloc productionize** — `b0fe736` is on `feat/numba-coloc-costes` (#62), part of the
  stale numba stack; rides the same rebase. Optional follow-up: featurize-level routing through
  `get_correlation_all(features=enabled)` (careful: per-channel-pair symmetric/asymmetric semantics).
- [ ] **BACKEND UNIFICATION** (new, big) — plan in `cp_measure_wt_integration/tasks/plan_unified_backend_abstraction.md`.
  prep→compute→assemble seam: shared RUNNER + shared PREP (`PreparedImage`, also the cross-producer
  CSE lever) + per-backend KERNELs + shared ASSEMBLE; `Feature` descriptors. Phase 0 (shared assemble)
  is low-risk dedup, do on numpy now; Phases 1+ AFTER the numba stack rebases. Keep runner thin + perf-neutral.
- [ ] **NOTE on tasks/ location:** this session's artifacts (benchmarks, protos, plans) were written to
  `cp_measure_wt_integration/tasks/` (the integration worktree), NOT this `cp_measure/tasks/`. Consolidate later.

## Numpy PR cleanup #74-78 (review-driven) — 2026-06-09/10

### Done
- [x] **Max-effort `/code-review` of the numpy PRs (#74-77 combined diff)** → 15 findings. Then
  reviewed the resulting PLAN itself (`tasks/numpy_prs_cleanup_plan.md`) with another max review,
  which caught: foreground_scatter fits NO caller (bbox-min/circle-radii/CSR all differ),
  `moment_feature_dict` is 5-arg not 2, impossible F-first sequencing, and an A/B self-contradiction.
  Rewrote as **plan v2** (no foundation PR; extend `label_to_idx_lut`; in-place per-PR fixes).
- [x] **Decisions:** Q1 #75 → co-shaped (Option A, raise on shape mismatch). Q2 → match **PyPI
  0.1.19** column order (verified grouped: all Spatial, then Central, then Normalized, ...).
- [x] **#77 sizeshape** (`perf/sizeshape-moments-scatter` @ 6d961eb): eigenvalue **clip** (skimage
  parity), `label_to_idx_lut(return_bbox=True)` → bbox origin replaces 1<<31 sentinel/minimum.at,
  grouped `moment_feature_dict` (ported from numba, single source), single `inertia_2d` shared by
  axes+tensor, `advanced=` gate skips normalized/Hu, dropped redundant `numpy.unique`, `_ORDER`
  powers, honest docstring, order-pin + thin-object-clip regression tests. **VERIFIED bit-exact vs
  0.1.19** (column order + values, max|diff|=0.00). 95 tests pass.
- [x] **#74 zernike** (@ 2c1a1f1): `errstate` around single-pixel `/radii` (no warning under -W error).
- [x] **#75 radial** (@ ee7ed1e): co-shaped `ValueError` guard + test; rebased onto new #74.
- [x] **#76 granularity** (@ 27d963b): boundary bug FIX — fused operator dropped out-of-bounds
  pixels to match `map_coordinates(mode='constant')` cval=0 (counts still span all foreground);
  overshoot golden test (160→64) added. Pre-fix diverged ~0.08 on edge objects.
- [x] **#78 numba sizeshape** (@ 289e997): same eigenvalue clip + **grouped** `moment_feature_dict`
  (fixed its latent INTERLEAVE column-order bug — no test caught it, only moment arrays were tested);
  also fixed a pre-existing `SyntaxError` (unclosed paren) in `test_backend_correctness.py` that
  masked a stale assertion ("sizeshape stays numpy" — wrong, #78 makes it numba-backed). 25+11 pass.
- [x] **2 real correctness bugs found + fixed:** (1) granularity boundary divergence (#76, ~1.3% of
  size/subsample combos, ALL edge objects); (2) missing eigenvalue clip → NaN axis_minor/ecc>1 (#77
  + #78, ~4% of thin objects). Both were SILENT (no test caught them).
- [x] **PR #59 rebased/slimmed** (866→163 lines): cherry-picked only the `to_bzyx` commit onto
  current main (the rest was already merged via #54/#69-73), force-pushed; PR now CLEAN/mergeable.
- [x] **"Everything merged" benchmark REDONE** with latest PR heads + fused coloc. Built
  `bench/numpy-all-v2` (main + #74/#75/#76/#77) and `bench/numba-all-v2` (feat/numba-sizeshape +
  coloc fusion b0fe736 merged). 3 pinned-identical-dep venvs (cp_bench_pypi 0.1.19 / cp_bench_numpy
  / cp_bench_numba). `bench_3env_v2.py` + `combine_3env_v2.py` add a FUSED `coloc_all` row (numba
  get_correlation_all once; numpy/pypi sum the 5). LARGE joint: numba **34.7×** vs pypi / **5.0×**
  vs numpy; fused coloc **112×** vs pypi (24ms vs ~97ms summed). Artifacts in
  `cp_measure_wt_integration/tasks/`.

### Open follow-ups (this session)
- [ ] **Refresh stale PR descriptions** for #76, #77, #78 (substantial scope changes vs original bodies).
- [ ] **_moments.py convergence** (at the numba-stack rebase): #77 (numpy) and #78 (numba) BOTH now
  have the clip + grouped `moment_feature_dict`, but #77 additionally has the `label_to_idx_lut`
  bbox path + `advanced=` gate that #78 lacks. Unify into one file when both reach main.
- [ ] **Prune worktrees/venvs** when done: `cp_74 cp_75 cp_76 cp_77 cp_78 cp_bench_{pypi,numpy,numba}
  cp_77 pr59-rebased pr74 pr75` (+ their .venvs). `git worktree remove` / `git worktree prune`.
- [ ] **CI watch:** #74/#76/#77 showed `unstable` (checks pending, NOT conflicts); #75/#78 `clean`.
- [ ] **Plan v2 lives in `cp_measure/tasks/numpy_prs_cleanup_plan.md`** (this repo's tasks, not the
  integration worktree) — the canonical spec for the above.
