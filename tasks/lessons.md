# cp_measure — Lessons

## `featurizer.py` is only on the feature branch, not main
- The featurizer module (`src/cp_measure/featurizer.py`, tests, conftest) only exists on `feat/multi-mask-featurizer`, not `main`. Any new feature branches that build on it must branch from there, not `main`.
- **Why:** Main has different code (multimask/overlap refactoring). Branching from main and stash-popping causes delete/modify conflicts.

## Running tests requires PYTHONPATH
- Tests must be run with `PYTHONPATH=src:test pytest ...` because the package uses a `src/` layout and the test conftest uses direct imports.
- **Why:** The package isn't installed in editable mode in the current environment; `python -m pytest` alone can't find `cp_measure`.

## Optional dependencies: use `_lazy_import` pattern
- For optional output formats, use lazy imports inside converter functions with clear `ImportError` messages pointing to `pip install cp_measure[extra]`.
- **Why:** Keeps the core package lightweight; pandas/pyarrow/anndata are large dependencies users may not need.

## SLURM: submit with plain `sbatch` + positional args, not `srun` or `--export=ALL`
- Submit benchmark jobs as `sbatch [flags] script.sbatch arg1 arg2` (args read as `$1 $2`). Do NOT launch GPU work with `srun` from inside the harness's own allocation, and do NOT use `--export=ALL`/`--export=NONE,VARS`.
- **Why:** `srun` from within the harness's cpu_p allocation cannot acquire a GPU (it reuses the CPU allocation) → hangs forever. `--export=ALL` triggers this cluster's login-shell env retrieval which fails → every job `user_env_retrieval_failed_requeued_held`. Plain sbatch with positional args avoids both.

## Single-thread fairness = pin BLAS, not numba
- For fair single-thread CPU benchmarks, export `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=NUMBA_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1` (and for jax-cpu also `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`).
- **Why:** cp_measure's numba kernels are all `@njit(cache=True)` serial (no `parallel=`/`prange`), so they're already 1-thread. The only silent multi-threading is numpy's BLAS (lstsq/corrcoef in zernike/pearson/costes), which defaults to the visible core count when unpinned → unfair boost to baseline.

## A `try/except` fallback can silently hide a broken fast path
- `cp_measure_jax`'s `get_intensity` wrapped the JAX path in `try/except: <numpy fallback>`. A float64-index bug made the JAX path raise every call → it silently ran the slow numpy loop. The "GPU" timings were numpy-on-CPU; the GPU was never used.
- **Why:** When benchmarking an accelerated path, call it DIRECTLY (so it raises) or assert the backend/log on fallback. Otherwise a bug masquerades as "the GPU doesn't help."

## GPU single-image intensity is orchestration-bound, not compute-bound
- On H100, the intensity GPU *compute* is ~8ms (transfer 2.7ms, lexsort 1.7ms, segment_sum 2ms), but the full call is ~23ms — the rest is host `bincount`, CPU `find_boundaries`, ~21 device→host `np.asarray` syncs, and two separate jit kernels.
- **Why:** GPUs win on one big fused computation with amortised launch/transfer (batching). At single-image scale these fixed per-call overheads dominate; numba (no transfer) is competitive or better.

## scipy.ndimage.maximum_position labeled tie-break is a non-reproducible quirk
- For exact-value ties it returns neither the first nor last pixel in raster order; the labeled call differs from the global call. Real microscopy data has ~no exact ties (so a numba/jax `>=`-last kernel is bit-exact), but synthetic/saturated images (the `tiny` tier) will differ in `Location_MaxIntensity_X/Y`.
- **Why:** Don't burn effort matching it; accept the synthetic-only diff and note it.

## cp_measure cost is concentrated; partial ports hit Amdahl hard
- On `large` (1080², 142 obj, default subsampling) baseline time: intensity 53%, granularity 13%, zernike 11%. A variant that doesn't accelerate intensity is capped near its Amdahl ceiling (jax was 1.64× purely because intensity stayed at baseline). Best-of-breed routing → ~57× ceiling.
- **Why:** Always check where baseline spends time before judging a variant's global speedup. Also: granularity defaults to `subsample_size=0.25` (270² for 1080²) on main, so it's cheap by default; it only dominates at `subsample_size=1.0` (full-res).

## Don't trust a static inventory for "which functions differ" — run them all
- The benchmark's RUN_SET (from an Explore-agent inventory) marked most fast/jax functions `=base` and skipped them. WRONG: `fast` accelerates ~all functions (every core module differs from baseline; coloc uses bincount vs reference's per-object loop), `jax` also pearson/costes. This under-counted fast (1.3×→8.4×) and jax, and is exactly what hid the jax-intensity fallback bug (an un-run function reveals neither speedup nor bug). `speed` was *thought* to be the only correctly-scoped variant (intensity-only) — but a 2026-05-30 source mining (`primitive_existence_matrix.md`) found speed ALSO changed `measuregranularity.py` functionally (re-enables `subsample_size<1` resampling baseline had commented out), so its RUN_SET should include granularity too.
- **Why:** Run every function for every variant and detect `=base` EMPIRICALLY (measured output ≈ baseline + measured time), never from a guessed list. Diff source against baseline to confirm scope.

## SLURM `--cpus-per-task=N` gives N *logical* CPUs (hyperthreads), not N physical cores
- `--cpus-per-task=8` on this cluster allocated 4 physical cores + their 4 SMT siblings. Parallel scaling looked like a "wall" at 8 (59% eff) but it was just hyperthreading: ~linear to the 4 physical cores, then ~1.3× for the 4 HT siblings. NOT a bug and NOT memory bandwidth (my first guess was wrong). Verified with diag_scaling.py reading /sys/.../topology/thread_siblings_list.
- **Why:** To benchmark "N cores" honestly, either request `--hint=nomultithread` (N physical) or report it as "N logical (≈N/2 physical + HT)". A sub-linear scaling curve that's embarrassingly parallel with negligible IPC ⇒ check topology (HT) before blaming memory bandwidth.

## GPU under-utilisation: single-image cp_measure leaves the GPU ~halfidle, and multi-process packing is a VRAM-bound dead-end
- A single 1080² tile keeps the H100 busy only ~part of the wall time (intensity: ~8ms compute vs ~22ms wall). Multi-process packing (N jax procs sharing one GPU via mem-fraction) plateaus at ~4.9 MP/s with only **54% mean GPU util** and uses 69/80GB at N=8 → can't pack more.
- **Why:** Host-side serial code (find_boundaries/bincount/sort dispatch) + per-tile launch/transfer starve the SMs. The reliable fix is in-process BATCHING (one VRAM footprint, fused work), not more processes. Also: jax preallocates ~75% VRAM by default → set `XLA_PYTHON_CLIENT_MEM_FRACTION` when sharing a GPU or it OOMs.

## Throughput is not pixel-linear; pick the tier matching your real field size
- cp_measure cost = per-pixel (texture, morphology) + per-object (intensity, coloc, sizeshape) + near-fixed (jax-cpu granularity_fullres ~12s regardless of size) terms. So s/MP differs across tile sizes; the per-image global speedup is only valid at that tile size. jax-H100 throughput peaks at ~1-2k tiles and DROPS at 4k (object-count-bound funcs get worse per-MP; +OOM risk on small GPUs).
- **Why:** Never rescale a speedup/throughput across tile sizes by megapixels. Benchmark the tile size the data actually uses.

## Reliability via a single code path: single image = batch of 1
- Two separate code paths (single vs batch, or accelerated vs fallback) silently diverge — that's the bug class behind the jax-intensity numpy fallback AND the mislabeled functions. The robust design for a batching API is ONE batched implementation where a single image is `batch=1`, so single and batch agree by construction.
- **Why:** Prefer one path even at a small N=1 overhead. When benchmarking an accelerated path, call it DIRECTLY (so it raises) rather than through a try/except wrapper that can mask a broken path as "working but slow."

## Verify a "regression" against the actual diff — and check WHAT differs (typing vs algorithm)
- The benchmark reported `speed` variant feret 0.04× / radial_zernikes 0.33× "regressions". Those `feret`/`radial_zernikes` files (`measureobjectsizeshape.py`, `measureobjectintensitydistribution.py`) differ from baseline by **typing/cosmetics only** (functionally identical), so the numbers are measurement artifacts (warmup/contention/per-call variance), not code. Chasing a "dense intermediate that cratered feret" would have been wasted effort — there is no such code.
- **CORRECTION (2026-05-30, `primitive_existence_matrix.md`):** the earlier claim that speed is "byte-identical except intensity" was wrong on both counts. ALL 6 core files differ; intensity is the real rewrite, **granularity is ALSO a real functional change** (subsampling re-enabled), and the other 4 differ by typing only. md5 alone would have flagged all 6 as "differs" — so a raw diff is necessary but not sufficient.
- **Why:** A speedup/slowdown number is only meaningful if the code differs *algorithmically*. Before debugging a per-function regression, diff against baseline AND read the diff — a typing-only diff means the delta is noise; a functional diff (like speed's granularity) means the timing isn't apples-to-apples and the RUN_SET/scope must be updated.

## Backend dispatch: detect capabilities once with find_spec + flags, NEVER try/except
- The ecosystem refactor bans `try/except` entirely. Backend availability is checked ONCE at import via `importlib.util.find_spec("numba")`/`find_spec("jax")` (+ `jax.default_backend()` for GPU), setting boolean flags. Dispatch reads the flags; the resolved path is called directly and unguarded. A backend is present (flag → used) or absent (flag → never attempted) — there is no error-driven fallback.
- **Why:** `try/except` around an accelerated path is exactly what hid the jax float64-index bug for a whole benchmark round (it silently ran numpy). Removing it deletes the silent-fallback bug class outright instead of mitigating it: a flagged-present backend that raises is a real bug and MUST surface. Trade-off (intended): backend health is an install/config responsibility, not papered over at call time.

## A dispatcher whose "valid set" and "handled branches" live in two files will silently return None
- PR #49's `set_accelerator` validates against `_VALID_ACCELERATORS` (in `__init__.py`) while `_dispatch` handles each value with `if/elif` (in `bulk.py`), with NO trailing `else`/`raise`. Today every valid value has a branch so it's fine — but add a value to the tuple and forget the branch → `_dispatch` returns `None`, caller does `None["intensity"]` → TypeError far from the cause. Fix = trailing `raise` (exhaustiveness by construction).
- **Why:** Same silent-divergence bug class as the try/except fallback. When the set of valid inputs and the set of handled inputs are maintained separately, the dispatcher must raise on the unhandled case, never fall through. (Aside, verified: a function-local `from module import GLOBAL` re-reads the attribute each call, so it DOES pick up a `global`-rebind — runtime-switchable dispatch works.)

## "Repo X does technique Y" claims from mining must be verified against source before becoming design guidance
- Our notes encoded "rust intensity = single fused pass", "rust computes the hull once and shares it", "rust feret = rotating calipers". A line-level re-read found ALL THREE wrong: rust intensity is two-phase scatter-then-reduce (cross-sums in the reduce loop), the hull is recomputed 3×, and feret is O(n²) brute force. The *transferable ideas* survived (scatter-once + running-sums; share the hull) but as things WE'd build, not things rust already does.
- **Why:** A second-hand "X does Y" hardens into a plan assumption and then a spec. Before encoding a mined technique as design guidance, confirm what the source actually does vs what we'd be *introducing* — the distinction changes effort estimates and who owns the optimization.

## CI lint is `nix fmt` (treefmt: ruff-format + ruff-check + nixfmt + dprint), not the uv/pytest loop
- The `lint` workflow runs `nix fmt` then `git diff --exit-code`, so ANY reformatting fails it. Our local loop was `uv run pytest` only — it never ran the formatter, so a few >88-col lines I wrote slipped through and broke CI lint (purely cosmetic). ruff IS a dev dep; we just had no format step.
- **Why:** Before pushing to this repo, run `uvx ruff@0.12.1 format --check . && uvx ruff@0.12.1 check .` (matches the CI ruff gate without needing Nix locally). ruff default line-length is 88; there's no `[tool.ruff]` override in pyproject.

## Local test env for upstream cp_measure: provision uv (no Nix/pixi project here)
- The repo has `uv.lock` but uv wasn't installed. `pixi global install uv` (→ `/home/icb/tim.treis/.pixi-home/bin/uv`), then `uv sync --extra numba --extra test`, run with `uv run pytest`. Package installs editable, so NO `PYTHONPATH=src:test` needed (unlike the old cp_measure_fast repo). numba 0.65.1, numpy 1.26.4, skimage 0.26.0.
- **Why:** Don't reach for the cp_measure_fast pixi env or micromamba base (no numba+pytest). The uv env is the project's own toolchain.

## zsh does NOT word-split unquoted variables
- `FILES="a b c"; ruff $FILES` passed the whole string as ONE path on this zsh shell (error: "No such file or directory"). Bash would word-split; zsh does not by default.
- **Why:** In Bash tool commands (shell is zsh), list args explicitly or use `${=FILES}` / an array — never rely on unquoted `$VAR` splitting.

## skimage disk erosion/dilation is border-mode-insensitive; granularity correctness specifics
- For a disk footprint, `skimage.morphology.erosion/dilation` give bit-identical results under reflect/symmetric/edge padding (min/max selection can't introduce a new extremum from reflected/clamped copies of border values). Verified over 300 tie-heavy/border-hot trials. So pad once with `edge` (clamp) on the host and run a border-free kernel.
- Granularity output is indexed DENSELY by label `1..max(mask)` (NaN at gaps), NOT compacted to present objects like intensity — so use `bincount`, NOT `label_to_idx_lut`. The per-object mean is the only non-bit-exact spot (bincount vs scipy.ndimage.mean summation order) → compare within tol, equal_nan=True.
- The numpy baseline collapses dims at subsample (`int(Z*subsample_size)`); a thin 3D volume at the 0.25 default hits Z=0 and errors — a shared baseline limitation, not ours.
- **Why:** These are the exact-match conventions to honour when porting any morphology/per-object feature; getting the dense-vs-compact indexing or border mode wrong silently diverges from baseline.

## Reconstruction: Vincent FIFO-hybrid is O(N); raster-until-convergence is O(passes·N)
- Morphological reconstruction by dilation via alternating raster passes needed up to 247 passes for ONE granularity spectrum step at fullres (1080²) — it was 83% of fullres time. The Vincent (1993) FIFO-hybrid (1 fwd + 1 bwd raster, then a FIFO of still-propagating pixels) is O(N) and BIT-IDENTICAL (both compute the exact reconstruction; min/max only). Swapped it in → fullres 4.01×→6.86×. Safe FIFO sizing: ring buffer of N+1 with a per-pixel in-queue flag bounds live entries to N.
- **Why:** When a "correct but simple" iterate-until-stable kernel is the hot path, profile the pass count — geodesic propagation distance can be O(H+W), making the simple version pathological. The hybrid is exact, so the bit-exact-vs-skimage tests are the safety net for the swap.

## A "pure batch normaliser" that erases dimensionality forces backends to re-infer it (with edge cases)
- `to_bzyx` deliberately maps both a 2D image `(H,W)` and a single-slice volume `(1,Y,X)` to the same `(1,Y,X)` canonical form. That's clean for batching, but it ERASES the original ndim, so each backend must re-infer "2D-ness" as `Z==1`. For intensity this drives `mad_frac = 1/ndim`: the inference matches the numpy baseline for every real input (2D `(H,W)`→1/2, Z>1 volume→1/3) but DIVERGES for a degenerate `(1,Y,X)` volume (baseline 1/3 vs inferred 1/2) — affecting only `Intensity_MADIntensity`. The golden test (DEPTH_3D=8) never exercises Z=1, so the divergence is silent.
- **Why:** When a shared normaliser collapses two input shapes into one, any per-shape behaviour downstream becomes a re-inference with a blind spot. Either carry the discriminating fact through the normaliser (here: per-element original ndim) or accept+document the collapsed-case divergence — but know it exists and that your tests may not cover the collapsed case. Don't push the dimension-aware logic back INTO the normaliser (that re-couples it); fix it by carrying the fact alongside, in a dedicated cross-consumer change.

## Profile the WHOLE function, not just the kernel — slow host helpers dominate a "fast" numba path
- The fused numba zernike kernel was ~6 ms, but radial_zernikes was only 1.27× because the HOST helpers dominated: `cp_measure.utils.masks_to_ijv` did `np.where(masks==L)` per label over the full image (O(n_labels·HW) = **347 ms**), and `numpy.unique(labels)` was an **8.9 ms** full-image sort. Replacing them (one `numpy.nonzero` pass; `label_to_idx_lut` via `find_objects` = 2.3 ms) took radial 1.27×→20.9×.
- **Why:** a fast kernel guarantees nothing about the function. Always break the function into pieces and time each; a per-label `np.where`, a full-image `np.unique`/sort, or a per-pixel Python construct loop is often the real cost. Prefer one `nonzero` pass + `label_to_idx_lut` over `masks_to_ijv`/`np.unique`.

## A/B a reimplementation against the import-alternative before judging it "not worth it"
- I hedged that reimplementing the Zernike basis (vs importing `centrosome.construct_zernike_polynomials`) was marginal — based on a stale recollection of the `fast` variant's numbers. An A/B on identical host setup refuted it: fused kernel **6.3 ms** vs centrosome-construct + numba-sum **61.8 ms** (~10× kernel win; centrosome's construct alone is a 25.7 ms Python loop over the K polynomials).
- **Why:** recollected speedups from sibling variants conflate multiple optimizations and mislead. When deciding whether to reimplement something importable, measure the marginal contribution directly. Decision rule that held up: reimplement the mechanical, bit-exact-test-guardable part (evaluation); IMPORT the numerically-sensitive coefficients and the high-risk geometry (`minimum_enclosing_circle`) — that boundary is why we're not the rust 5-6%-drift situation.

## Stacked PRs already show isolated diffs; extraction buys MERGE-order flexibility, not review isolation
- GitHub shows a PR's diff as base…head, so a PR stacked on another already shows ONLY its own changes (the base's work is excluded). So "extract the shared base so features can be reviewed in isolation" is a non-reason — they already are. Extracting a shared base PR buys *merge-order independence* (any feature can land first once the base merges) and a smaller gating PR — a narrower (still real) benefit.
- **Why:** don't justify a PR restructure with "unblocks review" when the stack already isolates review. Be honest that the benefit is merge ordering + a tiny review-first gate, and weigh it against the cost (+1 PR, rebase surgery, and it converts stack-ordered conflict resolution into per-merge append conflicts).

## Editing a BASE PR in a stack requires re-rebasing dependents with `--onto` (capture the old base SHA first)
- After amending the base PR (#59), the feature branches still pointed at the OLD base commit; their GitHub diff would show the base's change as a REVERT until re-rebased. Re-rebase each with `git rebase --onto <new-base> <OLD-base-SHA> <feature>` — replaying ONLY the feature's own commits. A plain `git rebase <new-base>` would try to replay the old base commit (which re-adds the changed file) → add/add conflict.
- **Why:** in a stacked setup, capture the old base tip SHA BEFORE rewriting it, then `--onto` it away. Also: `git rebase --onto <base> <upstream> -X theirs` cleanly auto-resolves the identical add/add when first extracting a file out of a branch into a new base it now shares.

## A "golden vs numpy" test must not assert bit-equality where the numpy baseline is dtype-buggy
- The numba coloc backend upcasts pixels to float64. The numpy colocalization reference does NOT: with a uint8 image `fi*si` overflows (Overlap exceeded 1.0 — impossible under Cauchy-Schwarz) and the Pearson slope's `lstsq` design matrix is uint8 → a *float32* result. So integer-dtype golden tests failed on pearson/overlap while manders/rwc (no pixel products) passed.
- **Why:** the float64 path is strictly more correct; matching an overflow/precision artifact is the wrong target. Distinguish "exercise a code path" (rank ties) from "the dtype itself" — drive ties with integer-VALUED float64 arrays, and document the genuine integer-dtype divergence rather than encoding the bug into the kernel. When a backend rewrite is *more* accurate than its reference, the test compares on realistic inputs (float) and the divergence is noted, not asserted away.

## Many small per-object sorts beat one big keyed global sort (profile, don't assume)
- RWC needs per-object dense ranks (144 objects × ~4489 px). The "obvious" optimisation — replace 144 in-kernel numba argsorts with ONE numpy `np.lexsort((value, segment))` + an O(M) linear dense-rank pass — produced bit-identical ranks but was **2.3× SLOWER** (220 vs 94 ms). Reason: 144 sorts of log≈12 each beat one 646k-element sort of log≈19, and lexsort runs two passes (one per key). The win I *did* find was elsewhere: `np.bincount` for label prep (one C pass → lut+offsets) replacing scipy `find_objects` + a separate count scan, a clean 2× on the shared floor that ~doubled the three sort-free coloc features.
- **Why:** profile the real components before optimising (find_objects 3.5 / flatten 2.6 / kernel 4.2 / rwc-sort 93 ms made the targets obvious), and benchmark the "better" algorithm against the naive one — asymptotically-equal work can lose badly on constants and call overhead. The biggest dominant cost (rwc's sort) was the one with no good fix; the win was in the cheap-looking shared prep.

## "The work is dead, short-circuit it" must be verified across the FULL input domain
- costes' iterative threshold search costs ~7ms (the bulk of the kernel). On float pixels scale=1 the [1,1] bisection window *looked* degenerate, and a 200-seed check vs the reference confirmed a closed-form result (bisection→(0,b)) — so I short-circuited it. The end-to-end golden test then failed on a 3D object: the 200 seeds were all positively correlated (regression slope a>0), but an anti-correlated object (a<0) drives `left=mid-1=0`, makes the next `mid`=0, and the loop runs 2-3 more iterations reaching a different threshold. The pearson was NOT dead. Reverted; no exact speedup exists.
- **Why:** when claiming computation is redundant and short-circuiting it, the verification must span the whole input domain, not a plausible-looking sample — here both signs of the regression slope. The end-to-end golden (numba vs reference on float64) is what caught it, which is why a behaviour-changing "optimisation" must stay gated by an exact end-to-end test before you trust it. Cf. [[the bincount win, which WAS exact]] — profile to find the cost, but prove the shortcut over all inputs.

## Profile-first can KILL a lane: when the imported part dominates, Amdahl caps the numba win
- radial_distribution looked like a fine numba target (binned histograms + wedge-CV). Step-0 profiling before writing any kernel showed the centrosome geometry we (correctly, per the import-don't-reimplement boundary) KEEP — distance_to_edge + propagate — is ~97% of runtime; the reducible reductions are ~3%. So a numba kernel buys ~1.03×. Lane dropped before a line of kernel code. Contrast coloc (fully reducible → 30-68×).
- The profile also surfaced the real, orthogonal win: running the geometry per-object on small crops is 1.65× faster than one whole-image pass AND fixes Issue #22 — a numpy bug-fix, not numba.
- **Why:** the reimplement/import boundary (import numerically-sensitive geometry) interacts with Amdahl: if the imported part is the bottleneck, accelerating the reducible remainder is pointless. Always profile the baseline's import-vs-reducible split BEFORE committing to a lane. Reach for numba only where the reducible work dominates (coloc, GLCM/texture), not where imported geometry does (radial_distribution, and likely sizeshape's regionprops geometry).

## Refinement: "imported geometry dominates → un-numba-able" is only true if it's NOT a shortest-path
- The earlier radial_distribution lesson (profile killed the lane; numba can't touch imported geometry) was HALF wrong. Profiling correctly found propagate = 80%. But before concluding "un-reimplementable numerically-sensitive geometry," I should have asked WHAT propagate actually computes: it's `propagate(image=zeros, weight=1)` = the 1/√2 chamfer SHORTEST-PATH within the mask. Shortest-path is algorithm-independent — Dijkstra (centrosome's C heap) and Bellman-Ford raster sweeps (numba) find the identical minimum. A POC numba geodesic matched centrosome BIT-EXACTLY (maxdiff 0.0, all shapes) at 35.8×. The lane was revived.
- The genuinely un-reimplementable transforms are EUCLIDEAN EDT (Felzenszwalb separable, scipy C — hard + exactness-fragile) and numerically-tuned filters — NOT a shortest-path/geodesic. Earlier I'd tested propagate vs *Euclidean* EDT-from-seed (which DID diverge 1-3%) and wrongly generalized to "propagate is un-reimplementable"; the right comparison was vs the *chamfer* metric, which is exact.
- **Why:** "import, don't reimplement numerically-sensitive geometry" is a real boundary, but classify the primitive first. A shortest-path/argmin has one correct answer reachable by any algorithm → free to reimplement in numba and bit-exact. Only metric-defined-by-the-implementation transforms (exact Euclidean EDT, tuned chamfers with unknown weights) are off-limits. Probe the metric (a 5-shape exact-match check) before ruling numba out.

## Replacing the bottleneck exposes Amdahl: the geodesic was 35x but end-to-end only 9.5x
- radial_distribution's propagate was 80% of runtime; the numba chamfer geodesic replaced it 35x bit-exact. But end-to-end was only 9.5x — because the work that used to be ~5% (per-object host numpy glue, 144 scipy.maximum_position calls, scipy EDT) became ~90% of a much smaller total. Quoting the geodesic's 35x as the headline under-sold AND over-promised: it's the right number for the kernel, the wrong number for the function.
- Fix: profile AGAIN after killing the first bottleneck. The second profile (geodesic 9% / host-glue 33% / maxpos 25% / EDT 19%) pointed straight at folding the per-object host glue + maximum_position into one numba kernel → 9.5x → 21x. The new floor is scipy's exact-Euclidean EDT (~19%), which we correctly don't reimplement.
- **Why:** Amdahl — accelerating a part by N caps total speedup at 1/(1-p) where p is that part's fraction. After the big win, re-profile; the headline speedup of a sub-component is never the function's speedup. And per-object Python/numpy glue (mgrid, np.where, per-object scipy calls, list+concatenate) is itself a major cost at scale — fold it into the kernel.

## A symmetry property can delete the hot loop (texture: HXY1=HXY2=2·HX)
- texture's numba kernel was 4.8× — the bottleneck was two per-direction O(fm1²≈256²) loops computing the Haralick InfoMeas cross-entropies (HXY1, HXY2) over the fm1×fm1 outer product of the marginals. But the GLCM is always SYMMETRIC (mahotas builds it C+Cᵀ), so px==py, and both cross-entropies collapse algebraically to 2·HX (a marginal-only O(fm1) quantity). Verified vs mahotas to ~1e-15, swapped in → 16.8× (bit-exact). Similarly T (the GLCM total after ignore_zeros) was an O(fm1²) sum but is `total − 2·row0 + cmat[0,0]` computable during the build in O(fm1).
- **Why:** before optimizing a hot numeric loop, check whether a structural invariant of the input (here: GLCM symmetry) makes the quantity reduce to something far cheaper. Reading the math (HXY1 = −Σ p·log2(px·py) factors into marginal terms because Σ_j p[i,j] = py[i]) beat any micro-optimization of the double loop. Pair with the Amdahl lesson: profile to find the hot loop, then look for an exact identity that removes it, not just a faster way to run it.

## The bottleneck can be plumbing, not the algorithm — and the geometry floor can be shrunk bit-exactly
- feret looked like a geometry lane (convex hull + antipodal Feret diameter — both centrosome, both import-boundary). Step-0 profiling showed the bottleneck was NOT the geometry: `utils.masks_to_ijv` (a per-label `np.where` scan that re-reads the whole image once per object) was **86%** of runtime; the hull was 13%, the Feret antipodal scan 0.6%. The whole lane's win came from replacing that plumbing with a single numba counting-scatter (same `(i,j,label)` rows/order, bit-identical) — the same per-label-scan anti-pattern zernike already killed.
- Then the *imported* geometry floor (the hull, 13%) was itself shrunk bit-exactly without reimplementing it: hull(object) == hull(boundary(object)) because an interior pixel (all 8 neighbours share its label) can never be a hull vertex. Feeding `convex_hull_ijv` only the boundary pixels (~6% of pixels) left both the hull AND `feret_diameter` bit-identical while cutting that call 78.9→6.2ms. End result 43.4×, all centrosome geometry untouched.
- **Why:** (1) profile the baseline before assuming the named-hard part (geometry) is the cost — it's often the surrounding plumbing (per-label scans, list+concatenate, per-object Python calls). (2) You can accelerate an imported numerically-sensitive primitive WITHOUT reimplementing it by shrinking its input along a property that leaves the output identical (here: hull is determined by boundary pixels). That respects the import-don't-reimplement boundary while still beating Amdahl. 8-connectivity + image-edge detection is load-bearing for the boundary set (a diagonal-staircase corner and an edge-clipped pixel are real hull vertices).

## A reducible-dominated lane can still be NO-GO if the dominant primitive is import-geometry and the reducible one is cheap (sizeshape)
- sizeshape's `regionprops_table` is 73% of runtime — reducible-LOOKING (it's "just" region measurements). But decomposing it by property group: the single hot primitive is the CONVEX HULL (area_convex+solidity, +187ms/34% of the function) = numerically-sensitive computational geometry (import). The one mechanically-reducible group, the moment cascade (spatial/central/normalized/Hu/inertia = polynomial pixel sums), is +4.7ms — negligible. The only cleanly-reducible work left (the per-object EDT max/mean/median radius reductions) caps the lane at **1.13×**. Verdict: documented NO-GO, no kernel forced.
- Contrast feret (same "geometry tail" prior): there the dominant cost was reducible plumbing (masks_to_ijv 86%) → 43×. Same profiling step, opposite verdict.
- **Why:** "reducible part is large" is necessary but not sufficient — check whether the dominant primitive WITHIN it is import-boundary geometry (convex hull, perimeter_crofton, euler) vs an algorithm-independent reduction (histogram, shortest-path, moments). And check the reducible primitive isn't trivially cheap (moments were free here). Profile-first + classify-the-primitive is what separates a 43× lane from a 1.13× no-go that looks identical from the outside.

## Validate perf proposals by building+benchmarking end-to-end; component wins rarely = function wins
- A 7-agent deep perf-mining pass over the numba PRs produced many plausible in-kernel optimizations. I built and benchmarked each OLD-vs-NEW end-to-end instead of trusting the component-level estimates. Result: 4 of 5 were non-wins — texture sparse-stack REGRESSED the common case (−10%, build-branch mispredict on large objects) while helping only dense fields; granularity numba label_mean was bit-exact but negligible (map_coordinates is 67% of runtime); zernike skip-vi-on-m==0 was bit-exact but SLOWER (the inner-loop `if m==0` branch breaks numba's vectorization of the k-loop); radial FIFO `%`→branch gave no win (the geodesic isn't radial's bottleneck — scipy EDT + host glue are). Only coloc's pass-fusion (merging two independent per-object passes into one sweep) was a real, bit-identical ~3-4% win.
- **Why:** (1) Amdahl — a kernel sped up Nx only moves the function by its fraction of total time, and most lanes' dominant end-to-end cost is host glue / scipy EDT / sorts / map_coordinates / regionprops, NOT the numba kernel. (2) Adding a data-dependent branch inside a hot numeric inner loop can prevent numba/LLVM vectorization, making the loop slower even though it does less arithmetic. (3) An optimization that wins for one input regime (many small objects) can regress another (few large objects). Always build the change and benchmark BOTH regimes end-to-end with a bit-identical check before shipping; revert anything that isn't a clear, non-regressing win. The real remaining wins were structural (shared-flatten, gather, shared geometry), not micro-opts.

## A per-object-crop "isolation" fix isolates via the `== label` mask, not by the crop
- The Issue #22 fix (radial_distribution) crops each object to its bbox+1px pad. The crop does NOT exclude neighbours spatially (bboxes overlap all the time); isolation comes from `numpy.pad(labels[sl] == label, 1)` — a boolean where ONLY this label is True, so any neighbour pixels in the bbox become background. The geometry (EDT/propagate) then sees only this object; intensity (`pix`) still spatially contains neighbours but the kernel only reads pixels where the mask is True. The crop is pure efficiency; the 1px pad guarantees a clean background edge when the object touches the bbox boundary.
- **Why:** if the fix relied on the crop spatially excluding neighbours it would be fragile. It relies on masking, so it's robust regardless of what's in the crop — including objects that physically abut (the shared boundary becomes a real edge = "object in isolation").

## Testing per-object independence: compare new-vs-new, not new-vs-legacy (centre tie-break)
- To assert the #22 property, compare `new(multi)[obj]` vs `new(object-alone)` — BOTH the per-object path — which is BIT-EXACT (same crop array → identical). Comparing `new(multi)` against the WHOLE-IMAGE `legacy(isolated)` fails by large (1e-1) amounts on symmetric objects: an even-sized square has a centre *plateau* of equally-deep pixels, and `scipy maximum_position` picks a different one in the crop vs the whole image, shifting the centre/bins. The exact `new == legacy` match only holds for UNIQUE-centre (odd-sized) objects.
- **Why:** large diffs in a per-object-isolation test usually mean a centre/tie-break artifact from array-layout differences, not a bug. Use unique-centre (odd) objects for exact-equality tests and same-path comparisons for the independence property. (`np.testing.assert_allclose` also defaults to atol=0/rtol=1e-7, much stricter than `np.allclose` rtol=1e-5 — a "failure" can be pure tolerance.)

## `legacy=True` means "reproduce upstream/CellProfiler behaviour", not an arbitrary toggle
- For BOTH the intensity and radial toggles, `legacy=True` reproduces what CellProfiler/centrosome genuinely produce: intensity's `n*q` percentile convention; radial's whole-image label-leak (Alan traced #22 to scipy → centrosome upstream, so the leak is inherent to the CellProfiler algorithm stack, not a cp_measure orchestration bug). So `legacy` is the right name (a rename like `per_object` would mislabel a faithfulness toggle as a structural one); `legacy=False` is a *deliberate divergence* from CellProfiler for correctness.
- **Why:** before naming a behaviour flag, check whether the "old" branch is faithful upstream behaviour (→ `legacy`) or an arbitrary cp_measure choice. Verify provenance (Alan's issue comments said #22 originates in scipy) rather than assuming. Document that the new default diverges from CellProfiler so nobody expecting bit-exact CP values is surprised.

## numpy.percentile cannot express the CellProfiler n*q convention; parameterize the existing block
- The legacy intensity quartiles use position `n*q` (centrosome/CellProfiler); `numpy.percentile` 'linear' uses `(n-1)*q` (none of numpy's 9 H&F methods is `n*q`). So the legacy path can't use `numpy.percentile`; the cleanest toggle parameterizes the EXISTING cumsum-offset interpolation block with one variable: `span = areas if legacy else (areas-1)` (and `mad_fraction = 1/ndim if legacy else 0.5`), reusing the same clamp/lerp machinery. numba mirrors it via `_interp(..., legacy)` with `pos = n*frac if legacy else (n-1)*frac`. The two backends are kept in lock-step by a golden test parametrized over both flag values — not by shared code (different languages).
- **Why:** when a "convention" has two defensible definitions that differ only in a position multiplier, don't fork into two code paths or reach for a stdlib function that only does one — parameterize the multiplier. Also: a speed PR's win can come from a STRUCTURE change (per-object crops) not the new primitive (numpy.percentile) it happens to introduce; separate the convention change from the perf change.

## Validate perf claims by measuring — `numpy.add.at` is slower than per-column `bincount`
- Twice this session a review agent proposed batching scatter-sums via `numpy.add.at` on a stacked
  `(npix, K)` weights array "to avoid K separate bincounts". Measured: 16 `bincount` = 12 ms,
  `add.at` = 66 ms (5× slower), sparse matmul = 40 ms. `add.at` is the *unbuffered* ufunc path.
- **Why:** `numpy.bincount` (buffered C) beats `numpy.add.at` for segment sums; don't trust
  "fewer calls = faster" for `.at`. Always micro-bench the alternatives before applying a perf
  agent's suggestion — component speedups ≠ function speedups, and `.at` is a known trap.

## CI mypy is stricter than `mypy <file>` — reproduce the CI invocation before pushing
- PR #77 passed `mypy src/cp_measure/primitives/_moments.py` locally but CI failed: CI runs
  `mypy src/cp_measure/core/ --python-version=3.X --ignore-missing-imports`, which type-checks
  THROUGH the import with stricter numpy shape generics. A `raw = numpy.zeros((n,4,4))` then
  `raw = _moment_matrix(...)` reassignment was a shape-narrowing error CI caught.
- **Why:** before pushing, run the CI's exact mypy command (`mypy src/cp_measure/core/ --python-version=3.11 --ignore-missing-imports`), not `mypy <file>`. numpy's shape-typed stubs narrow on `zeros((a,b,c))` literals; assign such arrays once, or annotate `NDArray[floating]`.

## skimage `regionprops` marginal-cost profiling is unreliable — use cProfile
- Adding/removing a property from `regionprops_table` and timing the delta gives noise (often
  NEGATIVE deltas) because skimage caches/shares computation across properties. The "convex hull is
  free / moments are cheap" conclusions from marginal timing were wrong; cProfile (per-function
  tottime) showed convex_hull_image ≈ 39 ms and the einsum moment path ≈ 40–120 ms.
- **Why:** for skimage regionprops cost attribution, trust cProfile tottime, not add-a-prop deltas.
  Also: `regionprops` `centroid` is moment-FREE (uses `coords.mean`), so cheap shape props
  (area/bbox/centroid/extent/area_filled) trigger 0 ms einsum.

## area_convex bit-exact in numba: monotone-chain hull on diamond-offset boundary + skimage raster
- skimage `area_convex` = pixel count inside the convex hull built over pixel *corners* (±0.5
  "diamond" offsets), counted by `grid_points_in_poly`. To reproduce bit-exact AND fast: feed only
  BOUNDARY pixels' diamond-offset points (scaled ×2 to stay integer) to a numba monotone-chain hull,
  then KEEP skimage's `grid_points_in_poly` for the raster (do NOT port pnpoly). 142/142 exact.
  Two gotchas that gave 0/142 until fixed: (1) rasterize in each object's BBOX-LOCAL frame (subtract
  bbox-min), and (2) the hull MUST be over the corner-offset points, not pixel centers (centrosome's
  center-hull diverges ~1%). The numba hull (10 ms) beats scipy QHull; pure-python monotone-chain
  was 17× *slower* than skimage — the hull must be compiled.
- **Why:** for a discretized-geometry feature (pixel count inside a hull), replace the slow
  *construction* and keep the library's exact *rasterizer* — that sidesteps the convention-matching
  risk while still getting the win. And a python proxy can't tell you if a numba kernel will be fast.

## numba can't index a heterogeneous tuple with a variable
- `cells = (a, b, c, d); for i in range(4): lab = cells[i]` fails in `@njit` when a,b,c,d have
  different inferred int widths (e.g. int32 from the array + int64 from `... if ... else 0`). Numba
  needs homogeneous tuples for variable indexing.
- **Why:** in numba kernels, unroll small fixed-size per-cell logic explicitly (or cast all to one
  dtype) instead of building a tuple and indexing it by a loop variable.

## A "sizeshape-only" numba PR had to fix a broken shared file — flag it, don't hide it
- The `integration/all-numba` branch's `bulk._numba_registries` was un-parseable (the merge of all
  numba lanes left interleaved docstrings / em-dashes / duplicate returns — a SyntaxError that only
  surfaces when `bulk` is imported, not on `import cp_measure`). The sizeshape lane couldn't register
  its dispatch without rewriting it.
- **Why:** when a feature PR must repair shared infra to function, do the fix but call it out
  explicitly in the commit/PR body and the handoff (it belongs at the merge/integration level, not
  buried in a feature PR). Also: a Python SyntaxError in a lazily-imported module hides from
  top-level import — test the actual entry point (`featurize`/dispatch), not just `import pkg`.

---

## Session 2026-06-07

### A "perf" PR can REGRESS if it leaves a residual that re-triggers the work it removed
#77 stripped explicit moment props from regionprops' `desired_properties` but kept
`axis_major_length`/`axis_minor_length`/`eccentricity`/`orientation` — and regionprops derives THOSE
from `moments_central`, so the per-region einsum still ran AND the new scatter was added on top → 2D
sizeshape got SLOWER than the pre-opt baseline (198→216ms). Fix = "option B": derive those four from
the scatter's central moments too (`axes_eccentricity_orientation`), making regionprops fully
moment-free. **Why:** removing a feature from a regionprops request does NOT remove its compute if
another requested prop lazily depends on it. Verify with cProfile that the einsum is actually gone,
not just unrequested.

### A single-feature benchmark penalizes a fused multi-feature kernel re-run per call
The numba coloc "lost" to numpy on manders/overlap/rwc only because each of the 5 `get_correlation_*`
re-ran the whole `coloc_per_object` kernel (which computes ALL features). Timed one-at-a-time, numba
did ~5× redundant work. One fused call (`get_correlation_all`) = 38ms vs 140ms numpy (3.7×). **Why:**
when a kernel computes many outputs in one pass, benchmarking/exposing it per-output throws away the
fusion. The fix is compute-once (a fused producer), not a numpy fallback.

### Collapsing a per-feature registry to a single entry breaks featurize's per-group selection
Registering numba correlation as one `{"correlation": fused}` entry broke `featurize` with
`KeyError 'pearson'` — `featurizer._collect_correlation_features` looks up `corr_funcs["pearson"]`,
`["rwc"]`, etc. by group key. Keep per-group registry keys; make them gated wrappers over the fused
producer; expose the fused producer as a direct API for callers who want one pass. **Why:** the
registry keys are a contract consumed by the featurizer's selection, not just a dispatch table. A
dispatch test that only checks registry shape won't catch this — test through `featurize`.

### Fusion is backend-specific, not universal — it only pays where expensive work is SHARED
Numba coloc fusion = 2.7× (the fused per-pixel kernel was the whole cost). Numpy coloc fusion =
1.02× (no win): #69 already made the shared prep cheap (`find_objects` bbox), and the per-feature math
(corrcoef/lstsq/lexsort/costes) is distinct and unshareable. **Why:** before "fusing" a numpy lane,
measure how much is actually shared — if the prep is already cheap and the per-feature compute
dominates, fusion buys nothing.

### Watch out for legacy code paths still in the tree when prototyping against "the numpy backend"
First numpy coloc fusion prototype showed 3.8× SLOWER — it accidentally used the pre-#69 per-object
binmask `_ind` functions (still present, superseded by #69's `_iter_label_pixels`). **Why:** an
optimization PR often leaves the old helpers importable; benchmark against the CURRENT public path
(check which branch/commit has the optimization), not the legacy internals.

### Fair cross-version benchmarking: pin deps, vary only the code
For "pre-opt vs optimized", run all variants on the IDENTICAL dependency set (e.g. numpy 1.26.4) and
swap only cp_measure code — a fresh `pip install pkg==old` pulled numpy 2.4 and would confound the
comparison. Reuse the dev venv via `PYTHONPATH` to the right `src/`, or pin the isolated venv's deps to
match (`uv pip install -c dev_constraints`). Also: separate-process columns drift with machine state
(CPU contention/turbo) — near-1× rows need interleaved re-runs; big wins are robust.

### scipy.ndimage.map_coordinates(mode='constant') zeros the WHOLE point for a coord even 1 ULP outside [0, n-1]
Not a partial bilinear blend at the boundary — a source coordinate of `63.00000000000001` (n=64)
returns `cval=0`, not `≈rec[63]`. The fused upsample's scale `(new-1)/(orig-1)` then `*(orig-1)`
floats the last row/col coordinate ~1 ULP past `new-1` for ~1.3% of (image-size, subsample) combos
(e.g. 158×0.4→63, 80×0.2→16), so every edge-touching object diverged ~0.08. **Why:** to reproduce
map_coordinates exactly you must DROP out-of-bounds pixels (contribute 0), not clamp them in; keep
the per-object count over ALL foreground so the dropped pixel still lands in the mean's denominator.

### skimage clips inertia eigenvalues to >=0; a closed-form half_trace±disc does not
Thin / oblique objects have a near-singular inertia tensor; float error makes the minor eigenvalue
tiny-negative, so `4*sqrt(eig_minor)` is NaN and `sqrt(1 - eig_minor/eig_major)` exceeds 1. skimage's
`inertia_tensor_eigvals` does `np.clip(eigvals, 0, None)`. ~4% of random thin lines trigger it.
**Why:** any analytic eigendecomposition replacing skimage's must replicate the clip, or degenerate
objects silently emit NaN axis_minor / eccentricity>1 (and a RuntimeWarning).

### Preserve the RELEASE column order when porting a feature-assembly dict
get_sizeshape's moment columns ship GROUPED (all SpatialMoment, then all CentralMoment, ...). The
numba `moment_feature_dict` built them INTERLEAVED by (p,q) — a latent column-order bug in the numba
backend that NO test caught (the sizeshape test checks the moment *arrays*; the backend test was
uncollectable from a syntax error). **Why:** column order is an implicit API (featurize builds
columns from dict insertion order); when porting/sharing an assembler, match the released order and
pin it with a hardcoded key-list test against PyPI 0.1.19.

### Reviewing your own PLAN (not just the code) catches load-bearing wrong assumptions
A max `/code-review` of the cleanup plan caught: a wrong API signature (`moment_feature_dict` is
5-arg, not the 2-arg the plan wrote → would TypeError), an over-generalized "one seam" primitive
that fit none of its three callers, an impossible "foundation-PR-first" sequencing (it owned code
that only existed in the consumer PRs), and an internal A/B contradiction. **Why:** plans assert
facts about the code; verify each against the actual source before executing — and verify disputed
empirical claims by RUNNING the code (two finders statically declared the granularity bug
non-existent; reproduction proved it real).

### Installing an old PyPI release in a modern pinned env: pip --no-deps + matching python
`uv pip install cp_measure==0.1.19` re-resolves and rejects the pinned modern scipy (0.1.19 caps it);
`uv ... --no-deps` still resolves. Use real `python -m pip install --no-deps pkg==X` (pip --no-deps
skips resolution entirely), AND pin the venv python to satisfy the wheel's Requires-Python (0.1.19
needs <3.15; uv defaulted the new env to 3.9 → "no matching distribution"). **Why:** benchmarking a
released baseline against modern deps needs deps layered first, then the package with deps ignored.
