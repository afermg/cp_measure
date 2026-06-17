# cp_measure — Implementation & Benchmark Inventory

_Generated 2026-05-29. Covers the canonical upstream and all 5 local variants._

## A. Implementations

| Repo (dir) | GitHub | HEAD | Acceleration tech | Role |
|---|---|---|---|---|
| `cp_measure` | `afermg/cp_measure` (origin/main) | `54e5c09` v0.1.19 | none — numpy/scipy/skimage/centrosome/mahotas | **canonical baseline** |
| `cp_measure_fast` | `timtreis/cp_measure_fast` | `f41b5c5` | Numba `@njit` (selective) | numba variant |
| `cp_measure_jax` | `timtreis/cp_measure_jax` | `ed8495c` | JAX (jit/lax) + some Numba; GPU-conditional | jax variant |
| `cp_measure_speed` | `timtreis/cp_measure_speed` | `2c09c98` | pure NumPy vectorization (numba dep present but unused) | speed variant |
| `cp_measure_rust` | `timtreis/cp_measure_rust` | `03247cb` | Rust + PyO3 (`_cpmeasurerust`) | rust variant |

### Per-function backend — what is ACTUALLY used

Legend: ✅ = genuinely accelerated with that variant's tech · 🟡 = partial/mixed (some jitted, rest numpy; or GPU-only) · ⬜ = numpy passthrough (lives in the variant but unchanged) · ❌ = not implemented/not ported

| Function | true (baseline) | fast (numba) | jax | speed | rust |
|---|---|---|---|---|---|
| **intensity** | numpy | ⬜ numpy | ⬜ numpy | ✅ **numpy-vectorized 32×** | ✅ rust |
| **sizeshape** | numpy/skimage | ⬜ numpy (skimage regionprops) | 🟡 numba+numpy | ⬜ unchanged | 🟡 partial (feret+zernike only) |
| **zernike** | numpy/centrosome | 🟡 numba complex-sum kernel | 🟡 jax GPU + numpy CPU | ⬜ unchanged | ✅ rust |
| **radial_distribution** | numpy/scipy.sparse | ⬜ numpy | 🟡 jax `lax.while_loop` + numpy hist | ⬜ unchanged | ❌ (only zernike-radial ported) |
| **radial_zernikes** | centrosome | 🟡 numba complex-sum + boundary kernel | 🟡 jax GPU + numpy CPU | ⬜ unchanged | ✅ rust |
| **texture** | mahotas/skimage | ⬜ numpy (mahotas) | 🟡 jax GPU GLCM + numpy CPU | ⬜ unchanged | ✅ rust |
| **granularity** | skimage morphology | ✅ **numba (2D); 3D falls back** | 🟡 jax morphology kernels + numpy spectrum | ⬜ unchanged | ❌ not implemented |
| **pearson** | numpy | ⬜ numpy | ⬜ numpy | ⬜ unchanged | ✅ rust |
| **manders_fold** | numpy/scipy | ⬜ numpy | 🟡 jax GPU-only / numpy CPU | ⬜ unchanged | ✅ rust |
| **rwc** | numpy | ⬜ numpy | 🟡 jax GPU-only / numpy CPU | ⬜ unchanged | ✅ rust |
| **costes** | numpy/scipy | ⬜ numpy | ⬜ numpy (sequential bisection) | ⬜ unchanged | ✅ rust |
| **neighbors** | scipy/centrosome | ⬜ numpy | ❌ delegates to baseline | ⬜ unchanged | ❌ not implemented |
| **overlap** | centrosome/numpy | ⬜ numpy | ❌ delegates to baseline | ⬜ unchanged | ✅ rust |

### Key takeaways
- **fast (numba):** only **granularity** (2D) is a real jitted hot path. **zernike / radial_zernikes** are numba-*partial* (complex-sum + boundary kernels). The other ~10 functions are pure numpy — present but unaccelerated.
- **jax:** genuinely JAX only when a GPU backend is active. **granularity** + **radial_distribution** use jax kernels even on CPU; **zernike/radial_zernikes/texture/manders/rwc** are GPU-path jax with numpy CPU fallback; **intensity/pearson/costes** are numpy; **neighbors/overlap** not ported. So a CPU run is mostly numpy.
- **speed:** intentionally numpy-only ("no Rust/Cython/Numba" per CLAUDE.md). **intensity** is the main optimization (32× large / 11× small). CORRECTION (2026-05-30, `primitive_existence_matrix.md`): **granularity is also functionally changed** (re-enables `subsample_size<1` resampling baseline commented out) — its timing isn't apples-to-apples with baseline. The other 4 core files differ by typing/cosmetics only (functionally identical). [NB: this whole inventory predates the corrected benchmark; treat the `fast`/`jax` lines above as superseded by `lessons.md` + the corrected REPORT.md — `fast` actually accelerates ~all functions.]
- **rust:** the most complete port — **11/13** functions native Rust. Missing: **granularity**, **neighbors** (and `radial_distribution` proper — only the zernike-radial is ported; `sizeshape` is partial: feret + zernike only).

> ⚠️ The "numba version" and "jax version" labels are aspirational for many functions: per the matrix, most functions in each fall back to numpy. Benchmarks must record the **actual** backend per function (this matrix), not the repo label.

## B. Benchmark options

### The 3-tier benchmark (confirmed)
Defined in `cp_measure_speed/benchmarks/bench_harness.py:12-15` (replicated in `cp_measure_fast`):

| Tier | Image | Objects | Source |
|---|---|---|---|
| `tiny` | 256×256 synthetic | 2 | `cp_measure.examples` |
| `small` | 540×540 real crop | ~50 cells | CellPainting AASDHPPT_01 |
| `large` | 1080×1080 real full | ~100 cells | CellPainting AASDHPPT_01 |

`cp_measure_jax/benchmarks/compare_implementations.py` extends this with a 4th tier **`xlarge`** = `large` batched ×100 (GPU throughput).

### Benchmark entrypoints
| Repo | File | Invocation | Tiers | Compares |
|---|---|---|---|---|
| `cp_measure_speed` | `benchmarks/bench_harness.py` | `uv run python benchmarks/bench_harness.py --all --mode {tiny\|small\|large}` | 3-tier | self vs reference, + correctness (rtol 1e-8) |
| `cp_measure_fast` | `benchmarks/bench_harness.py` | same CLI | 3-tier | numba vs reference |
| `cp_measure_jax` | `benchmarks/compare_implementations.py` | `python … --all --mode {tiny\|small\|large\|xlarge} [--batch-size N]` | 4-tier | reference vs numba vs jax (rtol 1e-5) |
| `cp_measure_bench` | `bench.py` | `python bench.py` | object-count sweep `[50,150,400]` @ 1080² (+ scaling phase) | self-profiling, plots/reports |
| `cp_measure_bench` | `compare_rust_python*.py` (×5) | `python compare_rust_python_<fn>.py` | object counts `[50,150,400]` | Rust vs Python (intensity, ferret, texture, correlation) |
| `cp_measure_speed` | `benchmarks/benchmark.py` | `python benchmark.py` | full dataset | cp_measure vs **CellProfiler** (parquet, Pearson r/R²) |

### Recommendation
- **Cross-variant timing + correctness:** `bench_harness.py` 3-tier (`tiny/small/large`) is the canonical, consistent harness — extend it to also drive `speed` and `rust`.
- **Object-count scaling + Rust comparison + report generation:** `cp_measure_bench` is the most complete reporting suite.
- **Gap:** no single harness yet runs all 5 implementations together; `cp_measure_bench` (rust+python) and `compare_implementations.py` (ref+numba+jax) cover different subsets.
