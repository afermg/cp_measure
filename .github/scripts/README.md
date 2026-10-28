# PR benchmarks

The `benchmark` workflow runs `benchmark.py` from the PR checkout against both the
PR and main environments, then posts a sticky timing comment. Both environments
use the same generator and workload selection, including when main predates a
benchmark-driver change.

## Representative workloads

These are **explicit image-size/object-count pairs**, not a Cartesian product:

| Workload           | Image size  | Objects | Motivation                                        |
| ------------------ | ----------- | ------- | ------------------------------------------------- |
| Tiles              | 256 × 256   | 16      | Small tile-based approaches                       |
| Intermediate       | 512 × 512   | 64      | Intermediate image scale                          |
| Cell Painting-like | 1024 × 1024 | 128     | Object density closer to our usual JUMP workloads |
| Spatial-omics-like | 1024 × 1024 | 256     | A denser workload proxy                           |

The first, second and fourth pairs are retained; the Cell Painting-like pair is
added between them. CI previously crossed three sizes with three object counts
(nine pairs). It now uses these four representative pairs: **12 seeded scenes per
backend/revision instead of 27**, with no extra 128-object cases at smaller sizes.

All cases use the existing synthetic 2D ellipse-grid generator and two Gaussian-blob
intensity channels. These labels describe workload proxies, **not real JUMP or
spatial-omics datasets**, biological realism, or 3D coverage. Ellipse dimensions
scale with the grid: fewer objects at the same image size also means larger
objects, not just fewer fixed-size objects.

## Timing and comparison

- Seeds remain `0, 1, 2`; each function/scene has one excluded warmup and three
  timed repetitions, with the existing 120-second case limit.
- Core and correlation functions, including the explicit Costes threshold grid,
  are unchanged. NumPy runs on every PR; Numba-targeting PRs also exercise the
  same workloads through the optional Numba passes in `run_benchmark.sh`.
- JSON retains the existing scene keys and adds readable workload names and seed
  metadata. Comparison output lists only measured pairs, in workload order; it
  also accepts historical JSON without the new metadata.
- Each reported speedup is the ratio of medians over the available successful
  seed/repetition timings. The existing 1.05× movement filter is unchanged;
  Numba inclusion is still filtered against Numba main while displayed speedups
  use NumPy main as the baseline.
- Warm-call timings exclude imports and JIT warmup. Shared-runner noise can
  affect small changes; these are per-function results, not end-to-end pipeline
  measurements or confidence intervals. Missing successful timings display `—`
  for a function included by another workload.

To run the driver in an environment with `cp_measure` installed:

```bash
python .github/scripts/benchmark.py run --out timings.json
python .github/scripts/benchmark.py run --accelerator numba --out timings-numba.json
python .github/scripts/benchmark.py compare --base base.json --head head.json
```

For the complete CI head-vs-main workflow, from a clean checkout:

```bash
bash .github/scripts/run_benchmark.sh bench-out "$(git rev-parse HEAD)"
```

Driver regressions are covered by `test/test_ci_benchmark.py` and
`test/test_costes_benchmarks.py`; these tests do not run the full timing battery.
