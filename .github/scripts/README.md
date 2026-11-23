# PR benchmarks

`run_benchmark.sh` installs the actual PR head and current main in separate
virtual environments and runs the same head-owned harness in both. Library
threads are pinned to one. JSON measurements, NPZ correctness outputs and
Markdown tables are uploaded even when the benchmark job fails.

- `benchmark.py`: existing individual-function timings on a synthetic matrix.
- `benchmark_colocalization.py`: small multi-channel **featurizer** benchmark,
  exercising actual channel-pair dispatch rather than only single-pair functions.

The featurizer case uses a seeded 64×64, five-channel uint8 image, four objects,
all four colocalization measurements, and an explicit 255-level Costes grid.
Both default and CP-matched colocalization settings are measured. “Matched” does
not mean that CellProfiler runs in CI or that full-profile equivalence is proven.
Each policy includes the package's default pair selection and an ordered control;
on older main without `costes_pair_mode`, the default already supplies that control.

An untimed instrumented call warms the full path and counts actual pair tasks:
20 Pearson, 10 Manders, 10 RWC, and 10 unordered or 20 ordered Costes pairs.
These are pair executions, not per-object fit counts. Three subsequent calls
are timed without instrumentation, input generation, imports or serialization.
Tables report median and min/max, including when no speedup was observed.

Correctness failures fail CI: retained outputs must match exactly across main/head
and default/ordered controls, including NaN and signed-infinity patterns. Only
extra reverse-fit Costes columns may disappear. Rows, dtypes, input hashes,
non-pair settings and relevant dependency versions are checked. This intentionally
requires revisiting the gate for a future deliberate colocalization numerical
change, rather than silently treating changed outputs as a speedup.

Timings are informational, **not** regression thresholds. Shared runners and
three repetitions do not establish confidence intervals. This tiny fixture
complements, rather than replaces, real-data full-profile/colocalization studies.

NumPy always runs. Changes to featurizer/dispatch, colocalization, Numba code or
this harness also require Numba featurizer validation. The older exploratory
Numba function table remains best-effort; the new correctness pass is not.

For a local run with the intended package available in each interpreter:

```sh
python .github/scripts/benchmark_colocalization.py run \
  --backend numpy --revision HEAD_SHA --out head.json
# Run the same script with the main environment/interpreter:
python .github/scripts/benchmark_colocalization.py run \
  --backend numpy --revision MAIN_SHA --out main.json
python .github/scripts/benchmark_colocalization.py compare \
  --base main.json --head head.json --md comparison.md
```

Set the same single-thread environment variables as `run_benchmark.sh` and repeat
with `--backend numba` when applicable. Keep the adjacent `.npz` files with JSON.
