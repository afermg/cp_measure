#!/usr/bin/env bash
# Benchmark the PR head's measurement functions against current `main`.
#
# Strategy: time every get_* function in two isolated envs on IDENTICAL inputs, where only the
# measurement code differs. The generator (synth.py) and the bench tooling (_bench/) are vendored
# from HEAD into both checkouts, so a PR that changes the generator/tooling cannot skew the
# comparison — only cp_measure.core.* differs between the two runs.
#
# Usage: run_benchmark.sh <matrix-preset> <out-dir>   (run from the HEAD checkout; needs origin/main)
set -euo pipefail

MATRIX="${1:-ci}"
OUT="${2:-bench-out}"
HEAD_DIR="$(pwd)"
WORK="$(mktemp -d)"
SHARED="$WORK/fixtures"
mkdir -p "$OUT"

# Single-thread timing: representative (the repo forbids in-function parallelism) and low-noise.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1

vendor_tooling() {  # copy HEAD's generator + bench tooling into a worktree so both runs share them
  local dest="$1"
  cp "$HEAD_DIR/src/cp_measure/synth.py" "$dest/src/cp_measure/synth.py"
  rm -rf "$dest/src/cp_measure/_bench"
  cp -r "$HEAD_DIR/src/cp_measure/_bench" "$dest/src/cp_measure/_bench"
}

echo "::group::HEAD env: install, build fixtures, run"
uv venv "$WORK/venv-head"
uv pip install --python "$WORK/venv-head/bin/python" -e "$HEAD_DIR"
"$WORK/venv-head/bin/python" -m cp_measure._bench.fixtures --out "$SHARED" --matrix "$MATRIX"
"$WORK/venv-head/bin/python" -m cp_measure._bench.run --fixtures "$SHARED" --out "$OUT/head.json"
echo "::endgroup::"

echo "::group::main worktree: vendor HEAD tooling, install, run (same fixtures)"
git fetch --no-tags --depth=1 origin main
git worktree add --detach "$WORK/main" origin/main
vendor_tooling "$WORK/main"
uv venv "$WORK/venv-main"
uv pip install --python "$WORK/venv-main/bin/python" -e "$WORK/main"
"$WORK/venv-main/bin/python" -m cp_measure._bench.run --fixtures "$SHARED" --out "$OUT/main.json"
git worktree remove --force "$WORK/main" || true
echo "::endgroup::"

echo "::group::compare"
"$WORK/venv-head/bin/python" -m cp_measure._bench.compare \
  --base "$OUT/main.json" --head "$OUT/head.json" --md "$OUT/table.md"
cat "$OUT/table.md"
echo "::endgroup::"
