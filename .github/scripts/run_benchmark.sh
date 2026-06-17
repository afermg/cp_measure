#!/usr/bin/env bash
# Time every get_* in two isolated envs on IDENTICAL inputs, where only cp_measure.core differs.
# Runs FROM a `main` checkout (which carries the tooling); the PR head supplies only the measurement
# code. main's synth.py + _bench/ are vendored into the head worktree so a PR that doesn't carry the
# tooling (the normal case) is still benchmarked, and generator/tooling changes can't skew it.
# Usage: run_benchmark.sh <matrix-preset> <head-ref-or-sha> <out-dir>   (run from the main checkout)
set -euo pipefail

MATRIX="${1:-ci}"
HEAD_REF="$2"
OUT="${3:-bench-out}"
MAIN_DIR="$(pwd)"
WORK="$(mktemp -d)"
SHARED="$WORK/fixtures"
mkdir -p "$OUT"
trap 'git worktree remove --force "$WORK/head" 2>/dev/null || true; rm -rf "$WORK"' EXIT

# Single-thread timing: representative (the repo forbids in-function parallelism) and low-noise.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1

vendor_tooling() { # copy main's generator + bench tooling into the head worktree
  cp "$MAIN_DIR/src/cp_measure/synth.py" "$1/src/cp_measure/synth.py"
  rm -rf "$1/src/cp_measure/_bench"
  cp -r "$MAIN_DIR/src/cp_measure/_bench" "$1/src/cp_measure/_bench"
}

echo "::group::main env: install, build fixtures, run"
uv venv "$WORK/venv-main"
uv pip install --python "$WORK/venv-main/bin/python" -e "$MAIN_DIR"
"$WORK/venv-main/bin/python" -m cp_measure._bench.fixtures --out "$SHARED" --matrix "$MATRIX"
"$WORK/venv-main/bin/python" -m cp_measure._bench.run --fixtures "$SHARED" --out "$OUT/main.json"
echo "::endgroup::"

echo "::group::head worktree: vendor main tooling, install, run (same fixtures)"
git fetch --no-tags --depth=1 origin "$HEAD_REF"
git worktree add --detach "$WORK/head" FETCH_HEAD
vendor_tooling "$WORK/head"
uv venv "$WORK/venv-head"
uv pip install --python "$WORK/venv-head/bin/python" -e "$WORK/head"
"$WORK/venv-head/bin/python" -m cp_measure._bench.run --fixtures "$SHARED" --out "$OUT/head.json"
echo "::endgroup::"

echo "::group::compare"
"$WORK/venv-main/bin/python" -m cp_measure._bench.compare \
  --base "$OUT/main.json" --head "$OUT/head.json" --md "$OUT/table.md"
cat "$OUT/table.md"
echo "::endgroup::"
