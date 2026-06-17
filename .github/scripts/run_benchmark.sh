#!/usr/bin/env bash
# Time every get_* in two isolated envs on identical inputs, where only cp_measure.core differs.
# Runs from the PR-head checkout (which carries the tooling); main is fetched as a worktree and the
# head's synth.py + _bench/ vendored into it. Usage: run_benchmark.sh <matrix-preset> <out-dir>
set -euo pipefail

MATRIX="${1:-ci}"
OUT="${2:-bench-out}"
HEAD_DIR="$(pwd)"
WORK="$(mktemp -d)"
SHARED="$WORK/fixtures"
mkdir -p "$OUT"
trap 'git worktree remove --force "$WORK/main" 2>/dev/null || true; rm -rf "$WORK"' EXIT

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1

# six is a centrosome runtime dep not declared in its metadata; install it into the bench venvs only.
echo "::group::head env: install, build fixtures, run"
uv venv "$WORK/venv-head"
uv pip install --python "$WORK/venv-head/bin/python" -e "$HEAD_DIR" six
"$WORK/venv-head/bin/python" -m cp_measure._bench.fixtures --out "$SHARED" --matrix "$MATRIX"
"$WORK/venv-head/bin/python" -m cp_measure._bench.run --fixtures "$SHARED" --out "$OUT/head.json"
echo "::endgroup::"

echo "::group::main worktree: vendor head tooling, install, run (same fixtures)"
git fetch --no-tags --depth=1 origin main
git worktree add --detach "$WORK/main" origin/main
cp "$HEAD_DIR/src/cp_measure/synth.py" "$WORK/main/src/cp_measure/synth.py"
rm -rf "$WORK/main/src/cp_measure/_bench"
cp -r "$HEAD_DIR/src/cp_measure/_bench" "$WORK/main/src/cp_measure/_bench"
uv venv "$WORK/venv-main"
uv pip install --python "$WORK/venv-main/bin/python" -e "$WORK/main" six
"$WORK/venv-main/bin/python" -m cp_measure._bench.run --fixtures "$SHARED" --out "$OUT/main.json"
echo "::endgroup::"

echo "::group::compare"
"$WORK/venv-head/bin/python" -m cp_measure._bench.compare \
  --base "$OUT/main.json" --head "$OUT/head.json" --md "$OUT/table.md"
cat "$OUT/table.md"
echo "::endgroup::"
