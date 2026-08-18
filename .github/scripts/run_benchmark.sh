#!/usr/bin/env bash
# Install the PR head and main into separate venvs and run benchmark.py (from this checkout) in
# each, then compare. Each run regenerates the same seeded inputs, so nothing is shared on disk.
# When the PR targets the numba backend, also times the numba backend head-vs-main.
# Usage: run_benchmark.sh <out-dir> <head-commit-sha>
set -euo pipefail

# Pin BLAS/OpenMP to one thread so timings reflect algorithmic cost, not incidental parallelism
# (cp_measure keeps core functions single-threaded; the batch layer is what parallelises).
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

OUT="${1:-bench-out}"
COMMIT="${2:-}"
HEAD_DIR="$(pwd)"
WORK="$(mktemp -d)"
BENCH="$HEAD_DIR/.github/scripts/benchmark.py"
mkdir -p "$OUT"
trap 'git worktree remove --force "$WORK/main" 2>/dev/null || true; rm -rf "$WORK"' EXIT

# Detect whether this PR targets the numba backend: does it touch a numba implementation or the
# dispatch registry? If so, install [numba] on both venvs and add a numba head-vs-main pass below.
git fetch --no-tags origin main
BASE="$(git merge-base HEAD origin/main || echo origin/main)"
NUMBA=0
if git diff --name-only "$BASE" HEAD | grep -qE '^src/cp_measure/(core/numba/|bulk\.py)$'; then
  NUMBA=1
fi
echo "numba-targeting PR: $NUMBA"
# On a numba PR both venvs get [numba] so each backend can be timed head-vs-its-own-main.
EXTRA=""
[ "$NUMBA" = 1 ] && EXTRA="[numba]"

# six is a centrosome runtime dep not declared in its metadata; install it into the bench venvs only.
echo "::group::PR head env"
uv venv "$WORK/venv-head"
uv pip install --python "$WORK/venv-head/bin/python" -e "${HEAD_DIR}${EXTRA}" six
"$WORK/venv-head/bin/python" "$BENCH" run --out "$OUT/head.json"
echo "::endgroup::"

echo "::group::main env"
git worktree add --detach "$WORK/main" origin/main
uv venv "$WORK/venv-main"
uv pip install --python "$WORK/venv-main/bin/python" -e "${WORK}/main${EXTRA}" six
"$WORK/venv-main/bin/python" "$BENCH" run --out "$OUT/main.json"
echo "::endgroup::"

"$WORK/venv-head/bin/python" "$BENCH" compare \
  --base "$OUT/main.json" --head "$OUT/head.json" --commit "$COMMIT" \
  --base-name main --head-name head --md "$OUT/table.md"

# numba PRs: the numpy table above tracks the numpy backend head-vs-main; this one tracks the numba
# backend head-vs-main (numba@main is the current baseline — un-ported features fall back to numpy —
# so this measures how far this PR moves the numba backend forward).
if [ "$NUMBA" = 1 ]; then
  echo "::group::numba runs"
  "$WORK/venv-head/bin/python" "$BENCH" run --accelerator numba --out "$OUT/head-numba.json"
  "$WORK/venv-main/bin/python" "$BENCH" run --accelerator numba --out "$OUT/main-numba.json"
  "$WORK/venv-head/bin/python" "$BENCH" compare \
    --base "$OUT/main-numba.json" --head "$OUT/head-numba.json" --commit "$COMMIT" \
    --base-name "numba main" --head-name "numba head" --md "$OUT/table-numba.md"
  { printf '\n\n'; cat "$OUT/table-numba.md"; } >> "$OUT/table.md"
  echo "::endgroup::"
fi

cat "$OUT/table.md"
