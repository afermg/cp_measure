#!/usr/bin/env bash
# Install the PR head and main into separate venvs and run benchmark.py (from this checkout) in
# each, then compare. Each run regenerates the same seeded inputs, so nothing is shared on disk.
# When the PR targets the numba backend, additionally times numba-vs-numpy on the head build.
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
# dispatch registry? If so, install the [numba] extra on head and add a numba-vs-numpy pass below.
git fetch --no-tags origin main
BASE="$(git merge-base HEAD origin/main || echo origin/main)"
NUMBA=0
if git diff --name-only "$BASE" HEAD | grep -qE '^src/cp_measure/(core/numba/|bulk\.py)$'; then
  NUMBA=1
fi
echo "numba-targeting PR: $NUMBA"
HEAD_EXTRA=""
[ "$NUMBA" = 1 ] && HEAD_EXTRA="[numba]"

# six is a centrosome runtime dep not declared in its metadata; install it into the bench venvs only.
echo "::group::PR head env"
uv venv "$WORK/venv-head"
uv pip install --python "$WORK/venv-head/bin/python" -e "${HEAD_DIR}${HEAD_EXTRA}" six
"$WORK/venv-head/bin/python" "$BENCH" run --out "$OUT/head.json"
echo "::endgroup::"

echo "::group::main env"
git worktree add --detach "$WORK/main" origin/main
uv venv "$WORK/venv-main"
uv pip install --python "$WORK/venv-main/bin/python" -e "$WORK/main" six
"$WORK/venv-main/bin/python" "$BENCH" run --out "$OUT/main.json"
echo "::endgroup::"

"$WORK/venv-head/bin/python" "$BENCH" compare \
  --base "$OUT/main.json" --head "$OUT/head.json" --commit "$COMMIT" --md "$OUT/table.md"

# numba PRs: numpy-vs-numpy above catches regressions; this pass shows the actual backend speedup,
# numba vs numpy on the same head build (independent of what is merged on main).
if [ "$NUMBA" = 1 ]; then
  echo "::group::numba head run"
  "$WORK/venv-head/bin/python" "$BENCH" run --accelerator numba --out "$OUT/head-numba.json"
  "$WORK/venv-head/bin/python" "$BENCH" compare \
    --base "$OUT/head.json" --head "$OUT/head-numba.json" --commit "$COMMIT" \
    --base-name numpy --head-name numba --md "$OUT/table-numba.md"
  { printf '\n\n'; cat "$OUT/table-numba.md"; } >> "$OUT/table.md"
  echo "::endgroup::"
fi

cat "$OUT/table.md"
