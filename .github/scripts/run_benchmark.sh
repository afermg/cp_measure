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
# Capture the diff first, then grep the string: `git … | grep -q` under `pipefail` can report the
# pipe's SIGPIPE (141) when grep exits early on a match, spuriously flipping detection to 0.
CHANGED="$(git diff --name-only "$BASE" HEAD)"
if grep -qE '^src/cp_measure/core/numba/|^src/cp_measure/bulk\.py$' <<< "$CHANGED"; then
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

# numba PRs: the numpy table above tracks the numpy backend head-vs-main. This one times the numba
# backend head-vs-main and reports the speedup vs numpy main for the features THIS PR changes in
# numba. --filter-base picks those features (head-numba differs from numba@main), while --base makes
# the printed speedup numpy@main/numba-head. numba@main falls back to numpy for un-ported features,
# so for a greenfield feature this IS the whole story; the day a follow-up PR improves an existing
# numba impl, add a second table with --base main-numba.json for the incremental gain.
# The numba backend is optional: a failure to install or run it (e.g. main not yet carrying the
# backend, or a wheel/arch gap) must NOT sink the numpy table — the sticky-comment step only runs
# on success. Run the pass in a subshell so any failure is caught here, not propagated by `set -e`.
if [ "$NUMBA" = 1 ]; then
  echo "::group::numba runs"
  if (
      "$WORK/venv-head/bin/python" "$BENCH" run --accelerator numba --out "$OUT/head-numba.json" &&
      "$WORK/venv-main/bin/python" "$BENCH" run --accelerator numba --out "$OUT/main-numba.json" &&
      "$WORK/venv-head/bin/python" "$BENCH" compare \
        --base "$OUT/main.json" --filter-base "$OUT/main-numba.json" --head "$OUT/head-numba.json" \
        --commit "$COMMIT" --base-name "numpy main" --head-name "numba head" \
        --md "$OUT/table-numba.md"
    ); then
    # Append when this PR moved a numba feature. A bulk.py-only PR that touches no numba path moves
    # nothing here, so nothing misleading is appended.
    grep -q "No function moved" "$OUT/table-numba.md" ||
      { printf '\n\n'; cat "$OUT/table-numba.md"; } >> "$OUT/table.md"
  else
    printf '\n\n_numba benchmark failed for this PR; the numpy results above are unaffected._\n' \
      >> "$OUT/table.md"
  fi
  echo "::endgroup::"
fi

cat "$OUT/table.md"
