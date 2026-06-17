#!/usr/bin/env bash
# Install the PR head and main into separate venvs and run benchmark.py (from this checkout) in
# each, then compare. Each run regenerates the same seeded inputs, so nothing is shared on disk.
# Usage: run_benchmark.sh <out-dir> <head-commit-sha>
set -euo pipefail

OUT="${1:-bench-out}"
COMMIT="${2:-}"
HEAD_DIR="$(pwd)"
WORK="$(mktemp -d)"
BENCH="$HEAD_DIR/.github/scripts/benchmark.py"
mkdir -p "$OUT"
trap 'git worktree remove --force "$WORK/main" 2>/dev/null || true; rm -rf "$WORK"' EXIT

# six is a centrosome runtime dep not declared in its metadata; install it into the bench venvs only.
echo "::group::PR head env"
uv venv "$WORK/venv-head"
uv pip install --python "$WORK/venv-head/bin/python" -e "$HEAD_DIR" six
"$WORK/venv-head/bin/python" "$BENCH" run --out "$OUT/head.json"
echo "::endgroup::"

echo "::group::main env"
git fetch --no-tags --depth=1 origin main
git worktree add --detach "$WORK/main" origin/main
uv venv "$WORK/venv-main"
uv pip install --python "$WORK/venv-main/bin/python" -e "$WORK/main" six
"$WORK/venv-main/bin/python" "$BENCH" run --out "$OUT/main.json"
echo "::endgroup::"

"$WORK/venv-head/bin/python" "$BENCH" compare \
  --base "$OUT/main.json" --head "$OUT/head.json" --commit "$COMMIT" --md "$OUT/table.md"
cat "$OUT/table.md"
