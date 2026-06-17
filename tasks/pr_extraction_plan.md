# Extract `to_bzyx` into its own PR (#59) — restructuring runbook

Goal: pull the shared `(B,Z,Y,X)` batch-shape helper out of the granularity PR so it is a tiny,
review-first PR; rebase the 3 feature PRs onto it so none gates another.

## Target topology
```
#59  feat/bzyx-shape   to_bzyx only (shapes.py + test_primitives_shapes.py)   base: origin/main
#56  granularity       base: feat/bzyx-shape
#57  intensity→bzyx    base: feat/bzyx-shape
#58  zernike           base: feat/bzyx-shape
```
After #59 merges to main, the 3 features become independent siblings on main.

## Fixed SHAs (verified 2026-06-03)
- `origin/main` = `4ca0a35` (base for #59; shapes.py/granularity NOT yet on main — confirmed)
- #56 `feat/accelerator-numba-granularity` tip = **`8d84546`**  ← the cutoff for replaying #57/#58
- #57 `feat/intensity-bzyx` tip = `5d08a15`
- #58 `feat/numba-zernike` tip = `b073742`
- (local `main` is a stale `199b4e5` — IGNORE it; always use `origin/main`.)

## Conflict footprint (measured — small)
- `primitives/shapes.py` + `test_primitives_shapes.py`: only in #56 → move to #59.
- #57 touches ONLY `test_backend_correctness.py` (+28 lines, intensity tests) → ~1 trivial conflict.
- #58 touches `bulk.py` (+14/-9 registry), `__init__.py` (+17), `test_backend_correctness.py` (+6) →
  conflicts where its additions sat next to granularity's; resolve by keeping ONLY zernike's lines.
- All `uv run` via `export PATH="/home/icb/tim.treis/.pixi-home/bin:$PATH"`.

## Phase 0 — safety net (local backups; remotes are the real backup until force-push)
```
git branch backup/56 feat/accelerator-numba-granularity
git branch backup/57 feat/intensity-bzyx
git branch backup/58 feat/numba-zernike
git fetch origin
```
All 3 branches are pushed and working trees clean (only untracked tasks/). Rollback = reset each
branch to its backup/SHA and force-push.

## Phase 1 — create #59 (to_bzyx)
```
git checkout -b feat/bzyx-shape origin/main
git checkout feat/accelerator-numba-granularity -- \
    src/cp_measure/primitives/shapes.py test/test_primitives_shapes.py
git add -A && git commit -m "feat(primitives): canonical (B,Z,Y,X) batch shape helper (to_bzyx)"
uv run pytest test/test_primitives_shapes.py -q        # expect 11 passed
uvx ruff@0.12.1 format --check src/cp_measure/primitives/shapes.py test/test_primitives_shapes.py
uvx ruff@0.12.1 check src/cp_measure/primitives/shapes.py test/test_primitives_shapes.py
git push -u origin feat/bzyx-shape
gh pr create --repo afermg/cp_measure --draft --base main --head feat/bzyx-shape \
  --title "feat(primitives): (B,Z,Y,X) batch-shape helper (to_bzyx)" \
  --body "<foundational shared helper used by the numba intensity/granularity/zernike backends; \
review-first so the feature PRs (#56/#57/#58) unblock>"
```

## Phase 2 — rebase #56 (granularity) onto #59
```
git checkout feat/accelerator-numba-granularity
git rebase --onto feat/bzyx-shape origin/main
```
- Expect an **add/add conflict** on `shapes.py` + `test_primitives_shapes.py` (identical content,
  now in the base). Resolve by keeping either side (identical): `git checkout --theirs <files>;
  git add <files>; git rebase --continue`.
- Verify the net PR diff no longer lists those two files:
  `git diff feat/bzyx-shape...HEAD --stat`  (should show only granularity files + wiring)
- `uv run pytest -q` (full suite green) ; ruff check.
- `git push --force-with-lease`
- `gh pr edit 56 --repo afermg/cp_measure --base feat/bzyx-shape`

## Phase 3 — rebase #57 (intensity) onto #59  (drops granularity from its base)
```
git checkout feat/intensity-bzyx
git rebase --onto feat/bzyx-shape 8d84546        # replay ONLY #57's own commits
```
- Conflict likely in `test_backend_correctness.py`: #57's intensity tests were added next to #56's
  granularity dispatch test. Resolve by keeping ONLY #57's intensity-test additions (granularity's
  lines belong to #56, not here).
- Verify: `git diff feat/bzyx-shape...HEAD --stat` → only intensity migration + intensity tests.
- `uv run pytest test/test_backend_correctness.py -q` ; ruff.
- `git push --force-with-lease` ; `gh pr edit 57 --repo afermg/cp_measure --base feat/bzyx-shape`

## Phase 4 — rebase #58 (zernike) onto #59  (drops granularity from its base)
```
git checkout feat/numba-zernike
git rebase --onto feat/bzyx-shape 8d84546        # replay ONLY #58's own commits
```
- Conflicts in `bulk.py` (`_numba_registries`), `core/numba/__init__.py`, `test_backend_correctness.py`:
  #58 added zernike/radial_zernikes entries on top of #56's granularity entries. On the #59 base the
  registry/init/test have NEITHER granularity nor zernike → resolve to keep ONLY zernike's additions
  (intensity stays from main; granularity is #56's, not here). Most conflict-prone phase but small
  (+14/+17/+6 lines).
- Verify: `git diff feat/bzyx-shape...HEAD --stat` → only zernike files + zernike wiring/tests;
  bulk.py adds ONLY zernike/radial_zernikes (NOT granularity).
- `uv run pytest test/test_zernike_kernels.py test/test_zernike_backend.py test/test_backend_correctness.py -q`
  ; ruff.
- `git push --force-with-lease` ; `gh pr edit 58 --repo afermg/cp_measure --base feat/bzyx-shape`

## Phase 5 — verify topology + clean up
- `gh pr list` → #59 base main; #56/#57/#58 base feat/bzyx-shape.
- Each PR diff contains ONLY its own content (no shapes.py in #56/#57/#58; no granularity in #57/#58).
- Merge order: **#59 first** → then #56/#57/#58 in any order. As each feature lands on main, the next
  resolves a trivial append conflict in bulk.py/__init__/test (one-line). After #59 merges, GitHub
  retargets the feature PR bases to main automatically (or `gh pr edit --base main`).
- Delete local `backup/*` branches once the new branches are confirmed pushed + green.

## If the rebases get hairy (fallback for #57/#58 only)
Since #57/#58 are draft and don't need granularity, reconstruct fresh instead of rebasing through
granularity conflicts:
- `git checkout -b feat/intensity-bzyx-new feat/bzyx-shape`, re-apply the intensity changes by
  `git checkout backup/57 -- <intensity source files>`, then hand-merge ONLY the intensity additions
  into the shared `test_backend_correctness.py` (which on #59 is main's version). Recommit, repoint PR.
- Lower-fidelity history but avoids untangling granularity context. Use only if Phase 3/4 conflicts
  are worse than expected.

## Risks / notes
- Force-push rewrites #56/#57/#58 history. They are DRAFT with no review comments yet → low risk of
  detaching review threads. Do it before Alan starts reviewing.
- `--force-with-lease` (not `--force`) so a surprise remote update aborts rather than clobbers.
- Net behaviour unchanged: this is pure PR-boundary restructuring; the full suite must stay green at
  every phase (195 passed baseline).
- Cost ≈ 30-45 min of careful git. Benefit: #59 reviews in minutes and unblocks the fan-out
  (radial_distribution is the next feature PR that will also sit on #59 / main).
