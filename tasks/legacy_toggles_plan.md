# Plan: `legacy: bool` convention toggles (intensity + radial_distribution)

Source: cross-repo legacy-toggle mining (`tasks/legacy_toggle_candidates.md`). Two toggles
are worth building (A1+A2 intensity, A3 radial). They are different change-types, so this
is **two coordinated PRs sharing one `legacy` convention**, not one bundle.

## Shared convention (both PRs, both backends)
- `legacy: bool = False`. **`False` (default) = corrected/standard behavior; `True` = exact
  byte-for-byte reproduction of the pre-change `main` output.**
- Same flag name + meaning across functions → consistent API (future option: a repo-wide
  `legacy` umbrella that fans out — NOT now, keep minimal).
- **Linchpin contract for every toggle:** the existing `test_*_matches_numpy` golden test is
  parametrized over `legacy ∈ {False, True}` and asserts numba == numpy for BOTH. Plus a
  `legacy=True` snapshot anchor proving it equals current `main`.

---

# PR 1 — intensity `legacy` (numpy + numba, against `main`)  ★ ready, small

The clean, self-contained one. Both backends exist on `main`.

**Convention toggled** (4 output keys only): `Intensity_{LowerQuartile, Median, UpperQuartile, MAD}Intensity`.

| | quartiles / median | MAD |
|---|---|---|
| `legacy=False` (new default) | `(n-1)·q` linear interp (= numpy.percentile) | `median(\|x−median\|)`, centered on the `(n-1)·q` median |
| `legacy=True` (CellProfiler, today's main) | `n·q` linear interp | `(1/ndim)`-quantile of `\|x−median\|`, centered on the `n·q` median |

**Elegant trick — parameterize, don't fork:**
- **numpy** (`core/measureobjectintensity.py`): the existing vectorized block already does
  `qindex = indices + areas*fraction`. Replace with `span = areas if legacy else (areas-1)`;
  MAD: `mad_fraction = (1.0/pixels.ndim) if legacy else 0.5`. No new code path, no perf change,
  no `numpy.percentile` call needed (the block already does type-7 linear interp).
- **numba** (`primitives/_segment_numba.py`): `_interp(seg, n, frac, legacy)` →
  `pos = n*frac if legacy else (n-1)*frac`. `segment_quantiles(..., mad_frac, legacy)` picks
  `_interp(ad, cnt, mad_frac, True)` vs `_interp(ad, cnt, 0.5, False)`.
- **wrapper**: add `legacy=False` to `get_intensity`, thread to `segment_quantiles`.

**Consolidation note:** `legacy_mad` already exists on `origin/speedups` (`cb05ade`) on BOTH
numpy + numba. This PR SUBSUMES it — one `legacy` flag covers quartiles+MAD. Coordinate with
Alan: he drops `legacy_mad` and rebases #55's intensity perf/coloc/sizeshape on this flag.

**Files:** `core/measureobjectintensity.py` · `core/numba/measureobjectintensity.py` ·
`primitives/_segment_numba.py` · `test/test_backend_correctness.py` · changelog.

**Commits:** (1) primitives `_interp`/`segment_quantiles` gain `legacy` (default wired to legacy
→ no behavior change yet); (2) numpy `span`/`mad_fraction` parameterization; (3) numba wrapper
threads `legacy`; (4) flip defaults to `False` + parametrize golden test + `main` snapshot
anchor; (5) docs/changelog.

**Risk:** default-output change for the 4 features (esp. 3D MAD: true median vs `1/3`-quantile).
Mitigate via changelog + `legacy=True` escape hatch. Optional softer rollout: ship default
`legacy=True` for one release with a deprecation note, then flip.

---

# PR 2 — radial_distribution `legacy` (Issue #22), coordinated with #63

The heavier one — a two-sided **algorithm-parity** job, not a reindex. Best done **in / on top
of the #63 numba lane** (which owns the #22 fix), NOT as a standalone main PR.

**Behavior toggled** (whole feature set of `get_radial_distribution`):

| | geometry |
|---|---|
| `legacy=False` (new default) | per-object crop + 1px pad → results **independent** of other labels (the #22 fix) |
| `legacy=True` (today's main) | whole-image `propagate` → multi-object **leakage** (the documented #22 bug) |

**Why it's two-sided / bigger than intensity:** today `main` numpy = whole-image only; #63 numba
= per-object only. For numba==numpy under BOTH flags, each backend must gain its missing path:
- **numpy** `get_radial_distribution`: add the per-object-crop path as the new default (port the
  #22 fix), keep the existing whole-image block under `if legacy:`.
- **numba** (#63): add a whole-image legacy path (it currently only does per-object).

**Recommended scoping (minimal):** fold the `legacy` flag into **#63** and add the matching
numpy default-flip in the same PR (so #63 becomes "numba radial lane + #22 fix as default + legacy
whole-image in both backends"). This keeps all radial/#22 reasoning in one place and avoids a
standalone main PR that duplicates #63's geometry.

**Alternative (even more minimal, if porting #22 to numpy is out of scope now):** keep the toggle
**numba-only** — numba gets `legacy=True` = whole-image (matches the unchanged numpy baseline),
`legacy=False` = per-object. numpy stays old-only (no flag). Downside: the #22 fix stays
numba-only; numpy users can't get it. Acceptable as a first step; revisit numpy port later.

**Files (recommended scope):** `core/measureobjectintensitydistribution.py` (numpy) ·
`core/numba/measureobjectintensitydistribution.py` + `core/numba/_radial.py` (#63) · radial tests.

---

## Sequencing
1. **PR 1 (intensity) first** — independent of everything, lands on `main`, unblocks #55's rebase.
2. **#55 rebases** onto PR 1 (drops `legacy_mad`, keeps coloc/sizeshape + intensity perf).
3. **PR 2 (radial)** rides with **#63** (after #59 stack ordering), using the same `legacy`
   convention PR 1 established.

## Decisions to confirm with Alan
- Flag name `legacy` (vs `legacy_percentiles` / per-axis names). One umbrella name, recommended.
- Default flip now vs deprecation period (intensity 4 features + radial geometry both change default).
- Radial scoping: fold into #63 (recommended) vs numba-only first step.
