# Implementation plan — vectorized numpy `get_zernike` (~8×)

## Context
`get_zernike` (`src/cp_measure/core/measureobjectsizeshape.py`) is the 2nd-most expensive
feature on the large tier (793 ms / 21% of a 3730 ms pipeline). It delegates to
`centrosome.zernike.zernike`, which (a) scatters the per-pixel Zernike basis into a **full
`(H,W,K)` complex array (~560 MB at 1080², K≈30)** and (b) scores via **~60 full-image
`scipy.ndimage.sum`** calls (K polynomials × real+imag). Both scale with image area, not object
pixels, so most work is on background. A pure-numpy rewrite that keeps the basis on the masked
`(npix,)` vectors and segment-sums by label removes both. **Prototype
(`tasks/proto_zernike_vectorized.py`) measured 8.0× on large (782→98 ms), 13× on m2160, with a
max divergence of 2.2e-16 (machine epsilon).** No new deps — reuses centrosome's coefficient
LUT + geometry. Goal: land it as a single-concern `perf(zernike)` PR matching the #69–73 style.

## Why it's safe (fidelity)
- Existing tests (`test/test_core_measurements.py`, `test_featurizer.py`) assert only output
  **shape / non-triviality / 3D-empty** for zernike — **no numerical golden**. The 2.2e-16
  change pins nothing.
- `zernike` is **not** in `bulk._LEGACY_FEATURES` (only `intensity` is). The divergence is
  float round-off from summation order, not a semantic change ⇒ **no `legacy` flag needed**
  (unlike radial #22 / intensity MAD). A new golden test locks it within a tolerance.

## Change (one file: `measureobjectsizeshape.py`)
Replace the body of `get_zernike` (keep signature `(masks, pixels, zernike_numbers=9)`, the
`ndim==3 → {}` guard, the `pixels`-unused behaviour, and the `Zernike_{n}_{m}` output keys):

1. `idx = unique(masks)>0`; `n = len(idx)`. **Use the actual label values as indices** (not the
   current `range(1, n+1)`, which silently assumes contiguous 1..n). Equivalent on contiguous
   masks (all real/test data); strictly more correct otherwise. Order results by sorted `idx`.
2. Empty guard: `n == 0` → return `{f"Zernike_{nn}_{mm}": numpy.zeros(0) for (nn,mm) in zidx}`.
3. `zidx = centrosome.zernike.get_zernike_indexes(zernike_numbers + 1)`;
   `centers, radii = centrosome.zernike.minimum_enclosing_circle(masks, idx)`.
4. `rev` map (size `masks.max()+1`, −1 default, `rev[idx]=arange(n)`); `mask = rev[masks] != -1`.
   Build masked, per-object-normalised coords: `ym=(yy[mask]-centers[ri,0])/radii[ri]`,
   `xm=(xx[mask]-centers[ri,1])/radii[ri]` where `ri=rev[masks[mask]]`.
5. Basis: `lut = centrosome.zernike.construct_zernike_lookuptable(zidx)`. Run centrosome's exact
   Horner inner loop on the `(npix,)` vectors — `r2=xm²+ym²`, `z=ym+1j*xm`, per index Horner
   accumulate, `s[r2>1]=0`, `m==0`→real else `s*z**m` (cache `z**m`). **Copy verbatim** from
   `construct_zernike_polynomials` so coefficients/cutoff/`z_pows` match bit-for-bit.
6. Score: `np.add.at(out_re[:,k], ri, zf.real)` / `out_im` per index (segment-sum), then
   `score = sqrt(out_re²+out_im²) / (pi*radii²)[:,None]` — same formula as `score_zernike`.
7. Return `{f"Zernike_{nn}_{mm}": score[:, i] for i,(nn,mm) in enumerate(zidx)}`.

Factor steps 4–6 into a private `_zernike_scores(masks, idx, zidx, weight=None)` returning the
`(n, K)` real/imag sums, so **`radial_zernikes` (sibling follow-up) can reuse it** with
`weight=pixels` and a magnitude+phase reduction. (radial_zernikes is intensity-weighted and
emits phase, so it shares the *pattern*/geometry, not the final reduction — out of scope here.)

## Edge cases to cover in the golden test
- empty mask (0 objects) → empty arrays, right keys.
- single-pixel object → `minimum_enclosing_circle` r may be 0 ⇒ `pi*r²=0` division. centrosome's
  `score_zernike` divides too, so both yield the same inf/nan — assert parity (use `equal_nan`,
  and treat inf consistently) rather than masking it.
- non-contiguous labels (e.g. {1,3,5}) — document the corrected behaviour; compare against
  centrosome called with `indices=unique_labels` (its intended use), not the old `range`.
- object touching the image edge; fully-masked / all-background image.
- `zernike_numbers` other than the default 9 (K changes).

## Tests (`test/test_zernike.py`, new)
- **Golden**: `get_zernike(m,px)` vs a reference that calls `centrosome.zernike.zernike` the old
  way, on tiny/small/large + the edge cases — assert `numpy.allclose(atol=1e-10, rtol=1e-10,
  equal_nan=True)` per key (≫ the 2.2e-16 observed, ≪ any real signal). Pin key set + order.
- Keep `test_core_measurements.py` green (shape/3D/non-trivial assertions unchanged).
- Quick perf assertion optional (not in CI): reuse `tasks/proto_zernike_vectorized.py`.

## Workflow
- **New branch off `origin/main`** (`perf/zernike-vectorize`) — independent single-concern PR,
  NOT stacked on the radial branch.
- Lint before push: `uvx ruff@0.12.1 format --check . && uvx ruff@0.12.1 check .` (line-len 88).
- Run `.venv/bin/python -m pytest test/test_zernike.py test/test_core_measurements.py -q`.
- Benchmark large/m2160 before+after via `tasks/proto_zernike_vectorized.py`; put the table +
  the machine-eps fidelity note in the PR body (flag the round-off change explicitly, à la the
  granularity-gather PR #56 comment).
- Follow-up PR: `radial_zernikes` reusing `_zernike_scores` (weight=pixels, mag+phase), and a
  shared single `minimum_enclosing_circle` pass when both run.

## Risk / rollback
Low. One function, no API/key change, no dep change, no test currently pins the values. Sole
behaviour delta is ≤2.2e-16 round-off + the non-contiguous-label correction. Trivial revert
(restore the `centrosome.zernike.zernike` call).
