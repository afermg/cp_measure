# Changelog

Notable user-facing changes are documented here.

## [Unreleased]

### Changed

- **Breaking:** require Python 3.11 or newer. Remove the project's Python upper
  bound; CI covers Python 3.11–3.14 on Ubuntu and macOS.

### Fixed

- Costes now maps exact reciprocal-normalized integer-grid values (including
  `img_as_float` outputs) to the existing division encoding, preventing round-off
  from changing coefficients solely because of the normalization path (#117).
  Other float intensities, threshold-search rules and NaN handling are unchanged.

## [0.3.0]

### Added

- Experimental Numba Feret backend, selected with `set_accelerator("numba")`. It preserves
  NumPy's 2D results and raw 3D behavior (`{}`), including single-slice volumes.
  Byte-order normalization remains at the backend boundary; #111 tracks a
  possible shared normalization layer.

### Changed

- **Breaking:** `featurize` now takes explicit batches: images `(B, C, Y, X)`
  and masks `(B, M, Y, X)`, or `(B, C, Z, Y, X)` / `(B, M, Z, Y, X)` with
  `is_3d=True`. `M` counts mask types, not labelled objects. Single images still
  require `B=1`. Batch size and all spatial dimensions must match; masks are
  never repeated or broadcast automatically.
- **Breaking:** replace `image_id` with `image_ids`, one identifier per batch
  item. Omitted IDs now produce zero-based batch indices instead of `None`.
- Empty batch items are skipped with a warning while retaining the remaining
  items' IDs; an entirely empty batch raises `ValueError`.
- Require centrosome 1.3.4 or newer, which removes its undeclared `six` dependency;
  test and benchmark environments no longer install `six` as a workaround.

### Fixed

- Exclude generated Python/Numba cache files from built packages.
- Costes now separates acquisition scale from normalized float dtype. Both public
  APIs accept keyword-only `scale_max`, normalize integer pixels internally, and
  reject float inputs without an explicit scale instead of silently searching a
  one-step grid. Matching unsigned integer dtypes infer their storage maximum;
  lower-bit-depth data in larger containers need an explicit override.
- Undefined Costes regressions/background subsets and zero signal denominators
  now warn with `RuntimeWarning` and return NaN. Constant bisection
  candidates no longer cause silent convergence on stale bounds.
- Added integer/float, 2D/3D, bulk/featurizer and degenerate-input regressions,
  plus 18 fixtures from pinned CellProfiler threshold methods. This does not
  establish full CellProfiler pipeline equivalence.

### Migration

- For a formerly unbatched image/mask pair, use
  `featurize(image[None], masks[None], config)`. For volumes, also pass
  `is_3d=True`. Replace `image_id="sample"` with `image_ids=["sample"]`.
  Supply caller-unique IDs when combining results from separate calls.
- If intentionally applying a 2D mask throughout a volume, explicitly create
  `masks_3d = np.repeat(masks_2d[:, :, None], image.shape[2], axis=2)` before
  calling `featurize` with a 5D image. The featurizer does not add or repeat Z.
- **Breaking:** normalized float callers must supply `scale_max` to Costes, or set
  `correlation_costes_params={"scale_max": ...}` in the featurizer. Use the
  acquisition maximum (e.g. 65535 for 16-bit), not an observed image maximum.
  Missing scale raises ValueError; other correlation families are unchanged.

### Credits

- Tim Treis (@timtreis) authored the batched-featurizer and Numba Feret
  implementations (#106 and #65). Alán F. Muñoz (@afermg) authored the maintainer
  review follow-ups, compatibility tests/documentation, and v0.3.0 preparation.

## [0.2.0] - 2026-08-13

### Added

- Public bulk and featurizer entry points now accept arbitrary positive label IDs;
  labels are sanitized internally without mutating the input, while featurizer rows
  retain the original IDs.
- Added `legacy=True` support for the historical CellProfiler/cp_measure intensity
  percentile and MAD conventions.
- Added a dedicated 3D measurement registry so the featurizer skips unsupported 2D
  features correctly for volumetric input.
- Added initial experimental Numba/backend scaffolding. It is not yet a supported
  accelerator.
- Added a self-contained PR benchmark workflow.

### Changed

- Intensity quartiles now default to NumPy's `(n - 1) * q` linear percentile
  convention, and MAD defaults to the textbook median absolute deviation. Use
  `legacy=True` for the previous convention.
- Radial distribution now resolves tied center pixels deterministically using the
  first position in C order. This makes each object's result independent of other
  labels; symmetric objects may differ from historical unstable results.

### Fixed

- Fixed 3D feature dispatch in the featurizer.
- Fixed granularity behavior when subsampling collapses an image axis.
- Fixed radial-distribution measurements changing when unrelated labels were added
  or removed.

### Performance

- Reworked intensity, colocalization, Feret, size/shape, Zernike, radial Zernike,
  granularity, and mask-conversion paths to reduce per-object Python/SciPy work.
- In the synthetic release benchmark, median speedups across the image/object matrix
  ranged from **1.28x** for radial distribution to **28.07x** for intensity.
  Granularity improved **2.16x**, shape Zernikes **3.89x**, radial Zernikes **3.41x**,
  Pearson **3.26x**, and Feret **4.49x** at the median benchmark cell.
- Texture was broadly unchanged, with results ranging from 0.92x to 1.15x.
- See the full methodology, per-cell results, incremental CI measurements, and raw
  data in [`benchmarks/releases/v0.2.0.md`](benchmarks/releases/v0.2.0.md).
