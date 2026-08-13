# Changelog

Notable user-facing changes are documented here.

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
