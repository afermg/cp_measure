# Pre-batch featurizer reference

`featurizer_pre_batch.json.gz` records the NumPy outputs of the single-image featurizer at commit `1f2ef4dc1fa0bfd81c2b3203b9e31af8d4e2e8c0`, before PR #106. It is a cp_measure regression reference, **not** an independent CellProfiler validation.

The deterministic cases contain two distinct images, two channels, two mask types, different object counts, and non-contiguous object IDs. Both 2D and genuine multi-slice 3D data are covered, with modern and legacy intensity percentiles. Intensity, size/shape, and directional Pearson measurements exercise the per-channel, shape, and correlation loops. Other measurements are outside this fixture's coverage.

Columns and complete row metadata are saved in order. JSON `null` represents an undefined numerical value and is loaded as `NaN`. Numerical comparisons use `rtol=atol=1e-10` to allow floating-point library differences; metadata is compared exactly.

The fixture records the source commit, featurizer SHA256, and numerical-library versions. Regenerate with `test/make_featurizer_reference.py` and a clean checkout of that commit on `PYTHONPATH`; the generator verifies the source checkout before writing. The JSON is minified and gzip-compressed with a fixed timestamp, without changing any values or metadata. Inspect it with `gzip -dc test/data/featurizer_pre_batch.json.gz | python -m json.tool`. Tests load it using Python's standard library, without invoking Git or executing the old code.

This is temporary migration coverage: remove the snapshot and historical-baseline assertion after v0.3.0, retaining the focused numerical, column-order, row-metadata, and batch-versus-singleton tests. When retiring the generator, move its shared case builder into a test helper.
