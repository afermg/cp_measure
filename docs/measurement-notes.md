# Measurement notes

For essential input and performance guidance, see the [README](../README.md#important-notes).

## 3D size and shape

The featurizer supports both single- and multichannel volumes with `is_3d=True`. A 2D mask stack `(B, M, Y, X)` is repeated across the image's Z slices before measurement. This extrudes each 2D label into a single 3D object: intensity and shape measurements use the entire volume, not separate 2D measurements per slice. The expanded mask is a copy; the caller's image and mask arrays are unchanged.

Batch size and Y/X dimensions must match exactly. Explicitly 3D masks `(B, M, Z, Y, X)` must also match Z; a singleton Z axis is not stretched. The input image always needs all five axes and `is_3d=True`, regardless of mask rank.

The existing `sizeshape` measurement uses a surface-area calculation that requires at least two voxels along each spatial axis. In particular, a single-slice volume (`Z=1`) raises `ValueError: Input array must be at least 2x2x2.` with size/shape enabled. Disable it with `make_featurizer_config(..., sizeshape=False)`, or explicitly provide a 2D batch and use `is_3d=False` if 2D measurements are intended. This limitation predates the batch featurizer.

## Texture quantization

Intensities use a fixed image-wide mapping, not per-object min/max scaling. Dim variation can disappear, and pairs containing zero are excluded. See the [#100 explanation](https://github.com/afermg/cp_measure/issues/100#issuecomment-5939719548) and [`get_texture` Notes](../src/cp_measure/core/measuretexture.py) for CellProfiler compatibility and zero/NaN behavior.

## Legacy percentile convention

`legacy=True` restores the original intensity quantile conventions; the default uses NumPy's linear quartiles and the usual median absolute deviation. See the [`get_intensity` docstring](../src/cp_measure/core/measureobjectintensity.py#L139-L163) for formulas, including the different legacy MAD convention in 3D.

## Radial-distribution centers

Center ties use the first pixel in C order, so symmetric objects may differ from older releases. See the [#22 discussion](https://github.com/afermg/cp_measure/issues/22) for the tie-breaking fix.

## Floating-point differences

Some features differ by approximately `1e-16` when using one versus multiple masks. See the [#18 explanation](https://github.com/afermg/cp_measure/issues/18#issuecomment-4593709963) for the upstream centrosome discrepancy.
