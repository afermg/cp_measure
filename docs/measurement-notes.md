# Measurement notes

For essential input and performance guidance, see the [README](../README.md#important-notes).

## Texture quantization

Intensities use a fixed image-wide mapping, not per-object min/max scaling. Dim variation can disappear, and pairs containing zero are excluded. See the [#100 explanation](https://github.com/afermg/cp_measure/issues/100#issuecomment-5939719548) and [`get_texture` Notes](../src/cp_measure/core/measuretexture.py) for CellProfiler compatibility and zero/NaN behavior.

## Legacy percentile convention

`legacy=True` restores the original intensity quantile conventions; the default uses NumPy's linear quartiles and the usual median absolute deviation. See the [`get_intensity` docstring](../src/cp_measure/core/measureobjectintensity.py#L139-L163) for formulas, including the different legacy MAD convention in 3D.

## Radial-distribution centers

Center ties use the first pixel in C order, so symmetric objects may differ from older releases. See the [#22 discussion](https://github.com/afermg/cp_measure/issues/22) for the tie-breaking fix.

## Floating-point differences

Some features differ by approximately `1e-16` when using one versus multiple masks. See the [#18 explanation](https://github.com/afermg/cp_measure/issues/18#issuecomment-4593709963) for the upstream centrosome discrepancy.
