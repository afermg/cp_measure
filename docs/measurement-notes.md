# Measurement notes

For essential input and performance guidance, see the [README](../README.md#important-notes). For dimensions, label requirements, empty masks, and Feret batch shapes, see the [measurement support table](measurement-support.md).

## 3D size and shape

The featurizer supports both single- and multichannel volumes with `is_3d=True`. Both images `(B, C, Z, Y, X)` and masks `(B, M, Z, Y, X)` must have all five axes, with matching batch size and spatial dimensions. A 2D mask stack is rejected; singleton spatial axes are not stretched.

To use a 2D mask through a volume, the caller must explicitly repeat it before calling `featurize`, for example `masks_3d = np.repeat(masks_2d[:, :, None], image.shape[2], axis=2)`. Each repeated label becomes one 3D object, so intensity and shape measurements use the entire volume rather than separate 2D measurements per slice. The featurizer does not perform this conversion or modify its inputs.

The existing `sizeshape` measurement uses a surface-area calculation that requires at least two voxels along each spatial axis. In particular, a single-slice volume (`Z=1`) raises `ValueError: Input array must be at least 2x2x2.` with size/shape enabled. Disable it with `make_featurizer_config(..., sizeshape=False)`, or explicitly provide a 2D batch and use `is_3d=False` if 2D measurements are intended. This limitation predates the batch featurizer.

## Texture quantization

Intensities use a fixed image-wide mapping, not per-object min/max scaling. Dim variation can disappear, and pairs containing zero are excluded. See the [#100 explanation](https://github.com/afermg/cp_measure/issues/100#issuecomment-5939719548) and [`get_texture` Notes](../src/cp_measure/core/measuretexture.py) for CellProfiler compatibility and zero/NaN behavior.

## Legacy percentile convention

`legacy=True` restores the original intensity quantile conventions; the default uses NumPy's linear quartiles and the usual median absolute deviation. See the [`get_intensity` docstring](../src/cp_measure/core/measureobjectintensity.py#L139-L163) for formulas, including the different legacy MAD convention in 3D.

## Radial-distribution centers

Center ties use the first pixel in C order, so symmetric objects may differ from older releases. See the [#22 discussion](https://github.com/afermg/cp_measure/issues/22) for the tie-breaking fix.

## Radial Zernike inputs and phases

Radial Zernikes measure one intensity channel at a time. For a 2D label image `(Y, X)`, pass a matching 2D intensity image. The Numba backend rejects mismatched shapes, including `(C, Y, X)` or `(1, Y, X)` intensities paired with a 2D mask, rather than selecting the first plane. This also applies to 2D entries in list/tuple batches. Stacked 2D batches use matching `(B, 1, Y, X)` arrays. Volume masks retain the existing empty-result behavior.

Phases follow CellProfiler's `arctan2(real, imag)` convention, not the usual reversed argument order. Compare meaningful phases modulo `2π`: `+π` and `-π` represent the same angle. Near-zero moments have numerically unstable phases, which can differ across backends, platforms, or even repeated NumPy calls. NumPy-computed azimuthal powers reduce these differences in the Numba backend but do not guarantee identical phases.

Backend parity tests compare finite moments' phases only when their magnitude exceeds `1e-9` times the largest finite magnitude **of the same object**, with wrapped angular tolerance `1e-6` radians. This object-local threshold prevents a bright or nonfinite object from hiding phase errors in another object. Magnitudes and phase NaN patterns are checked regardless of the threshold. These are comparison rules only: returned phases are not rounded, zeroed, or masked.

## Percentage-of-maximum colocalization thresholds

Manders, overlap/K and RWC use array-valued per-object maxima when calculating percentage thresholds. This preserves the historical CellProfiler float32 arithmetic under both NumPy 1.x and 2.x. Scalar promotion in NumPy 1.x could otherwise change a threshold by one float32 ULP and include or exclude boundary pixels (#122). Correcting it can change coefficients on such inputs; pixels themselves are not rounded or adjusted, and comparisons remain `>=`.

This does not change RWC's ranking definition, introduce an epsilon, or change Costes threshold fitting/normalization. Float64 and integer-input threshold behavior and undefined-result conventions remain unchanged.

## Floating-point differences

Some features differ by approximately `1e-16` when using one versus multiple masks. See the [#18 explanation](https://github.com/afermg/cp_measure/issues/18#issuecomment-4593709963) for the upstream centrosome discrepancy.
