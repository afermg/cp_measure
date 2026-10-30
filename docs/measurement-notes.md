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

## Floating-point differences

Some features differ by approximately `1e-16` when using one versus multiple masks. See the [#18 explanation](https://github.com/afermg/cp_measure/issues/18#issuecomment-4593709963) for the upstream centrosome discrepancy.

## Numba colocalization

Pearson, Manders and RWC keep the same default bulk feature names and ordering as
NumPy. Overlap/K remains available through the direct module API only. RWC ranks
pixels independently within each object; changing another object's signal does
not change these ranks.

The accelerated float64 path groups paired pixels once per measurement and uses
a fused kernel. Pearson retains NumPy's mean reductions, including their behavior
for constant signals. Other pixel dtypes explicitly use the NumPy implementation:
float32 threshold membership and integer multiplication overflow must not change
just because an accelerator is selected. Undefined outputs also use the reference
path to preserve warnings and errors (including the existing zero-denominator
Overlap/K error). These compatibility paths are not advertised as accelerated.

All three inputs must have matching spatial ranks and extents. A 2D mask is not
broadcast over Z, nor is the first intensity plane selected implicitly. Matching
singleton-Z inputs are supported. Direct functions require contiguous labels;
bulk sanitation handles arbitrary IDs. Missing raw label slots retain NumPy's
zero values.

Two independent follow-ups are deliberately not included here:

- [#124](https://github.com/afermg/cp_measure/pull/124) restores array-valued
  float32 threshold arithmetic. Its boundary fixture currently retains the
  installed NumPy generation's baseline result in both backends. Once that fix
  lands, the dtype compatibility path will inherit it without a separate JIT
  threshold policy.
- [#126](https://github.com/afermg/cp_measure/pull/126) adds an opt-in
  `rank_scope="object_set"`. This backend retains the current signature and
  ordinary object-local default; that follow-up will also need to add keyword
  forwarding to the Numba wrapper (with a reference path for the new scope, or
  a separately validated implementation). It must not silently change the default.

### Costes

The Numba Costes wrapper accepts the same `scale_max`, `fast_costes` and unused
`thr` parameters as NumPy. Normalized floats require an explicit acquisition
maximum; same-dtype unsigned integers infer their storage maximum. Integer
normalization uses the shared NumPy preparation, not the old dtype-derived
`scale=1` float shortcut. All three search modes are supported.

NumPy regression and coefficient reductions surround the compiled threshold
search. This preserves threshold inclusion: even last-bit changes in the fitted
line can change the result substantially. Constant/nonfinite or insufficient
signals retain NaN results and standard `RuntimeWarning` diagnostics. Missing
raw Costes labels likewise yield NaN and warnings, unlike the zero-valued slots
of the other colocalization metrics. Undefined searches are replayed per object
by the reference to preserve precise diagnostics. Near-constant subsets and
correlations within a reduction-error bound of a search decision also use this
compatibility path; no correlation is rounded or search cutoff moved.

The pending [#118](https://github.com/afermg/cp_measure/pull/118) normalization
canonicalization is **not** included. Each encoding currently matches NumPy,
without promising equality between division and reciprocal-multiplication
encodings. Both backends share image-scale preparation, but #118 canonicalizes
inside its per-object `_costes_pair`, **not** `_prepare_costes_inputs`. Its
integration must call the same canonicalization helper in the Numba
`_costes_pair` before validation/regression/search; merging only #118's NumPy
change would leave finite accelerated searches on the old policy. That
propagation is already requested, not a new policy decision. Its regression
case is checked against current main in the accelerator tests.

No fused `get_correlation_all` dispatch is introduced. Costes supports matching
2D, singleton-Z and true 3D inputs, and the existing explicit batch convention;
it does not repeat masks or select an intensity plane implicitly.
