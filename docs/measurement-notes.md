# Measurement notes

For essential input and performance guidance, see the [README](../README.md#important-notes). For dimensions, label requirements, empty masks, and Feret batch shapes, see the [measurement support table](measurement-support.md).

## 3D size and shape

The featurizer supports both single- and multichannel volumes with `is_3d=True`. Both images `(B, C, Z, Y, X)` and masks `(B, M, Z, Y, X)` must have all five axes, with matching batch size and spatial dimensions. A 2D mask stack is rejected; singleton spatial axes are not stretched.

To use a 2D mask through a volume, the caller must explicitly repeat it before calling `featurize`, for example `masks_3d = np.repeat(masks_2d[:, :, None], image.shape[2], axis=2)`. Each repeated label becomes one 3D object, so intensity and shape measurements use the entire volume rather than separate 2D measurements per slice. The featurizer does not perform this conversion or modify its inputs.

The existing `sizeshape` measurement uses a surface-area calculation that requires at least two voxels along each spatial axis. In particular, a single-slice volume (`Z=1`) raises `ValueError: Input array must be at least 2x2x2.` with size/shape enabled. Disable it with `make_featurizer_config(..., sizeshape=False)`, or explicitly provide a 2D batch and use `is_3d=False` if 2D measurements are intended. This limitation predates the batch featurizer.

## Numba intensity batches

The direct Numba `get_intensity` function accepts matching 2D images or 3D volumes and returns one feature dictionary. Matching `(B, Z, Y, X)` arrays or lists/tuples of equally shaped 2D/3D images return a list of dictionaries. A 3D array is one volume, never an implicit batch. Empty or ragged batches and mismatched input ranks/shapes are rejected; masks are not broadcast or repeated.

`legacy` and `edge_measurements` apply to every image. Legacy MAD depends on the original spatial rank: a 2D list item uses the 2D convention, while a `(B, 1, Y, X)` batch contains singleton-Z volumes and retains the 3D convention. Batch items are processed sequentially using the existing kernels; this API refactor does not introduce parallel batch execution or promise a speedup. The NumPy direct function remains a single-image API, and featurizer input conventions are unchanged.

## Texture quantization

Intensities use a fixed image-wide mapping, not per-object min/max scaling. Dim variation can disappear, and pairs containing zero are excluded. See the [#100 explanation](https://github.com/afermg/cp_measure/issues/100#issuecomment-5939719548) and [`get_texture` Notes](../src/cp_measure/core/measuretexture.py) for CellProfiler compatibility and zero/NaN behavior.

## Legacy percentile convention

`legacy=True` restores the original intensity quantile conventions; the default uses NumPy's linear quartiles and the usual median absolute deviation. See the [`get_intensity` docstring](../src/cp_measure/core/measureobjectintensity.py#L139-L163) for formulas, including the different legacy MAD convention in 3D.

## Singleton-Z measurement eligibility

The 2D-only families (`radial_distribution`, `radial_zernikes`, shape `zernike`, and `feret`) use their 2D calculations when there is exactly one Z slice. They are not computed when there is more than one Z slice.

- Matching `(Y, X)` inputs use the 2D measurements as usual.
- Matching `(1, Y, X)` inputs use the 2D measurements on their sole slice.
- Matching `(Z, Y, X)` inputs with `Z > 1` return an empty result (`{}`).
- Apply the same rule per image in list/tuple and `(B, Z, Y, X)` batches: `Z=1` is measured; `Z>1` is not.

The same eligibility rule applies in both backends and through the featurizer. NumPy raw functions remain per-image APIs; Numba also accepts batches. Labels and intensities must match for intensity-dependent features; do not repeat masks through Z or select the first slice of a multi-slice volume.

This does not reinterpret the input as 2D for other measurements. Intensity, texture, granularity, size/shape, and colocalization keep their 3D input and existing conventions. In particular, the size/shape singleton-axis limitation above remains. Granularity's Numba backend uses the NumPy 3D path for singleton-Z volumes instead of silently selecting its 2D kernel.

## Radial-distribution centers

Center ties use the first pixel in C order, so symmetric objects may differ from older releases. See [#22 discussion](https://github.com/afermg/cp_measure/issues/22). Intensities are accumulated in float64 by both backends, avoiding integer overflow.

Labels and intensities must have matching shapes and spatial rank. Singleton-Z and multi-slice eligibility is specified above. Bin counts must be non-negative integers. For nonempty 2D inputs with zero or NaN unscaled radii, the Numba wrapper delegates to the NumPy reference: its non-finite-to-integer bin conversion can produce results or raise depending on the installed NumPy/platform. These exceptional inputs never reach the unchecked kernel; regular radii remain accelerated.

## Radial Zernike inputs and phases

Radial Zernikes measure one intensity channel at a time. For a 2D label image `(Y, X)`, pass a matching 2D intensity image. Both backends reject mismatched shapes, including `(C, Y, X)` or `(1, Y, X)` intensities paired with a 2D mask, rather than selecting the first plane. This also applies to 2D entries in list/tuple batches. Stacked singleton-Z batches use matching `(B, 1, Y, X)` arrays. The same eligibility rule applies to individual volume masks: use the 2D measurements for `Z=1`, and return an empty result for `Z>1`.

Phases follow CellProfiler's `arctan2(real, imag)` convention, not the usual reversed argument order. Compare meaningful phases modulo `2π`: `+π` and `-π` represent the same angle. Near-zero moments have numerically unstable phases, which can differ across backends, platforms, or even repeated NumPy calls. NumPy-computed azimuthal powers reduce these differences in the Numba backend but do not guarantee identical phases.

Backend parity tests compare finite moments' phases only when their magnitude exceeds `1e-9` times the largest finite magnitude **of the same object**, with wrapped angular tolerance `1e-6` radians. This object-local threshold prevents a bright or nonfinite object from hiding phase errors in another object. Magnitudes and phase NaN patterns are checked regardless of the threshold. These are comparison rules only: returned phases are not rounded, zeroed, or masked.

## Percentage-of-maximum colocalization thresholds

Manders, overlap/K and RWC accept keyword-only `precision="input"` (default) or `precision="float64"`, including NumPy's single-mask `_ind` APIs and Numba batches. Unknown modes raise `ValueError`, even without objects.

`"input"` retains the dtype/arithmetic paths immediately before #129 (main `8513bf3`, including #124). This is **not** a promise of exact released-0.3.0 or universal CellProfiler parity. It does not force every intermediate to the input dtype. Threshold maxima remain array-valued as in #124; stored float32 values retain float32 threshold rounding.

`"float64"` opts into widening float16/float32 pixels **before** threshold calculation, comparison, products and sums. Explicit threshold-scalar conversion avoids NumPy 1.x/2.x promotion ambiguities in this mode. Stored values are widened exactly, not renormalized from acquisition integers; inputs are not mutated. Float64 inputs need no promotion, and wider floats are not narrowed. Integer pixels are not promoted; their product/reduction dtype conventions remain unchanged.

Opting in may change boundary membership substantially: the documented float32 fixture's Manders coefficient changes from approximately 1/3 to 1. Comparisons remain `>=`; there is no epsilon, clipping or final rounding. RWC remains object-local. Pearson, Costes and unrelated families are unaffected. True undefined/zero-denominator conventions remain; artifacts caused solely by float32 product overflow/underflow can disappear in float64 mode.

## Granularity intermediate precision

Granularity accepts the same keyword-only `precision` option. The default `"input"` preserves each backend's pre-#129 path, including existing NumPy/Numba float32 differences: Numba's path was already mixed-precision. This mode does **not** incorporate #125's proposed float32 cast-back or promise new cross-backend/CellProfiler parity.

With `precision="float64"`, both backends widen float16/float32 inputs before the first resampling, background morphology, interpolation or subtraction. This is not a final-output cast. The option also reaches the NumPy 3D fallback. Original dimensional dispatch, singleton-axis undefined spectra, and integer-pixel handling are unchanged. No new shape-validation policy is bundled here.

Higher precision does not recover acquisition information or establish biological accuracy. It reduces arithmetic rounding and float32 overflow/underflow, at the cost of float64 working buffers (eight bytes per promoted pixel, possibly alongside retained source arrays) and workload-dependent runtime. Numba batch promotion is per image, not a whole-batch float64 stack. Float64 mode intentionally does not reproduce all CellProfiler float32 outputs.

### Selecting and recording precision

Direct and bulk calls accept the option, for example `get_granularity(mask, pixels, precision="float64")`. Through the featurizer, use the existing per-feature parameter dictionaries:

```python
config = make_featurizer_config(
    channels=["DNA", "AGP"],
    granularity_params={"precision": "float64"},
    correlation_manders_fold_params={"precision": "float64"},
    correlation_rwc_params={"precision": "float64"},
)
```

`make_featurizer_config` records an explicit `"input"` default in these dictionaries when the option is omitted, without modifying caller-owned dictionaries. Save the config alongside results, package/dependency versions, backend and input provenance. Existing manually constructed configs without this key still use input mode. No output columns or global accelerator settings are changed. Overlap remains direct-API-only; the option is independent of intensity's `legacy` percentile flag.

## Floating-point differences

Some features differ by approximately `1e-16` when using one versus multiple masks. See the [#18 explanation](https://github.com/afermg/cp_measure/issues/18#issuecomment-4593709963) for the upstream centrosome discrepancy.

## Numba colocalization

Pearson, Manders and RWC keep the same default bulk feature names and ordering as
NumPy. Overlap/K remains available through the direct module API only. RWC ranks
pixels independently within each object; changing another object's signal does
not change these ranks.

The accelerated float64 path groups paired pixels once per measurement and uses
a fused kernel. Pearson retains NumPy's mean reductions, including their behavior
for constant signals. With `precision="float64"`, percentage-threshold metrics
promote float16/float32 inputs per image and use the same float64 kernel. With
`precision="input"`, they keep their existing dtype-based reference fallbacks.
Pearson's non-float64 inputs and integer or wider-float inputs retain the reference path. Undefined outputs also use
the reference to preserve warnings and errors (including the existing true
zero-denominator Overlap/K error). Reference fallbacks are not advertised as
accelerated.

All three inputs must have matching spatial ranks and extents. A 2D mask is not
broadcast over Z, nor is the first intensity plane selected implicitly. Matching
singleton-Z inputs are supported. Direct functions require contiguous labels;
bulk sanitation handles arbitrary IDs. Missing raw label slots retain NumPy's
zero values.

Both backends' float64 mode is checked against independent scalar-double
calculations on the same stored float32 boundary values. Default/input mode is
checked against the historical boundary behavior. The original CellProfiler
fixture remains unchanged; the float64 expectation is a separate oracle, not a
rewritten reference. No separate JIT threshold policy is introduced.

Object-set RWC remains an independent follow-up:

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
