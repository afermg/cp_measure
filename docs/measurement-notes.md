# Measurement notes

For essential input and performance guidance, see the [README](../README.md#important-notes). For dimensions, label requirements, empty masks, and Feret batch shapes, see the [measurement support table](measurement-support.md).

## Feature coverage policy

Maximize valid feature coverage: prefer 3D measurements; where unavailable for singleton 3D inputs, generate supported 2D features. Apply fallback per measurement family, retain 2D feature names and meanings, and keep schemas consistent across backends and empty/nonempty inputs.

## 3D size and shape

The featurizer supports both single- and multichannel volumes with `is_3d=True`. Both images `(B, C, Z, Y, X)` and masks `(B, M, Z, Y, X)` must have all five axes, with matching batch size and spatial dimensions. A 2D mask stack is rejected; singleton spatial axes are not stretched.

To use a 2D mask through a volume, the caller must explicitly repeat it before calling `featurize`, for example `masks_3d = np.repeat(masks_2d[:, :, None], image.shape[2], axis=2)`. Each repeated label becomes one 3D object, so intensity and shape measurements use the entire volume rather than separate 2D measurements per slice. The featurizer does not perform this conversion or modify its inputs.

For `Z=1`, size/shape measures the sole plane in both backends, including through the bulk wrappers and featurizer. It returns 2D feature names and pixel-based values: 78 default features, not the 212-feature 3D schema. All-background raw masks use the same 2D schema with zero-length arrays. This replaces the previous nonempty-input error and changes the previous empty singleton-Z schema. As for explicit 2D inputs, `spacing` does not scale these measurements.

For `Z>1`, size/shape retains its 3D implementation and spacing convention. Its surface-area calculation still requires at least two voxels along each spatial axis; singleton X/Y axes are not squeezed. Raw size/shape calls remain per-image, not batch APIs.

## Numba size and shape

With `set_accelerator("numba")`, size/shape accelerates convex-hull construction and perimeter/Euler reductions for 2D and singleton-Z inputs. Hull rasterization, Euclidean distance transforms and the existing NumPy moment calculations remain unchanged. Euler output stays integer; returned values are not rounded or clipped.

2D moment comparisons use object-local scaling: raw/central moments of total order k are divided by `Area**(1+k/2)` and compared with `rtol=1e-7, atol=1e-12`. Normalized and Hu moments use the same tolerance directly. Orientations are compared modulo 180 degrees with absolute tolerance `1e-8` degrees. These are test comparison rules only; other geometry, dtype and nonfinite checks are unchanged.

Both raw functions require contiguous labels. The featurizer and default bulk wrappers sanitize arbitrary positive IDs without modifying inputs. Multi-slice volumes use the NumPy fallback. Warmed gains depend on object size and density; first calls also pay compilation cost.

## Texture quantization

Intensities use a fixed image-wide mapping, not per-object min/max scaling. Dim variation can disappear, and pairs containing zero are excluded. See the [#100 explanation](https://github.com/afermg/cp_measure/issues/100#issuecomment-5939719548) and [`get_texture` Notes](../src/cp_measure/core/measuretexture.py) for CellProfiler compatibility and zero/NaN behavior.

## Legacy percentile convention

`legacy=True` restores the original intensity quantile conventions; the default uses NumPy's linear quartiles and the usual median absolute deviation. See the [`get_intensity` docstring](../src/cp_measure/core/measureobjectintensity.py#L139-L163) for formulas, including the different legacy MAD convention in 3D.

## Costes integer-to-float normalization

For integer images, Costes divides by `scale_max`; normalized floats require the same scale explicitly. Equivalent floating-point expressions such as `level / scale_max` and `level * (1 / scale_max)` can differ at round-off, which can otherwise change threshold decisions substantially (see [#117](https://github.com/afermg/cp_measure/issues/117)).

Both public Costes functions recognize **exact** reciprocal-multiply encodings of integer acquisition-grid levels, as produced by `img_as_float`, and map them to the existing division encoding before fitting thresholds and computing coefficients. This is performed per object, without modifying caller inputs. Other float values are left unchanged: there is no blanket quantization or epsilon-based snapping of nearby values. Explicit lower-bit-depth scales, such as `4095` for 12-bit images in `uint16`, use the same rule.

This resolves conversion-path differences for these grid encodings; it does not promise continuity for genuinely different pixel values, identical results across floating-point precisions, or equivalence between different Costes search modes. Existing threshold comparisons and undefined-result/NaN rules are unchanged.

## Singleton-Z measurement eligibility

The 2D-only families (`radial_distribution`, `radial_zernikes`, shape `zernike`, and `feret`) use their 2D calculations when there is exactly one Z slice. They are not computed when there is more than one Z slice.

- Matching `(Y, X)` inputs use the 2D measurements as usual.
- Matching `(1, Y, X)` inputs use the 2D measurements on their sole slice.
- Matching `(Z, Y, X)` inputs with `Z > 1` return an empty result (`{}`).
- Apply the same rule per image in list/tuple and `(B, Z, Y, X)` batches: `Z=1` is measured; `Z>1` is not.

The same eligibility rule applies in both backends and through the featurizer. NumPy raw functions remain per-image APIs; Numba also accepts batches. Labels and intensities must match for intensity-dependent features; do not repeat masks through Z or select the first slice of a multi-slice volume.

Size/shape also uses 2D measurements for singleton-Z inputs, as described above. Intensity, texture, granularity and colocalization retain their 3D input and existing conventions. Granularity's Numba backend uses the NumPy 3D path for singleton-Z volumes instead of silently selecting its 2D kernel.

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

One compiled pass measures every object: canonicalization, validation, the
regression line, the threshold search and the coefficients. The regression and
coefficient sums reproduce NumPy's pairwise summation exactly (including the
buffer-sized chunks NumPy used before 2.3), because even last-bit changes in the
fitted line can change which pixels count and so the result. Constant/nonfinite
or insufficient signals retain NaN results and standard `RuntimeWarning`
diagnostics, emitted by the wrapper with the reference's messages. Missing raw
Costes labels likewise yield NaN and warnings, unlike the zero-valued slots of
the other colocalization metrics. Near-constant subsets, correlations within a
reduction-error bound of a search decision, and ill-conditioned regressions are
replayed per object by the reference to preserve its diagnostics; no
correlation is rounded or search cutoff moved.

Both backends apply the same per-object normalization canonicalization before
validation, regression and threshold search. Exact reciprocal-multiply
acquisition-grid encodings therefore follow the same division encoding in the
NumPy and Numba paths; arbitrary off-grid floats are unchanged. This is the
normalization rule described above, not an epsilon-based threshold adjustment.

No fused `get_correlation_all` dispatch is introduced. Costes supports matching
2D, singleton-Z and true 3D inputs, and the existing explicit batch convention;
it does not repeat masks or select an intensity plane implicitly.
