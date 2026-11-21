# Measurement notes

For essential input and performance guidance, see the [README](../README.md#important-notes). For dimensions, label requirements, empty masks, and Feret batch shapes, see the [measurement support table](measurement-support.md).

## Matching CellProfiler

Some cp_measure defaults intentionally differ from CellProfiler 4.2.8. Each difference has an option that restores CellProfiler's behavior. The featurizer's top-level `legacy=True` covers **only** the intensity percentiles; set the others yourself.

Inputs must be float images scaled to `[0, 1]` (e.g. uint16 ÷ 65535), as in CellProfiler. Beyond that:

| Measurement | cp_measure default | To match CellProfiler | Why the default differs |
|---|---|---|---|
| [Intensity percentiles](#legacy-percentile-convention) | `legacy=False`: standard linear quantiles, `(n - 1) * q` | `legacy=True`: CellProfiler's `n * q` | Standard quantiles match NumPy. CellProfiler's 3D MAD also has a bug: it uses the ⅓ quantile instead of the median. `legacy=True` reproduces that bug. |
| [Granularity](#granularity-of-objects-without-signal) | `zero_signal_policy="nan"`: NaN for objects without signal | `zero_signal_policy="epsilon_floor"` | CellProfiler floors the object's mean at machine epsilon, giving 0, or values around 1e15 when a bright neighbor's signal leaks in. NaN marks the value as undefined. |
| [Costes](#explicit-cellprofiler-costes-compatibility) | `scope="object"`, `precision="float64"`: thresholds fitted per object, in float64 | `scope="image"`, `precision="input"`, `scale_max` matching the data | With per-object thresholds, an object's values stay the same when other objects are added. float64 reduces rounding. |
| [RWC](#rwc-ranking-scope) | `rank_scope="object"`: each object ranked on its own | `rank_scope="object_set"`: ranks shared across all objects | An object's coefficient stays the same when other objects are added. |
| [Zernike, radial Zernikes](#zernike-boundary-compatibility) | `legacy_boundary=False`: includes pixels on the enclosing circle | `legacy_boundary=True`: strict `r² > 1` cutoff | The strict cutoff drops boundary pixels depending on float rounding, so values change when an unchanged object moves in the image. |

Also use your pipeline's own module settings, e.g. texture `scale`. Through the featurizer:

```python
config = make_featurizer_config(
    channels,
    legacy=True,  # intensity percentiles only
    granularity_params={"zero_signal_policy": "epsilon_floor"},
    correlation_costes_params={"scope": "image", "precision": "input", "scale_max": 65535},
    correlation_rwc_params={"rank_scope": "object_set"},
    zernike_params={"legacy_boundary": True},
    radial_zernikes_params={"legacy_boundary": True},
    texture_params={"scale": 15},  # your pipeline's setting
)
```

Granularity and Costes have been compared directly with CellProfiler 4.2.8 on real images (in [cpm_benchmark](https://github.com/afermg/cpm_benchmark)) and agree within numerical tolerance, not bit for bit. `legacy_boundary=True` restores cp_measure's earlier Zernike convention rather than being checked against CellProfiler directly.

## Feature coverage policy

Maximize valid feature coverage: prefer 3D measurements; where unavailable for singleton 3D inputs, generate supported 2D features. Apply fallback per measurement family, retain 2D feature names and meanings, and keep schemas consistent across backends and empty/nonempty inputs.

## 3D size and shape

The featurizer supports both single- and multichannel volumes with `is_3d=True`. Both images `(B, C, Z, Y, X)` and masks `(B, M, Z, Y, X)` must have all five axes, with matching batch size and spatial dimensions. A 2D mask stack is rejected; singleton spatial axes are not stretched.

To use a 2D mask through a volume, the caller must explicitly repeat it before calling `featurize`, for example `masks_3d = np.repeat(masks_2d[:, :, None], image.shape[2], axis=2)`. Each repeated label becomes one 3D object, so intensity and shape measurements use the entire volume rather than separate 2D measurements per slice. The featurizer does not perform this conversion or modify its inputs.

For `Z=1`, size/shape measures the sole plane in both backends, including through the bulk wrappers and featurizer. It returns 2D feature names and pixel-based values: 78 default features, not the 224-feature 3D schema. All-background raw masks use the same 2D schema with zero-length arrays. This replaces the previous nonempty-input error and changes the previous empty singleton-Z schema. As for explicit 2D inputs, `spacing` does not scale these measurements.

For `Z>1`, size/shape retains its 3D implementation and spacing convention. Its surface-area calculation still requires at least two voxels along each spatial axis; singleton X/Y axes are not squeezed. Raw size/shape calls remain per-image, not batch APIs.

## Numba size and shape

With `set_accelerator("numba")`, size/shape accelerates convex-hull construction and perimeter/Euler reductions for 2D and singleton-Z inputs. Hull rasterization, Euclidean distance transforms and the existing NumPy moment calculations remain unchanged. Euler output stays integer; returned values are not rounded or clipped.

2D moment comparisons use object-local scaling: raw/central moments of total order k are divided by `Area**(1+k/2)` and compared with `rtol=1e-7, atol=1e-12`. Normalized and Hu moments use the same tolerance directly. Orientations are compared modulo 180 degrees with absolute tolerance `1e-8` degrees. These are test comparison rules only; other geometry, dtype and nonfinite checks are unchanged.

Both raw functions require contiguous labels. The featurizer and default bulk wrappers sanitize arbitrary positive IDs without modifying inputs. Multi-slice volumes use the NumPy fallback. Warmed gains depend on object size and density; first calls also pay compilation cost.

## Featurizer batch execution

`featurize` calls each enabled measurement once per mask type over the supplied FOVs and channels or channel pairs. Numba can schedule these items together and reuse per-mask preparation; NumPy measures them serially through the shared batch wrappers. Shape-only calls with no image channels skip channel measurements. Output remains a C-contiguous float64 matrix, with rows ordered by image, mask type and original label.

Larger batches expose more parallel work but retain more intermediate results and worker scratch, especially with many channel pairs. Split large workloads into smaller calls to limit peak memory, passing explicit `image_ids` to preserve provenance. The batch layout alone does not establish a speedup over the current `main` implementation; the original PR's whole-stack timings used an older baseline.

## Numba intensity batches

The direct Numba `get_intensity` function accepts matching 2D or 3D images and returns one feature dictionary. Matching `(B, Z, Y, X)` arrays or lists/tuples of equally shaped 2D/3D images return a list of dictionaries. A 3D array is one image, never an implicit batch. Empty or ragged batches and mismatched input ranks/shapes are rejected; masks are not broadcast or repeated.

`legacy` and `edge_measurements` apply to every image. Legacy MAD depends on the original spatial rank: a 2D list item uses the 2D convention, while a `(B, 1, Y, X)` batch contains singleton-Z volumes and retains the 3D convention. Batch items run on a thread pool, each with the existing kernels and per-item views. The NumPy direct function accepts the same batch forms and measures each item in turn; featurizer input conventions are unchanged.

Both intensity backends require matching label and image shapes. The NumPy function previously broadcast a 2D mask across a 3D image; callers who want that measurement can now pass `np.broadcast_to(mask_2d, pixels_3d.shape)` explicitly. Each repeated label is measured as one object across the volume.

For tied maximum intensities, both backends report the last maximum pixel in raster (C) order. NumPy previously selected the first one, so `Location_MaxIntensity_X/Y/Z` can change on tied inputs; the intensity value itself does not change.

## Numba granularity batches

Batch images should share one dtype. A same-dtype batch (including any `(B, Z, Y, X)` array) gives the same values as measuring each image separately. Mixed-dtype lists or tuples are the caller's responsibility: they are not checked or rejected, and their results are not guaranteed. Currently granularity measures such a batch at its common dtype, so a mixed float32/float64 list uses float64 arithmetic even with `precision="input"`. The `precision="float64"` option widens any remaining float16/float32 data before arithmetic. To preserve each input dtype, measure the images in separate calls.

## Granularity intermediate precision

The Numba granularity backend preserves the input dtype through background opening, background resampling and subtraction, matching the NumPy path (#123). The extrema kernels may compute in float64 internally, but their selected values are restored to the incoming background dtype before interpolation. Without that boundary conversion, float32 inputs could produce small but observable backend differences (approximately `6.5e-6` in the real-image pilot). Float64 behavior is unchanged; this is not arbitrary rounding of final measurements or a relaxed comparison tolerance. Integer images are measured as their exact float64 values, so the background subtraction cannot wrap around.

For 2D float16 images with `subsample_size=1`, the Numba backend retains its historical float64 widening, including when the background is resampled. This exception also applies to same-dtype batches with `precision="input"`; float32 and float64 arithmetic is unchanged. Float16 image subsampling still requires `precision="float64"`.

3D volumes use the same numba scheme as 2D (ball opening, 6-connectivity reconstruction) and follow the NumPy path's upsampling and per-object means voxel by voxel, so results are bit-identical to NumPy, including its float32 rounding of the upsampled reconstruction.

Kernel bit-parity checks require identical dtype, shape, NaN locations, and all non-NaN bits (including signed zeros and infinities). NaN sign and payload encodings are excluded: CI observed different NaN signs between SciPy and Numba for the same non-finite interpolation inputs, without differences in NaN locations or non-NaN values.

## Granularity of objects without signal

Each `Granularity_k` is a percentage of the object's raw mean intensity, so both backends return NaN for an object whose raw mean is at or below float64 eps (#132). CellProfiler's MeasureGranularity floors that mean at eps instead. With the floor, an object without signal gets 0 when nothing leaks into it, and values around 1e15 when the bilinear upsampling of each spectrum step spreads intensity from a bright neighbour into its border pixels. Pass `zero_signal_policy="epsilon_floor"` to either backend, or `granularity_params={"zero_signal_policy": "epsilon_floor"}` to the featurizer, to restore CellProfiler's floor.

The upsampling leak itself is unchanged, so a dim object next to a bright one can still get values outside 0 to 100. For example, an object at 1% of its neighbour's brightness gets `Granularity_1` near -57. With `subsample_size=1` there is no upsampling and no leak.

## Texture quantization

Intensities use a fixed image-wide mapping, not per-object min/max scaling. Dim variation can disappear, and pairs containing zero are excluded. See the [#100 explanation](https://github.com/afermg/cp_measure/issues/100#issuecomment-5939719548) and [`get_texture` Notes](../src/cp_measure/core/measuretexture.py) for CellProfiler compatibility and zero/NaN behavior.

## RWC ranking scope

`get_correlation_rwc` defaults to `rank_scope="object"`: each object's channel intensities are ranked independently. This preserves existing results and keeps an object's RWC independent of other objects in the image.

Use `rank_scope="object_set"` for CellProfiler 4.2.8's ranking definition: compute zero-based dense ranks over all foreground pixels of the supplied object set, excluding background, then reduce the weighted intensities per object. The maximum rank count is shared across the object set, so changing other objects can change an object's coefficient. This is **not** image-wide ranking that includes background, nor a change to the default.

The option is available through the bulk function and `make_featurizer_config(..., correlation_rwc_params={"rank_scope": "object_set"})`. Each supplied mask type/image is its own object set. The binary single-mask API has only one object and needs no scope selection. Raw multi-label calls expect contiguous IDs; the bulk/featurizer sanitization handles noncontiguous labels.

Object-set mode requires finite foreground intensities (nonfinite background is ignored). Existing CPM conventions remain: absent objects or objects with no jointly above-threshold pixels return zero; a zero total signal with qualifying pixels yields NaN. This mode reproduces CP's ranking definition for ordinary finite inputs, not all of its image-mask or degenerate-export policies. The existing object-local threshold path is unchanged by this option; #122 separately addresses float32 threshold precision.

## Legacy percentile convention

`legacy=True` restores the original intensity quantile conventions; the default uses NumPy's linear quartiles and the usual median absolute deviation. See the [`get_intensity` docstring](../src/cp_measure/core/measureobjectintensity.py#L139-L163) for formulas, including the different legacy MAD convention in 3D.

## Zernike boundary compatibility

Shape `get_zernike` and intensity-weighted `get_radial_zernikes` include pixels on the enclosing circle by default, allowing a squared normalized radius up to `1 + 1e-9`. This avoids dropping boundary pixels solely because of floating-point rounding when an unchanged object moves in the image. It changes some historical measurements, not the feature names or normalization.

Pass `legacy_boundary=True` to either function to restore the previous strict `r² > 1` exclusion. In the Numba backend this also restores the historical centrosome coordinate calculation: changing only the cutoff while using a different circle algorithm would not reproduce the same boundary membership. Legacy mode retains compiled moment evaluation and batch support, but its coordinate preparation is not the new GIL-free kernel. The flag applies to single images, singleton-Z volumes, and every item in a batch. It is independent of intensity's `legacy` percentile option.

For the featurizer, use `zernike_params={"legacy_boundary": True}` and/or `radial_zernikes_params={"legacy_boundary": True}`. Omitting the flag, or setting it to `False`, keeps the boundary-rounding fix. Legacy compatibility is to the previous numerical convention, not a promise of bit-identical floating-point results across backends, library versions, or ill-defined near-zero Zernike phases.

## Costes integer-to-float normalization

For integer images, Costes divides by `scale_max`; normalized floats require the same scale explicitly. Equivalent floating-point expressions such as `level / scale_max` and `level * (1 / scale_max)` can differ at round-off, which can otherwise change threshold decisions substantially (see [#117](https://github.com/afermg/cp_measure/issues/117)).

By default, both public Costes functions recognize **exact** reciprocal-multiply encodings of integer acquisition-grid levels, as produced by `img_as_float`, and map them to the existing division encoding before fitting thresholds and computing coefficients. This is performed per object, without modifying caller inputs. Other float values are left unchanged: there is no blanket quantization or epsilon-based snapping of nearby values. Explicit lower-bit-depth scales, such as `4095` for 12-bit images in `uint16`, use the same rule. The explicit compatibility path below does not apply this canonicalization.

This normalization correction resolves conversion-path differences for these grid encodings; it does not promise continuity for genuinely different pixel values, identical results across floating-point precisions, or equivalence between different Costes search modes. It does not itself change threshold comparisons or NaN rules. The separate boundary-policy rollback is described below.

## Costes boundary results and downstream quality control

NumPy and Numba retain CellProfiler-style numerical boundary results instead of
vetoing them based on the final background subset. This selectively reverses the
background-search safeguards added in #110, not its acquisition-scale correction:

- Neither search rechecks final background size, variation or correlation before
  returning its thresholds. A final background with 0–2 pixels or a constant
  channel can therefore produce finite coefficients, including zero or one.
- Faster/bisection retains the `valid=1` fallback: if no nonnegative candidate is
  found, the thresholds are `(0, fitted_intercept)`. During the search, subsets
  with at most two pixels still move the lower bound; a constant subset leaves
  the bounds unchanged, matching CP's numerical comparisons.
- Fast/Accurate linear searches accept two-pixel correlations. Fewer than two
  pixels end the search at the current thresholds; constant subsets follow the
  ordinary one-step descent rather than causing an explicit NaN veto.

Finite results are **not** a confidence or quality flag; filtering belongs to the
caller. This does not promise the absence of NaNs: empty/missing labels,
nonfinite/constant/insufficient input signals (including fewer than three usable
input pairs), undefined or ill-conditioned regressions, nonconvergence, and zero
coefficient denominators retain their existing diagnostics. Constant background
subsets are handled without calling SciPy, so CP's incidental constant-input
warnings are not reproduced. No NaNs are imputed or replaced after calculation.

The label APIs of both backends emit one `RuntimeWarning` per undefined reason
for each image, with the number of objects, e.g. `Undefined Costes result for 412
objects: constant channel in the regression subset`. The reasons come in the
order of the first object that has them, and a batch warns for each image. If a
call raises, for example for pixels outside [0, 1], it emits no such summary. The
single-object functions (`get_correlation_costes_ind`, `linear_costes` and
`bisection_costes`) still warn with `Undefined Costes result: <reason>`.

Per-object threshold fitting, scale validation, normalization, coefficient
formulas and input immutability are unchanged. CP's object workflow still fits
thresholds image-wide, so this rollback does not establish full CP/CPM pipeline
equivalence. Historical NaN rates and timings must not be reused as measurements
of this new policy.

## Explicit CellProfiler Costes compatibility

The defaults are **per-object threshold fitting and float64 arithmetic** on both
backends. Fitting scope and arithmetic precision are independent controls:

- `scope="object"` (default) fits each object independently. `scope="image"`
  fits once over the whole image, including label-zero background. Coefficients
  remain per-object; changing background or other objects can affect them only
  with image-wide fitting.
- `precision="float64"` (default) keeps the native float64, normalization and
  undefined-result policies. `precision="input"` selects the historical
  arithmetic described below **without changing scope**. Using it with object
  scope does not reproduce CellProfiler's image-wide measurement definition.
  Input precision does not mean every intermediate uses that dtype: Pearson
  still uses float64, and integer inputs normalize to float32. This uses the
  same option names as other colocalization APIs, but Costes defaults to
  `"float64"` rather than `"input"`.

To reproduce the tested **CellProfiler 4.2.8 normalized-float32 workflow**, use
image scope, float32 input images and the compatibility arithmetic together:

```python
get_correlation_costes(
    first.astype(numpy.float32),
    second.astype(numpy.float32),
    labels,
    scale_max=65535,
    scope="image",
    precision="input",
)
```

Here `first` and `second` are already normalized to [0, 1]; casting raw integers
to float32 does not normalize them. Alternatively, pass integer images directly
with the correct acquisition scale. Through the featurizer, use
`correlation_costes_params={"scale_max": 65535, "scope": "image", "precision": "input"}`
with matching input images. Image scope alone, or float32 input alone on the
native path, is insufficient for compatibility. Casting previously rounded
float64 data cannot reconstruct the original loader's float32 values.

The compatibility arithmetic has these rules:

- Target **CellProfiler 4.2.8 with NumPy 1.x / SciPy 1.9** arithmetic. Preserve
  float32/float64 regression reductions, historical scalar promotions and
  threshold comparisons, and the SciPy 1.9 Pearson coefficient calculation.
  These rules are explicit so NumPy 2 does not silently change the policy.
- Normalize integer inputs to float32 using the acquisition scale. Already
  normalized float32/float64 arrays retain their values and dtype; no grid
  canonicalization, epsilon snapping or output rounding is applied. Other float
  precisions are rejected rather than silently treated as a reference format.
- Use the same serial reference implementation from NumPy and Numba, including
  for batches and volumes. `scipy_ties` has no effect in this mode: the
  compatibility search always uses the historical Pearson calculation.
- Keep the current feature keys and float64 output arrays. Reference boundary
  outcomes can include zeros for undefined regressions or an empty joint
  threshold mask, and NaNs for individual zero denominators. Do not interpret
  finite values as quality guarantees. Repeated identical warnings are emitted
  once per image; CellProfiler's warning counts/text are not an API promise.

Both scopes work with either arithmetic policy. Native image scope also uses
the shared serial reference, always with float64 and current SciPy decisions;
it is not the fused Numba object kernel. Native image-wide fits retain the
native invalid-signal guards, so a nonfinite background pixel can invalidate
the fit for all objects. Compatibility fitting excludes NaN pairs instead.
Compatibility object scope leaves missing labels and objects with fewer than
two usable paired pixels as NaN and aggregates their warnings per image.

Matching acquisition scale and normalized input values is necessary. This is
not a CellProfiler pipeline or image-loader emulator: caller inputs must have
matching shapes, labels must be nonnegative, and CellProfiler's separate image
masks and cropping are not modeled. NaNs are excluded from threshold fitting;
infinities and out-of-range signals are rejected. Array dtype alone does not
identify an acquisition scale for normalized floats, so `scale_max` remains
required. Compatibility does not imply bit-identical results on every BLAS,
platform or dependency version, nor a speedup over the ordinary path.

Live CellProfiler comparisons and their input/environment provenance belong in
`cpm_benchmark` (`benchmarks/extract_costes_compatibility.py` and
`benchmarks/compare_costes_compatibility.py`), not a CellProfiler dependency or
fixture generator in this package's tests.

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

The accelerated path groups paired pixels once per measurement and uses a fused
float64 kernel. Pearson retains NumPy's mean reductions, including their behavior
for constant signals. With `precision="float64"`, percentage-threshold metrics
promote float16/float32 inputs per image and use the same float64 kernel.

Float32 inputs, including one float32 channel paired with a float64 one, are also
accelerated with `precision="input"`. A float32 channel's percentage threshold is
computed with the reference's own expression on float32 maxima, so
the same pixels pass each threshold as in NumPy. The sums are float64, whereas the
NumPy reference sums float32 pixels in float32, so values differ by float32 rounding
(about `1e-7` relative). The float64 sums are the more accurate of the two. On the
#129 boundary fixture they reproduce CellProfiler's stored Manders value exactly,
where NumPy's input mode returns it rounded to float32. NumPy's own float32 sums also
change with the NumPy version for objects above 8192 pixels (NumPy 2.3 dropped the
8192-element chunking), so input-mode float32 values are not bit-stable across NumPy
versions in either backend.

Float16, integer and wider-float inputs retain the reference path. For accelerated
float32/float64 Pearson, Manders and RWC, undefined objects keep the kernel's NaN
results without replaying the whole image through NumPy. Incidental NumPy runtime
warnings from those calculations are therefore no longer guaranteed. Overlap/K
replays only nonfinite objects through the single-object reference, preserving
the existing true zero-denominator error. Reference fallbacks are not advertised
as accelerated. These changes do not alter the separate Costes warning policy.

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

Both backends accept `rank_scope="object_set"`. The Numba wrapper delegates this
opt-in scope to the NumPy reference per image, including in explicit batches;
it does not rank across batch items. The default `rank_scope="object"` retains
its accelerated object-local calculation. Bulk and featurizer calls forward
the explicit scope without changing the default.

Both scopes accept `precision="input"` (default) or `precision="float64"`.
Object-set float64 mode widens stored float16/float32 values before ranking,
thresholding and reduction; it does not promise CellProfiler float32 parity.

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
the other colocalization metrics. Small/constant search backgrounds use the
permissive boundary policy above, not an output veto. Near-constant subsets and
correlations within a reduction-error bound of a search decision are
replayed per object by the reference when `scipy_ties=True` to preserve its
diagnostics. Ill-conditioned regressions retain their reference fallback
independently of this flag; no correlation is rounded or search cutoff moved.

Some search steps are ties: the background correlation is within rounding of
a decision point (r = 0, or Fast mode's 0.25/0.35/0.45), which on small
integer-grid objects is often exactly 0, or the subset is nearly constant. The
NumPy backend's answer there follows SciPy's last bits (its sums and dot
product), not the data. By default the numba backend decides these steps with
its own arithmetic, so the backends can differ on such objects (8 of 6000
randomized test images), and it does not emit SciPy's
`NearConstantInputWarning`. With `scipy_ties=True` the numba backend asks SciPy
for exactly those steps and resumes its own search, so results and warnings
are bit-identical to the NumPy backend.

The default remains `scipy_ties=False` to avoid SciPy callbacks for ambiguous
steps in the compiled search, not because an overall speedup was established.
The tested real-image workload showed no meaningful timing advantage. Use
`True` when agreement with the NumPy backend matters. Revisit the default if
further workloads show large numerical discrepancies, a speed penalty, or no
useful speedup from keeping these steps in Numba. This setting neither removes
SciPy as a dependency nor disables unrelated exceptional-case reference fallbacks.

The fused searches share #167's exact positive-slope entry calculation: a
pixel joins the sorted background at the first representable threshold that
satisfies the original strict comparisons. Bisection uses bounded grid buckets
for objects with at least 32768 pixels when the acquisition scale is small
enough relative to the object's size, otherwise its one-pass moment scan.
Sorting ordinary objects slowed Faster mode relative to the fused scan in
matched tests, so that #167 dispatch is deliberately not carried over.
Long linear searches retain
their delayed sort and guarded negative-slope sweep. All paths use the same
rounding guard, mode-specific cutoffs and `scipy_ties` replay policy; acceleration
does not change the policy or introduce image-wide thresholds. Sorted scratch
is capped at 2^22 pixels and grid scratch at scale 2^20; larger inputs scan when
neither bounded path is eligible. Dispatch constants are frozen in Numba's
compiled kernels, not runtime tuning parameters.

Both backends apply the same per-object normalization canonicalization before
validation, regression and threshold search. Exact reciprocal-multiply
acquisition-grid encodings therefore follow the same division encoding in the
NumPy and Numba paths; arbitrary off-grid floats are unchanged. This is the
normalization rule described above, not an epsilon-based threshold adjustment.

No fused `get_correlation_all` dispatch is introduced. Costes supports matching
2D, singleton-Z and true 3D inputs, and the existing explicit batch convention;
it does not repeat masks or select an intensity plane implicitly.
