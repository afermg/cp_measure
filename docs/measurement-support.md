# Measurement support

This table covers the object measurements exposed by the bulk registries. The featurizer uses the 3D registry for volumes. For `Z=1`, it additionally enables the 2D-only families on the sole slice; for `Z>1`, it skips them with a warning. Both backends use this rule. This is not a 3D implementation of those measurements; genuinely 3D features retain their input rank.

| Measurement           | 2D  | Multi-slice 3D | Notes                                                                             |
| --------------------- | --- | -------------- | --------------------------------------------------------------------------------- |
| `intensity`           | Yes | Yes            | Per channel; `legacy` selects percentile conventions.                             |
| `sizeshape`           | Yes | Yes            | Shape-only; 3D surface area requires at least two voxels along each spatial axis. |
| `texture`             | Yes | Yes            | Per channel; see [quantization notes](measurement-notes.md#texture-quantization). |
| `granularity`         | Yes | Yes            | Per channel.                                                                      |
| `radial_distribution` | Yes | No             | Measures `Z=1`; returns `{}` for `Z>1`.                                           |
| `radial_zernikes`     | Yes | No             | Measures `Z=1`; returns `{}` for `Z>1`.                                           |
| `zernike`             | Yes | No             | Shape-only; measures `Z=1`, returns `{}` for `Z>1`.                               |
| `feret`               | Yes | No             | Shape-only; measures `Z=1`, returns `{}` for `Z>1` in both backends.              |
| `pearson`             | Yes | Yes            | Two channels.                                                                     |
| `costes`              | Yes | Yes            | Two channels; see the API's intensity-scale requirements.                         |
| `manders_fold`        | Yes | Yes            | Two channels.                                                                     |
| `rwc`                 | Yes | Yes            | Two channels.                                                                     |

“Supported” describes dimensionality, not a guarantee that every feature is defined for every object. Constant intensities, tiny objects, and other degenerate inputs may produce undefined values. Empty masks have no uniform raw-function contract; for example, raw Feret raises `ValueError` for an all-background 2D mask.

## Labels and entry points

- The featurizer and default bulk wrappers accept non-contiguous positive integer IDs, relabel each mask independently, and leave the input unchanged. Background is `0`.
- Raw functions (including bulk registries with `sanitize=False`) require contiguous labels `1..N`. Some accept additional inputs, such as a boolean mask for raw Feret; these are not part of the bulk wrapper's integer-label contract.
- The batched featurizer skips empty images and rejects an all-empty batch. That behavior does not change the underlying raw measurements.

## Numba Feret batch inputs

The raw Numba Feret backend and its bulk wrapper support these inputs. These are **not** the featurizer's batch/channel shapes.

| Masks input                     | Result                                                     |
| ------------------------------- | ---------------------------------------------------------- |
| `(Y, X)`                        | Dictionary of Feret arrays, one value per object.          |
| `(1, Y, X)`                     | Dictionary of 2D Feret measurements on the sole slice.     |
| `(Z, Y, X)` with `Z>1`          | `{}`; multi-slice volumes are unsupported.                 |
| List/tuple of same-shaped masks | List of per-image results using the same Z-depth rule.     |
| `(B, Z, Y, X)`                  | `Z=1`: measurements per volume; `Z>1`: empty dictionaries. |

The NumPy raw Feret function processes one image at a time. The Numba bulk wrapper sanitizes labels separately for each batch item. Numba's native-byte-order conversion remains at the backend boundary so raw and bulk callers also work; it does not rely on a featurizer call.

Multi-slice volumes return `{}` rather than raising. Singleton-Z inputs now opt into the 2D calculations consistently; this is an explicit eligibility change, not merely an acceleration.
