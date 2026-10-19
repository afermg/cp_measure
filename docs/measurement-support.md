# Measurement support

This table covers the object measurements exposed by the bulk registries. The featurizer uses the 3D registry for volumes and skips requested 2D-only features with a warning. Choosing an accelerator does not make a 2D measurement support 3D.

| Measurement           | 2D  | 3D  | Notes                                                                             |
| --------------------- | --- | --- | --------------------------------------------------------------------------------- |
| `intensity`           | Yes | Yes | Per channel; `legacy` selects percentile conventions.                             |
| `sizeshape`           | Yes | Yes | Shape-only; 3D surface area requires at least two voxels along each spatial axis. |
| `texture`             | Yes | Yes | Per channel; see [quantization notes](measurement-notes.md#texture-quantization). |
| `granularity`         | Yes | Yes | Per channel.                                                                      |
| `radial_distribution` | Yes | No  | Raw function returns `{}` for a 3D mask.                                          |
| `radial_zernikes`     | Yes | No  | Raw function returns `{}` for a 3D mask.                                          |
| `zernike`             | Yes | No  | Shape-only; raw function returns `{}` for a 3D mask.                              |
| `feret`               | Yes | No  | Shape-only; both NumPy and Numba raw functions return `{}` for a 3D mask.         |
| `pearson`             | Yes | Yes | Two channels.                                                                     |
| `costes`              | Yes | Yes | Two channels; see the API's intensity-scale requirements.                         |
| `manders_fold`        | Yes | Yes | Two channels.                                                                     |
| `rwc`                 | Yes | Yes | Two channels.                                                                     |

“Supported” describes dimensionality, not a guarantee that every feature is defined for every object. Constant intensities, tiny objects, and other degenerate inputs may produce undefined values. Empty masks have no uniform raw-function contract; for example, raw Feret raises `ValueError` for an all-background 2D mask.

## Labels and entry points

- The featurizer and default bulk wrappers accept non-contiguous positive integer IDs, relabel each mask independently, and leave the input unchanged. Background is `0`.
- Raw functions (including bulk registries with `sanitize=False`) require contiguous labels `1..N`. Some accept additional inputs, such as a boolean mask for raw Feret; these are not part of the bulk wrapper's integer-label contract.
- The batched featurizer proposed in [#106](https://github.com/afermg/cp_measure/pull/106) skips empty images and rejects an all-empty batch. That behavior does not change the underlying raw measurements.

## Numba Feret batch inputs

The raw Numba Feret backend and its bulk wrapper support these inputs. These are **not** the featurizer's batch/channel shapes.

| Masks input                              | Result                                            |
| ---------------------------------------- | ------------------------------------------------- |
| `(Y, X)`                                 | Dictionary of Feret arrays, one value per object. |
| `(Z, Y, X)`, including `Z=1`             | `{}`; the input is a volume, not a 2D batch.      |
| List/tuple of same-shaped `(Y, X)` masks | List of dictionaries, one per image.              |
| `(B, Z, Y, X)`                           | List of empty dictionaries, one per volume.       |

The NumPy raw Feret function processes one image at a time. The Numba bulk wrapper sanitizes labels separately for each batch item. Numba's native-byte-order conversion remains at the backend boundary so raw and bulk callers also work; it does not rely on a featurizer call.

Returning `{}` for 3D raw Feret preserves the existing NumPy behavior. Replacing it with an exception would be a separate API change, not an acceleration-only change.
