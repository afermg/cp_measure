<div align="center">
<img src="./logos/cpm.svg" width="150px">
</div>

# cp_measure: Morphological features for imaging data

Do you need to use [CellProfiler](https://github.com/CellProfiler) features, but you want to do it in a programmatic way? Look no more, this package was developed by and for the click-a-phobic scientists.

### Preprint

[Here](https://arxiv.org/abs/2507.01163) is the preprint. Published as a workshop paper for ICML 2025's CODEML.

<details>
<summary>Please cite using the following .bib entry</summary>

```
@article{munoz2025cpmeasure,
  title={cp\_measure: API-first feature extraction for image-based profiling workflows},
  author={Mu{\~n}oz, Al{\'a}n F and Treis, Tim and Kalinin, Alexandr A and Dasgupta, Shatavisha and Theis, Fabian and Carpenter, Anne E and Singh, Shantanu},
  journal={arXiv preprint arXiv:2507.01163},
  year={2025}
}
```

</details>

# Quick overview

## Installation

```bash
pip install cp-measure
```

## Usage

We provide three entry points: (Featurizer) We orchestrate the images x combinations, (Bulk API) we give you all features and you orchestrate, and (Low-level) you directly import the functions.

### Featurizer (Recommended for small datasets)

<details>

The simplest way to extract all features from an image and its masks:

```python
import numpy as np
from cp_measure.featurizer import featurize

# Canonical input: image (B, C, Y, X), masks (B, N_masks, Y, X). A single image is B=1.
# Dimensionality is declared with is_3d (default False), never inferred from shape.
image = np.random.default_rng(42).random((1, 2, 240, 240))
masks = np.zeros((1, 1, 240, 240), dtype=np.int32)
masks[0, 0, 50:100, 50:100] = 1
masks[0, 0, 150:200, 150:200] = 2

data, columns, rows = featurize(image, masks)  # is_3d=False
# data:    np.ndarray of shape (n_objects, n_features)
# columns: feature names (e.g. "Area", "Intensity_MeanIntensity__ch0", ...)
# rows:    [(0, "object", 1), (0, "object", 2)]  — (image_id, object_name, label) per row
#          image_id defaults to the batch index; pass image_ids=[...] for your own ids
```

To customise which features are extracted, or to name your channels and masks, use `make_featurizer_config`. Channel names are matched positionally to the image's channel axis (axis 1) and control how per-channel features are labeled in the output columns (e.g. "Intensity_MeanIntensity__DNA"). If omitted, channels are auto-named `ch0`, `ch1`, ...

```python
import numpy as np
from cp_measure.featurizer import featurize, make_featurizer_config

# Recreate variables from previous examples for this block to run in isolation
image = np.random.default_rng(42).random((1, 2, 240, 240))
masks = np.zeros((1, 1, 240, 240), dtype=np.int32)
masks[0, 0, 50:100, 50:100] = 1
masks[0, 0, 150:200, 150:200] = 2

# Disable texture features, name channels explicitly
config = make_featurizer_config(["DNA", "ER"], texture=False)
data, columns, rows = featurize(image, masks, config)
```

Multiple mask types (e.g. nuclei and cells) are supported by stacking them along the mask-type axis (axis 1):

```python
import numpy as np
from cp_measure.featurizer import featurize, make_featurizer_config

# Recreate variables from previous examples for this block to run in isolation
image = np.random.default_rng(42).random((1, 2, 240, 240))

config = make_featurizer_config(["DNA", "ER"], objects=["nuclei", "cells"])

masks = np.zeros((1, 2, 240, 240), dtype=np.int32)
masks[0, 0, 50:100, 50:100] = 1    # nucleus 1
masks[0, 1, 40:110, 40:110] = 1    # cell 1
masks[0, 1, 150:200, 150:200] = 2  # cell 2
masks[0, 1, 175:180, 180:210] = 2  # Minor asymmetries on bottom right edge of cells

data, columns, rows = featurize(image, masks, config)
# rows: [(0, "nuclei", 1), (0, "cells", 1), (0, "cells", 2)]
```

Volumetric data is supported: pass `(B, C, Z, Y, X)` image / `(B, M, Z, Y, X)` masks together with `is_3d=True` (dimensionality is declared, not inferred — so a single-channel volume is never mistaken for a multichannel 2D image). The featurizer automatically skips 2D-only features (`radial_distribution`, `radial_zernikes`, `zernike`, `feret`). All other features (`intensity`, `sizeshape`, `texture`, `granularity`, correlations) work for both 2D and 3D.

The output is plain numpy + lists, so converting to a DataFrame is straightforward:

```python notest
import pandas as pd
row_names = [f"{img}__{obj}__{label}" for img, obj, label in rows]
df = pd.DataFrame(data, index=row_names, columns=columns)
```

Note: DataFrame libraries must be installed independently, to keep the dependency tree low.

</details>

### Bulk API (Access all measurements at once)

<details>

For more control over individual measurements, or to call specific functions directly, use the bulk API. It operates on single images and masks following the scikit-image convention.

cp_measure currently provides two types of measurements based on their inputs:

- Type 1: 1 image + 1 set of masks (e.g., intensity)
- Type 2: 2 images + 1 set of masks (e.g., colocalization)

```python
import numpy as np
from cp_measure.bulk import get_core_measurements

measurements = get_core_measurements()
# print(measurements.keys())
# dict_keys(['radial_distribution', 'radial_zernikes', 'intensity', 'sizeshape', 'zernike', 'feret', 'texture', 'granularity'])

# Create synthetic data
size = 240
rng = np.random.default_rng(42)
pixels = rng.integers(low=1, high=255, size=(size, size))

# Create two similar-sized objects
masks = np.zeros_like(pixels)
masks[50:100, 50:100] = 1
masks[150:200, 150:200] = 2

measurements = get_core_measurements()
results = {}
for name, func in measurements.items():
    results = {**results, **func(masks, pixels)}

"""
{'RadialDistribution_FracAtD_1of4': array([0.03673493, 0.05640786]),
 'RadialDistribution_MeanFrac_1of4': array([1.02857809, 1.15072037]),
 'RadialDistribution_RadialCV_1of4': array([0.05539421, 0.04635982]),
 ...
 'Granularity_16': array([97.65759629, 97.64371833])
}
"""
```

</details>

### Low-level access

<details>

Individual measurement functions can be imported directly. Each returns a dictionary of arrays.

```python
import numpy as np
from cp_measure.core.measureobjectsizeshape import get_sizeshape

mask = np.zeros((50, 50), dtype=np.int32)
mask[5:-6, 5:-6] = 1
get_sizeshape(mask, None)
```

```
measureobjectintensitydistribution.get_radial_zernikes
measureobjectintensity.get_intensity
measureobjectsizeshape.get_zernike
measureobjectsizeshape.get_feret
measuregranularity.get_granularity
measuretexture.get_texture
measurecolocalization.get_correlation_pearson
measurecolocalization.get_correlation_manders_fold
measurecolocalization.get_correlation_rwc
measurecolocalization.get_correlation_costes
measurecolocalization.get_correlation_overlap
```

</details>

### Important notes

- **Labels**: Use positive integers for objects and `0` for background. `featurize` and the bulk registries relabel non-contiguous IDs internally without modifying your array; `featurize` reports the original IDs. Raw measurement functions require contiguous `1..N` labels; wrap them with `cp_measure._sanitize.sanitize` when needed.
- **Image shapes**: `featurize` requires dense `(B, C, *spatial)` images and `(B, M, *spatial)` masks. A single image still needs `B=1`. Ragged (differently-sized) batches are not supported; normalise to a common shape and stack first.
- **Fidelity**: Use float intensities in `[0, 1]` to match CellProfiler's input convention (e.g. divide uint16 values by `65535`). Matching the original intensity quantile measurements also requires `legacy=True`; see the [legacy percentile convention](docs/measurement-notes.md#legacy-percentile-convention).
- **Speed**: v0.2.0 speeds up the default NumPy/SciPy implementation without adding required runtime dependencies; see the [performance report](benchmarks/releases/v0.2.0.md). Optional Numba and JAX backends remain under development and are not yet supported.

See [Measurement notes](docs/measurement-notes.md) for texture quantization, percentile formulas, radial-distribution centers, and floating-point caveats.

## Similar projects

- [spacr](https://github.com/EinarOlafsson/spacr): Library to analyse screens, it provides measurements (independent implementation) and a GUI.
- [ScaleFEX](https://github.com/NYSCF/ScaleFEx): Python pipeline that includes measurements, designed for the cloud.
- [CharmFeatures](https://gitlab.com/iggman/charm-features): Library, Python module, and command-line utility for extracting Wnd-Charm image features from large TIFF collections.
- [thyme](https://github.com/tomouellette/thyme): Rust library to extract a subset of CellProfiler's features efficiently (independent implementation).
- [CellProfiler Library](https://github.com/CellProfiler/CellProfiler/tree/main/src/subpackages/library): WIP library that isolates CellProfiler image-processing functions from its frontend.

<details>
<summary>Current work</summary>

You can follow progress [here](https://docs.google.com/spreadsheets/d/1_7jQ8EjPwOr2MUnO5Tw56iu4Y0udAzCJEny-LQMgRGE/edit?usp=sharing).

Most features are implemented, but Type 3 measurements (e.g., `ObjectNeighbors`) does not have a wrapper. We do not plan to implement `ObjectSkeleton`.

</details>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
