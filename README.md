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

### Featurizer (Recommended for small datasets)

The simplest way to extract all features from an image and its masks:

```python
import numpy as np
from cp_measure.featurizer import featurize

# image: (C, H, W) float array, masks: (N_masks, H, W) integer labels
image = np.random.default_rng(42).random((2, 240, 240))
masks = np.zeros((1, 240, 240), dtype=np.int32)
masks[0, 50:100, 50:100] = 1
masks[0, 150:200, 150:200] = 2

data, columns, rows = featurize(image, masks)
# data:    np.ndarray of shape (n_objects, n_features)
# columns: feature names (e.g. "Area", "Intensity_MeanIntensity__ch0", ...)
# rows:    [(None, "object", 1), (None, "object", 2)]  — (image_id, object_name, label) per row
```

To customise which features are extracted, or to name your channels and masks, use `make_featurizer_config`. Channel names are matched positionally to the image's first axis and control how per-channel features are labeled in the output columns (e.g. "Intensity_MeanIntensity__DNA"). If omitted, channels are auto-named `ch0`, `ch1`, ...

```python
import numpy as np
from cp_measure.featurizer import featurize, make_featurizer_config

# Recreate variables from previous examples for this block to run in isolation
image = np.random.default_rng(42).random((2, 240, 240))
masks = np.zeros((1, 240, 240), dtype=np.int32)
masks[0, 50:100, 50:100] = 1
masks[0, 150:200, 150:200] = 2

# Disable texture features, name channels explicitly
config = make_featurizer_config(["DNA", "ER"], texture=False)
data, columns, rows = featurize(image, masks, config)
```

Multiple mask types (e.g. nuclei and cells) are supported by stacking them along the first axis:

```python
import numpy as np
from cp_measure.featurizer import featurize, make_featurizer_config

# Recreate variables from previous examples for this block to run in isolation
image = np.random.default_rng(42).random((2, 240, 240))

config = make_featurizer_config(["DNA", "ER"], objects=["nuclei", "cells"])

masks = np.zeros((2, 240, 240), dtype=np.int32)
masks[0, 50:100, 50:100] = 1    # nucleus 1
masks[1, 40:110, 40:110] = 1    # cell 1
masks[1, 150:200, 150:200] = 2  # cell 2
masks[1, 175:180, 180:210] = 2  # Minor asymmetries on bottom right edge of cells

data, columns, rows = featurize(image, masks, config)
# rows: [(None, "nuclei", 1), (None, "cells", 1), (None, "cells", 2)]
```

Volumetric `(C, Z, H, W)` data is supported. The featurizer automatically skips 2D-only features (`radial_distribution`, `radial_zernikes`, `zernike`, `feret`). All other features (`intensity`, `sizeshape`, `texture`, `granularity`, correlations) work for both 2D and 3D.

The default output is plain numpy + lists. Use `return_as` to get structured output directly:

```python notest
# pip install cp_measure[pandas]
df = featurize(image, masks, config, return_as="pandas")
#   image_id object_type  label   Area  BoundingBoxArea  ...
#       None      object      1 2500.0           2500.0
#       None      object      2 2500.0           2500.0
# Shape: (2, 123) — 3 metadata columns + 120 feature columns

# pip install cp_measure[anndata]
adata = featurize(image, masks, config, return_as="anndata")
# AnnData object with n_obs x n_vars = 2 x 120
#     obs: 'image_id', 'object_type', 'label'
#     var: 'feature_group', 'feature_type', 'feature_name', 'channel', 'channel_2'
#     uns: 'config', 'channels', 'objects', 'is_3d'

# pip install cp_measure[pyarrow]
table = featurize(image, masks, config, return_as="pyarrow")
# pyarrow.Table with 2 rows, 123 columns
# Per-column metadata in schema fields, table-level metadata:
#   'cp_measure_config', 'channels', 'objects', 'is_3d'
```

These are optional dependencies — install only what you need (or `pip install cp_measure[all]`).

## Important notes

- **Contiguous labels**: The input labels must be sequential (e.g., `[1,2,3]`, not `[1,3,4]`). You can use `skimage.segmentation.relabel_sequential` to ensure compliance.
- **Fidelity**: If you need to match CellProfiler measurements 1:1, you must convert your image arrays to float values between 0 and 1. For instance, if you have an array of data type uint16, you must divide them all by 65535. This is important for radial distribution measurements.
- **Speed**: The Granularity measurement is particularly slow (~80% of the compute time). Skip this one it if speed is of utmost importance.

<details>
<summary>API Overview (develop your own pipelines)</summary>

### Bulk API

For more control over individual measurements, or to call specific functions directly, use the bulk API. It operates on single images and masks following the scikit-image convention.

cp\_measure currently provides two types of measurements based on their inputs:

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

### Import a subset of measurements

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

For Type 3 functions:

```
measureobjectoverlap.measureobjectoverlap
measureobjectneghbors.measureobjectneighboors
```

## Similar projects

- [spacr](https://github.com/EinarOlafsson/spacr): Library to analyse screens, it provides measurements (independent implementation) and a GUI.
- [ScaleFEX](https://github.com/NYSCF/ScaleFEx): Python pipeline that includes measurements, designed for the cloud.
- [thyme](https://github.com/tomouellette/thyme): Rust library to extract a subset of CellProfiler's features efficiently (independent implementation).

</details>

<details>
<summary>Current work</summary>

You can follow progress [here](https://docs.google.com/spreadsheets/d/1_7jQ8EjPwOr2MUnO5Tw56iu4Y0udAzCJEny-LQMgRGE/edit?usp=sharing).

Most features are implemented, but Type 3 measurements (e.g., `ObjectNeighbors`) does not have a wrapper. We do not plan to implement `ObjectSkeleton`.

</details>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
