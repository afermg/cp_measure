<div align="center">
<img src="./logos/cpm.svg" width="150px">
</div>

# cp_measure: Morphological features for imaging data
Do you need to use [CellProfiler](https://github.com/CellProfiler) features, but you want to do it in a programmatic way? Look no more, this package was developed by and for the click-a-phobic scientists.


### Preprint
[Here](https://arxiv.org/abs/2507.01163) is the current version of the preprint.

If you used cp\_measure in your project, please cite using the following bib entry:

```
@article{munoz2025cpmeasure,
  title={cp\_measure: API-first feature extraction for image-based profiling workflows},
  author={Mu{\~n}oz, Al{\'a}n F and Treis, Tim and Kalinin, Alexandr A and Dasgupta, Shatavisha and Theis, Fabian and Carpenter, Anne E and Singh, Shantanu},
  journal={arXiv preprint arXiv:2507.01163},
  year={2025}
}
```

## Quick overview


### Installation

```bash
pip install cp-measure
```

### Usage

#### Featurizer (recommended)

The featurizer wraps all measurements into a two-step workflow: configure once with `make_featurizer`, then call `featurize` on each image. It returns plain numpy arrays and Python lists — no extra dependencies required.

```python
import numpy as np
from cp_measure.featurizer import make_featurizer, featurize

# image: (C, H, W) float array, masks: (N_masks, H, W) integer labels
image = np.random.default_rng(42).random((5, 240, 240))
masks = np.zeros((1, 240, 240), dtype=np.int32)
masks[0, 50:100, 50:100] = 1
masks[0, 150:200, 150:200] = 2

# Channel names are matched positionally to the image's first axis.
# They control how per-channel features are labeled in the output columns
# (e.g. "Intensity_MeanIntensity__DNA"). If omitted, channels are auto-named ch0, ch1, ...
config = make_featurizer(["DNA", "ER", "RNA", "AGP", "Mito"])

data, columns, rows = featurize(image, masks, config)
# data:    np.ndarray of shape (n_objects, n_features)
# columns: ["Area", "Intensity_MeanIntensity__DNA", ...] — feature names
# rows:    [(None, "object", 1), (None, "object", 2)]  — (image_id, object_name, label) per row
```

When you have multiple segmentation masks (e.g. nuclei and cells), pass them stacked along the first axis and name them with `objects`. Each mask can have a different number of labels; all rows share the same feature columns.

```python
config = make_featurizer(["DNA", "ER"], objects=["nuclei", "cells"])

masks = np.zeros((2, 240, 240), dtype=np.int32)
masks[0, 50:100, 50:100] = 1    # nucleus 1
masks[1, 40:110, 40:110] = 1    # cell 1
masks[1, 150:200, 150:200] = 2  # cell 2

data, columns, rows = featurize(image[:2], masks, config)
# rows: [(None, "nuclei", 1), (None, "cells", 1), (None, "cells", 2)]
```

The output is plain numpy + lists, so converting to a DataFrame is straightforward:

```python
import pandas as pd
row_names = [f"{img}__{obj}__{label}" for img, obj, label in rows]
df = pd.DataFrame(data, index=row_names, columns=columns)
```

#### Bulk API

Users usually want to calculate all the features. There are four type of measurements, based on their inputs:

-   Type 1: 1 image + 1 set of masks (e.g., intensity)
-   Type 2: 2 images + 1 set of masks (e.g., colocalization)
-   Type 3: 2 sets of masks (e.g., number of neighbors)
-   Type 4: 1 image + 2 sets of masks (e.g., skeleton)

This shows the simplest way to use the first set (1 image, 1 mask set), which currently follows the style of scikit-image (1 image, 1 matrix with non-overlapping labels). **IMPORTANT:** If you need to match CellProfiler measurements 1:1, you must convert your image arrays to float values between 0 and 1. For instance, if you have an array of data type uint16, you must divide them all by 65535. This is important for radial distribution measurements.

NOTE: The input labels must be sequential (e.g., `[1,2,3]`, not `[1,3,4]`). You can use `skimage.segmentation.relabel_sequential` to ensure compliance.

```python
import numpy as np

from cp_measure.bulk import get_core_measurements

measurements = get_core_measurements()
print(measurements.keys())
# dict_keys(['radial_distribution', 'radial_zernikes', 'intensity', 'sizeshape', 'zernike', 'feret', 'texture', 'granularity'])

import numpy as np
import pandas as pd
from cp_measure.bulk import get_core_measurements

# Create synthetic data
size = 240
rng = np.random.default_rng(42)
pixels = rng.integers(low=1, high=255, size=(size, size))

# Create two similar-sized objects
masks = np.zeros_like(pixels)
masks[50:100, 50:100] = 1  # First square 50x50
masks[80:120, 90:120] = 1  # Major asymmetries on bottom right edge
masks[150:200, 150:200] = 2  # Second square 50x50
masks[175:180, 180:210] = 2  # Minor asymmetries on bottom right edge

# Get measurements
measurements = get_core_measurements()

results = {}
for name, v in measurements.items():
    results = {**results, **v(masks, pixels)}

"""
{'RadialDistribution_FracAtD_1of4': array([0.03673493, 0.05640786]),
 'RadialDistribution_MeanFrac_1of4': array([1.02857809, 1.15072037]),
 'RadialDistribution_RadialCV_1of4': array([0.05539421, 0.04635982]),
 ...
 'Granularity_16': array([97.65759629, 97.64371833])}
"""
```


#### Call specific measurements

If you need a specific measurement/feature you can just import it. Note that measurements come in sets, so you have to fetch the one that you specifically require from the resultant dictionary. Any available measurement can be found using code as follows:

```python
import numpy as np

from cp_measure.minimal.measureobjectsizeshape import get_sizeshape

mask = np.zeros((50, 50))
mask[5:-6, 5:-6] = 1
get_sizeshape(mask, None) # pixels, the second argument, is not necessary for this particular measurement
```

The other available functions are as follows:

```
measureobjectintensitydistribution.get_radial_zernikes
measureobjectintensity.get_intensity
measureobjectsizeshape.get_zernike
measureobjectsizeshape.get_feret
measuregranularity.get_granularity
measuretexture.get_texture
```

And for Type 2 functions:

```
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

### Contribute

Please use GitHub issues to report bugs and issues or submit a Pull Request.

### Development installation

If you want to install it for development use [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:afermg/cp_measure.git
cd cp_measure
uv sync --all-groups
```

## Current work

You can follow progress [here](https://docs.google.com/spreadsheets/d/1_7jQ8EjPwOr2MUnO5Tw56iu4Y0udAzCJEny-LQMgRGE/edit?usp=sharing).


### Done

-   Type 1 and 2 in sklearn style (multiple integer labels in one mask array)

### Pending

-   Add a wrapper for type 3 measurements
-   Type 4 measurements (ObjectSkeleton). We don't know if it is worth implementing.


# Design notes

- cp\_measure is not optimised for efficiency (yet). We aim to reproduce the 'vanilla' results of CellProfiler with minimal code changes. Optimisations will be implemented once we come up with a standard interface for functionally-focused CellProfiler components.
- The Image-wide functions will not be implemented directly, they were originally implemented independently to the Object (mask) functions. We will adjust the existing functions assume that an image-wide measurement is the same as measuring an object with the same size as the intensity image.
- The functions do not include guardrails (e.g., checks of type or value). They will fail if provided with empty masks. Not all functions will fail if provided with masks only.
