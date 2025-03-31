Do you need to use [CellProfiler](https://github.com/CellProfiler) features, but you want to do it in a programmatic way? Look no more, this package was developed by and for the click-a-phobic scientists.


# Quick overview


## Installation

```bash
pip install cp-measure
```


### Poetry

If you want a development environment.

```bash
git clone git@github.com:afermg/cp_measure.git
cd cp_measure
poetry install 
```


## Usage

Users usually want to calculate all the features. There are four type of measurements, based on their inputs:

-   Type 1: 1 image + 1 set of masks (e.g., intensity)
-   Type 2: 2 images + 1 set of masks (e.g., colocalization)
-   Type 3: 2 sets of masks (e.g., number of neighbors)
-   Type 4: 1 image + 2 sets of masks (e.g., skeleton)

This shows the simplest way to use the first set (1 image, 1 mask set), which currently follows the style of scikit-image (1 image, 1 matrix with non-overlapping labels).

```python
import numpy as np

from cp_measure.bulk import get_core_measurements

measurements = get_core_measurements()
print(measurements.keys())
# dict_keys(['radial_distribution', 'radial_zernikes', 'intensity', 'sizeshape', 'zernike', 'ferret', 'texture', 'granularity'])

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


### Call specific measurements

If you need a specific measurement/feature you can just import it. Note that measurements come in sets, so you have to fetch the one that you specifically require from the resultant dictionary. Any available measurement can be found using code as follows:

```python
import numpy as np

from cp_measure.minimal.measureobjectsizeshape import get_sizeshape

mask = np.zeros((50, 50))
mask[5:-6, 5:-6] = 1
get_sizeshape(mask, None) # pixels, the second argument, is not necessary for this particular measurement
```

The other available functions are as follows:

-   measureobjectintensitydistribution.get<sub>radial</sub><sub>zernikes</sub>,
-   measureobjectintensity.get<sub>intensity</sub>,
-   measureobjectsizeshape.get<sub>zernike</sub>,
-   measureobjectsizeshape.get<sub>ferret</sub>,
-   measuregranularity.get<sub>granularity</sub>,
-   measuretexture.get<sub>texture</sub>,

And for Type 2 functions:

-   measurecolocalization.get<sub>correlation</sub><sub>pearson</sub>
-   measurecolocalization.get<sub>correlation</sub><sub>manders</sub><sub>fold</sub>
-   measurecolocalization.get<sub>correlation</sub><sub>rwc</sub>
-   measurecolocalization.get<sub>correlation</sub><sub>costes</sub>
-   measurecolocalization.get<sub>correlation</sub><sub>overlap</sub>

For Type 3 functions:

-   measureobjectoverlap.measureobjectoverlap
-   measureobjectneghbors.measureobjectneighboors


# Work in Progress

You can follow progress [here](https://docs.google.com/spreadsheets/d/1_7jQ8EjPwOr2MUnO5Tw56iu4Y0udAzCJEny-LQMgRGE/edit?usp=sharing).


### Done

-   Type 1, 2 and 3 measurements in sklearn style (multiple masks per image)


### Pending

-   Add a wrapper for type 3 measurements
-   Type 4 measurements (ObjectSkeleton). We don't know if it is worth implementing.


### Additional notes

The Image-wide functions will not be implemented directly, they were originally implemented independently to the Object (mask) functions. We will adjust the existing functions assume that an image-wide measurement is the same as measuring an object with the same size as the intensity image.


# Additional notes

-   This is not optimised for efficiency (yet). We aim to reproduce the 'vanilla' results of CellProfiler with minimal code changes. Optimisations will be implemented once we come up with a standard interface for functionally-focused CellProfiler components.
-   The functions exposed perform minimal checks. They will fail if provided with empty masks. Not all functions will fail if provided with masks only.


# Similar projects

-   [spacr](https://github.com/EinarOlafsson/spacr): Library to analyse screens, it provides measurements (independently implemented) and a GUI.
-   [ScaleFEX](https://github.com/NYSCF/ScaleFEx): Python pipeline that includes measurements, designed for the cloud.
