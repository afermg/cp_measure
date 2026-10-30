"""Numba-accelerated MeasureObjectIntensityDistribution backend.

``get_radial_zernikes`` mirrors :func:`cp_measure.core.measureobjectintensitydistribution.get_radial_zernikes`
but replaces its per-moment Horner array operations and ``bincount`` reductions
with a single fused Numba kernel
(:func:`cp_measure.core.numba._zernike.zernike_moments`). Centrosome still provides
the radial LUT (degree-only) and ``minimum_enclosing_circle`` (host). Batch
handling is :func:`cp_measure.core.numba._zernike.zernike_per_image`.

``get_radial_distribution`` processes each object on its own cropped + 1px-padded
sub-image, so the dominant geometry runs on small arrays, and replaces centrosome's
``propagate`` (the 80% cost) with a bit-exact numba chamfer geodesic plus the
centre and histogram/wedge-CV reductions, all fused into one per-crop kernel
(:func:`cp_measure.core.numba._radial.radial_object`). Only the exact-Euclidean
``distance_transform_edt`` stays host-side (scipy).
Results match the NumPy reference, which is label-independent since #96 (Issue #22).
"""

import numpy
import scipy.ndimage
from numpy.typing import NDArray

from cp_measure.core.measureobjectintensitydistribution import (
    MF_FRAC_AT_D,
    MF_MEAN_FRAC,
    MF_RADIAL_CV,
    OF_FRAC_AT_D,
    OF_MEAN_FRAC,
    OF_RADIAL_CV,
    _radial_zernike_features,
)
from cp_measure.core.numba._radial import (
    radial_object,
)
from cp_measure.core.numba._zernike import (
    zernike_per_image,
)
from cp_measure.primitives.shapes import to_bzyx


def get_radial_zernikes(labels, pixels, zernike_degree: int = 9):
    """Radial Zernike magnitude/phase per object; single image/volume or batch.

    Measured 2D entries require matching label and intensity shapes; mismatches
    raise ``ValueError`` rather than selecting an intensity plane. Compare phases
    modulo 2π; near-zero moments have numerically unstable phases.
    """
    return zernike_per_image(
        labels,
        pixels,
        zernike_degree,
        lambda vr, vi, _radii, counts, indexes: _radial_zernike_features(
            vr, vi, counts, indexes
        ),
    )


def get_radial_distribution(
    labels,
    pixels,
    scaled: bool = True,
    bin_count: int = 4,
    maximum_radius: int = 100,
):
    """Radial intensity distribution (FracAtD / MeanFrac / RadialCV) per object.

    Matches the NumPy reference; each object is processed on its own crop. Single
    image/volume or batch; 2D only.
    """
    # Volumes give {} as in NumPy, including Z=1, as for the numba Zernikes.
    if not isinstance(labels, (list, tuple)) and numpy.ndim(labels) == 3:
        return {}
    masks_zyx, pixels_zyx, unwrap = to_bzyx(labels, pixels)
    results = [
        {}
        if isinstance(labels, (list, tuple)) and numpy.ndim(labels[i]) == 3
        else _radial_distribution_2d(m, p, scaled, bin_count, maximum_radius)
        for i, (m, p) in enumerate(zip(masks_zyx, pixels_zyx))
    ]
    return unwrap(results)


# (per-bin name template, overflow-bin name) per feature, in result order. The
# triple index (0,1,2) selects the (frac_at_d, mean_frac, radial_cv) array.
_RADIAL_FEATURES = (
    (MF_FRAC_AT_D, OF_FRAC_AT_D),
    (MF_MEAN_FRAC, OF_MEAN_FRAC),
    (MF_RADIAL_CV, OF_RADIAL_CV),
)


def _radial_features(scaled, bin_count):
    """Yield ``(col, name, bin)`` for each output feature, in order — ``col`` picks
    the ``(frac_at_d, mean_frac, radial_cv)`` array, ``bin`` its column."""
    for b in range(bin_count + (0 if scaled else 1)):
        for col, (mf, of) in enumerate(_RADIAL_FEATURES):
            yield col, (of if b == bin_count else mf % (b + 1, bin_count)), b


def _radial_distribution_2d(
    labels_zyx: NDArray[numpy.integer],
    pixels_zyx: NDArray[numpy.floating],
    scaled: bool,
    bin_count: int,
    maximum_radius: int,
) -> dict[str, NDArray[numpy.floating]]:
    # The kernel has no bounds checks, so a mismatched image would be read out of
    # bounds; NumPy raises on it too.
    if labels_zyx.shape != pixels_zyx.shape:
        raise ValueError(
            f"labels {labels_zyx.shape} and pixels {pixels_zyx.shape} differ in shape"
        )
    if labels_zyx.shape[0] > 1:  # Z > 1 -> 3D volume; baseline returns {} for ndim==3
        return {}
    labels = labels_zyx[0]
    pixels = pixels_zyx[0]
    # cp_measure labels are the contiguous 1..n of relabel_sequential, so the object
    # count is max() and the segment index is label-1 (matching the numpy baseline,
    # which indexes its (nobjects, ...) arrays by label-1).
    n = int(labels.max()) if labels.size else 0
    if n == 0:  # no objects (fringe) -> empty array per key
        return {
            name: numpy.zeros(0) for _, name, _ in _radial_features(scaled, bin_count)
        }

    # Fixed scalar types keep the cached kernel to one compiled signature; the
    # radius is unused (and may be None, as NumPy allows) when scaled.
    scaled, bin_count = bool(scaled), int(bin_count)
    radius = 1.0 if scaled else float(maximum_radius)
    slices = scipy.ndimage.find_objects(labels)
    nb = bin_count + 1
    frac_at_d = numpy.zeros((n, nb))
    mean_frac = numpy.zeros((n, nb))
    wedge_sums = numpy.zeros((n, nb, 8))
    wedge_counts = numpy.zeros((n, nb, 8))
    for label in range(1, n + 1):
        sl = slices[label - 1]
        if sl is None:
            continue
        # Crop to the object's bbox grown by 1px and clipped to the image: the
        # margin gives the EDT its background edge, and the clip keeps the image
        # border from counting as one, as in the whole-image reference.
        sl = tuple(
            slice(max(s.start - 1, 0), min(s.stop + 1, dim))
            for s, dim in zip(sl, labels.shape)
        )
        m = numpy.ascontiguousarray(labels[sl] == label)
        pix = numpy.ascontiguousarray(pixels[sl], dtype=numpy.float64)
        d_to_edge = scipy.ndimage.distance_transform_edt(m)
        (
            frac_at_d[label - 1],
            mean_frac[label - 1],
            wedge_sums[label - 1],
            wedge_counts[label - 1],
        ) = radial_object(m, pix, d_to_edge, scaled, bin_count, radius)

    arrays = (frac_at_d, mean_frac, _radial_cv(wedge_sums, wedge_counts))
    return {
        name: arrays[col][:, b] for col, name, b in _radial_features(scaled, bin_count)
    }


def _radial_cv(wedge_sums, wedge_counts):
    """RadialCV per (object, bin) with the reference's own ``numpy.ma`` expression.

    ``numpy.ma`` division masks zero or non-finite denominators and keeps the
    numerator, so all-dark rings give 0 rather than NaN; evaluating the same
    expression (vectorised over objects) keeps those cases identical.
    """
    radial_cv = numpy.zeros(wedge_sums.shape[:2])
    with numpy.errstate(invalid="ignore", divide="ignore"):
        for b in range(wedge_sums.shape[1]):
            mask = wedge_counts[:, b] == 0
            means = numpy.ma.MaskedArray(wedge_sums[:, b] / wedge_counts[:, b], mask)
            cv = numpy.std(means, 1) / numpy.mean(means, 1)
            cv[numpy.sum(~mask, 1) == 0] = 0
            radial_cv[:, b] = numpy.array(cv)
    return radial_cv
