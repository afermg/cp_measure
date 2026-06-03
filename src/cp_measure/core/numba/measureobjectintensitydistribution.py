"""Numba-accelerated MeasureObjectIntensityDistribution backend.

``get_radial_zernikes`` mirrors :func:`cp_measure.core.measureobjectintensitydistribution.get_radial_zernikes`
but replaces its per-moment Horner array operations and ``bincount`` reductions
with a single fused Numba kernel
(:func:`cp_measure.core.numba._zernike.zernike_moments`). Centrosome still provides
the radial LUT (degree-only) and ``minimum_enclosing_circle`` (host). Batch
handling is :func:`cp_measure.core.numba._zernike.zernike_per_image`.

``get_radial_distribution`` processes each object on its own cropped + 1px-padded
sub-image — which **fixes Issue #22** (per-object results no longer depend on other
labels in the field) and lets the dominant geometry run on small arrays — and
replaces centrosome's ``propagate`` (the 80% cost) with a bit-exact numba chamfer
geodesic (:func:`cp_measure.core.numba._radial.geodesic_chamfer_fifo`) plus numba
histogram/wedge-CV reductions. Exact-Euclidean ``distance_transform_edt`` and the
centre (``maximum_position_of_labels``) stay host-side (scipy/centrosome).
NOTE: because of the #22 fix, this diverges from the current (buggy) numpy baseline
on multi-object fields; it equals the baseline run on each object in ISOLATION.
"""

import centrosome.cpmorphology
import centrosome.zernike
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
    UNREACHED,
    geodesic_chamfer_fifo,
    radial_reduce,
)
from cp_measure.core.numba._zernike import (
    zernike_per_image,
)
from cp_measure.primitives.shapes import to_bzyx


def get_radial_zernikes(labels, pixels, zernike_degree: int = 9):
    """Radial Zernike magnitude/phase per object; single image/volume or batch."""
    return zernike_per_image(
        labels,
        pixels,
        zernike_degree,
        lambda vr, vi, _radii, counts, indexes: _radial_zernike_features(
            vr, vi, counts, indexes
        ),
    )


def get_radial_distribution(
    masks,
    pixels,
    scaled: bool = True,
    bin_count: int = 4,
    maximum_radius: int = 100,
):
    """Radial intensity distribution (FracAtD / MeanFrac / RadialCV) per object.

    Per-object cropped processing fixes Issue #22 (results independent of other
    labels) and equals the numpy baseline run on each object in ISOLATION. Single
    image/volume or batch; 2D only.
    """
    masks_zyx, pixels_zyx, unwrap = to_bzyx(masks, pixels)
    results = [
        _radial_distribution_2d(m, p, scaled, bin_count, maximum_radius)
        for m, p in zip(masks_zyx, pixels_zyx)
    ]
    return unwrap(results)


def _radial_keys(scaled, bin_count):
    """The ordered output feature names for the given (scaled, bin_count)."""
    names = []
    for b in range(bin_count + (0 if scaled else 1)):
        for mf, of in (
            (MF_FRAC_AT_D, OF_FRAC_AT_D),
            (MF_MEAN_FRAC, OF_MEAN_FRAC),
            (MF_RADIAL_CV, OF_RADIAL_CV),
        ):
            names.append(of if b == bin_count else mf % (b + 1, bin_count))
    return names


def _radial_distribution_2d(
    labels_zyx: NDArray[numpy.integer],
    pixels_zyx: NDArray[numpy.floating],
    scaled: bool,
    bin_count: int,
    maximum_radius: int,
) -> dict[str, NDArray[numpy.floating]]:
    if labels_zyx.shape[0] > 1:  # Z > 1 -> 3D volume; baseline returns {} for ndim==3
        return {}
    labels = labels_zyx[0]
    pixels = pixels_zyx[0]
    n = int(labels.max()) if labels.size else 0
    if n == 0:  # no objects (fringe) -> empty array per key
        return {k: numpy.zeros(0) for k in _radial_keys(scaled, bin_count)}

    slices = scipy.ndimage.find_objects(labels)
    vals, segs, bins, wedges = [], [], [], []
    for label in range(1, n + 1):
        sl = slices[label - 1]
        if sl is None:
            continue
        # Crop to the object's bbox + a 1px background border, so the imported
        # scipy EDT and the chamfer geodesic on the crop are bit-identical to the
        # object computed in isolation (the Issue #22 semantics).
        m = numpy.pad(labels[sl] == label, 1)
        pix = numpy.pad(pixels[sl].astype(numpy.float64), 1)
        d_to_edge = scipy.ndimage.distance_transform_edt(m)
        ci_a, cj_a = centrosome.cpmorphology.maximum_position_of_labels(
            d_to_edge, m.astype(numpy.int32), indices=numpy.array([1])
        )
        ci, cj = int(ci_a[0]), int(cj_a[0])
        d_from = geodesic_chamfer_fifo(numpy.ascontiguousarray(m), ci, cj)
        good = m & (d_from < UNREACHED)

        nd = numpy.zeros(m.shape)
        if scaled:
            nd[good] = d_from[good] / (d_from[good] + d_to_edge[good] + 0.001)
        else:
            nd[good] = d_from[good] / maximum_radius
        bin_idx = (nd * bin_count).astype(int)
        bin_idx[bin_idx > bin_count] = bin_count

        ii, jj = numpy.mgrid[0 : m.shape[0], 0 : m.shape[1]]
        wedge = (
            (ii > ci).astype(int)
            + (jj > cj).astype(int) * 2
            + (numpy.abs(ii - ci) > numpy.abs(jj - cj)).astype(int) * 4
        )
        gy, gx = numpy.where(good)
        vals.append(pix[gy, gx])
        segs.append(numpy.full(gy.size, label - 1, numpy.int64))
        bins.append(bin_idx[gy, gx].astype(numpy.int64))
        wedges.append(wedge[gy, gx].astype(numpy.int64))

    values = numpy.ascontiguousarray(numpy.concatenate(vals), numpy.float64)
    frac_at_d, mean_frac, radial_cv = radial_reduce(
        values,
        numpy.concatenate(segs),
        numpy.concatenate(bins),
        numpy.concatenate(wedges),
        n,
        bin_count,
    )
    cols = {
        MF_FRAC_AT_D: frac_at_d,
        MF_MEAN_FRAC: mean_frac,
        MF_RADIAL_CV: radial_cv,
        OF_FRAC_AT_D: frac_at_d,
        OF_MEAN_FRAC: mean_frac,
        OF_RADIAL_CV: radial_cv,
    }
    results = {}
    for b in range(bin_count + (0 if scaled else 1)):
        for mf, of in (
            (MF_FRAC_AT_D, OF_FRAC_AT_D),
            (MF_MEAN_FRAC, OF_MEAN_FRAC),
            (MF_RADIAL_CV, OF_RADIAL_CV),
        ):
            name = of if b == bin_count else mf % (b + 1, bin_count)
            results[name] = cols[of][:, b]
    return results
