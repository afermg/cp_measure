"""Numba-accelerated MeasureObjectIntensityDistribution backend.

``get_radial_zernikes`` mirrors :func:`cp_measure.core.measureobjectintensitydistribution.get_radial_zernikes`
but replaces its per-moment Horner array operations and ``bincount`` reductions
with a single fused Numba kernel
(:func:`cp_measure.core.numba._zernike.zernike_moments`). Centrosome still provides
the radial LUT (degree-only) and ``minimum_enclosing_circle`` (host). Batch
handling is :func:`cp_measure.core.numba._zernike.zernike_per_image`.

``get_radial_distribution`` processes each object on its bounding-box crop,
grown by one pixel and clipped to the image. It replaces centrosome's
``propagate`` with a numba chamfer geodesic plus centre and histogram/wedge
reductions, fused into one per-crop kernel
(:func:`cp_measure.core.numba._radial.radial_object`). The exact Euclidean
distance transform is the bit-identical numba port in
:mod:`cp_measure.primitives._edt_numba`; the masked ``RadialCV`` stays host-side.
Results match the NumPy reference, which is label-independent since #96 (Issue #22).
"""

import operator

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
    get_radial_distribution as _numpy_radial_distribution,
)
from cp_measure.core.numba._radial import (
    radial_all,
    radial_object,
)
from cp_measure.core.numba._zernike import (
    zernike_per_image,
)
from cp_measure.primitives._edt_numba import distance_transform_edt
from cp_measure.primitives.parallel import map_images
from cp_measure.primitives.shapes import each_volume, memory_key, native


def get_radial_zernikes(labels, pixels, zernike_degree: int = 9, *, max_workers=None):
    """Radial Zernike magnitude/phase per object; single image/volume or batch.

    Measured 2D entries require matching label and intensity shapes; mismatches
    raise ``ValueError`` rather than selecting an intensity plane. Compare phases
    modulo 2π; near-zero moments have numerically unstable phases. ``max_workers``
    caps the batch thread pool (default: the CPUs this process may use); pass it when nesting
    under an outer parallel map.
    """
    return zernike_per_image(
        labels,
        pixels,
        zernike_degree,
        lambda vr, vi, _radii, counts, indexes: _radial_zernike_features(
            vr, vi, counts, indexes
        ),
        max_workers=max_workers,
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
    image/volume or batch; 2D only. Measured entries require matching label and
    intensity shapes and spatial rank, before axis promotion. Singleton-Z
    volumes use the 2D measurements; other volumes return {}, including in batches.
    """
    is_sequence = isinstance(labels, (list, tuple))
    if not is_sequence and numpy.ndim(labels) == 3 and labels.shape[0] != 1:
        return {}
    # Views, no batch copy: images that share a mask keep sharing its memory.
    # each_volume also checks that labels and pixels match in shape and rank.
    volumes, unwrap = each_volume(labels, pixels)
    # Images that share a mask (the featurizer passes one mask per channel) are
    # measured together, so each object's crop and distance transform are built once.
    groups: dict = {}
    for i, (m, _, _) in enumerate(volumes):
        if m.shape[0] == 1:
            groups.setdefault(memory_key(m), []).append(i)

    def one(items):
        return _radial_measure(
            volumes[items[0]][0],
            [volumes[i][1] for i in items],
            scaled,
            bin_count,
            maximum_radius,
        )

    # Groups are independent and the per-object kernel releases the GIL, so
    # map_images runs a batch on threads.
    measured: list = [{} for _ in volumes]
    grouped = list(groups.values())
    for items, outs in zip(
        grouped,
        map_images(
            one,
            grouped,
            pixels=sum(volumes[i][0].size for items in grouped for i in items),
        ),
    ):
        for i, out in zip(items, outs):
            measured[i] = out
    return unwrap(_with_radial_cv(measured, scaled, bin_count))


def _with_radial_cv(measured, scaled, bin_count):
    """Name each image's arrays, adding RadialCV.

    RadialCV's ``numpy.ma`` expression holds the GIL: run per image on the pool, the
    threads only queue for it (on a 20-image batch, 8 threads were 3x slower than
    one). It runs once over all images' objects here instead, on the calling
    thread. Each row is reduced on its own, so the values are bit-identical.
    """
    parts = [m for m in measured if isinstance(m, tuple)]
    if parts:
        cv = _radial_cv(
            numpy.concatenate([p[2] for p in parts]),
            numpy.concatenate([p[3] for p in parts]),
        )
        cvs = iter(numpy.split(cv, numpy.cumsum([len(p[2]) for p in parts])[:-1]))
    results = []
    for m in measured:
        if isinstance(m, tuple):
            arrays = (m[0], m[1], next(cvs))
            m = {
                name: arrays[col][:, b]
                for col, name, b in _radial_features(scaled, bin_count)
            }
        results.append(m)
    return results


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


def _radial_measure(
    labels_zyx: NDArray[numpy.integer],
    channels: list[NDArray[numpy.floating]],
    scaled: bool,
    bin_count: int,
    maximum_radius: int,
) -> list:
    """Per channel image measured with ``labels_zyx``: its ``(frac_at_d, mean_frac,
    wedge_sums, wedge_counts)``, or its finished result dict when no kernel ran (no
    objects, or the NumPy fallback)."""
    # Validate before the empty-image path and allocation. Unlike int(), index()
    # rejects fractional bin counts instead of silently changing the requested bins.
    bin_count = operator.index(bin_count)
    if bin_count < 0:
        raise ValueError("bin_count must be >= 0")
    labels = labels_zyx[0]
    # cp_measure labels are the contiguous 1..n of relabel_sequential, so the object
    # count is max() and the segment index is label-1 (matching the numpy baseline,
    # which indexes its (nobjects, ...) arrays by label-1).
    n = int(labels.max()) if labels.size else 0
    if n == 0:  # no objects (fringe) -> empty array per key
        return [
            {name: numpy.zeros(0) for _, name, _ in _radial_features(scaled, bin_count)}
            for _ in channels
        ]

    # Fixed scalar types keep the cached kernel to one compiled signature; the
    # radius is unused (and may be None, as NumPy allows) when scaled.
    scaled, bin_count = bool(scaled), int(bin_count)
    radius = 1.0 if scaled else float(maximum_radius)
    if not scaled and (radius == 0 or numpy.isnan(radius)):
        # Non-finite-to-int bin casts in NumPy are platform/version-dependent.
        # Preserve its result or exception without exposing unchecked kernel indices.
        return [
            _numpy_radial_distribution(
                labels,
                p[0],
                scaled=scaled,
                bin_count=bin_count,
                maximum_radius=maximum_radius,
            )
            for p in channels
        ]
    # (C, H, W); one image's (1, H, W) view needs no copy. Numba has no float16;
    # widening is exact (the kernel reads float64).
    stack = channels[0] if len(channels) == 1 else numpy.stack([p[0] for p in channels])
    stack = native(stack)
    if stack.dtype == numpy.float16:
        stack = stack.astype(numpy.float64)
    frac_at_d, mean_frac, wedge_sums, wedge_counts, fallback = radial_all(
        numpy.ascontiguousarray(native(labels)), stack, n, scaled, bin_count, radius
    )
    # A crop without background (an object filling its image-clipped box) gets
    # scipy's distance transform, whose result for that case differs.
    if fallback.any():
        slices = scipy.ndimage.find_objects(labels)
        for k in numpy.flatnonzero(fallback):
            sl = tuple(
                slice(max(s.start - 1, 0), min(s.stop + 1, dim))
                for s, dim in zip(slices[k], labels.shape)
            )
            m = numpy.ascontiguousarray(labels[sl] == k + 1)
            d_to_edge = distance_transform_edt(m)
            for c, p in enumerate(channels):
                pix = numpy.ascontiguousarray(p[0][sl], dtype=numpy.float64)
                (
                    frac_at_d[c, k],
                    mean_frac[c, k],
                    wedge_sums[c, k],
                    wedge_counts[c, k],
                ) = radial_object(m, pix, d_to_edge, scaled, bin_count, radius)

    return [
        (frac_at_d[c], mean_frac[c], wedge_sums[c], wedge_counts[c])
        for c in range(len(channels))
    ]


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
