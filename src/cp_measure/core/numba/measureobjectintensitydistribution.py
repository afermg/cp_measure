"""Numba-accelerated MeasureObjectIntensityDistribution backend.

``get_radial_zernikes`` mirrors the reference but replaces the per-(n,m)
``scipy.ndimage.sum_labels`` AND centrosome's per-pixel
``construct_zernike_polynomials`` with a single fused numba kernel
(:func:`cp_measure.core.numba._zernike.zernike_moments`).

``get_radial_distribution`` processes each object on its own cropped + 1px-padded
sub-image — which **fixes Issue #22** (per-object results no longer depend on other
labels in the field) and lets the dominant geometry run on small arrays — and
replaces centrosome's ``propagate`` (the 80% cost) with a bit-exact numba chamfer
geodesic plus the centre and histogram/wedge-CV reductions, all fused into one
per-crop kernel (:func:`cp_measure.core.numba._radial.radial_object`). Only the
exact-Euclidean ``distance_transform_edt`` stays host-side (scipy).
NOTE: because of the #22 fix, this diverges from the current (buggy) numpy baseline
on multi-object fields; it equals the baseline run on each object in ISOLATION.

Both are batch-shaped via the canonical ``(B, Z, Y, X)`` form (single image =
``B == 1``); 2D-only (a ``Z > 1`` volume returns ``{}``, matching ``ndim == 3``).
"""

import centrosome.zernike
import numpy
import scipy.ndimage
from numpy.typing import NDArray

from cp_measure.core.measureobjectintensitydistribution import (
    M_CATEGORY,
    MF_FRAC_AT_D,
    MF_MEAN_FRAC,
    MF_RADIAL_CV,
    OF_FRAC_AT_D,
    OF_MEAN_FRAC,
    OF_RADIAL_CV,
)
from cp_measure.core.numba._radial import radial_object
from cp_measure.core.numba._zernike import zernike_coeffs, zernike_moments_per_object
from cp_measure.primitives.shapes import to_bzyx


def get_radial_zernikes(masks, pixels, zernike_degree: int = 9):
    """Radial Zernike magnitude/phase per object; single image/volume or batch."""
    masks_zyx, pixels_zyx, unwrap = to_bzyx(masks, pixels)
    # degree-only work, hoisted out of the per-image loop (computed once per batch)
    zernike_indexes = centrosome.zernike.get_zernike_indexes(zernike_degree + 1)
    coeffs = zernike_coeffs(zernike_indexes)
    results = [
        _radial_zernikes_2d(m, p, zernike_indexes, coeffs)
        for m, p in zip(masks_zyx, pixels_zyx)
    ]
    return unwrap(results)


def _radial_zernikes_2d(
    labels_zyx: NDArray[numpy.integer],
    pixels_zyx: NDArray[numpy.floating],
    zernike_indexes: NDArray[numpy.integer],
    coeffs: tuple,
) -> dict[str, NDArray[numpy.floating]]:
    if labels_zyx.shape[0] > 1:  # Z > 1 -> 3D volume; baseline returns {} for ndim==3
        return {}

    vr, vi, _radii, seg0 = zernike_moments_per_object(
        labels_zyx[0], pixels_zyx[0], coeffs
    )
    if (
        vr.shape[0] == 0
    ):  # no objects (fringe) -> empty arrays per key, matches baseline
        return {
            f"{M_CATEGORY}_Zernike{mag_or_phase}_{nn}_{mm}": numpy.zeros(0)
            for mag_or_phase in ("Magnitude", "Phase")
            for nn, mm in zernike_indexes
        }

    areas = numpy.bincount(seg0, minlength=vr.shape[0]).astype(
        numpy.float64
    )  # px/object
    results = {}
    for i, (nn, mm) in enumerate(zernike_indexes):
        results[f"{M_CATEGORY}_ZernikeMagnitude_{nn}_{mm}"] = (
            numpy.sqrt(vr[:, i] ** 2 + vi[:, i] ** 2) / areas
        )
        # baseline's (real, imag) arctan2 arg order
        results[f"{M_CATEGORY}_ZernikePhase_{nn}_{mm}"] = numpy.arctan2(
            vr[:, i], vi[:, i]
        )
    return results


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

    slices = scipy.ndimage.find_objects(labels)
    nb = bin_count + 1
    frac_at_d = numpy.zeros((n, nb))
    mean_frac = numpy.zeros((n, nb))
    radial_cv = numpy.zeros((n, nb))
    for label in range(1, n + 1):
        sl = slices[label - 1]
        if sl is None:
            continue
        # Crop to the object's bbox + a 1px background border, so the imported
        # scipy EDT (exact Euclidean, kept host-side) and the in-kernel chamfer
        # geodesic are bit-identical to the object computed in isolation (the
        # Issue #22 semantics). The fused radial_object kernel then does the centre,
        # geodesic, histograms and wedge-CV with no per-object host numpy.
        # numpy.pad returns C-contiguous arrays, so the kernel needs no further copy.
        m = numpy.pad(labels[sl] == label, 1)
        pix = numpy.pad(pixels[sl].astype(numpy.float64), 1)
        d_to_edge = scipy.ndimage.distance_transform_edt(m)
        fad, mfr, cv = radial_object(
            m, pix, d_to_edge, scaled, bin_count, maximum_radius
        )
        frac_at_d[label - 1] = fad
        mean_frac[label - 1] = mfr
        radial_cv[label - 1] = cv

    arrays = (frac_at_d, mean_frac, radial_cv)
    return {
        name: arrays[col][:, b] for col, name, b in _radial_features(scaled, bin_count)
    }
