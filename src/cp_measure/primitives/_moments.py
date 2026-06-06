"""Per-object spatial-moment matrices and their derived quantities.

``skimage.measure.regionprops_table`` computes ``moments`` / ``moments_central`` /
``moments_normalized`` / ``moments_hu`` (and the inertia tensor) per region with an
``einsum``-based routine whose contraction path is re-derived for every object — on a 1080² /
142-object tile this dominates ``get_sizeshape``. The moments are plain reductions, so they are
computed in one pass (numpy ``bincount`` scatter here; a fused numba kernel in
``core/numba/_sizeshape.py``) and share the same derivation algebra below.

Raw spatial moments are bit-exact vs regionprops; the centroid-dependent matrices match to
floating-point round-off (~1e-13 relative — moments reach ~1e8 magnitude). Objects are ordered by
ascending label, exactly as ``regionprops_table``.
"""

import numpy
from numpy.typing import NDArray

# Moments up to order 3 (indices 0..3), matching skimage's order-3 regionprops matrices.
_ORDER = 4


def _moment_matrix(
    obj: NDArray[numpy.integer],
    r: NDArray[numpy.floating],
    c: NDArray[numpy.floating],
    n: int,
) -> NDArray[numpy.floating]:
    """Segment-sum ``r**p * c**q`` per object into an ``(n, 4, 4)`` moment matrix."""
    r_pow = [numpy.ones_like(r), r, r * r, r * r * r]
    c_pow = [numpy.ones_like(c), c, c * c, c * c * c]
    moments = numpy.zeros((n, _ORDER, _ORDER))
    for p in range(_ORDER):
        for q in range(_ORDER):
            moments[:, p, q] = numpy.bincount(
                obj, weights=r_pow[p] * c_pow[q], minlength=n
            )
    return moments


def normalized_from_central(
    central: NDArray[numpy.floating],
) -> NDArray[numpy.floating]:
    """Normalized central moments (skimage convention): NaN where ``p + q < 2``."""
    n = central.shape[0]
    normalized = numpy.full((n, _ORDER, _ORDER), numpy.nan)
    mu00 = central[:, 0, 0]
    for p in range(_ORDER):
        for q in range(_ORDER):
            if p + q >= 2:
                normalized[:, p, q] = central[:, p, q] / mu00 ** ((p + q) / 2 + 1)
    return normalized


def hu_from_normalized(nu: NDArray[numpy.floating]) -> NDArray[numpy.floating]:
    """The 7 Hu invariants from normalized central moments (skimage convention)."""
    n20, n02, n11 = nu[:, 2, 0], nu[:, 0, 2], nu[:, 1, 1]
    n30, n03, n21, n12 = nu[:, 3, 0], nu[:, 0, 3], nu[:, 2, 1], nu[:, 1, 2]
    a, b = n30 + n12, n21 + n03  # recurring (rotation-coupled) pairs
    hu = numpy.zeros((nu.shape[0], 7))
    hu[:, 0] = n20 + n02
    hu[:, 1] = (n20 - n02) ** 2 + 4 * n11**2
    hu[:, 2] = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    hu[:, 3] = a**2 + b**2
    hu[:, 4] = (n30 - 3 * n12) * a * (a**2 - 3 * b**2) + (3 * n21 - n03) * b * (
        3 * a**2 - b**2
    )
    hu[:, 5] = (n20 - n02) * (a**2 - b**2) + 4 * n11 * a * b
    hu[:, 6] = (3 * n21 - n03) * a * (a**2 - 3 * b**2) - (n30 - 3 * n12) * b * (
        3 * a**2 - b**2
    )
    return hu


def derive_normalized_hu(
    central: NDArray[numpy.floating],
) -> tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
    """``(normalized, hu)`` from per-object central moments — shared by the numpy and numba
    accumulators, so the derivation algebra has a single source of truth."""
    normalized = normalized_from_central(central)
    return normalized, hu_from_normalized(normalized)


def inertia_2d(
    central: NDArray[numpy.floating],
) -> tuple[
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
]:
    """2D inertia tensor and its eigenvalues from per-object central moments.

    Matches ``skimage.measure.regionprops`` ``inertia_tensor`` / ``inertia_tensor_eigvals``. The
    tensor is ``[[c, -b], [-b, a]]`` with ``a = mu20/mu00``, ``b = mu11/mu00``, ``c = mu02/mu00``;
    eigenvalues descending. Returns ``(t00, t_offdiag, t11, eig_major, eig_minor)``.
    """
    mu00 = central[:, 0, 0]
    a = central[:, 2, 0] / mu00
    b = central[:, 1, 1] / mu00
    c = central[:, 0, 2] / mu00
    half_trace = (a + c) / 2
    disc = numpy.sqrt(((c - a) / 2) ** 2 + b**2)
    return c, -b, a, half_trace + disc, half_trace - disc


def axes_eccentricity_orientation(
    central: NDArray[numpy.floating],
) -> tuple[
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
]:
    """``(axis_major, axis_minor, eccentricity, orientation)`` from per-object central moments.

    Matches ``skimage.measure.regionprops`` to floating-point round-off, including the symmetric
    fallback (``it00 == it11`` -> ±pi/4). Derived from the same inertia tensor / eigenvalues as
    :func:`inertia_2d`, so requesting these no longer forces regionprops' moment einsum.
    """
    it00, it_off, it11, eig_major, eig_minor = inertia_2d(central)
    axis_major = 4 * numpy.sqrt(eig_major)
    axis_minor = 4 * numpy.sqrt(eig_minor)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        eccentricity = numpy.where(
            eig_major == 0, 0.0, numpy.sqrt(1 - eig_minor / eig_major)
        )
    orientation = numpy.where(
        it00 - it11 == 0,
        numpy.where(it_off < 0, numpy.pi / 4, -numpy.pi / 4),
        0.5 * numpy.arctan2(-2 * it_off, it11 - it00),
    )
    return axis_major, axis_minor, eccentricity, orientation


def spatial_moments_2d(
    labels: NDArray[numpy.integer],
) -> tuple[
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
]:
    """numpy scatter accumulator: per-object ``(raw, central, normalized, hu)`` moments (2D)."""
    unique_labels = numpy.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    n = len(unique_labels)
    if n == 0:
        empty = numpy.zeros((0, _ORDER, _ORDER))
        return empty, empty, empty, numpy.zeros((0, 7))

    rows, cols = numpy.nonzero(labels)
    obj = numpy.searchsorted(unique_labels, labels[rows, cols])
    rmin = numpy.full(n, 1 << 31)
    cmin = numpy.full(n, 1 << 31)
    numpy.minimum.at(rmin, obj, rows)
    numpy.minimum.at(cmin, obj, cols)
    local_r = (rows - rmin[obj]).astype(float)
    local_c = (cols - cmin[obj]).astype(float)

    raw = _moment_matrix(obj, local_r, local_c, n)
    centre_r = raw[:, 1, 0] / raw[:, 0, 0]
    centre_c = raw[:, 0, 1] / raw[:, 0, 0]
    central = _moment_matrix(obj, local_r - centre_r[obj], local_c - centre_c[obj], n)

    normalized, hu = derive_normalized_hu(central)
    return raw, central, normalized, hu


def moment_feature_dict(
    raw: NDArray[numpy.floating],
    central: NDArray[numpy.floating],
    normalized: NDArray[numpy.floating],
    hu: NDArray[numpy.floating],
    inertia: tuple[
        NDArray[numpy.floating],
        NDArray[numpy.floating],
        NDArray[numpy.floating],
        NDArray[numpy.floating],
        NDArray[numpy.floating],
        NDArray[numpy.floating],
    ],
) -> dict[str, NDArray[numpy.floating]]:
    """Assemble the ``calculate_advanced`` moment + inertia features of 2D ``get_sizeshape``.

    Single source of truth for these 53 feature names and the ``(p, q)`` orders exposed, shared by
    the numpy and numba sizeshape backends (the key strings match the ``F_*`` constants in
    ``core.measureobjectsizeshape``; the sizeshape golden test cross-checks the two). ``raw`` and
    ``central`` are ``(n, 4, 4)`` with only ``p in {0, 1, 2}`` exposed; ``normalized`` is the full
    ``(n, 4, 4)``; ``hu`` is ``(n, 7)``. ``inertia`` is the tuple
    ``(it_0_0, it_0_1, it_1_0, it_1_1, eigenvalue_0, eigenvalue_1)``.
    """
    it_0_0, it_0_1, it_1_0, it_1_1, eig_0, eig_1 = inertia
    features: dict[str, NDArray[numpy.floating]] = {}
    for p in range(3):  # spatial / central expose p in {0,1,2}, q in {0,1,2,3}
        for q in range(_ORDER):
            features[f"SpatialMoment_{p}_{q}"] = raw[:, p, q]
            features[f"CentralMoment_{p}_{q}"] = central[:, p, q]
    for p in range(_ORDER):  # normalized full 4x4
        for q in range(_ORDER):
            features[f"NormalizedMoment_{p}_{q}"] = normalized[:, p, q]
    for k in range(7):
        features[f"HuMoment_{k}"] = hu[:, k]
    features["InertiaTensor_0_0"] = it_0_0
    features["InertiaTensor_0_1"] = it_0_1
    features["InertiaTensor_1_0"] = it_1_0
    features["InertiaTensor_1_1"] = it_1_1
    features["InertiaTensorEigenvalues_0"] = eig_0
    features["InertiaTensorEigenvalues_1"] = eig_1
    return features
