"""Per-object spatial-moment matrices via a single label-scatter pass.

``skimage.measure.regionprops_table`` computes ``moments`` / ``moments_central`` /
``moments_normalized`` / ``moments_hu`` per region with an ``einsum``-based routine whose
contraction path is re-derived for every object — on a 1080² / 142-object tile this dominates
``get_sizeshape`` (~120 ms, ~40 ms of which is pure ``einsum_path`` overhead). The moments are
plain reductions, so the whole set is one scatter over the foreground pixels.

The raw spatial moments match ``regionprops`` to floating-point round-off — NOT bit-exact: the
``bincount`` summation order differs from skimage's ``einsum``, so the divergence grows with object
size (worst seen ~3e-12 relative on a 1000² object). The centroid-dependent matrices
(``central`` / ``normalized`` / ``hu``) likewise match to round-off. Objects are ordered by
ascending label, exactly as ``regionprops_table``.
"""

import numpy
from numpy.typing import NDArray

from cp_measure.primitives.segment import label_to_idx_lut

# Moments up to order 3 (indices 0..3), matching skimage's order-3 regionprops matrices.
_ORDER = 4


def _moment_matrix(
    obj: NDArray[numpy.integer],
    r: NDArray[numpy.floating],
    c: NDArray[numpy.floating],
    n: int,
) -> NDArray[numpy.floating]:
    """Segment-sum ``r**p * c**q`` per object into an ``(n, 4, 4)`` moment matrix."""
    r_pow = [r**k for k in range(_ORDER)]
    c_pow = [c**k for k in range(_ORDER)]
    moments = numpy.zeros((n, _ORDER, _ORDER))
    for p in range(_ORDER):
        for q in range(_ORDER):
            moments[:, p, q] = numpy.bincount(
                obj, weights=r_pow[p] * c_pow[q], minlength=n
            )
    return moments


def _hu_from_normalized(nu: NDArray[numpy.floating]) -> NDArray[numpy.floating]:
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


def spatial_moments_2d(
    labels: NDArray[numpy.integer],
    *,
    advanced: bool = True,
) -> tuple[
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating] | None,
    NDArray[numpy.floating] | None,
]:
    """Per-object ``(raw, central, normalized, hu)`` spatial moments for a 2D label image.

    ``raw`` / ``central`` / ``normalized`` are ``(n, 4, 4)`` and ``hu`` is ``(n, 7)``, ordered by
    ascending label. Drop-in for the matching ``regionprops_table`` columns
    (``moments-p-q`` etc.); ``normalized`` is NaN where ``p + q < 2`` (skimage convention).

    ``advanced=False`` returns ``(raw, central, None, None)`` — the centroid-axes path only needs
    ``central``, so the normalized / Hu moments are skipped when the caller won't emit them.
    """
    lut, n, origins = label_to_idx_lut(labels, return_bbox=True)
    if n == 0:
        empty = numpy.zeros((0, _ORDER, _ORDER))
        if advanced:
            return empty, empty, empty, numpy.zeros((0, 7))
        return empty, empty, None, None

    rows, cols = numpy.nonzero(labels)
    obj = lut[labels[rows, cols]]

    # skimage takes moments in each object's local (bounding-box) frame; the per-object bbox
    # origin comes straight from label_to_idx_lut's find_objects pass (no minimum.at).
    local_r = (rows - origins[obj, 0]).astype(float)
    local_c = (cols - origins[obj, 1]).astype(float)

    raw = _moment_matrix(obj, local_r, local_c, n)
    centre_r = raw[:, 1, 0] / raw[:, 0, 0]
    centre_c = raw[:, 0, 1] / raw[:, 0, 0]
    # Central moments by direct centred summation (binomial-from-raw loses ~1e-4 to cancellation).
    central = _moment_matrix(obj, local_r - centre_r[obj], local_c - centre_c[obj], n)
    if not advanced:
        return raw, central, None, None

    normalized = numpy.full((n, _ORDER, _ORDER), numpy.nan)
    mu00 = central[:, 0, 0]
    for p in range(_ORDER):
        for q in range(_ORDER):
            if p + q >= 2:
                normalized[:, p, q] = central[:, p, q] / mu00 ** ((p + q) / 2 + 1)

    return raw, central, normalized, _hu_from_normalized(normalized)


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

    Matches ``skimage.measure.regionprops`` ``inertia_tensor`` / ``inertia_tensor_eigvals`` to
    floating-point round-off. The tensor is ``[[c, -b], [-b, a]]`` with ``a = mu20/mu00``,
    ``b = mu11/mu00``, ``c = mu02/mu00`` (skimage's row/col convention); eigenvalues are returned
    in descending order. Reuses the central moments from :func:`spatial_moments_2d`, so the
    inertia features need no separate regionprops einsum.

    Returns ``(t00, t_offdiag, t11, eig_major, eig_minor)`` — ``t_offdiag`` is both
    off-diagonal entries (the tensor is symmetric).
    """
    mu00 = central[:, 0, 0]
    a = central[:, 2, 0] / mu00
    b = central[:, 1, 1] / mu00
    c = central[:, 0, 2] / mu00
    half_trace = (a + c) / 2
    disc = numpy.sqrt(((c - a) / 2) ** 2 + b**2)
    # Clip eigenvalues to >= 0 (skimage does the same): tiny-negative float error on degenerate /
    # thin objects would otherwise give NaN axis lengths and eccentricity > 1.
    eig_major = numpy.clip(half_trace + disc, 0.0, None)
    eig_minor = numpy.clip(half_trace - disc, 0.0, None)
    return c, -b, a, eig_major, eig_minor


def axes_eccentricity_orientation(
    inertia: tuple[
        NDArray[numpy.floating],
        NDArray[numpy.floating],
        NDArray[numpy.floating],
        NDArray[numpy.floating],
        NDArray[numpy.floating],
    ],
) -> tuple[
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
]:
    """``(axis_major, axis_minor, eccentricity, orientation)`` from the per-object inertia tuple.

    Takes the output of :func:`inertia_2d` (so the eigendecomposition is computed once and shared
    with the inertia features) and matches ``skimage.measure.regionprops`` to floating-point
    round-off, including the symmetric fallback (``it00 == it11`` -> ±pi/4). CellProfiler reports
    orientation in degrees; callers apply the ``180/pi`` conversion.
    """
    it00, it_off, it11, eig_major, eig_minor = inertia
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

    Single source of truth for these 53 feature names and the ``(p, q)`` orders. Keys are emitted
    in the *grouped* order of the CellProfiler / PyPI release (all Spatial, then Central, then
    Normalized, then Hu, then the inertia tensor) so the output column order is unchanged. ``raw``
    and ``central`` are ``(n, 4, 4)`` with only ``p in {0, 1, 2}`` exposed; ``normalized`` is the
    full ``(n, 4, 4)``; ``hu`` is ``(n, 7)``. ``inertia`` is
    ``(it_0_0, it_0_1, it_1_0, it_1_1, eigenvalue_0, eigenvalue_1)`` (both off-diagonals passed
    explicitly — they are equal for the symmetric tensor).
    """
    it_0_0, it_0_1, it_1_0, it_1_1, eig_0, eig_1 = inertia
    features: dict[str, NDArray[numpy.floating]] = {}
    for p in range(3):  # spatial / central expose p in {0,1,2}, q in {0,1,2,3}
        for q in range(_ORDER):
            features[f"SpatialMoment_{p}_{q}"] = raw[:, p, q]
    for p in range(3):
        for q in range(_ORDER):
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
