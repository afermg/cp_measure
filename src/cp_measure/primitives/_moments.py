"""Per-object spatial-moment matrices via a single label-scatter pass.

``skimage.measure.regionprops_table`` computes ``moments`` / ``moments_central`` /
``moments_normalized`` / ``moments_hu`` per region with an ``einsum``-based routine whose
contraction path is re-derived for every object — on a 1080² / 142-object tile this dominates
``get_sizeshape`` (~120 ms, ~40 ms of which is pure ``einsum_path`` overhead). The moments are
plain reductions, so the whole set is one scatter over the foreground pixels.

The raw spatial moments are bit-exact vs regionprops; the centroid-dependent matrices
(``central`` / ``normalized`` / ``hu``) match to floating-point round-off (~1e-13 relative — the
moments reach ~1e8 magnitude). Objects are ordered by ascending label, exactly as
``regionprops_table``.
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
) -> tuple[
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
]:
    """Per-object ``(raw, central, normalized, hu)`` spatial moments for a 2D label image.

    ``raw`` / ``central`` / ``normalized`` are ``(n, 4, 4)`` and ``hu`` is ``(n, 7)``, ordered by
    ascending label. Drop-in for the matching ``regionprops_table`` columns
    (``moments-p-q`` etc.); ``normalized`` is NaN where ``p + q < 2`` (skimage convention).
    """
    unique_labels = numpy.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    n = len(unique_labels)
    if n == 0:
        empty = numpy.zeros((0, _ORDER, _ORDER))
        return empty, empty, empty, numpy.zeros((0, 7))

    rows, cols = numpy.nonzero(labels)
    obj = numpy.searchsorted(unique_labels, labels[rows, cols])

    # skimage takes moments in each object's local (bounding-box) frame, so reduce to the
    # per-object minimum row/col. Init the accumulators above any pixel index for minimum.at.
    sentinel = 1 << 31
    rmin = numpy.full(n, sentinel)
    cmin = numpy.full(n, sentinel)
    numpy.minimum.at(rmin, obj, rows)
    numpy.minimum.at(cmin, obj, cols)
    local_r = (rows - rmin[obj]).astype(float)
    local_c = (cols - cmin[obj]).astype(float)

    raw = _moment_matrix(obj, local_r, local_c, n)
    centre_r = raw[:, 1, 0] / raw[:, 0, 0]
    centre_c = raw[:, 0, 1] / raw[:, 0, 0]
    # Central moments by direct centred summation (binomial-from-raw loses ~1e-4 to cancellation).
    central = _moment_matrix(obj, local_r - centre_r[obj], local_c - centre_c[obj], n)

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
    :func:`inertia_2d`, so requesting these no longer forces regionprops' per-region moment einsum
    (CellProfiler reports orientation in degrees; callers apply the ``180/pi`` conversion).
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
