"""Numba Zernike-moment kernel (fused basis-eval + weighted-complex segment-sum).

Both Zernike features — shape (`get_zernike`, weight=1) and intensity-weighted
(`get_radial_zernikes`, weight=pixel) — reduce to the same per-object
weighted-complex sum of the Zernike polynomial basis ``V_nm`` over the object's
pixels. This module fuses Horner basis evaluation and the segment-sum into one
single-pass ``@njit`` kernel (no ``(M, K)`` intermediate). The radial path computes
azimuthal powers with NumPy to reduce phase-rounding differences, not to guarantee
identical phases. Meaningful phases agree modulo 2π; phases of near-zero moments
are numerically unstable (see ``docs/measurement-notes.md``).

The polynomial *coefficients* come from centrosome (``construct_zernike_lookuptable``,
degree-only, cheap) via :func:`zernike_coeffs`; the per-pixel evaluation follows
``centrosome.zernike.construct_zernike_polynomials``:

- radial part ``R_nm(r)/r^m`` via Horner over ``r² = xm²+ym²`` using the LUT row,
  looping exactly ``(n-m)//2 + 1`` terms (forward order, LUT[k,0] = highest power),
- azimuthal factor ``z^m`` with ``z = ym + 1j*xm`` (ym = normalised row offset,
  xm = normalised column offset),
- unit-disk cutoff: zero where ``r² > 1 + 1e-9``, so pixels on the enclosing circle
  count regardless of rounding (see ``cp_measure.utils._UNIT_DISK_EDGE``).

Pixel coordinates come from :func:`_object_coords`, a numba port of the NumPy
backend's :func:`cp_measure.utils._zernike_coords` with the same arithmetic; the
minimum enclosing circle is the numba :func:`enclosing_circles`.
"""

import centrosome.zernike
import numba
import numpy as np
from numba import njit
from numpy.typing import NDArray

from cp_measure.primitives.parallel import map_images
from cp_measure.primitives.shapes import each_volume, memory_key, native
from cp_measure.utils import _UNIT_DISK_EDGE


def zernike_coeffs(
    zernike_indexes: NDArray[np.integer],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Repack centrosome's radial LUT for the Horner kernel.

    Returns ``(lut, nterms, m_arr)`` where ``lut[k, :]`` are the radial-polynomial
    coefficients of index ``k`` (highest power first), ``nterms[k] = (n-m)//2 + 1``
    is exactly how many of them to consume in the Horner loop (trailing LUT entries
    are zero and must NOT be iterated — they would spuriously multiply by r²), and
    ``m_arr[k]`` is the azimuthal order ``m``.
    """
    lut = centrosome.zernike.construct_zernike_lookuptable(zernike_indexes).astype(
        np.float64
    )
    n = zernike_indexes[:, 0].astype(np.int64)
    m_arr = zernike_indexes[:, 1].astype(np.int64)
    nterms = ((n - m_arr) // 2 + 1).astype(np.int64)
    return lut, nterms, m_arr


# nogil is load-bearing: it lets map_images run this across a thread pool for a
# batch. Thread-safety holds only because every call allocates its own vr/vi/zr/zi
# and reads lut/nterms/m_arr (and the hoisted coeffs/zernike_indexes) read-only.
# A future shared scratch/output buffer would make batches race — keep it per-call.
@njit(cache=True, nogil=True)
def zernike_moments(weights, xm, ym, seg0, lut, nterms, m_arr, n, powers=None):
    """Fused per-object weighted-complex Zernike sum (single pass, no (M,K) buffer).

    For each pixel: evaluate every polynomial's ``V_nm`` from ``lut``/``m_arr``
    (Horner over r² + ``z^m``, ``z = ym + 1j*xm``; zero the basis outside the unit disk)
    and scatter-add ``weight·Re/Im`` into ``vr``/``vi`` ``(n, K)``. ``seg0[i]`` is the
    0-based object index of pixel ``i`` (``< 0`` ⇒ skip). Returns ``(vr, vi)``.
    Optional ``powers[m, i]`` supplies NumPy-computed azimuthal powers ``z^m``;
    otherwise the scalar recurrence is evaluated in the kernel.
    """
    K = lut.shape[0]
    M = xm.shape[0]
    vr = np.zeros((n, K))
    vi = np.zeros((n, K))
    max_m = 0
    for k in range(K):
        if m_arr[k] > max_m:
            max_m = m_arr[k]
    zr = np.empty(max_m + 1)
    zi = np.empty(max_m + 1)
    for i in range(M):
        seg = seg0[i]
        if seg < 0:
            continue
        x = xm[i]
        y = ym[i]
        r2 = x * x + y * y
        w = weights[i]
        zr[0] = 1.0
        zi[0] = 0.0
        if powers is None:
            for mm in range(1, max_m + 1):  # z^mm = z^(mm-1) * (y + i x)
                zr[mm] = zr[mm - 1] * y - zi[mm - 1] * x
                zi[mm] = zr[mm - 1] * x + zi[mm - 1] * y
        else:
            for mm in range(1, max_m + 1):
                zr[mm] = powers[mm, i].real
                zi[mm] = powers[mm, i].imag
        for k in range(K):
            s = 0.0
            nt = nterms[k]
            for t in range(nt):  # Horner: R_nm(r)/r^m
                s = s * r2 + lut[k, t]
            if r2 > _UNIT_DISK_EDGE:
                s = 0.0  # cutoff, before weighting (including NaN/Inf)
            m = m_arr[k]
            s *= w
            if m == 0:
                # The NumPy reference keeps the m=0 imaginary sum identically
                # zero, even for nonfinite weights or zero-radius objects.
                vr[seg, k] += s
            else:
                vr[seg, k] += s * zr[m]
                vi[seg, k] += s * zi[m]
    return vr, vi


@njit(cache=True, nogil=True)
def zernike_moments_channels(weights, xm, ym, seg0, lut, nterms, m_arr, n, powers=None):
    """:func:`zernike_moments` for several images measured with one mask: the basis
    is evaluated once per pixel and accumulated for every row of ``weights``
    ``(C, M)``, each with :func:`zernike_moments`' arithmetic (bit-identical per row).
    Returns ``(vr, vi)`` of shape ``(C, n, K)``. Kept separate so the one-image
    kernel (shape moments, single calls) has no channel loop in its hot path.
    """
    K = lut.shape[0]
    M = xm.shape[0]
    C = weights.shape[0]
    vr = np.zeros((C, n, K))
    vi = np.zeros((C, n, K))
    max_m = 0
    for k in range(K):
        if m_arr[k] > max_m:
            max_m = m_arr[k]
    zr = np.empty(max_m + 1)
    zi = np.empty(max_m + 1)
    w = np.empty(C)
    for i in range(M):
        seg = seg0[i]
        if seg < 0:
            continue
        x = xm[i]
        y = ym[i]
        r2 = x * x + y * y
        for c in range(C):
            w[c] = weights[c, i]
        zr[0] = 1.0
        zi[0] = 0.0
        if powers is None:
            for mm in range(1, max_m + 1):  # z^mm = z^(mm-1) * (y + i x)
                zr[mm] = zr[mm - 1] * y - zi[mm - 1] * x
                zi[mm] = zr[mm - 1] * x + zi[mm - 1] * y
        else:
            for mm in range(1, max_m + 1):
                zr[mm] = powers[mm, i].real
                zi[mm] = powers[mm, i].imag
        for k in range(K):
            s = 0.0
            nt = nterms[k]
            for t in range(nt):  # Horner: R_nm(r)/r^m
                s = s * r2 + lut[k, t]
            if r2 > _UNIT_DISK_EDGE:
                s = 0.0  # cutoff, before weighting (including NaN/Inf)
            m = m_arr[k]
            for c in range(C):
                sw = s * w[c]
                if m == 0:
                    # The NumPy reference keeps the m=0 imaginary sum identically
                    # zero, even for nonfinite weights or zero-radius objects.
                    vr[c, seg, k] += sw
                else:
                    vr[c, seg, k] += sw * zr[m]
                    vi[c, seg, k] += sw * zi[m]
    return vr, vi


# error_model="numpy": a single-pixel object has radius 0, and 0/0 must give NaN
# coordinates (and NaN features), as in the NumPy backend, not raise.
@njit(cache=True, nogil=True, error_model="numpy")
def _object_coords(labels):
    """:func:`cp_measure.utils._zernike_coords` without the GIL: for each foreground pixel
    in raster order, its 0-based object index ``seg``, its row and column, and its
    row/column offsets ``ym``/``xm`` from the object's enclosing-circle centre divided by
    the radius; plus the ``(n,)`` radii and pixel counts. Labels are ``1..n``."""
    H, W = labels.shape
    n = 0
    size = 0
    for r in range(H):
        for c in range(W):
            lab = labels[r, c]
            if lab > 0:
                size += 1
                n = max(n, lab)
    centers, radii = enclosing_circles(labels, n)
    seg = np.empty(size, np.int64)
    rows = np.empty(size, np.int64)
    cols = np.empty(size, np.int64)
    ym = np.empty(size)
    xm = np.empty(size)
    counts = np.zeros(n)
    i = 0
    for r in range(H):
        for c in range(W):
            lab = labels[r, c]
            if lab > 0:
                o = lab - 1
                seg[i] = o
                rows[i] = r
                cols[i] = c
                ym[i] = (r - centers[o, 0]) / radii[o]
                xm[i] = (c - centers[o, 1]) / radii[o]
                counts[o] += 1.0
                i += 1
    return seg, rows, cols, ym, xm, radii, counts


@njit(cache=True, nogil=True)
def _shape_sums(labels, lut, nterms, m_arr):
    """Unweighted (shape) Zernike sums ``(vr, vi)`` of shape ``(n, K)``, the radii and
    the pixel counts, in one call."""
    seg, _rows, _cols, ym, xm, radii, counts = _object_coords(labels)
    vr, vi = zernike_moments(
        np.ones(seg.size), xm, ym, seg, lut, nterms, m_arr, radii.size
    )
    return vr, vi, radii, counts


@njit(cache=True, nogil=True)
def _weighted_coords(labels, pixels):
    """:func:`_object_coords` plus each foreground pixel's value in every image of the
    tuple ``pixels``, as float64 ``(C, M)`` weights."""
    seg, rows, cols, ym, xm, radii, counts = _object_coords(labels)
    weights = np.empty((len(pixels), seg.size))
    for c in range(len(pixels)):
        image = pixels[c]
        for i in range(seg.size):
            weights[c, i] = image[rows[i], cols[i]]
    return seg, ym, xm, radii, counts, weights


def _pixel_tuple(pixels):
    """The images as a tuple numba can index with a loop variable: all of one array type,
    so they are passed as views; otherwise as float64 copies (the weights are float64).
    Non-native byte orders, which numba cannot type, are converted first."""
    pixels = [native(p) for p in pixels]
    if len({numba.typeof(p) for p in pixels}) == 1:
        return tuple(pixels)
    return tuple(np.ascontiguousarray(p, dtype=np.float64) for p in pixels)


def zernike_moments_per_object(labels, pixels, coeffs):
    """Per-object weighted-complex Zernike sums shared by both 2D backends.

    Same contract as :func:`cp_measure.utils._zernike_scores` for every intensity image
    in ``pixels`` (a list; the weighted, radial case) or for ``None`` (the unweighted
    shape case). The coordinates and the basis are computed once for all images.
    ``coeffs`` is :func:`zernike_coeffs`. Returns ``(vr, vi, radii, counts)`` with
    ``vr``/``vi`` of shape ``(C, n, K)``.
    """
    lut, nterms, m_arr = coeffs
    # native-endian, contiguous intp: one compiled signature (as _enclosing_circle)
    labels = np.ascontiguousarray(labels, dtype=np.intp)
    if pixels is None:
        vr, vi, radii, counts = _shape_sums(labels, lut, nterms, m_arr)
        return vr[None], vi[None], radii, counts
    seg, ym, xm, radii, counts, weights = _weighted_coords(labels, _pixel_tuple(pixels))
    n = len(radii)
    if n == 0:
        empty = np.zeros((len(pixels), 0, len(lut)))
        return empty, empty, radii, counts
    # Mirror-symmetric objects have moments whose real (or imaginary) part is
    # zero by symmetry, so atan2 returns +pi or -pi on the sign of round-off.
    # Building z^m with NumPy's complex multiply, as _zernike_scores does,
    # reduces these sign differences but cannot guarantee identical phases:
    # even repeated NumPy calls can differ at the branch cut or near zero.
    # Only the (max_m + 1, M) powers are materialised, not the (M, K) basis.
    z = ym + 1j * xm
    powers = np.empty((int(m_arr.max()) + 1, len(z)), dtype=np.complex128)
    powers[0] = 1
    for m in range(1, len(powers)):
        powers[m] = z if m == 1 else powers[m - 1] * z
    if len(pixels) == 1:
        vr, vi = zernike_moments(weights[0], xm, ym, seg, lut, nterms, m_arr, n, powers)
        return vr[None], vi[None], radii, counts
    vr, vi = zernike_moments_channels(
        weights, xm, ym, seg, lut, nterms, m_arr, n, powers
    )
    return vr, vi, radii, counts


def zernike_per_image(labels, pixels, degree, features, *, max_workers=None):
    """Batch driver shared by the numba ``get_zernike`` and ``get_radial_zernikes``.

    Accepts a single 2D image or a batch (list/tuple of images, or a 4D
    ``(B, Z, Y, X)`` array) and returns a dict or a list of dicts. Singleton-Z
    volumes use the 2D measurements; other volumes give ``{}``, consistently for
    individual inputs and batches. ``pixels`` is ``None`` for the unweighted
    shape moments. For measured entries, weighted
    inputs must have matching shapes and dimensionality, before axis promotion.
    ``features(vr, vi, radii, counts, zernike_indexes)`` names and normalises one
    image's moment sums.
    """
    is_sequence = isinstance(labels, (list, tuple))
    if not is_sequence and np.ndim(labels) == 3 and labels.shape[0] != 1:
        return {}
    weighted = pixels is not None
    # Views, no batch copy: images that share a mask keep sharing its memory.
    # each_volume also checks that labels and pixels match in shape and rank.
    volumes, unwrap = each_volume(labels, pixels if weighted else labels)
    masks_bzyx = [m for m, _, _ in volumes]
    pixels_bzyx = [p for _, p, _ in volumes]
    # degree-only work, hoisted out of the per-image loop (computed once per batch)
    zernike_indexes = centrosome.zernike.get_zernike_indexes(degree + 1)
    coeffs = zernike_coeffs(zernike_indexes)

    measured = [i for i in range(len(masks_bzyx)) if masks_bzyx[i].shape[0] == 1]
    # Images that share a mask (the featurizer passes one mask per channel) share
    # its coordinates and basis: one group per mask, keyed by the memory it views.
    groups: dict = {}
    for i in measured:
        groups.setdefault(memory_key(masks_bzyx[i]), []).append(i)

    def one(items):
        sums = zernike_moments_per_object(
            masks_bzyx[items[0]][0],
            [pixels_bzyx[i][0] for i in items] if weighted else None,
            coeffs,
        )
        vr, vi, radii, counts = sums
        if weighted:
            return [
                features(vr[c], vi[c], radii, counts, zernike_indexes)
                for c in range(len(items))
            ]
        result = features(vr[0], vi[0], radii, counts, zernike_indexes)
        # the first item keeps the arrays, the others get copies (callers may edit)
        return [result] + [
            {key: values.copy() for key, values in result.items()} for _ in items[1:]
        ]

    # Groups are independent; map_images runs them across a thread pool (the nogil
    # zernike_moments kernel releases the GIL) and a single group serially.
    results: list = [{} for _ in range(len(masks_bzyx))]
    grouped = list(groups.values())
    for items, outs in zip(
        grouped,
        map_images(
            one,
            grouped,
            max_workers=max_workers,
            pixels=sum(masks_bzyx[i].size for items in grouped for i in items),
        ),
    ):
        for i, out in zip(items, outs):
            results[i] = out
    return unwrap(results)


# --- Minimum enclosing circle ---------------------------------------------------------------
# centrosome's minimum_enclosing_circle (convex hull of the pixel centres, then Chrystal's
# algorithm) was ~1/3 of the coordinate setup. The minimum enclosing circle of a point set is
# unique, so any correct algorithm returns the same circle up to rounding: here the hull is
# built from each row's leftmost and rightmost pixel (whose hull is the object's), already in
# (row, col) order for Andrew's monotone chain, and the circle comes from Welzl's incremental
# algorithm on the hull vertices.


@njit(cache=True, nogil=True)
def _cross(ox, oy, ax, ay, bx, by):
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


@njit(cache=True, nogil=True)
def _covers(cx, cy, r, x, y):
    """Point (x, y) is inside the circle, allowing for rounding at the boundary."""
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r * (1.0 + 1e-12) + 1e-12


@njit(cache=True, nogil=True)
def _circle_through_2(ax, ay, bx, by):
    cx = (ax + bx) / 2.0
    cy = (ay + by) / 2.0
    return cx, cy, np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)


@njit(cache=True, nogil=True)
def _circle_through_3(ax, ay, bx, by, px, py):
    d = 2.0 * (ax * (by - py) + bx * (py - ay) + px * (ay - by))
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    p2 = px * px + py * py
    cx = (a2 * (by - py) + b2 * (py - ay) + p2 * (ay - by)) / d
    cy = (a2 * (px - bx) + b2 * (ax - px) + p2 * (bx - ax)) / d
    return cx, cy, np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)


@njit(cache=True, nogil=True)
def enclosing_circles(labels, n):
    """Minimum enclosing circle of each label ``1..n``'s pixel centres.

    Returns ``(centers, radii)`` with ``centers[:, 0]`` the row and ``[:, 1]`` the
    column, as centrosome's ``minimum_enclosing_circle``; an absent label gets a NaN
    centre and radius 0, also as centrosome.
    """
    H, W = labels.shape
    row_min = np.full((n + 1, H), W, np.int64)
    row_max = np.full((n + 1, H), -1, np.int64)
    for r in range(H):
        for c in range(W):
            lab = labels[r, c]
            if lab > 0:
                if c < row_min[lab, r]:
                    row_min[lab, r] = c
                if c > row_max[lab, r]:
                    row_max[lab, r] = c
    centers = np.full((n, 2), np.nan)
    radii = np.zeros(n)
    px = np.empty(2 * H)
    py = np.empty(2 * H)
    hx = np.empty(4 * H + 2)
    hy = np.empty(4 * H + 2)
    for lab in range(1, n + 1):
        m = 0  # extreme pixels per row, in (row, col) order
        for r in range(H):
            if row_max[lab, r] >= 0:
                px[m] = r
                py[m] = row_min[lab, r]
                m += 1
                if row_max[lab, r] != row_min[lab, r]:
                    px[m] = r
                    py[m] = row_max[lab, r]
                    m += 1
        if m == 0:
            continue
        k = 0  # Andrew's monotone chain: lower then upper hull
        for i in range(m):
            while (
                k >= 2
                and _cross(hx[k - 2], hy[k - 2], hx[k - 1], hy[k - 1], px[i], py[i])
                <= 0
            ):
                k -= 1
            hx[k] = px[i]
            hy[k] = py[i]
            k += 1
        lower = k + 1
        for i in range(m - 2, -1, -1):
            while (
                k >= lower
                and _cross(hx[k - 2], hy[k - 2], hx[k - 1], hy[k - 1], px[i], py[i])
                <= 0
            ):
                k -= 1
            hx[k] = px[i]
            hy[k] = py[i]
            k += 1
        h = k - 1 if k > 1 else k  # the chain repeats its first point
        cx, cy, rad = hx[0], hy[0], 0.0  # Welzl, iterative
        for i in range(1, h):
            if _covers(cx, cy, rad, hx[i], hy[i]):
                continue
            cx, cy, rad = hx[i], hy[i], 0.0
            for j in range(i):
                if _covers(cx, cy, rad, hx[j], hy[j]):
                    continue
                cx, cy, rad = _circle_through_2(hx[i], hy[i], hx[j], hy[j])
                for q in range(j):
                    if _covers(cx, cy, rad, hx[q], hy[q]):
                        continue
                    cx, cy, rad = _circle_through_3(
                        hx[i], hy[i], hx[j], hy[j], hx[q], hy[q]
                    )
        centers[lab - 1, 0] = cx
        centers[lab - 1, 1] = cy
        radii[lab - 1] = rad
    return centers, radii


def _enclosing_circle(masks, labels):
    """``centrosome.zernike.minimum_enclosing_circle`` signature over contiguous labels."""
    return enclosing_circles(np.ascontiguousarray(masks, dtype=np.intp), len(labels))
