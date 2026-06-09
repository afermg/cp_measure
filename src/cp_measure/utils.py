""" "
Utilities reused in multiple measurements.
"""

import centrosome.zernike
import numpy
from numpy.typing import NDArray

from cp_measure.primitives.segment import label_to_idx_lut


def _ensure_np_array(value):
    """Convert a result from scipy.ndimage to a numpy array

    scipy.ndimage has the annoying habit of returning a single, bare
    value instead of an array if the indexes passed in are of length 1.
    For instance:
    scind.maximum(image, labels, [1]) returns a float
    but
    scind.maximum(image, labels, [1,2]) returns a list
    """
    return numpy.array([value]) if numpy.isscalar(value) else numpy.array(value)


def _ensure_np_scalar(value):
    return value if numpy.isscalar(value) else numpy.array(value).squeeze()


def get_test_pixels_mask():
    pixels = numpy.random.randint(100, size=64**2).reshape((64, 64))
    mask = numpy.zeros_like(pixels, dtype=bool)
    mask[2:-3, 2:-3] = True
    return pixels, mask


def masks_to_ijv(masks: numpy.ndarray) -> numpy.ndarray:
    """
    input: 2d integer label array
    output: (n, 3) integer array of rows (i, j, label) sorted by label
    """
    i, j = numpy.nonzero(masks)
    v = masks[i, j]
    order = numpy.argsort(v, kind="stable")
    return numpy.column_stack((i[order], j[order], v[order])).astype(int, copy=False)


def labels_to_binmasks(masks: numpy.ndarray) -> numpy.ndarray:
    """
    Convert a label matrix to a boolean masks.

    Returns a list of binary masks.
    """
    labels = numpy.unique(masks)
    labels = labels[labels > 0]
    return masks == labels.reshape((-1,) + (1,) * masks.ndim)


def _zernike_scores(
    masks: NDArray[numpy.integer],
    zernike_indexes: NDArray[numpy.integer],
    weight: NDArray[numpy.floating] | None = None,
) -> tuple[
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
    NDArray[numpy.floating],
]:
    """Per-object real/imaginary Zernike moment sums, vectorised on foreground pixels.

    Mirrors ``centrosome.zernike.zernike`` without its two area-scaling costs: the basis is
    never scattered into a full ``(H, W, K)`` complex array, and each moment is segment-summed
    by label with one ``numpy.bincount`` instead of a whole-image ``scipy.ndimage.sum``. The
    Horner basis evaluation is copied from ``centrosome.construct_zernike_polynomials`` (same
    lookup table, ``r**2 > 1`` cutoff, ``z = y + i*x`` convention), so results match centrosome
    to round-off.

    ``weight`` (co-shaped with ``masks``, e.g. an intensity image) scales each pixel's
    contribution; ``None`` gives the unweighted shape moments. Returns
    ``(real_sums, imag_sums, radii, counts)`` of shapes ``(n, K)``, ``(n, K)``, ``(n,)`` and
    ``(n,)``, ordered by ascending label, where ``radii`` is the enclosing-circle radius and
    ``counts`` the object pixel count. ``get_zernike`` normalises by ``pi * radii**2``; the
    intensity-weighted radial Zernikes normalise by ``counts``.
    """
    lut, n = label_to_idx_lut(masks)
    k = len(zernike_indexes)
    labels = numpy.flatnonzero(lut >= 0)
    centers, radii = centrosome.zernike.minimum_enclosing_circle(masks, labels)
    radii = numpy.asarray(radii, dtype=float)
    real_sums = numpy.zeros((n, k))
    imag_sums = numpy.zeros((n, k))
    if n == 0:
        return real_sums, imag_sums, radii, numpy.zeros(0)

    # Foreground pixels, their object row, and unit-disk coordinates relative to each
    # object's enclosing circle — no full (H, W) coordinate grid is materialised.
    seg_full = lut[masks]
    keep = seg_full >= 0
    rows, cols = numpy.nonzero(keep)
    seg = seg_full[keep]
    counts = numpy.bincount(seg, minlength=n).astype(float)
    # Single-pixel objects have an enclosing-circle radius of 0; the resulting 0/0 yields NaN
    # coordinates (which the r**2 > 1 cutoff later discards), matching centrosome — suppress the
    # warning since it is expected, not a fault.
    with numpy.errstate(invalid="ignore", divide="ignore"):
        ym = (rows - centers[seg, 0]) / radii[seg]
        xm = (cols - centers[seg, 1]) / radii[seg]

    coeffs = centrosome.zernike.construct_zernike_lookuptable(zernike_indexes)
    r_square = xm * xm + ym * ym
    z = ym + 1j * xm
    w = None if weight is None else weight[keep].astype(float)
    z_pows = {m: z**m for m in numpy.unique(zernike_indexes[:, 1]) if m}
    for idx, (zn, zm) in enumerate(zernike_indexes):
        s = numpy.zeros_like(xm)
        for c in coeffs[idx, : (zn - zm) // 2 + 1]:  # Horner scheme on r**2
            s *= r_square
            s += c
        s[r_square > 1] = 0
        if w is not None:
            s *= w
        if zm == 0:  # purely real moment; the imaginary segment-sum is identically zero
            real_sums[:, idx] = numpy.bincount(seg, weights=s, minlength=n)
        else:
            zf = s * z_pows[zm]
            real_sums[:, idx] = numpy.bincount(seg, weights=zf.real, minlength=n)
            imag_sums[:, idx] = numpy.bincount(seg, weights=zf.imag, minlength=n)

    return real_sums, imag_sums, radii, counts
