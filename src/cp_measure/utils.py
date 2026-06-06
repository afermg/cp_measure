""" "
Utilities reused in multiple measurements.
"""

import centrosome.zernike
import numpy
from numpy.typing import NDArray


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
    indices: NDArray[numpy.integer],
    zernike_indexes: NDArray[numpy.integer],
    weight: NDArray[numpy.floating] | None = None,
) -> tuple[NDArray[numpy.floating], NDArray[numpy.floating], NDArray[numpy.floating]]:
    """Per-object real/imaginary Zernike moment sums, vectorised on foreground pixels.

    Mirrors ``centrosome.zernike.zernike`` but avoids its two area-scaling costs: it never
    scatters the basis into a full ``(H, W, K)`` complex array, and it segment-sums each
    moment by label with one ``numpy.bincount`` instead of a per-channel
    ``scipy.ndimage.sum`` over the whole image. The Horner basis evaluation is copied
    verbatim from ``centrosome.construct_zernike_polynomials`` (same lookup-table
    coefficients, ``r**2 > 1`` cutoff and ``z = y + i*x`` convention) so the result matches
    centrosome to floating-point round-off.

    Returns ``(real_sums, imag_sums, radii)`` with shapes ``(n, K)``, ``(n, K)`` and
    ``(n,)`` ordered by ``indices``. ``weight`` (e.g. intensity) multiplies each pixel's
    contribution; ``None`` gives the unweighted shape moments. Used by both ``get_zernike``
    (unweighted shape moments) and ``get_radial_zernikes`` (intensity-weighted moments).
    """
    n = len(indices)
    k = len(zernike_indexes)
    centers, radii = centrosome.zernike.minimum_enclosing_circle(masks, indices)
    radii = numpy.asarray(radii, dtype=float)
    real_sums = numpy.zeros((n, k))
    imag_sums = numpy.zeros((n, k))
    if n == 0:
        return real_sums, imag_sums, radii

    # Map each label to its row in [0, n); -1 for background / unselected labels.
    rev = numpy.full(int(masks.max()) + 1, -1, dtype=int)
    rev[indices] = numpy.arange(n)
    mask = rev[masks] != -1
    rows, cols = numpy.nonzero(mask)
    rev_idx = rev[masks[rows, cols]]

    # Coordinates relative to each object's enclosing circle (unit disk). Taken straight
    # from the foreground pixel indices — no full (H, W) coordinate grid is materialised.
    ym = (rows - centers[rev_idx, 0]) / radii[rev_idx]
    xm = (cols - centers[rev_idx, 1]) / radii[rev_idx]

    lut = centrosome.zernike.construct_zernike_lookuptable(zernike_indexes)
    r_square = xm * xm + ym * ym
    z = ym + 1j * xm
    w = None if weight is None else weight[mask].astype(float)

    z_pows: dict[int, NDArray] = {}
    for idx in range(k):
        zn, zm = zernike_indexes[idx]
        s = numpy.zeros_like(xm)
        for kk in range((zn - zm) // 2 + 1):  # Horner scheme on r**2
            s *= r_square
            s += lut[idx, kk]
        s[r_square > 1] = 0
        if w is not None:
            s *= w
        if zm == 0:
            zf = s.astype(complex)
        else:
            if zm not in z_pows:
                z_pows[zm] = z if zm == 1 else z**zm
            zf = s * z_pows[zm]
        real_sums[:, idx] = numpy.bincount(rev_idx, weights=zf.real, minlength=n)
        imag_sums[:, idx] = numpy.bincount(rev_idx, weights=zf.imag, minlength=n)

    return real_sums, imag_sums, radii
