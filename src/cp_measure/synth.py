"""Deterministic synthetic cell images for benchmarking.

``generate(image_size, n_objects, n_channels, seed)`` returns a cell-like integer label mask
plus intensity channels, built so that area/shape, intensity and texture features all carry real
signal:

- **Label mask** — organic, non-convex star-shaped cells (a base radius modulated by a few low
  Fourier harmonics), placed by dart-throwing with a minimum gap so they are separated (a few may
  sit close to form clusters, none overlap). Labels are contiguous ``1..n`` (cp_measure's
  contract) and every object is well above the degenerate-size floor — there are no 1-pixel cells.
- **Channels** — a shared smooth envelope over the cells (so intensity is object-correlated) plus,
  per channel, an *independent* field of multi-scale Gaussian splats (large gradients, mid blobs,
  tiny clustered puncta) and noise. The shared envelope drives a controlled positive correlation
  between channels (for colocalisation), while the independent splats keep it below 1.

The output is a pure function of ``(image_size, n_objects, n_channels, seed)`` for a fixed
scipy/skimage build. The result is **reproducible only against a pinned generator version**:
``__version__`` is stamped into benchmark comments so historical numbers stay comparable.

Placement is *capacity-checked*: an ``n_objects`` that cannot fit (with gaps) in ``image_size``
raises ``ValueError`` loudly rather than silently under-placing — callers that sweep an
object-count axis must choose counts feasible at their smallest image size.
"""

from __future__ import annotations

import numpy
import scipy.ndimage
from numpy.typing import NDArray

__version__ = "0.1.0"

# Cell geometry. Sizes are fixed in absolute pixels (NOT derived from n_objects) so that sweeping
# the object-count axis isolates per-object cost without also shrinking the cells.
_MEAN_RADIUS = 8.0  # median base radius of a cell, px
_RADIUS_LOGSIGMA = (
    0.35  # spread of the log-normal size distribution (big + small cells)
)
_MIN_RADIUS = (
    3.5  # lower clamp — keeps the smallest cell well above the degenerate floor
)
_MAX_RADIUS_FACTOR = (
    2.2  # clamp the log-normal upper tail to this * mean (also caps packing waste)
)
_GAP = 2.0  # minimum empty margin between two cells, px
_N_HARMONICS = 3  # low Fourier harmonics that make a boundary organic/non-convex
_HARMONIC_AMP = 0.12  # per-harmonic radial wobble as a fraction of the base radius
_FILL_LIMIT = 0.45  # max fraction of image area the (halo-padded) cells may claim before we refuse

# Channel synthesis.
_SHARED_SPLATS_PER_CELL = (
    6  # splats identical across channels (drive cross-channel correlation)
)
_INDEP_SPLATS_PER_CELL = 7  # splats drawn per channel (pull correlation back below 1)
_BG_LEVEL = 0.05  # flat background intensity
_ENVELOPE_WEIGHT = 0.5  # weight of the shared smooth envelope
_NOISE_LEVEL = 0.05  # additive Gaussian read noise (std)


def generate(
    image_size: int,
    n_objects: int,
    n_channels: int = 2,
    seed: int = 0,
) -> tuple[NDArray[numpy.int32], NDArray[numpy.float32]]:
    """Generate a synthetic ``(labels, channels)`` pair.

    Parameters
    ----------
    image_size : int
        Side length; the image is ``image_size x image_size``.
    n_objects : int
        Number of cells to place. Raises ``ValueError`` if they cannot fit.
    n_channels : int
        Number of intensity channels (>= 1). cp_measure core features consume channel 0;
        colocalisation consumes channels 0 and 1.
    seed : int
        Seeds all randomness; identical inputs give identical output.

    Returns
    -------
    labels : int32 array ``(image_size, image_size)``
        Contiguous labels ``1..n_objects`` (``0`` = background).
    channels : float32 array ``(n_channels, image_size, image_size)``
        Non-negative intensities.
    """
    if n_channels < 1:
        raise ValueError(f"n_channels must be >= 1, got {n_channels}")
    if n_objects < 0:
        raise ValueError(f"n_objects must be >= 0, got {n_objects}")

    rng = numpy.random.default_rng(seed)
    labels = _build_label_mask(rng, image_size, n_objects)
    channels = _build_channels(rng, labels, n_channels)
    return labels, channels


def _draw_radii(rng: numpy.random.Generator, n: int) -> NDArray[numpy.float64]:
    """Per-cell base radii from a clamped log-normal (variety: big + tiny cells)."""
    r = _MEAN_RADIUS * numpy.exp(rng.normal(0.0, _RADIUS_LOGSIGMA, size=n))
    return numpy.clip(r, _MIN_RADIUS, _MEAN_RADIUS * _MAX_RADIUS_FACTOR)


def _build_label_mask(
    rng: numpy.random.Generator, image_size: int, n_objects: int
) -> NDArray[numpy.int32]:
    labels = numpy.zeros((image_size, image_size), dtype=numpy.int32)
    if n_objects == 0:
        return labels

    radii = numpy.sort(_draw_radii(rng, n_objects))[
        ::-1
    ]  # place big cells first (easier packing)
    # The organic boundary can bulge to this radius, so packing and capacity use the bulge radius.
    bulge = radii * (1.0 + _N_HARMONICS * _HARMONIC_AMP)

    halo_area = float(numpy.sum(numpy.pi * (bulge + _GAP) ** 2))
    if halo_area > _FILL_LIMIT * image_size * image_size:
        raise ValueError(
            f"cannot fit {n_objects} cells in {image_size}x{image_size}: they need "
            f"{halo_area / image_size**2:.0%} of the image (limit {_FILL_LIMIT:.0%}). "
            f"Use a larger image or fewer objects."
        )

    centers = _place_centers(rng, image_size, bulge)

    # Rasterise each organic cell. Placement guarantees the bulge disks are disjoint, so labels
    # never collide; the `== 0` guard is belt-and-braces against boundary rounding.
    for k, (cy, cx, base_r) in enumerate(
        zip(centers[:, 0], centers[:, 1], radii), start=1
    ):
        amps = rng.uniform(-_HARMONIC_AMP, _HARMONIC_AMP, size=_N_HARMONICS)
        phases = rng.uniform(0.0, 2.0 * numpy.pi, size=_N_HARMONICS)
        win_mask, (y0, x0, y1, x1) = _rasterize_cell(
            image_size, cy, cx, base_r, amps, phases
        )
        sub = labels[y0:y1, x0:x1]
        sub[win_mask & (sub == 0)] = k

    realized = numpy.unique(labels)
    realized = realized[realized > 0]
    if realized.size != n_objects:
        raise ValueError(
            f"requested {n_objects} objects but rasterised {realized.size}; "
            f"a cell was fully occluded — widen _GAP or lower the count."
        )
    return labels


def _place_centers(
    rng: numpy.random.Generator, image_size: int, bulge: NDArray[numpy.float64]
) -> NDArray[numpy.float64]:
    """Dart-throw centers so bulge disks are pairwise separated by at least ``_GAP``.

    Big cells (front of ``bulge``) are placed first. Raises if the per-object attempt budget is
    exhausted — never returns fewer than requested.
    """
    n = bulge.size
    centers = numpy.empty((n, 2), dtype=numpy.float64)
    placed = 0
    budget = 200 * n + 1000
    attempts = 0
    while placed < n:
        r = bulge[placed]
        cy = rng.uniform(r, image_size - r)
        cx = rng.uniform(r, image_size - r)
        if placed == 0:
            ok = True
        else:
            dist = numpy.hypot(centers[:placed, 0] - cy, centers[:placed, 1] - cx)
            ok = bool(numpy.all(dist >= r + bulge[:placed] + _GAP))
        if ok:
            centers[placed] = (cy, cx)
            placed += 1
        attempts += 1
        if attempts > budget:
            raise ValueError(
                f"could not place {n} cells in {image_size}x{image_size} within the attempt "
                f"budget ({placed} placed); the layout is too dense — lower the count."
            )
    return centers


def _rasterize_cell(
    image_size: int,
    cy: float,
    cx: float,
    base_r: float,
    amps: NDArray[numpy.float64],
    phases: NDArray[numpy.float64],
) -> tuple[NDArray[numpy.bool_], tuple[int, int, int, int]]:
    """Boolean mask of one organic star-shaped cell, plus its ``(y0, x0, y1, x1)`` window."""
    reach = base_r * (1.0 + numpy.sum(numpy.abs(amps)))
    y0 = max(0, int(numpy.floor(cy - reach)))
    x0 = max(0, int(numpy.floor(cx - reach)))
    y1 = min(image_size, int(numpy.ceil(cy + reach)) + 1)
    x1 = min(image_size, int(numpy.ceil(cx + reach)) + 1)
    yy, xx = numpy.mgrid[y0:y1, x0:x1]
    dy = yy - cy
    dx = xx - cx
    dist = numpy.hypot(dy, dx)
    theta = numpy.arctan2(dy, dx)
    harmonics = numpy.arange(1, _N_HARMONICS + 1)[:, None, None]
    wobble = numpy.sum(
        amps[:, None, None]
        * numpy.cos(harmonics * theta[None] + phases[:, None, None]),
        axis=0,
    )
    boundary = base_r * (1.0 + wobble)
    return dist <= boundary, (y0, x0, y1, x1)


def _build_channels(
    rng: numpy.random.Generator, labels: NDArray[numpy.int32], n_channels: int
) -> NDArray[numpy.float32]:
    image_size = labels.shape[0]
    fg = labels > 0
    # Shared smooth envelope: bright on/near cells, identical across channels → drives a positive
    # cross-channel correlation that the independent splats then pull back below 1.
    envelope = scipy.ndimage.gaussian_filter(
        fg.astype(numpy.float64), sigma=_MEAN_RADIUS
    )
    if envelope.max() > 0:
        envelope /= envelope.max()

    fg_idx = numpy.flatnonzero(fg)
    n_objects = int(labels.max())
    # A splat field shared by every channel plus one independent field per channel: the shared
    # part correlates the channels (colocalisation signal), the independent part keeps r < 1.
    shared_splats = _splat_field(
        rng, image_size, fg_idx, n_objects * _SHARED_SPLATS_PER_CELL
    )
    base = _BG_LEVEL + _ENVELOPE_WEIGHT * envelope + shared_splats

    channels = numpy.empty((n_channels, image_size, image_size), dtype=numpy.float32)
    for c in range(n_channels):
        img = base + _splat_field(
            rng, image_size, fg_idx, n_objects * _INDEP_SPLATS_PER_CELL
        )
        img += rng.normal(0.0, _NOISE_LEVEL, size=img.shape)
        numpy.clip(img, 0.0, None, out=img)
        channels[c] = img.astype(numpy.float32)
    return channels


def _splat_field(
    rng: numpy.random.Generator,
    image_size: int,
    fg_idx: NDArray[numpy.intp],
    n_splats: int,
) -> NDArray[numpy.float64]:
    """Sum of multi-scale 2D Gaussians stamped inside the cells (intra-object texture signal)."""
    field = numpy.zeros((image_size, image_size), dtype=numpy.float64)
    if n_splats == 0 or fg_idx.size == 0:
        return field
    picks = fg_idx[rng.integers(0, fg_idx.size, size=n_splats)]
    ys, xs = numpy.divmod(picks, image_size)
    # Scale mixture: tiny puncta dominate (texture), with fewer mid blobs and a few large gradients.
    scale_choice = rng.choice(
        numpy.array([1.2, 3.0, 6.0]), size=n_splats, p=[0.6, 0.3, 0.1]
    )
    amps = rng.uniform(0.3, 1.0, size=n_splats)
    for cy, cx, sigma, amp in zip(ys, xs, scale_choice, amps):
        _add_gaussian(field, int(cy), int(cx), float(sigma), float(amp))
    return field


def _add_gaussian(
    field: NDArray[numpy.float64], cy: int, cx: int, sigma: float, amp: float
) -> None:
    """Add ``amp * exp(-r^2 / 2 sigma^2)`` onto a bounded window around ``(cy, cx)``."""
    rad = max(1, int(numpy.ceil(3.0 * sigma)))
    size = field.shape[0]
    y0, y1 = max(0, cy - rad), min(size, cy + rad + 1)
    x0, x1 = max(0, cx - rad), min(size, cx + rad + 1)
    yy, xx = numpy.mgrid[y0:y1, x0:x1]
    field[y0:y1, x0:x1] += amp * numpy.exp(
        -((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma)
    )
