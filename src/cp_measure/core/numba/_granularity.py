"""Numba 2D morphology kernels for the granularity backend.

Single-threaded ``@njit(cache=True)`` kernels, bit-exact with the skimage
operations the numpy baseline uses:

- :func:`disk_erosion_2d` / :func:`disk_dilation_2d` — greyscale disk min/max via
  a **row-decomposed van-Herk/Gil-Werman (VHG)** sliding 1-D min/max over the
  ``2r+1`` disk rows (``O(r·HW)``), matching ``skimage.morphology.erosion/dilation``
  with a ``disk(radius)`` footprint (used once for background opening).
- :func:`erosion_4conn_2d` / :func:`dilation_4conn_2d` — 5-tap (disk(1)/cross)
  min/max, matching ``skimage`` with a ``disk(1)`` footprint (the per-iteration
  erosion + the dilation used inside reconstruction).
- :func:`reconstruction_by_dilation_2d` — morphological reconstruction by dilation
  (4-connectivity) via raster-until-convergence, matching
  ``skimage.morphology.reconstruction(seed, mask, footprint=disk(1))``.

Borders match skimage's footprint ops via ``edge`` (clamp) padding applied on the
host before the kernel; min/max selection makes the result border-mode-insensitive
for disks (verified bit-exact vs skimage over reflect/symmetric/edge).
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray


def _disk_halfwidths(radius: int) -> NDArray[np.int64]:
    """Per-row half-widths of ``skimage.morphology.disk(radius)``.

    ``hx[dy] = floor(sqrt(radius² - dy²))`` for ``dy`` in ``0..radius`` — the disk
    row at vertical offset ``±dy`` spans columns ``[-hx[dy], +hx[dy]]``.
    """
    dy = np.arange(radius + 1, dtype=np.float64)
    return np.floor(np.sqrt(radius * radius - dy * dy)).astype(np.int64)


@njit(cache=True)
def _disk_reduce(P, radius, hx, H, W, is_max):
    """Row-decomposed VHG disk min/max over an ``edge``-padded image ``P``.

    ``P`` has shape ``(H+2r, W+2r)``. For each of the ``2r+1`` disk rows, a 1-D
    sliding min/max of radius ``hx[|dy|]`` is taken along the padded row via VHG
    (prefix ``g`` / suffix ``h`` over blocks of size ``2w+1``), then reduced across
    rows. ``is_max`` selects dilation (max) vs erosion (min).
    """
    L = W + 2 * radius
    out = np.full((H, W), -np.inf if is_max else np.inf)

    g = np.empty(L, np.float64)
    h = np.empty(L, np.float64)
    for dy in range(-radius, radius + 1):
        w = hx[dy] if dy >= 0 else hx[-dy]
        k = 2 * w + 1
        h_block_start = (L - 1) % k  # x % k of the last element (one modulo per row-band)
        for i in range(H):
            pr = i + radius + dy
            # forward prefix-min/max within blocks of size k (counter tracks x % k)
            c = 0
            for x in range(L):
                if c == 0:
                    g[x] = P[pr, x]
                else:
                    a = g[x - 1]
                    b = P[pr, x]
                    if is_max:
                        g[x] = a if a > b else b
                    else:
                        g[x] = a if a < b else b
                c += 1
                if c == k:
                    c = 0
            # backward suffix-min/max within blocks of size k (counter tracks x % k)
            c = h_block_start
            for x in range(L - 1, -1, -1):
                if x == L - 1 or c == k - 1:
                    h[x] = P[pr, x]
                else:
                    a = h[x + 1]
                    b = P[pr, x]
                    if is_max:
                        h[x] = a if a > b else b
                    else:
                        h[x] = a if a < b else b
                c -= 1
                if c < 0:
                    c = k - 1
            # combine windowed result into the output (center c = radius + j)
            for j in range(W):
                c = radius + j
                a = h[c - w]
                b = g[c + w]
                if is_max:
                    v = a if a > b else b
                    if v > out[i, j]:
                        out[i, j] = v
                else:
                    v = a if a < b else b
                    if v < out[i, j]:
                        out[i, j] = v
    return out


def disk_erosion_2d(img: NDArray, radius: int) -> NDArray[np.float64]:
    """Greyscale disk erosion, bit-exact with ``skimage…erosion(img, disk(radius))``."""
    a = np.ascontiguousarray(img, dtype=np.float64)
    if radius <= 0:
        return a.copy()
    P = np.pad(a, radius, mode="edge")
    return _disk_reduce(P, radius, _disk_halfwidths(radius), a.shape[0], a.shape[1], False)


def disk_dilation_2d(img: NDArray, radius: int) -> NDArray[np.float64]:
    """Greyscale disk dilation, bit-exact with ``skimage…dilation(img, disk(radius))``."""
    a = np.ascontiguousarray(img, dtype=np.float64)
    if radius <= 0:
        return a.copy()
    P = np.pad(a, radius, mode="edge")
    return _disk_reduce(P, radius, _disk_halfwidths(radius), a.shape[0], a.shape[1], True)


@njit(cache=True)
def _plus_reduce(P, H, W, is_max):
    """5-tap (disk(1)/cross) min/max over a 1-pixel ``edge``-padded image ``P``."""
    out = np.empty((H, W), np.float64)
    for i in range(H):
        for j in range(W):
            v = P[i + 1, j + 1]
            u = P[i, j + 1]
            d = P[i + 2, j + 1]
            le = P[i + 1, j]
            ri = P[i + 1, j + 2]
            if is_max:
                if u > v:
                    v = u
                if d > v:
                    v = d
                if le > v:
                    v = le
                if ri > v:
                    v = ri
            else:
                if u < v:
                    v = u
                if d < v:
                    v = d
                if le < v:
                    v = le
                if ri < v:
                    v = ri
            out[i, j] = v
    return out


def erosion_4conn_2d(img: NDArray) -> NDArray[np.float64]:
    """disk(1) erosion, bit-exact with ``skimage…erosion(img, disk(1))``."""
    a = np.ascontiguousarray(img, dtype=np.float64)
    P = np.pad(a, 1, mode="edge")
    return _plus_reduce(P, a.shape[0], a.shape[1], False)


def dilation_4conn_2d(img: NDArray) -> NDArray[np.float64]:
    """disk(1) dilation, bit-exact with ``skimage…dilation(img, disk(1))``."""
    a = np.ascontiguousarray(img, dtype=np.float64)
    P = np.pad(a, 1, mode="edge")
    return _plus_reduce(P, a.shape[0], a.shape[1], True)


@njit(cache=True)
def reconstruction_by_dilation_2d(seed, mask):
    """Morphological reconstruction by dilation (4-connectivity), seed under mask.

    Vincent (1993) hybrid: one forward (TL→BR) + one backward (BR→TL) raster
    geodesic dilation under ``mask``, the backward pass seeding a FIFO of pixels
    that can still propagate, then FIFO propagation until it drains. This computes
    the exact reconstruction in ``O(N)`` (independent of propagation distance),
    unlike raster-until-convergence which costs ``O(passes·N)`` and degrades to
    hundreds of passes on full-resolution images. Bit-identical to
    ``skimage.morphology.reconstruction(seed, mask, footprint=disk(1))`` — min/max
    selections only. ``seed`` is clamped to ``min(seed, mask)``.

    The FIFO is a ring buffer with a per-pixel ``in-queue`` flag, so at most ``N``
    entries are live at once and a buffer of ``N + 1`` cannot overflow.
    """
    H, W = seed.shape
    out = np.empty((H, W), np.float64)
    for i in range(H):
        for j in range(W):
            s = seed[i, j]
            m = mask[i, j]
            out[i, j] = s if s < m else m

    # forward raster: propagate from up/left causal neighbours
    for i in range(H):
        for j in range(W):
            v = out[i, j]
            if i > 0 and out[i - 1, j] > v:
                v = out[i - 1, j]
            if j > 0 and out[i, j - 1] > v:
                v = out[i, j - 1]
            m = mask[i, j]
            if v > m:
                v = m
            out[i, j] = v

    cap = H * W + 1
    queue = np.empty(cap, np.int64)
    inq = np.zeros((H, W), np.uint8)
    head = 0
    tail = 0

    # backward raster: propagate from down/right; seed the FIFO with pixels whose
    # down/right neighbour can still grow from them
    for i in range(H - 1, -1, -1):
        for j in range(W - 1, -1, -1):
            v = out[i, j]
            if i < H - 1 and out[i + 1, j] > v:
                v = out[i + 1, j]
            if j < W - 1 and out[i, j + 1] > v:
                v = out[i, j + 1]
            m = mask[i, j]
            if v > m:
                v = m
            out[i, j] = v
            seed_p = (i < H - 1 and out[i + 1, j] < v and out[i + 1, j] < mask[i + 1, j]) or (
                j < W - 1 and out[i, j + 1] < v and out[i, j + 1] < mask[i, j + 1]
            )
            if seed_p:
                queue[tail] = i * W + j
                inq[i, j] = 1
                tail += 1
                if tail == cap:
                    tail = 0

    # FIFO propagation to all 4 neighbours until drained
    while head != tail:
        code = queue[head]
        head += 1
        if head == cap:
            head = 0
        i = code // W
        j = code % W
        inq[i, j] = 0
        v = out[i, j]
        # up, down, left, right
        if i > 0:
            mq = mask[i - 1, j]
            nv = v if v < mq else mq
            if nv > out[i - 1, j]:
                out[i - 1, j] = nv
                if inq[i - 1, j] == 0:
                    inq[i - 1, j] = 1
                    queue[tail] = (i - 1) * W + j
                    tail += 1
                    if tail == cap:
                        tail = 0
        if i < H - 1:
            mq = mask[i + 1, j]
            nv = v if v < mq else mq
            if nv > out[i + 1, j]:
                out[i + 1, j] = nv
                if inq[i + 1, j] == 0:
                    inq[i + 1, j] = 1
                    queue[tail] = (i + 1) * W + j
                    tail += 1
                    if tail == cap:
                        tail = 0
        if j > 0:
            mq = mask[i, j - 1]
            nv = v if v < mq else mq
            if nv > out[i, j - 1]:
                out[i, j - 1] = nv
                if inq[i, j - 1] == 0:
                    inq[i, j - 1] = 1
                    queue[tail] = i * W + (j - 1)
                    tail += 1
                    if tail == cap:
                        tail = 0
        if j < W - 1:
            mq = mask[i, j + 1]
            nv = v if v < mq else mq
            if nv > out[i, j + 1]:
                out[i, j + 1] = nv
                if inq[i, j + 1] == 0:
                    inq[i, j + 1] = 1
                    queue[tail] = i * W + (j + 1)
                    tail += 1
                    if tail == cap:
                        tail = 0
    return out
