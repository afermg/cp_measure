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
  (4-connectivity) via the Vincent (1993) FIFO-hybrid, matching
  ``skimage.morphology.reconstruction(seed, mask, footprint=disk(1))``.

Borders match skimage's footprint ops via ``edge`` (clamp) padding applied on the
host before the kernel; min/max selection makes the result border-mode-insensitive
for disks (verified bit-exact vs skimage over reflect/symmetric/edge).
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(cache=True)
def readback_tables(mask, pixels, n, scale_y, scale_x, H, W):
    """One pass over the full-resolution mask for the spectrum's per-object means.

    Each object pixel ``(y, x)`` samples the ``(H, W)`` working grid bilinearly at
    ``(y * scale_y, x * scale_x)``, as ``scipy.ndimage.map_coordinates(order=1)``
    does (samples outside the grid, even by one ULP, contribute 0). Those samples
    are fixed across the spectrum steps, and a label's sum of bilinear samples is
    linear in the grid values, so the corner weights are summed once per label on a
    dense block over its bounding box in the grid. Each step then reads only that
    block (:func:`readback_sums`) instead of every object pixel.

    Returns ``(counts, sums, starts, cells, values)``, indexed by label ``0..n``:
    pixel counts, sums of ``pixels`` in raster order (as a weighted ``bincount``),
    and label ``L``'s nonzero weights ``values[starts[L]:starts[L + 1]]`` on the flat
    grid ``cells``, in raster order. With unit scales every weight is 1.0 at the
    pixel itself, so the per-label sums equal a ``bincount`` of the grid at the
    object pixels exactly.
    """
    Y, X = mask.shape
    counts = np.zeros(n + 1)
    sums = np.zeros(n + 1)
    row0 = np.full(n + 1, H, np.int64)
    col0 = np.full(n + 1, W, np.int64)
    row1 = np.full(n + 1, -1, np.int64)
    col1 = np.full(n + 1, -1, np.int64)
    for y in range(Y):
        iy = int(np.floor(y * scale_y))
        for x in range(X):
            lab = mask[y, x]
            if lab <= 0:
                continue
            counts[lab] += 1
            sums[lab] += np.float64(pixels[y, x])
            ix = int(np.floor(x * scale_x))
            row0[lab] = min(row0[lab], iy)
            col0[lab] = min(col0[lab], ix)
            row1[lab] = max(row1[lab], iy + 1)
            col1[lab] = max(col1[lab], ix + 1)
    rows = np.zeros(n + 1, np.int64)
    cols = np.zeros(n + 1, np.int64)
    offsets = np.zeros(n + 2, np.int64)
    for lab in range(n + 1):
        if row1[lab] >= 0:
            rows[lab] = row1[lab] - row0[lab] + 1
            cols[lab] = col1[lab] - col0[lab] + 1
        offsets[lab + 1] = offsets[lab] + rows[lab] * cols[lab]
    weights = np.zeros(offsets[n + 1])
    for y in range(Y):
        sy = y * scale_y
        iy = int(np.floor(sy))
        gy = sy - iy
        for x in range(X):
            lab = mask[y, x]
            if lab <= 0:
                continue
            sx = x * scale_x
            ix = int(np.floor(sx))
            gx = sx - ix
            # Test floor/frac separately so even a one-ULP overshoot is rejected.
            if (
                iy < 0
                or ix < 0
                or iy >= H
                or ix >= W
                or (iy == H - 1 and gy > 0)
                or (ix == W - 1 and gx > 0)
            ):
                continue
            k = offsets[lab] + (iy - row0[lab]) * cols[lab] + (ix - col0[lab])
            weights[k] += (1.0 - gx) * (1.0 - gy)
            if ix + 1 < W:
                weights[k + 1] += gx * (1.0 - gy)
            if iy + 1 < H:
                weights[k + cols[lab]] += (1.0 - gx) * gy
                if ix + 1 < W:
                    weights[k + cols[lab] + 1] += gx * gy
    # Keep only the nonzero weights, per label in raster order of the grid.
    nnz = 0
    for k in range(weights.size):
        if weights[k] != 0.0:
            nnz += 1
    starts = np.zeros(n + 2, np.int64)
    cells = np.empty(nnz, np.int64)
    values = np.empty(nnz)
    j = 0
    for lab in range(n + 1):
        starts[lab] = j
        k = offsets[lab]
        for a in range(rows[lab]):
            for b in range(cols[lab]):
                if weights[k] != 0.0:
                    cells[j] = (row0[lab] + a) * W + col0[lab] + b
                    values[j] = weights[k]
                    j += 1
                k += 1
    starts[n + 1] = j
    return counts, sums, starts, cells, values


@njit(cache=True)
def readback_sums(grid, starts, cells, values):
    """Per-label sums of the bilinear samples of ``grid``, from :func:`readback_tables`."""
    flat = grid.ravel()
    n = starts.size - 2
    out = np.zeros(n + 1)
    for lab in range(1, n + 1):
        acc = 0.0
        for j in range(starts[lab], starts[lab + 1]):
            acc += values[j] * flat[cells[j]]
        out[lab] = acc
    return out


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
        h_block_start = (
            L - 1
        ) % k  # x % k of the last element (one modulo per row-band)
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
    return _disk_reduce(
        P, radius, _disk_halfwidths(radius), a.shape[0], a.shape[1], False
    )


def disk_dilation_2d(img: NDArray, radius: int) -> NDArray[np.float64]:
    """Greyscale disk dilation, bit-exact with ``skimage…dilation(img, disk(radius))``."""
    a = np.ascontiguousarray(img, dtype=np.float64)
    if radius <= 0:
        return a.copy()
    P = np.pad(a, radius, mode="edge")
    return _disk_reduce(
        P, radius, _disk_halfwidths(radius), a.shape[0], a.shape[1], True
    )


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
def _geodesic_raster_fwd(out, mask):
    """One forward (TL→BR) geodesic-dilation raster of ``out`` under ``mask``."""
    H, W = out.shape
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


@njit(cache=True)
def _geodesic_raster_bwd(out, mask):
    """One backward (BR→TL) geodesic-dilation raster of ``out`` under ``mask``."""
    H, W = out.shape
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


@njit(cache=True)
def reconstruction_by_dilation_2d(seed, mask):
    """Morphological reconstruction by dilation (4-connectivity), seed under mask.

    Vincent (1993) / Robinson (2004) hybrid: two forward+backward raster
    geodesic-dilation passes under ``mask``, the last backward pass seeding a FIFO
    of pixels that can still propagate, then FIFO propagation until it drains. This
    computes the exact reconstruction in ``O(N)`` (independent of propagation
    distance), unlike raster-until-convergence which costs ``O(passes·N)`` and
    degrades to hundreds of passes on full-resolution images. Bit-identical to
    ``skimage.morphology.reconstruction(seed, mask, footprint=disk(1))`` — min/max
    selections only. ``seed`` is clamped to ``min(seed, mask)``.

    Two raster pairs precede the FIFO (the result is independent of how many, so
    this is purely a speed choice). Measured over the 16 spectrum steps: on real
    microscopy (skimage cells3d / human_mitosis) two pairs are 1.25-1.4x faster than
    three, and on synthetic Gaussian noise they are never slower; one pair is fastest
    on smooth real images but up to 2x slower on fine noise.

    FIFO entries use pointer-sized integers packed as ``(i << shift) | j``, with
    ``shift`` chosen from the image width.
    Shift/mask decoding avoids int-div/mod without imposing signed 16-bit coordinate
    limits. The FIFO is a ring buffer with a per-pixel ``in-queue`` flag, so at most
    ``N`` entries are live at once and a buffer of ``N + 1`` cannot overflow.
    """
    H, W = seed.shape
    out = np.empty((H, W), np.float64)
    for i in range(H):
        for j in range(W):
            s = seed[i, j]
            m = mask[i, j]
            out[i, j] = s if s < m else m

    # 2 raster pairs before FIFO seeding; the first pair does not seed
    _geodesic_raster_fwd(out, mask)
    _geodesic_raster_bwd(out, mask)
    _geodesic_raster_fwd(out, mask)

    cap = H * W + 1
    queue = np.empty(cap, np.intp)
    inq = np.zeros((H, W), np.uint8)
    shift = 0
    while (1 << shift) < W:
        shift += 1
    column_mask = (1 << shift) - 1
    head = 0
    tail = 0

    # final backward raster: propagate from down/right; seed the FIFO with pixels
    # whose down/right neighbour can still grow from them
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
            seed_p = (
                i < H - 1 and out[i + 1, j] < v and out[i + 1, j] < mask[i + 1, j]
            ) or (j < W - 1 and out[i, j + 1] < v and out[i, j + 1] < mask[i, j + 1])
            if seed_p:
                queue[tail] = (i << shift) | j
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
        i = code >> shift
        j = code & column_mask
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
                    queue[tail] = ((i - 1) << shift) | j
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
                    queue[tail] = ((i + 1) << shift) | j
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
                    queue[tail] = (i << shift) | (j - 1)
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
                    queue[tail] = (i << shift) | (j + 1)
                    tail += 1
                    if tail == cap:
                        tail = 0
    return out
