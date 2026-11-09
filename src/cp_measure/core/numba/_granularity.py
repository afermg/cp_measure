"""Numba 2D morphology kernels for the granularity backend.

Single-threaded ``@njit(cache=True, nogil=True)`` kernels, bit-exact with the
skimage operations the numpy baseline uses. ``nogil`` lets the batch path run images
on a thread pool (:func:`cp_measure.primitives.parallel.map_images`); every kernel
allocates its own outputs and only reads its inputs, so concurrent calls are safe.

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


@njit(cache=True, nogil=True)
def readback_tables(mask, n, scale_y, scale_x, H, W):
    """One pass over the full-resolution mask for the spectrum's per-object means.

    Each object pixel ``(y, x)`` samples the ``(H, W)`` working grid bilinearly at
    ``(y * scale_y, x * scale_x)``, as ``scipy.ndimage.map_coordinates(order=1)``
    does (samples outside the grid, even by one ULP, contribute 0). Those samples
    are fixed across the spectrum steps, and a label's sum of bilinear samples is
    linear in the grid values, so the corner weights are summed once per label on a
    dense block over its bounding box in the grid. Each step then reads only that
    block (:func:`readback_sums`) instead of every object pixel.

    Returns ``(counts, starts, cells, values)``, indexed by label ``0..n``: pixel
    counts, and label ``L``'s nonzero weights ``values[starts[L]:starts[L + 1]]`` on
    the flat grid ``cells``, in raster order. They depend only on the mask and the
    grid shape, so one set serves every channel measured with the same mask. With unit scales every weight is 1.0 at the
    pixel itself, so the per-label sums equal a ``bincount`` of the grid at the
    object pixels exactly.
    """
    Y, X = mask.shape
    counts = np.zeros(n + 1)
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
    return counts, starts, cells, values


@njit(cache=True, nogil=True)
def label_sums(mask, pixels, n):
    """Per-label sums of ``pixels`` in raster order, as ``bincount(weights=...)``."""
    sums = np.zeros(n + 1)
    Y, X = mask.shape
    for y in range(Y):
        for x in range(X):
            lab = mask[y, x]
            if lab > 0:
                sums[lab] += np.float64(pixels[y, x])
    return sums


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
def _pad1_edge(a):
    """``np.pad(a, 1, mode="edge")`` as float64."""
    H, W = a.shape
    P = np.empty((H + 2, W + 2), np.float64)
    for i in range(H + 2):
        si = min(max(i - 1, 0), H - 1)
        for j in range(W + 2):
            P[i, j] = a[si, min(max(j - 1, 0), W - 1)]
    return P


@njit(cache=True, nogil=True)
def spectrum_sums_2d(pixels, ng, starts, cells, values):
    """Readback sums of all ``ng`` spectrum steps, ``(ng, n + 1)``, in one GIL-free
    call: each step is :func:`erosion_4conn_2d` of the previous erosion, then
    :func:`reconstruction_by_dilation_2d` under the previous reconstruction (the
    first under ``pixels``), then :func:`readback_sums`. Small images spend most
    of a per-step Python loop holding the GIL, which serializes batch threads."""
    H, W = pixels.shape
    out = np.empty((ng, starts.size - 1))
    if ng == 0:
        return out
    # the first step's mask is ``pixels`` in its own dtype, the later ones float64
    ero = _plus_reduce(_pad1_edge(pixels), H, W, False)
    rec = reconstruction_by_dilation_2d(ero, pixels)
    out[0] = readback_sums(rec, starts, cells, values)
    for g in range(1, ng):
        ero = _plus_reduce(_pad1_edge(ero), H, W, False)
        rec = reconstruction_by_dilation_2d(ero, rec)
        out[g] = readback_sums(rec, starts, cells, values)
    return out


# --- 3D ------------------------------------------------------------------------
# The same morphology one dimension up: ``ball`` footprints instead of ``disk``,
# 6-connectivity instead of 4. The per-object means follow the baseline's 3D
# upsample + ``ndimage.mean`` voxel by voxel (:func:`upsample_label_sums_3d`).


@njit(cache=True, nogil=True)
def upsample_label_sums_3d(grid, mask, n, scales, round_float32):
    """Per-label sums of ``grid`` upsampled trilinearly to ``mask``'s shape, bit-exact
    with the baseline's ``map_coordinates(grid, (z*scale_z, y*scale_y, x*scale_x),
    order=1)`` (``scales`` is that ``(scale_z, scale_y, scale_x)`` tuple) followed by ``scipy.ndimage.mean``'s raster-order sums.

    Every object voxel is interpolated with scipy's arithmetic: weights ``1 - t`` and
    ``1 - (1 - t)``, each corner ``value * w_z * w_y * w_x``, corners summed in C
    order, and the sample rounded to float32 when the baseline's grid is float32
    (its ``map_coordinates`` output keeps that dtype). Samples outside the grid, even
    by one ULP, are 0 (``mode="constant"``). Unlike the 2D readback tables, this
    reads every object voxel per step; it is still a few ms per step on a
    30x256x256 volume, and it is exact where 3D's float32 rounding would otherwise
    differ.
    """
    scale_z, scale_y, scale_x = scales
    D, H, W = grid.shape
    Z, Y, X = mask.shape
    sums = np.zeros(n + 1)
    for z in range(Z):
        cz = z * scale_z
        iz = int(np.floor(cz))
        tz = cz - iz
        for y in range(Y):
            cy = y * scale_y
            iy = int(np.floor(cy))
            ty = cy - iy
            for x in range(X):
                lab = mask[z, y, x]
                if lab <= 0:
                    continue
                cx = x * scale_x
                ix = int(np.floor(cx))
                tx = cx - ix
                if (
                    iz < 0
                    or iy < 0
                    or ix < 0
                    or iz >= D
                    or iy >= H
                    or ix >= W
                    or (iz == D - 1 and tz > 0)
                    or (iy == H - 1 and ty > 0)
                    or (ix == W - 1 and tx > 0)
                ):
                    continue
                acc = 0.0
                for a in range(2):
                    wz = (1.0 - (1.0 - tz)) if a else 1.0 - tz
                    za = min(iz + a, D - 1)
                    for b in range(2):
                        wy = (1.0 - (1.0 - ty)) if b else 1.0 - ty
                        yb = min(iy + b, H - 1)
                        for c in range(2):
                            wx = (1.0 - (1.0 - tx)) if c else 1.0 - tx
                            acc += grid[za, yb, min(ix + c, W - 1)] * wz * wy * wx
                if round_float32:
                    acc = np.float64(np.float32(acc))
                sums[lab] += acc
    return sums


def _ball_halfwidths(radius: int) -> NDArray[np.int64]:
    """``hw[dz + r, dy + r]`` = x half-width of ``skimage.morphology.ball(radius)``'s
    row at offset ``(dz, dy)``, or -1 where the ball has no voxel."""
    d = np.arange(-radius, radius + 1, dtype=np.float64)
    rem = radius * radius - d[:, None] ** 2 - d[None, :] ** 2
    hw = np.full(rem.shape, -1, np.int64)
    inside = rem >= 0
    hw[inside] = np.floor(np.sqrt(rem[inside])).astype(np.int64)
    return hw


@njit(cache=True, nogil=True)
def _ball_reduce(P, radius, hw, is_max):
    """Ball min/max over an ``edge``-padded volume ``P`` of shape ``(D+2r, H+2r, W+2r)``:
    a 1-D VHG sliding min/max along x for every ``(dz, dy)`` row of the ball, as
    :func:`_disk_reduce` does per disk row."""
    D, H, W = P.shape[0] - 2 * radius, P.shape[1] - 2 * radius, P.shape[2] - 2 * radius
    L = W + 2 * radius
    out = np.full((D, H, W), -np.inf if is_max else np.inf)
    g = np.empty(L, np.float64)
    h = np.empty(L, np.float64)
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            w = hw[dz + radius, dy + radius]
            if w < 0:
                continue
            k = 2 * w + 1
            h_block_start = (L - 1) % k
            for z in range(D):
                pz = z + radius + dz
                for i in range(H):
                    pr = i + radius + dy
                    c = 0
                    for x in range(L):
                        if c == 0:
                            g[x] = P[pz, pr, x]
                        else:
                            a = g[x - 1]
                            b = P[pz, pr, x]
                            if is_max:
                                g[x] = a if a > b else b
                            else:
                                g[x] = a if a < b else b
                        c += 1
                        if c == k:
                            c = 0
                    c = h_block_start
                    for x in range(L - 1, -1, -1):
                        if x == L - 1 or c == k - 1:
                            h[x] = P[pz, pr, x]
                        else:
                            a = h[x + 1]
                            b = P[pz, pr, x]
                            if is_max:
                                h[x] = a if a > b else b
                            else:
                                h[x] = a if a < b else b
                        c -= 1
                        if c < 0:
                            c = k - 1
                    for j in range(W):
                        c = radius + j
                        a = h[c - w]
                        b = g[c + w]
                        if is_max:
                            v = a if a > b else b
                            if v > out[z, i, j]:
                                out[z, i, j] = v
                        else:
                            v = a if a < b else b
                            if v < out[z, i, j]:
                                out[z, i, j] = v
    return out


def ball_erosion_3d(img: NDArray, radius: int) -> NDArray[np.float64]:
    """Greyscale ball erosion, bit-exact with ``skimage…erosion(img, ball(radius))``."""
    a = np.ascontiguousarray(img, dtype=np.float64)
    if radius <= 0:
        return a.copy()
    P = np.pad(a, radius, mode="edge")
    return _ball_reduce(P, radius, _ball_halfwidths(radius), False)


def ball_dilation_3d(img: NDArray, radius: int) -> NDArray[np.float64]:
    """Greyscale ball dilation, bit-exact with ``skimage…dilation(img, ball(radius))``."""
    a = np.ascontiguousarray(img, dtype=np.float64)
    if radius <= 0:
        return a.copy()
    P = np.pad(a, radius, mode="edge")
    return _ball_reduce(P, radius, _ball_halfwidths(radius), True)


@njit(cache=True, nogil=True)
def _cross_min_3d(P):
    """7-tap (ball(1)) min over a 1-voxel ``edge``-padded volume ``P``."""
    D, H, W = P.shape[0] - 2, P.shape[1] - 2, P.shape[2] - 2
    out = np.empty((D, H, W), np.float64)
    for z in range(D):
        for i in range(H):
            for j in range(W):
                v = P[z + 1, i + 1, j + 1]
                for u in (
                    P[z, i + 1, j + 1],
                    P[z + 2, i + 1, j + 1],
                    P[z + 1, i, j + 1],
                    P[z + 1, i + 2, j + 1],
                    P[z + 1, i + 1, j],
                    P[z + 1, i + 1, j + 2],
                ):
                    if u < v:
                        v = u
                out[z, i, j] = v
    return out


def erosion_6conn_3d(img: NDArray) -> NDArray[np.float64]:
    """ball(1) erosion, bit-exact with ``skimage…erosion(img, ball(1))``."""
    a = np.ascontiguousarray(img, dtype=np.float64)
    return _cross_min_3d(np.pad(a, 1, mode="edge"))


@njit(cache=True, nogil=True)
def _geodesic_raster_3d(out, mask, forward):
    """One forward (or backward) geodesic-dilation raster of ``out`` under ``mask``,
    reading the three already-visited face neighbours."""
    D, H, W = out.shape
    s = -1 if forward else 1
    for zz in range(D):
        z = zz if forward else D - 1 - zz
        for ii in range(H):
            i = ii if forward else H - 1 - ii
            for jj in range(W):
                j = jj if forward else W - 1 - jj
                v = out[z, i, j]
                if 0 <= z + s < D and out[z + s, i, j] > v:
                    v = out[z + s, i, j]
                if 0 <= i + s < H and out[z, i + s, j] > v:
                    v = out[z, i + s, j]
                if 0 <= j + s < W and out[z, i, j + s] > v:
                    v = out[z, i, j + s]
                m = mask[z, i, j]
                out[z, i, j] = v if v < m else m


@njit(cache=True, nogil=True)
def reconstruction_by_dilation_3d(seed, mask):
    """Morphological reconstruction by dilation (6-connectivity), seed under mask.

    :func:`reconstruction_by_dilation_2d` one dimension up: raster passes, the last
    backward pass seeding a FIFO of voxels whose later neighbours can still grow,
    then FIFO propagation to the six face neighbours. Min/max selections only, so
    bit-identical to ``skimage.morphology.reconstruction(seed, mask, footprint=ball(1))``.
    FIFO entries are flat voxel indices in a ring buffer with an in-queue flag.
    """
    D, H, W = seed.shape
    out = np.empty((D, H, W), np.float64)
    for z in range(D):
        for i in range(H):
            for j in range(W):
                s = seed[z, i, j]
                m = mask[z, i, j]
                out[z, i, j] = s if s < m else m
    _geodesic_raster_3d(out, mask, True)
    _geodesic_raster_3d(out, mask, False)
    _geodesic_raster_3d(out, mask, True)

    N = D * H * W
    cap = N + 1
    queue = np.empty(cap, np.intp)
    inq = np.zeros(N, np.uint8)
    flat = out.ravel()  # views: the FIFO below reads and writes through them
    fmask = mask.ravel()
    head = 0
    tail = 0
    HW = H * W
    for z in range(D - 1, -1, -1):
        for i in range(H - 1, -1, -1):
            for j in range(W - 1, -1, -1):
                v = out[z, i, j]
                if z < D - 1 and out[z + 1, i, j] > v:
                    v = out[z + 1, i, j]
                if i < H - 1 and out[z, i + 1, j] > v:
                    v = out[z, i + 1, j]
                if j < W - 1 and out[z, i, j + 1] > v:
                    v = out[z, i, j + 1]
                m = mask[z, i, j]
                if v > m:
                    v = m
                out[z, i, j] = v
                if (
                    (
                        z < D - 1
                        and out[z + 1, i, j] < v
                        and out[z + 1, i, j] < mask[z + 1, i, j]
                    )
                    or (
                        i < H - 1
                        and out[z, i + 1, j] < v
                        and out[z, i + 1, j] < mask[z, i + 1, j]
                    )
                    or (
                        j < W - 1
                        and out[z, i, j + 1] < v
                        and out[z, i, j + 1] < mask[z, i, j + 1]
                    )
                ):
                    p = (z * H + i) * W + j
                    queue[tail] = p
                    inq[p] = 1
                    tail += 1
                    if tail == cap:
                        tail = 0

    while head != tail:
        p = queue[head]
        head += 1
        if head == cap:
            head = 0
        inq[p] = 0
        v = flat[p]
        j = p % W
        i = (p // W) % H
        z = p // HW
        for q, ok in (
            (p - HW, z > 0),
            (p + HW, z < D - 1),
            (p - W, i > 0),
            (p + W, i < H - 1),
            (p - 1, j > 0),
            (p + 1, j < W - 1),
        ):
            if not ok:
                continue
            mq = fmask[q]
            nv = v if v < mq else mq
            if nv > flat[q]:
                flat[q] = nv
                if inq[q] == 0:
                    inq[q] = 1
                    queue[tail] = q
                    tail += 1
                    if tail == cap:
                        tail = 0
    return out
