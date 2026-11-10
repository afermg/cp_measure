"""Numba intensity batches: every item equals its own call, from any thread."""

import subprocess
import sys

import numpy as np
import pytest
from conftest import SIZE_2D, _stamp_objects_2d, get_rng

pytest.importorskip("numba")

from cp_measure.core.numba.measureobjectintensity import get_intensity  # noqa: E402


def _batch(counts):
    rng = get_rng()
    masks, pix = [], []
    for n in counts:
        m = np.zeros((SIZE_2D, SIZE_2D), np.int32)
        if n > 0:
            _stamp_objects_2d(m, n_objects=n)
        masks.append(m)
        pix.append(rng.random((SIZE_2D, SIZE_2D)))
    return masks, pix


@pytest.mark.parametrize("legacy", [False, True], ids=["new", "legacy"])
@pytest.mark.parametrize("edge", [False, True], ids=["noedge", "edge"])
def test_list_batch_matches_single_calls(edge, legacy):
    """Distinct object counts, an empty image, and an all-non-finite image."""
    masks, pix = _batch([3, 5, 0, 2])
    pix[3] = np.full_like(pix[3], np.nan)
    batch = get_intensity(masks, pix, edge_measurements=edge, legacy=legacy)
    assert isinstance(batch, list) and len(batch) == len(masks)
    for got, m, p in zip(batch, masks, pix):
        ref = get_intensity(m, p, edge_measurements=edge, legacy=legacy)
        assert got.keys() == ref.keys()
        for key in ref:
            np.testing.assert_array_equal(got[key], ref[key], err_msg=key)


def test_bool_mask_batch_matches_single_calls():
    masks, pix = _batch([1, 1])
    masks = [m > 0 for m in masks]
    for got, m, p in zip(get_intensity(masks, pix), masks, pix):
        ref = get_intensity(m, p)
        for key in ref:
            np.testing.assert_array_equal(got[key], ref[key], err_msg=key)


def test_batches_from_python_threads_do_not_crash():
    """Intent: callers (dask, napari, thread pools) may run batches concurrently.
    A crash aborts the interpreter, so this runs in a subprocess."""
    code = """
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from cp_measure.core.numba.measureobjectintensity import get_intensity
m = np.zeros((64, 64), np.int32); m[5:20, 5:20] = 1; m[30:50, 30:60] = 2
p = np.random.default_rng(0).random((64, 64))
with ThreadPoolExecutor(4) as pool:
    list(pool.map(lambda _: get_intensity([m] * 8, [p] * 8), range(16)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize("depth", [0, 3], ids=["2d", "3d"])
def test_non_finite_pixels_match_numpy(depth):
    """Non-finite pixels are skipped as in NumPy: an object with none finite gets
    zeros, and edge features use only finite edge pixels."""
    from cp_measure.core.measureobjectintensity import get_intensity as reference

    masks, pix = _batch([3])
    mask, image = masks[0], pix[0].copy()
    if depth:
        mask = np.repeat(mask[None], depth, 0)
        # distinct planes: the backends break ties in the max location differently
        image = image[None] * np.arange(1, depth + 1)[:, None, None]
    labels = sorted(set(np.unique(mask)) - {0})
    image[mask == labels[0]] = np.nan  # an all-NaN object
    rows, cols = np.nonzero(mask[..., :, :].reshape(-1, mask.shape[-1]) == labels[1])
    flat = image.reshape(-1, image.shape[-1])
    flat[rows[0], cols[0]] = np.nan  # NaN on the boundary
    flat[rows[len(rows) // 2], cols[len(cols) // 2]] = np.inf  # +inf inside
    expected = reference(mask, image)
    single = get_intensity(mask, image)
    batch = get_intensity([mask, mask], [image, image])
    for got in (single, *batch):
        for key in expected:
            np.testing.assert_allclose(
                got[key], expected[key], rtol=1e-12, atol=1e-12, err_msg=key
            )
