"""Validate intensity batch structure before unchecked Numba indexing."""

import numpy as np
import pytest

pytest.importorskip("numba")
from cp_measure.core.numba import measureobjectintensity as accelerated
from cp_measure.core.measureobjectintensity import get_intensity as reference


@pytest.mark.parametrize("kind", ["single", "list", "tuple", "stacked"])
@pytest.mark.parametrize("edge_measurements", [False, True])
@pytest.mark.parametrize(
    "shapes", [((6, 7), (1, 6, 7)), ((1, 6, 7), (2, 6, 7)), ((6, 7), (6, 8))]
)
def test_mismatched_shapes_rejected_before_kernel(
    monkeypatch, kind, edge_measurements, shapes
):
    masks = np.ones(shapes[0], dtype=np.int32)
    pixels = np.ones(shapes[1])
    if kind in ("list", "tuple"):
        container = list if kind == "list" else tuple
        masks, pixels = container([masks]), container([pixels])
    elif kind == "stacked":
        # Keep both inputs explicitly batched even for original 2D shapes.
        masks, pixels = (
            masks.reshape((1, -1, *masks.shape[-2:])),
            pixels.reshape((1, -1, *pixels.shape[-2:])),
        )
        if masks.shape == pixels.shape:
            pixels = np.repeat(pixels, 2, axis=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid inputs reached unchecked volume kernel")

    monkeypatch.setattr(accelerated, "_intensity_volume", forbidden)
    with pytest.raises(ValueError, match="matching shapes"):
        accelerated.get_intensity(masks, pixels, edge_measurements=edge_measurements)


@pytest.mark.parametrize("shape", [(6, 7), (1, 6, 7), (2, 6, 7)])
@pytest.mark.parametrize("kind", ["list", "tuple", "stacked"])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("edge_measurements", [False, True])
def test_distinct_readonly_batch_matches_single_reference(
    shape, kind, legacy, edge_measurements
):
    rng = np.random.default_rng(57127)
    masks = [np.ones(shape, dtype=np.int32) for _ in range(2)]
    for mask in masks:
        mask[..., 3:] = 2
    pixels = [rng.random(shape), rng.random(shape) * 2 + 0.1]
    if kind == "stacked":
        masks = np.stack([m.reshape((-1, *shape[-2:])) for m in masks])
        pixels = np.stack([p.reshape((-1, *shape[-2:])) for p in pixels])
    elif kind == "tuple":
        masks, pixels = tuple(masks), tuple(pixels)
    originals = [(m.copy(), p.copy()) for m, p in zip(masks, pixels)]
    for m, p in zip(masks, pixels):
        m.setflags(write=False)
        p.setflags(write=False)
    kwargs = dict(legacy=legacy, edge_measurements=edge_measurements)
    actual = accelerated.get_intensity(masks, pixels, **kwargs)
    assert isinstance(actual, list) and len(actual) == 2
    for got, (m, p), before in zip(actual, zip(masks, pixels), originals):
        expected = reference(m, p, **kwargs)
        assert list(got) == list(expected)
        for key in expected:
            assert got[key].shape == expected[key].shape
            assert got[key].dtype == expected[key].dtype
            np.testing.assert_allclose(got[key], expected[key], rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(m, before[0])
        np.testing.assert_array_equal(p, before[1])


@pytest.mark.parametrize("kind", ["list", "tuple", "stacked"])
def test_batch_normalization_keeps_views(monkeypatch, kind):
    masks = [np.ones((1, 6, 7), dtype=np.int32) for _ in range(2)]
    pixels = [np.full((1, 6, 7), float(i)) for i in range(2)]
    if kind == "stacked":
        masks, pixels = np.stack(masks), np.stack(pixels)
    elif kind == "tuple":
        masks, pixels = tuple(masks), tuple(pixels)
    seen = []

    def volume(mask, image, *args):
        i = len(seen)
        assert np.shares_memory(mask, masks[i])
        assert np.shares_memory(image, pixels[i])
        seen.append(i)
        return {"index": np.array([i])}

    monkeypatch.setattr(accelerated, "_intensity_volume", volume)
    result = accelerated.get_intensity(masks, pixels)
    assert seen == [0, 1] and len(result) == 2


@pytest.mark.parametrize("kind", ["mixed", "count", "empty", "ragged"])
def test_invalid_batch_structure(kind):
    m, p = np.ones((6, 7), dtype=np.int32), np.ones((6, 7))
    pairs = {
        "mixed": ([m], p),
        "count": ([m, m], [p]),
        "empty": ([], []),
        "ragged": ([m, m[:3]], [p, p[:3]]),
    }
    with pytest.raises(ValueError):
        accelerated.get_intensity(*pairs[kind])
