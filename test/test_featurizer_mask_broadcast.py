"""2D masks for 3D images must behave like explicitly repeated 3D masks."""

import numpy as np
import pytest

import cp_measure
from cp_measure._detect import HAS_NUMBA
from cp_measure.featurizer import featurize, make_featurizer_config

from conftest import ALL_OFF
from make_featurizer_reference import make_reference_case


@pytest.fixture(
    params=[
        None,
        pytest.param(
            "numba",
            marks=pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed"),
        ),
    ],
    ids=["numpy", "numba"],
)
def backend(request):
    previous = cp_measure._ACCELERATOR
    cp_measure.set_accelerator(request.param)
    try:
        yield request.param
    finally:
        cp_measure.set_accelerator(previous)


@pytest.mark.parametrize("gapped_labels", [False, True], ids=["contiguous", "gapped"])
def test_broadcast_matches_repeated_volumes_and_singletons(backend, gapped_labels):
    """Cover distinct volumes, two channels/mask types, and varying object counts."""
    image, volume_masks, config = make_reference_case(is_3d=True)
    masks = volume_masks[:, :, 2].copy()
    if not gapped_labels:
        for batch_masks in masks:
            for mask in batch_masks:
                mask[:] = np.searchsorted(np.unique(mask), mask)
    # Exercise the other 3D-capable core measurements too. Costes is disabled.
    config.update(
        texture=True,
        granularity=True,
        granularity_params={"granular_spectrum_length": 2},
    )
    expected_rows = [
        (image_id, object_name, int(label))
        for image_id, batch_masks in zip(["A01", "A02"], masks)
        for object_name, mask in zip(["nuclei", "cells"], batch_masks)
        for label in np.unique(mask)
        if label != 0
    ]
    image_before, masks_before = image.copy(), masks.copy()
    # Callers may supply read-only arrays; broadcasting must not write through them.
    image.setflags(write=False)
    masks.setflags(write=False)
    repeated = np.repeat(masks[:, :, None], image.shape[2], axis=2)
    expected = featurize(image, repeated, config, is_3d=True, image_ids=["A01", "A02"])
    data, columns, rows = featurize(
        image, masks, config, is_3d=True, image_ids=["A01", "A02"]
    )
    assert columns == expected[1]
    assert rows == expected[2] == expected_rows
    np.testing.assert_allclose(
        data, expected[0], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    parts = [
        featurize(
            image[b : b + 1], masks[b : b + 1], config, is_3d=True, image_ids=[image_id]
        )
        for b, image_id in enumerate(["A01", "A02"])
    ]
    assert all(part[1] == columns for part in parts)
    assert rows == [row for part in parts for row in part[2]]
    np.testing.assert_allclose(
        data,
        np.vstack([part[0] for part in parts]),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(masks, masks_before)


@pytest.mark.parametrize("depth", [1, 3])
def test_broadcast_measures_all_slices_as_one_object(backend, depth):
    """Slice intensities differ, so measuring only one plane cannot pass."""
    image = np.broadcast_to(
        np.arange(1, depth + 1)[None, None, :, None, None], (1, 1, depth, 8, 8)
    ).astype(float)
    masks = np.zeros((1, 1, 8, 8), dtype=np.int32)
    masks[0, 0, 2:6, 2:6] = 7
    config = make_featurizer_config(["DNA"], **{**ALL_OFF, "intensity": True})
    data, columns, rows = featurize(image, masks, config, is_3d=True)
    assert rows == [(0, "object", 7)]
    assert data[0, columns.index("Intensity_MeanIntensity__DNA")] == pytest.approx(
        (depth + 1) / 2
    )
    assert data[
        0, columns.index("Intensity_IntegratedIntensity__DNA")
    ] == pytest.approx(16 * sum(range(1, depth + 1)))


def test_broadcast_empty_images_are_skipped_and_ids_preserved():
    image, volume_masks, config = make_reference_case(is_3d=True)
    masks = volume_masks[:, :, 2].copy()
    image = np.concatenate([image[:1], image[:1], image[1:]])
    masks = np.concatenate([masks[:1], np.zeros_like(masks[:1]), masks[1:]])
    with pytest.warns(UserWarning, match=r"1 of 3 image\(s\).*skipped"):
        data, columns, rows = featurize(image, masks, config, is_3d=True)
    assert [row[0] for row in rows] == [0, 0, 0, 2, 2, 2]
    assert data.shape == (6, len(columns))
    with pytest.raises(ValueError, match="all batch items were empty"):
        featurize(image, np.zeros_like(masks), config, is_3d=True)


@pytest.mark.parametrize(
    "masks, error, match",
    [
        (np.ones((1, 1, 8, 8), dtype=np.int32), ValueError, "batch size mismatch"),
        (np.ones((2, 1, 1, 8), dtype=np.int32), ValueError, "spatial dims mismatch"),
        (np.ones((2, 1, 8, 1), dtype=np.int32), ValueError, "spatial dims mismatch"),
        (np.ones((2, 1, 1, 8, 8), dtype=np.int32), ValueError, "spatial dims mismatch"),
        (np.ones((2, 8, 8), dtype=np.int32), ValueError, "masks must be"),
        (np.ones((2, 1, 8, 8), dtype=float), TypeError, "integer dtype"),
        (-np.ones((2, 1, 8, 8), dtype=np.int32), ValueError, "non-negative"),
    ],
)
def test_broadcast_does_not_relax_other_input_requirements(masks, error, match):
    image = np.ones((2, 1, 3, 8, 8))
    config = make_featurizer_config(["DNA"], **{**ALL_OFF, "intensity": True})
    with pytest.raises(error, match=match):
        featurize(image, masks, config, is_3d=True)
