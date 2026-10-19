"""3D masks must be explicit; repeating 2D labels is the caller's decision."""

import numpy as np
import pytest

import cp_measure
from cp_measure._detect import HAS_NUMBA
from cp_measure.featurizer import featurize, make_featurizer_config

from conftest import ALL_OFF


@pytest.mark.parametrize(
    "accelerator",
    [
        None,
        pytest.param(
            "numba",
            marks=pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed"),
        ),
    ],
    ids=["numpy", "numba"],
)
@pytest.mark.parametrize("depth", [1, 3])
def test_2d_masks_require_explicit_repetition(monkeypatch, accelerator, depth):
    """Reject 2D masks even for Z=1; caller-repeated labels span the full volume."""
    monkeypatch.setattr(cp_measure, "_ACCELERATOR", accelerator)
    image = np.broadcast_to(
        np.arange(1, depth + 1)[None, None, :, None, None], (1, 1, depth, 8, 8)
    ).astype(float)
    masks = np.zeros((1, 1, 8, 8), dtype=np.int32)
    masks[0, 0, 2:6, 2:6] = 7
    config = make_featurizer_config(["DNA"], **{**ALL_OFF, "intensity": True})
    with pytest.raises(ValueError, match=r"masks must be.*is_3d=True"):
        featurize(image, masks, config, is_3d=True)

    repeated = np.repeat(masks[:, :, None], depth, axis=2)
    image_before, masks_before = image.copy(), repeated.copy()
    data, columns, rows = featurize(image, repeated, config, is_3d=True)
    assert rows == [(0, "object", 7)]
    assert data[0, columns.index("Intensity_MeanIntensity__DNA")] == pytest.approx(
        (depth + 1) / 2
    )
    assert data[
        0, columns.index("Intensity_IntegratedIntensity__DNA")
    ] == pytest.approx(16 * sum(range(1, depth + 1)))
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(repeated, masks_before)


@pytest.mark.parametrize(
    "masks, error, match",
    [
        (np.ones((1, 1, 3, 8, 8), dtype=np.int32), ValueError, "batch size mismatch"),
        (np.ones((2, 1, 3, 1, 8), dtype=np.int32), ValueError, "spatial dims mismatch"),
        (np.ones((2, 1, 3, 8, 1), dtype=np.int32), ValueError, "spatial dims mismatch"),
        (np.ones((2, 1, 1, 8, 8), dtype=np.int32), ValueError, "spatial dims mismatch"),
        (np.ones((2, 8, 8), dtype=np.int32), ValueError, "masks must be"),
        (np.ones((2, 1, 3, 8, 8), dtype=float), TypeError, "integer dtype"),
        (-np.ones((2, 1, 3, 8, 8), dtype=np.int32), ValueError, "non-negative"),
    ],
)
def test_3d_masks_require_matching_shapes_and_integer_labels(masks, error, match):
    image = np.ones((2, 1, 3, 8, 8))
    config = make_featurizer_config(["DNA"], **{**ALL_OFF, "intensity": True})
    with pytest.raises(error, match=match):
        featurize(image, masks, config, is_3d=True)
