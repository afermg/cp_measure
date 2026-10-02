"""Tests for the unified batch featurizer (``featurize``) and its strict contract.

The public ``featurize`` enforces ``(B, C, *spatial)`` with ``is_3d`` declaring
dimensionality. These tests use only the public API.
"""

import numpy as np
import pytest

from cp_measure.featurizer import featurize, make_featurizer_config

from conftest import ALL_OFF

CONFIG_2CH = make_featurizer_config(
    ["DNA", "ER"], **{**ALL_OFF, "intensity": True, "sizeshape": True}
)
CONFIG_1CH = make_featurizer_config(
    ["DNA"], **{**ALL_OFF, "intensity": True, "sizeshape": True}
)


# ---------------------------------------------------------------------------
# Invariant: a batch == independent single-image (B=1) calls, stacked
# ---------------------------------------------------------------------------


def test_batch_equals_stacked_individual(image_2d_2ch, mask_2d):
    """Batching must not change any item's values vs running each image alone."""
    img_a = image_2d_2ch[None]
    img_b = (image_2d_2ch + 0.1)[None]
    mask = mask_2d[None]
    batch_img = np.concatenate([img_a, img_b])
    batch_masks = np.concatenate([mask, mask])

    data, _, rows = featurize(batch_img, batch_masks, CONFIG_2CH)
    data_a, _, _ = featurize(img_a, mask, CONFIG_2CH)
    data_b, _, _ = featurize(img_b, mask, CONFIG_2CH)

    np.testing.assert_array_equal(data, np.vstack([data_a, data_b]))
    # batch index lands in image_id; 2 objects per image
    assert [r[0] for r in rows] == [0, 0, 1, 1]


# ---------------------------------------------------------------------------
# is_3d declares dimensionality (no shape inference)
# ---------------------------------------------------------------------------


def test_is_3d_true_runs(image_3d_1ch, mask_3d):
    data, cols, rows = featurize(
        image_3d_1ch[None], mask_3d[None], CONFIG_1CH, is_3d=True
    )
    assert data.shape[0] == len(rows) == 2  # 2 objects
    assert data.shape[1] == len(cols)


# ---------------------------------------------------------------------------
# Strict shape / rank validation (hard errors, declared not inferred)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image, masks, is_3d, match",
    [
        # sub-canonical rank: bare 2-D / 3-D arrays are rejected
        (
            np.ones((10, 10)),
            np.ones((1, 10, 10), dtype=np.int32),
            False,
            "image must be",
        ),
        (
            np.ones((1, 10, 10)),
            np.ones((1, 10, 10), dtype=np.int32),
            False,
            "image must be",
        ),
        # image valid, masks rank disagrees
        (
            np.ones((1, 2, 10, 10)),
            np.ones((1, 1, 4, 10, 10), dtype=np.int32),
            False,
            "masks must be",
        ),
        # rank / is_3d mismatch, both directions
        (
            np.ones((1, 2, 10, 10)),
            np.ones((1, 1, 10, 10), dtype=np.int32),
            True,
            "is_3d=True",
        ),
        (
            np.ones((1, 1, 4, 10, 10)),
            np.ones((1, 1, 4, 10, 10), dtype=np.int32),
            False,
            "is_3d=False",
        ),
    ],
)
def test_canonical_rank_errors(image, masks, is_3d, match):
    with pytest.raises(ValueError, match=match):
        featurize(image, masks, is_3d=is_3d)


@pytest.mark.parametrize(
    "image, masks, exc, match",
    [
        (
            np.ones((2, 2, 16, 16)),
            np.ones((1, 1, 16, 16), dtype=np.int32),
            ValueError,
            "batch size mismatch",
        ),
        (
            np.ones((1, 2, 16, 16)),
            np.ones((1, 1, 8, 8), dtype=np.int32),
            ValueError,
            "spatial dims mismatch",
        ),
        (
            np.ones((1, 2, 16, 16)),
            np.ones((1, 1, 16, 16), dtype=float),
            TypeError,
            "integer dtype",
        ),
    ],
)
def test_canonical_validation_errors(image, masks, exc, match):
    with pytest.raises(exc, match=match):
        featurize(image, masks, CONFIG_2CH)


# ---------------------------------------------------------------------------
# Empty / invalid mask stacks
# ---------------------------------------------------------------------------


def test_empty_image_skipped_and_reported(image_2d_2ch, mask_2d):
    empty = np.zeros_like(mask_2d)
    image = np.stack([image_2d_2ch, image_2d_2ch, image_2d_2ch])  # (3, C, Y, X)
    masks = np.stack([mask_2d, empty, mask_2d])  # (3, M, Y, X), middle empty

    with pytest.warns(UserWarning, match=r"1 of 3 image\(s\) had no labelled objects"):
        data, cols, rows = featurize(image, masks, CONFIG_2CH)

    # only items 0 and 2 contribute; batch indices preserved (no renumbering)
    assert [r[0] for r in rows] == [0, 0, 2, 2]
    assert data.shape[0] == 4


def test_all_empty_raises_without_warning(mask_2d, image_2d_2ch):
    """All-empty batch raises; the skip-warning must not pre-empt it under -W error."""
    empty = np.zeros_like(mask_2d)
    with pytest.raises(ValueError, match="no images had labelled objects"):
        featurize(image_2d_2ch[None], empty[None], CONFIG_2CH)


def test_negative_labels_raise_not_skipped(image_2d_2ch, mask_2d):
    """An all-negative mask stack is surfaced (contract), not silently skipped as empty."""
    neg = np.where(mask_2d > 0, -mask_2d, 0)
    with pytest.raises(ValueError, match="non-negative"):
        featurize(image_2d_2ch[None], neg[None], CONFIG_2CH)


# ---------------------------------------------------------------------------
# image_ids
# ---------------------------------------------------------------------------


def test_image_ids_propagate_and_validate(image_2d_2ch, mask_2d):
    image = np.stack([image_2d_2ch, image_2d_2ch])
    masks = np.stack([mask_2d, mask_2d])
    _, _, rows = featurize(image, masks, CONFIG_2CH, image_ids=["A01", "A02"])
    assert [r[0] for r in rows] == ["A01", "A01", "A02", "A02"]

    with pytest.raises(ValueError, match="image_ids has"):
        featurize(image, masks, CONFIG_2CH, image_ids=["only_one"])
