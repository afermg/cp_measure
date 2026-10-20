"""Boolean labels retain their integer 0/1 interpretation on NumPy 1 and 2."""

import numpy as np
import pytest

from cp_measure.utils import masks_to_ijv


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.int32, np.int64])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_masks_to_ijv_single_object(dtype, empty, strided):
    mask = np.zeros((8, 10), dtype=dtype)
    if not empty:
        mask[1:5, 3:8] = 1
    if strided:
        mask = mask[::-1, ::2]
    original = mask.copy()
    rows, cols = np.nonzero(mask)
    expected = np.column_stack((rows, cols, np.ones(len(rows), dtype=int)))
    actual = masks_to_ijv(mask)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(mask, original)


def test_masks_to_ijv_retains_label_order_and_ids():
    mask = np.array([[0, 7, 1], [1, 7, 0]], dtype=np.int32)
    expected = np.array([[0, 2, 1], [1, 0, 1], [0, 1, 7], [1, 1, 7]])
    np.testing.assert_array_equal(masks_to_ijv(mask), expected)
