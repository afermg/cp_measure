import itertools

import numpy as np
import pytest

from cp_measure.featurizer import _batch_results, _featurize_batch


@pytest.mark.parametrize("size", [1, 2, 3])
@pytest.mark.parametrize("count", [0, 6])
def test_rows_preserve_result_order_and_identity(monkeypatch, size, count):
    monkeypatch.delattr(itertools, "batched", raising=False)
    results = [{"feature": np.array([index])} for index in range(count)]
    rows = _batch_results(results, size)
    assert len(rows) == count // size
    assert all(len(row) == size for row in rows)
    for actual, expected in zip(
        itertools.chain.from_iterable(rows), results, strict=True
    ):
        assert actual is expected


@pytest.mark.parametrize("size", [2, 3])
def test_rows_reject_incomplete_backend_results(size):
    with pytest.raises(ValueError):
        _batch_results([{}] * (size + 1), size)


@pytest.mark.parametrize("size", [0, -1])
def test_rows_require_positive_width(size):
    with pytest.raises(ValueError, match="positive"):
        _batch_results([], size)


@pytest.mark.parametrize("family", ["shape", "channel", "pair"])
@pytest.mark.parametrize("extra_rows", [-2, -1, 1])
def test_measurements_require_one_result_per_requested_item(family, extra_rows):
    images = np.ones((2, 3, 4, 4))
    masks = np.ones((2, 1, 4, 4), dtype=np.int32)
    width = 1 if family == "shape" else 3

    def measure(*args, **kwargs):
        return [{"feature": np.ones(1)} for _ in range((2 + extra_rows) * width)]

    with pytest.raises((ValueError, RuntimeError)):
        _featurize_batch(
            images,
            masks,
            [0, 1],
            ["first", "second"],
            ["a", "b", "c"],
            ["object"],
            [(measure, {})] if family == "shape" else [],
            [(measure, {})] if family == "channel" else [],
            [(measure, {}, True)] if family == "pair" else [],
        )
