"""Precision is opt-in, validated, and carried by serializable configurations."""

import json
from pathlib import Path

import numpy as np
import pytest

import cp_measure
from cp_measure.featurizer import featurize, make_featurizer_config
from conftest import ALL_OFF


def modules(backend):
    if backend == "numba":
        pytest.importorskip("numba")
        from cp_measure.core.numba import measurecolocalization, measuregranularity
    else:
        from cp_measure.core import measurecolocalization, measuregranularity
    return measurecolocalization, measuregranularity


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("metric", ["manders_fold", "overlap", "rwc", "granularity"])
def test_invalid_precision_rejected_even_without_objects(backend, metric):
    coloc, gran = modules(backend)
    masks = np.zeros((8, 8), np.int32)
    pixels = np.ones(masks.shape, np.float32)
    fn = (
        gran.get_granularity
        if metric == "granularity"
        else getattr(coloc, "get_correlation_" + metric)
    )
    args = (masks, pixels) if metric == "granularity" else (pixels, pixels, masks)
    with pytest.raises(ValueError, match="precision"):
        fn(*args, precision="float32")


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("precision", [None, "input", "float64"])
def test_featurizer_precision_forwarding_and_config_roundtrip(
    monkeypatch, backend, precision
):
    coloc, gran = modules(backend)
    monkeypatch.setattr(
        cp_measure, "_ACCELERATOR", None if backend == "numpy" else "numba"
    )
    pixels = np.random.default_rng(51).random((2, 40, 40)).astype(np.float32)
    masks = np.ones((40, 40), np.int32)
    pixels.setflags(write=False)
    masks.setflags(write=False)
    params = {} if precision is None else {"precision": precision}
    gran_params = {"granular_spectrum_length": 2, **params}
    config = make_featurizer_config(
        ["A", "B"],
        **{
            **ALL_OFF,
            "granularity": True,
            "correlation_manders_fold": True,
            "correlation_rwc": True,
        },
        granularity_params=gran_params,
        correlation_manders_fold_params=params,
        correlation_rwc_params=params,
    )
    config = json.loads(json.dumps(config))
    for key in [
        "granularity_params",
        "correlation_manders_fold_params",
        "correlation_rwc_params",
    ]:
        assert config[key]["precision"] == (precision or "input")
    assert params == ({} if precision is None else {"precision": precision})
    assert gran_params == {"granular_spectrum_length": 2, **params}
    values, columns, rows = featurize(pixels[None], masks[None, None], config)
    expected = {}
    for channel, p in zip(["A", "B"], pixels):
        expected.update(
            {
                f"{k}__{channel}": v
                for k, v in gran.get_granularity(masks, p, **gran_params).items()
            }
        )
    for metric in ["manders_fold", "rwc"]:
        result = getattr(coloc, "get_correlation_" + metric)(
            pixels[0], pixels[1], masks, **params
        )
        expected.update({f"{k}__A__B": v for k, v in result.items()})
    assert columns == list(expected)
    assert len(rows) == 1
    for i, expected_values in enumerate(expected.values()):
        np.testing.assert_array_equal(values[:, i], expected_values)


@pytest.mark.parametrize("precision", ["input", "float64"])
@pytest.mark.parametrize("as_array", [False, True])
def test_numba_colocalization_precision_batches(precision, as_array):
    nb, _ = modules("numba")
    ref, _ = modules("numpy")
    a = np.array([[[3, 20, 10, 0]], [[4, 6, 2, 8]]], np.float32) / np.float32(65535)
    b = np.array([[[20, 3, 10, 0]], [[2, 5, 7, 11]]], np.float32) / np.float32(65535)
    masks = np.array([[[1, 1, 1, 1]], [[1, 1, 2, 2]]], np.int32)
    args = [x[:, None] if as_array else list(x) for x in (a, b, masks)]
    for metric in ["manders_fold", "overlap", "rwc"]:
        name = "get_correlation_" + metric
        actual = getattr(nb, name)(*args, precision=precision)
        assert len(actual) == 2
        for i, result in enumerate(actual):
            expected = getattr(ref, name)(a[i], b[i], masks[i], precision=precision)
            assert list(result) == list(expected)
            for key in expected:
                np.testing.assert_allclose(
                    result[key], expected[key], rtol=1e-12, atol=1e-12
                )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_input_granularity_preserves_pre_policy_results(backend):
    _, gran = modules(backend)
    reference = json.loads(
        (
            Path(__file__).parent / "data/granularity_input_precision_reference.json"
        ).read_text()
    )
    pixels = np.asarray(reference["pixels"], dtype=np.float32) / np.float32(65535)
    masks = np.ones(pixels.shape, np.int32)
    for case in reference["cases"]:
        for kwargs in [{}, {"precision": "input"}]:
            result = gran.get_granularity(masks, pixels, **case["params"], **kwargs)
            assert list(result) == list(case["expected"][backend])
            for key, values in result.items():
                np.testing.assert_allclose(
                    values, case["expected"][backend][key], rtol=1e-12, atol=1e-12
                )
