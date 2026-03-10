"""Cross-validation and unit tests for the Numba intensity implementation."""

import time

import numpy as np
import pytest

numba = pytest.importorskip("numba")

from cp_measure.core._intensity_numba import get_intensity_numba  # noqa: E402
from cp_measure.core.measureobjectintensity import (  # noqa: E402
    _get_intensity_python,
    get_intensity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_random_scene(rng, height=200, width=200, n_objects=7):
    """Create non-overlapping rectangular objects with random float64 pixels."""
    masks = np.zeros((height, width), dtype=np.int32)
    pixels = rng.random((height, width))

    # Place non-overlapping rectangles
    label = 1
    row = 5
    for _ in range(n_objects):
        h = rng.randint(8, 25)
        w = rng.randint(8, 25)
        col = rng.randint(5, width - w - 5)
        if row + h >= height - 5:
            break
        masks[row : row + h, col : col + w] = label
        label += 1
        row += h + 5  # gap between objects

    return masks, pixels


def _compare_results(result_numba, result_python, rtol=1e-10, atol=1e-12):
    """Assert all feature arrays match (values + NaN positions)."""
    assert set(result_numba.keys()) == set(result_python.keys()), (
        f"Key mismatch: {set(result_numba.keys()) ^ set(result_python.keys())}"
    )
    for key in result_numba:
        a = result_numba[key]
        b = result_python[key]
        assert len(a) == len(b), f"{key}: length {len(a)} != {len(b)}"
        nan_a = np.isnan(a)
        nan_b = np.isnan(b)
        np.testing.assert_array_equal(
            nan_a, nan_b, err_msg=f"{key}: NaN positions differ"
        )
        finite = ~nan_a
        if finite.any():
            np.testing.assert_allclose(
                a[finite],
                b[finite],
                rtol=rtol,
                atol=atol,
                err_msg=f"{key}: values differ",
            )


# ---------------------------------------------------------------------------
# Cross-validation against Python reference
# ---------------------------------------------------------------------------


class TestCrossValidation:
    """Compare Numba output against the pure-Python reference."""

    def test_random_scene(self):
        """Multiple non-overlapping objects, random intensities."""
        rng = np.random.RandomState(42)
        masks, pixels = _make_random_scene(rng, n_objects=7)

        result_numba = get_intensity_numba(masks, pixels)
        result_python = _get_intensity_python(masks, pixels)
        _compare_results(result_numba, result_python)

    def test_random_scene_varied_sizes(self):
        """Second seed with different object count."""
        rng = np.random.RandomState(123)
        masks, pixels = _make_random_scene(rng, height=300, width=300, n_objects=10)

        result_numba = get_intensity_numba(masks, pixels)
        result_python = _get_intensity_python(masks, pixels)
        _compare_results(result_numba, result_python)

    def test_elliptical_objects(self):
        """Non-rectangular objects exercise 8-connected edge detection."""
        rng = np.random.RandomState(99)
        height, width = 200, 200
        masks = np.zeros((height, width), dtype=np.int32)
        pixels = rng.random((height, width))

        # Draw circles (non-rectangular) to expose 4- vs 8-connected diffs
        for label, (cy, cx, r) in enumerate(
            [(30, 30, 12), (80, 80, 15), (140, 140, 10)], start=1
        ):
            for y in range(max(0, cy - r), min(height, cy + r + 1)):
                for x in range(max(0, cx - r), min(width, cx + r + 1)):
                    if (y - cy) ** 2 + (x - cx) ** 2 <= r**2:
                        masks[y, x] = label

        result_numba = get_intensity_numba(masks, pixels)
        result_python = _get_intensity_python(masks, pixels)
        _compare_results(result_numba, result_python)

    def test_nan_pixels(self):
        """NaN pixels are included in both paths (Python operator-precedence bug).

        NaN propagation differs between the two: scipy.ndimage.maximum returns
        NaN when any input is NaN, while Numba's ``v > mx`` with NaN returns
        False so NaN doesn't propagate to max/min.  We verify structural
        agreement (same keys, compatible lengths) and that the NaN-free
        features match.
        """
        masks = np.zeros((30, 30), dtype=np.int32)
        masks[5:15, 5:15] = 1
        pixels = np.random.RandomState(7).random((30, 30))
        pixels[8, 8] = np.nan

        result_numba = get_intensity_numba(masks, pixels)
        result_python = _get_intensity_python(masks, pixels)

        assert set(result_numba.keys()) == set(result_python.keys())
        for key in result_numba:
            assert len(result_numba[key]) == len(result_python[key])

        # Features unaffected by NaN comparison semantics must match exactly
        for key in (
            "Intensity_IntegratedIntensity",
            "Intensity_MeanIntensity",
            "Intensity_IntegratedIntensityEdge",
            "Intensity_MeanIntensityEdge",
        ):
            np.testing.assert_allclose(
                result_numba[key],
                result_python[key],
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"{key}: values differ with NaN pixel",
            )


# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------


class TestDispatch:
    """Verify get_intensity() dispatches to the Numba path for 2D."""

    def test_dispatch_matches_numba(self):
        """get_intensity() with 2D input returns same results as direct call."""
        rng = np.random.RandomState(55)
        masks, pixels = _make_random_scene(rng, n_objects=5)

        result_dispatch = get_intensity(masks, pixels)
        result_direct = get_intensity_numba(masks, pixels)

        assert set(result_dispatch.keys()) == set(result_direct.keys())
        for key in result_dispatch:
            np.testing.assert_array_equal(
                result_dispatch[key], result_direct[key], err_msg=key
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_mask(self):
        """All-zero mask -> empty output arrays."""
        masks = np.zeros((50, 50), dtype=np.int32)
        pixels = np.random.RandomState(0).random((50, 50))

        result = get_intensity_numba(masks, pixels)
        for key, arr in result.items():
            assert len(arr) == 0, f"{key} should be empty"

    def test_single_pixel_object(self):
        """One-pixel object: basic stats = pixel value, it IS an edge pixel
        (in-bounds neighbors have label 0 which differs from label 1)."""
        masks = np.zeros((20, 20), dtype=np.int32)
        masks[10, 10] = 1
        pixels = np.full((20, 20), 0.0)
        pixels[10, 10] = 0.75

        result = get_intensity_numba(masks, pixels)
        assert result["Intensity_IntegratedIntensity"][0] == 0.75
        assert result["Intensity_MeanIntensity"][0] == 0.75
        assert result["Intensity_StdIntensity"][0] == 0.0
        assert result["Intensity_MinIntensity"][0] == 0.75
        assert result["Intensity_MaxIntensity"][0] == 0.75
        # Single pixel surrounded by background IS an edge pixel
        assert result["Intensity_IntegratedIntensityEdge"][0] == 0.75
        assert result["Intensity_MeanIntensityEdge"][0] == 0.75

    def test_sparse_labels(self):
        """Labels {3, 7} -> output length = 7, missing indices = 0."""
        masks = np.zeros((30, 30), dtype=np.int32)
        masks[2:5, 2:5] = 3
        masks[10:15, 10:15] = 7
        pixels = np.ones((30, 30)) * 0.5

        result = get_intensity_numba(masks, pixels)
        for arr in result.values():
            assert len(arr) == 7

        # Labels 1, 2 should be zero (unused)
        assert result["Intensity_IntegratedIntensity"][0] == 0.0
        assert result["Intensity_IntegratedIntensity"][1] == 0.0
        # Label 3 (index 2) should have values
        assert result["Intensity_IntegratedIntensity"][2] > 0.0
        # Label 7 (index 6) should have values
        assert result["Intensity_IntegratedIntensity"][6] > 0.0

    def test_uniform_intensity(self):
        """Uniform intensity -> std~=0, mass_displacement~=0."""
        masks = np.zeros((30, 30), dtype=np.int32)
        masks[5:15, 5:15] = 1
        pixels = np.full((30, 30), 0.4)

        result = get_intensity_numba(masks, pixels)
        assert result["Intensity_StdIntensity"][0] == pytest.approx(0.0, abs=1e-14)
        assert result["Intensity_MassDisplacement"][0] == pytest.approx(0.0, abs=1e-12)
        assert result["Intensity_MedianIntensity"][0] == pytest.approx(0.4)
        assert result["Intensity_MADIntensity"][0] == pytest.approx(0.0, abs=1e-14)

    def test_zero_intensity(self):
        """All-zero pixel values -> cmi = NaN, mass_displacement = NaN."""
        masks = np.zeros((30, 30), dtype=np.int32)
        masks[5:15, 5:15] = 1
        pixels = np.zeros((30, 30))

        result = get_intensity_numba(masks, pixels)
        assert result["Intensity_IntegratedIntensity"][0] == 0.0
        assert np.isnan(result["Location_CenterMassIntensity_X"][0])
        assert np.isnan(result["Location_CenterMassIntensity_Y"][0])
        assert np.isnan(result["Intensity_MassDisplacement"][0])

    def test_z_coordinates_are_zero_and_independent(self):
        """2D input -> Z coordinates are always 0 and not aliased."""
        masks = np.zeros((30, 30), dtype=np.int32)
        masks[5:15, 5:15] = 1
        pixels = np.random.RandomState(0).random((30, 30))

        result = get_intensity_numba(masks, pixels)
        cmi_z = result["Location_CenterMassIntensity_Z"]
        max_z = result["Location_MaxIntensity_Z"]
        np.testing.assert_array_equal(cmi_z, np.zeros(1))
        np.testing.assert_array_equal(max_z, np.zeros(1))
        # Verify arrays are not aliased (B2)
        assert cmi_z is not max_z


# ---------------------------------------------------------------------------
# Benchmark (run with pytest -s -k benchmark to see output)
# ---------------------------------------------------------------------------


def _make_elliptical_objects(rng, height, width, n_objects):
    """Create non-overlapping elliptical objects for benchmarking."""
    masks = np.zeros((height, width), dtype=np.int32)
    pixels = rng.random((height, width)).astype(np.float64)

    placed = 0
    attempts = 0
    while placed < n_objects and attempts < n_objects * 20:
        attempts += 1
        ry = rng.randint(5, 20)
        rx = rng.randint(5, 20)
        cy = rng.randint(ry + 1, height - ry - 1)
        cx = rng.randint(rx + 1, width - rx - 1)

        # Check no overlap with existing objects
        y_lo = max(0, cy - ry)
        y_hi = min(height, cy + ry + 1)
        x_lo = max(0, cx - rx)
        x_hi = min(width, cx + rx + 1)
        if np.any(masks[y_lo:y_hi, x_lo:x_hi] > 0):
            continue

        # Draw ellipse
        for y in range(y_lo, y_hi):
            for x in range(x_lo, x_hi):
                if ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1.0:
                    masks[y, x] = placed + 1
        placed += 1

    return masks, pixels


class TestBenchmark:
    """Benchmark Numba vs Python (run with ``pytest -s -k benchmark``)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("n_objects", [50, 150])
    def test_benchmark(self, n_objects):
        """Time both implementations and print speedup."""
        rng = np.random.RandomState(0)
        height, width = 540, 540
        masks, pixels = _make_elliptical_objects(rng, height, width, n_objects)
        actual_objects = masks.max()

        # Warm up Numba JIT (first call compiles)
        _ = get_intensity_numba(masks, pixels)

        # Benchmark Numba
        n_runs = 5
        t0 = time.perf_counter()
        for _ in range(n_runs):
            result_numba = get_intensity_numba(masks, pixels)
        t_numba = (time.perf_counter() - t0) / n_runs

        # Benchmark Python
        t0 = time.perf_counter()
        for _ in range(n_runs):
            result_python = _get_intensity_python(masks, pixels)
        t_python = (time.perf_counter() - t0) / n_runs

        speedup = t_python / t_numba if t_numba > 0 else float("inf")
        print(
            f"\n  [{actual_objects} objects, {height}x{width}] "
            f"Python: {t_python:.3f}s, Numba: {t_numba:.4f}s, "
            f"Speedup: {speedup:.0f}x"
        )

        # Verify correctness alongside benchmark
        _compare_results(result_numba, result_python)
