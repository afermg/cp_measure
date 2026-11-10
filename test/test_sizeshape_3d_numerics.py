"""Numerical contracts for future 3D size/shape acceleration (not speed tests).

Geometry uses rtol=1e-6, atol=1e-8; counts/topology are exact. Moments use
rtol=1e-7, atol=1e-12 in object-local dimensionless units: raw/central moments
of order k are divided by N**(1+k/3). Normalized moments already have those
units. This avoids a fixed absolute budget in pixel**k units or a large
object hiding errors in a small one. Schema/dtypes/nonfinite values are
checked separately. NumPy and Numba dispatch may currently share a fallback;
independent analytic and rational references are therefore essential.
"""

from fractions import Fraction
from itertools import product
from math import prod, sqrt

import numpy as np
import pytest

import cp_measure
from cp_measure.bulk import get_core_measurements_3d


@pytest.fixture(params=[None, "numba"], ids=["numpy", "numba"])
def measure(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    previous = cp_measure._ACCELERATOR
    cp_measure.set_accelerator(request.param)
    try:
        yield get_core_measurements_3d()["sizeshape"]
    finally:
        cp_measure.set_accelerator(previous)


def _moment_scale(key, count):
    if key.startswith(("SpatialMoment_", "CentralMoment_")):
        order = sum(map(int, key.split("_")[1:]))
        return count ** (1 + order / 3)
    return 1.0


def _assert_result(actual, expected):
    assert list(actual) == list(expected)
    counts = {"Volume", "BoundingBoxVolume", "FilledArea", "ConvexArea", "EulerNumber"}
    for key, reference in expected.items():
        values = actual[key]
        assert values.shape == reference.shape, key
        assert values.dtype == reference.dtype, key
        np.testing.assert_array_equal(
            np.isnan(values), np.isnan(reference), err_msg=key
        )
        np.testing.assert_array_equal(
            np.isposinf(values), np.isposinf(reference), err_msg=key
        )
        np.testing.assert_array_equal(
            np.isneginf(values), np.isneginf(reference), err_msg=key
        )
        if key in counts or np.issubdtype(reference.dtype, np.integer):
            np.testing.assert_array_equal(values, reference, err_msg=key)
        elif "Moment_" in key:
            scale = _moment_scale(key, expected["Volume"])
            np.testing.assert_allclose(
                values / scale,
                reference / scale,
                rtol=1e-7,
                atol=1e-12,
                equal_nan=True,
                err_msg=key,
            )
        else:
            np.testing.assert_allclose(
                values, reference, rtol=1e-6, atol=1e-8, equal_nan=True, err_msg=key
            )


def _objects():
    masks = np.zeros((14, 20, 26), dtype=np.int32)
    masks[2:6, 2:8, 3:11] = 91
    masks[2, 2:4, 3:5] = 0  # Asymmetry and non-integer centroid.
    masks[8:11, 12:16, 17:22] = 7
    masks[10, 15, 21] = 0
    return masks


def _exact_moments(mask):
    """Python integers/Fraction, not regionprops or NumPy moment reductions."""
    points = [tuple(map(int, row)) for row in np.argwhere(mask)]
    origin = [min(p[d] for p in points) for d in range(3)]
    points = [tuple(p[d] - origin[d] for d in range(3)) for p in points]
    n = len(points)
    sums = [sum(p[d] for p in points) for d in range(3)]
    raw, central = {}, {}
    for powers in product(range(4), repeat=3):
        raw[powers] = sum(prod(p[d] ** powers[d] for d in range(3)) for p in points)
        numerator = sum(
            prod((n * p[d] - sums[d]) ** powers[d] for d in range(3)) for p in points
        )
        central[powers] = Fraction(numerator, n ** sum(powers))
    return n, raw, central


def test_cuboid_geometry(measure):
    masks = np.zeros((9, 12, 15), dtype=np.int32)
    masks[2:6, 3:9, 4:12] = 1
    got = measure(masks, None)
    n = 4 * 6 * 8
    for key in ("Volume", "BoundingBoxVolume", "FilledArea", "ConvexArea"):
        np.testing.assert_array_equal(got[key], [n])
    for axis, low, length in zip("ZYX", (2, 3, 4), (4, 6, 8)):
        np.testing.assert_array_equal(got[f"Center_{axis}"], [low + (length - 1) / 2])
        np.testing.assert_array_equal(got[f"BoundingBoxMinimum_{axis}"], [low])
        np.testing.assert_array_equal(got[f"BoundingBoxMaximum_{axis}"], [low + length])
    variance = [(length**2 - 1) / 12 for length in (4, 6, 8)]
    diagonal = [sum(variance) - value for value in variance]
    for i, j in product(range(3), repeat=2):
        np.testing.assert_allclose(
            got[f"InertiaTensor_{i}_{j}"],
            [diagonal[i] if i == j else 0],
            rtol=1e-6,
            atol=1e-8,
        )
    for i, value in enumerate(sorted(diagonal, reverse=True)):
        np.testing.assert_allclose(
            got[f"InertiaTensorEigenvalues_{i}"], [value], rtol=1e-6, atol=1e-8
        )
    for key, value in {
        "Extent": 1,
        "Solidity": 1,
        "EquivalentDiameter": (6 * n / np.pi) ** (1 / 3),
        "MajorAxisLength": sqrt(20 * max(variance)),
        "MinorAxisLength": sqrt(20 * min(variance)),
    }.items():
        np.testing.assert_allclose(got[key], [value], rtol=1e-6, atol=1e-8, err_msg=key)


def test_moments_against_exact_voxel_sums(measure):
    masks = _objects()
    got = measure(masks, None)
    for index, label in enumerate((7, 91)):
        n, raw, central = _exact_moments(masks == label)
        np.testing.assert_array_equal(got["Volume"][index], n)
        # Freeze the current exported set, including its omission of (0,3,r).
        # Adding those columns is a separate schema change, not acceleration.
        powers_exported = [p for p in product(range(4), repeat=3) if p[:2] != (0, 3)]
        assert {key for key in got if "Moment_" in key} == {
            f"{prefix}Moment_{p}_{q}_{r}"
            for prefix in ("Spatial", "Central", "Normalized")
            for p, q, r in powers_exported
        }
        for key, values in got.items():
            if "Moment_" not in key:
                continue
            powers = tuple(map(int, key.split("_")[1:]))
            order = sum(powers)
            if key.startswith("Normalized") and order < 2:
                assert np.isnan(values[index]), key
                continue
            if key.startswith("Spatial"):
                reference = float(raw[powers])
            else:
                reference = float(central[powers])
                if key.startswith("Normalized"):
                    reference /= n ** (1 + order / 3)
            scale = _moment_scale(key, n)
            np.testing.assert_allclose(
                values[index] / scale,
                reference / scale,
                rtol=1e-7,
                atol=1e-12,
                err_msg=f"{label}: {key}",
            )


def test_translation_layout_labels_and_immutability(measure):
    # Sparse, reversed IDs ensure output order is label order, not raster order.
    masks = _objects()
    pixels = np.arange(masks.size, dtype=float).reshape(masks.shape)
    original, original_pixels = masks.copy(), pixels.copy()
    masks.flags.writeable = False
    pixels.flags.writeable = False
    got = measure(masks, pixels)
    shift = (3, 2, 4)
    moved = np.pad(masks, tuple((s, 1) for s in shift))
    moved = np.asfortranarray(moved)
    expected = {key: values.copy() for key, values in got.items()}
    for axis, delta in zip("ZYX", shift):
        for prefix in ("Center", "BoundingBoxMinimum", "BoundingBoxMaximum"):
            expected[f"{prefix}_{axis}"] += delta
    for layout in (moved, moved[:, :, ::-1].copy()[:, :, ::-1]):
        layout.flags.writeable = False
        _assert_result(measure(layout, None), expected)
    for index, label in enumerate((7, 91)):
        isolated = measure((masks == label).astype(np.int32), None)
        _assert_result(
            isolated, {key: value[index : index + 1] for key, value in got.items()}
        )
    np.testing.assert_array_equal(masks, original)
    np.testing.assert_array_equal(pixels, original_pixels)


@pytest.mark.parametrize(
    "kind,euler", [("solid", 1), ("cavity", 2), ("tunnel", 0), ("disconnected", 2)]
)
def test_integer_topology(measure, kind, euler):
    masks = np.zeros((12, 12, 12), dtype=np.int32)
    masks[2:8, 2:8, 2:8] = 1
    if kind == "cavity":
        masks[4:6, 4:6, 4:6] = 0
    elif kind == "tunnel":
        masks[2:8, 4:6, 4:6] = 0
    elif kind == "disconnected":
        masks[:] = 0
        masks[2:5, 2:5, 2:5] = 1
        masks[7:10, 7:10, 7:10] = 1
    got = measure(masks, None)
    assert got["EulerNumber"].dtype == np.int64
    np.testing.assert_array_equal(got["EulerNumber"], [euler])
    np.testing.assert_array_equal(got["Volume"], [np.count_nonzero(masks)])
    if kind == "cavity":
        np.testing.assert_array_equal(got["FilledArea"], [6**3])


@pytest.mark.parametrize("lengths", [(1, 1, 1), (4, 6, 8)])
def test_surface_area_spacing_convention(measure, lengths):
    # Existing level=0 surface: a box of voxel centres with six offset faces,
    # twelve bevelled edges and eight triangular corners. A single voxel has
    # only the corners (an octahedron), not the six faces of a unit cube.
    masks = np.zeros(tuple(n + 4 for n in lengths), dtype=np.int32)
    masks[tuple(slice(2, n + 2) for n in lengths)] = 1
    unit = measure(masks, None)
    pairs = ((0, 1), (0, 2), (1, 2))
    for spacing in [(1, 1, 1), (2, 2, 2), (2, 1, 0.5), (0.5, 1, 2)]:
        edges = [(n - 1) * s for n, s in zip(lengths, spacing)]
        faces = 2 * sum(edges[i] * edges[j] for i, j in pairs)
        bevels = 4 * sum(
            edges[i] * sqrt(sum(spacing[j] ** 2 for j in range(3) if j != i))
            for i in range(3)
        )
        corners = 4 * sqrt(sum(spacing[i] ** 2 * spacing[j] ** 2 for i, j in pairs))
        got = measure(masks, None, spacing=spacing)
        np.testing.assert_allclose(
            got["SurfaceArea"], [faces + bevels + corners], rtol=1e-6, atol=1e-8
        )
        # Currently only SurfaceArea uses spacing; other features use voxels.
        _assert_result({**got, "SurfaceArea": unit["SurfaceArea"]}, unit)


@pytest.mark.parametrize(
    "advanced,new,count",
    [(False, False, 17), (False, True, 19), (True, False, 18), (True, True, 212)],
)
def test_empty_and_option_schemas(measure, advanced, new, count):
    masks = _objects()
    kwargs = dict(calculate_advanced=advanced, new_features=new)
    filled = measure(masks, None, **kwargs)
    empty = measure(np.zeros_like(masks), None, **kwargs)
    assert len(filled) == count
    assert list(empty) == list(filled)
    assert "Volume" in filled and "Area" not in filled
    for key in filled:
        dtype = (
            np.int64
            if key == "EulerNumber"
            or key.startswith(("BoundingBoxMinimum_", "BoundingBoxMaximum_"))
            else np.float64
        )
        assert filled[key].shape == (2,)
        assert empty[key].shape == (0,)
        assert filled[key].dtype == empty[key].dtype == dtype


def test_comparison_budget_does_not_hide_other_objects_or_nonfinite_changes():
    ref = {"Volume": np.array([1.0, 1e6]), "CentralMoment_2_2_2": np.zeros(2)}
    # A near-zero high-order moment can have raw accumulation error on a large
    # object; its size must never increase the allowance for the small object.
    _assert_result({**ref, "CentralMoment_2_2_2": np.array([0.0, 0.0025])}, ref)
    for values in ([1e-8, 0.0], [np.nan, 0.0], [np.inf, 0.0]):
        with pytest.raises(AssertionError):
            _assert_result({**ref, "CentralMoment_2_2_2": np.array(values)}, ref)
