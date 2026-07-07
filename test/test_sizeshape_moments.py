"""Equivalence tests for the scatter-based spatial moments in ``get_sizeshape``.

These lock **numerical fidelity to skimage regionprops**: the scatter ``spatial_moments_2d`` /
``inertia_2d`` must reproduce ``regionprops_table``'s ``moments`` / ``moments_central`` /
``moments_normalized`` / ``moments_hu`` / ``inertia_tensor`` columns, and the derived axis lengths /
eccentricity / orientation must match regionprops. Everything agrees to ~1 ULP once compared
against the right scale (see ``RTOL`` below). The rest of ``get_sizeshape`` is unchanged.
"""

import numpy
import skimage.measure

from cp_measure.core.measureobjectsizeshape import get_sizeshape
from cp_measure.primitives._moments import inertia_2d, spatial_moments_2d

# Relative equivalence tolerance. The batched matmul reorders sums vs regionprops' einsum, so
# every quantity matches to ~1 ULP *relative to its natural magnitude*; 1e-12 leaves a ~1000x
# margin. Central moments cancel toward 0, so they are compared against the raw-moment magnitude
# of the same order (the size of the summed terms) rather than their own near-zero value.
RTOL = 1e-12


def _generate_square_objects(size, n, gap_frac=0.7):
    masks = numpy.zeros((size, size), numpy.int32)
    step = size // n
    obj = int(step * gap_frac)
    lab = 0
    for a in range(n):
        for b in range(n):
            lab += 1
            masks[a * step : a * step + obj, b * step : b * step + obj] = lab
    return masks


def _generate_disk(size, cy, cx, r, label=1):
    yy, xx = numpy.mgrid[0:size, 0:size]
    masks = numpy.zeros((size, size), numpy.int32)
    masks[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = label
    return masks


def _assert_moments_match(masks):
    raw, central, normalized, hu = spatial_moments_2d(masks)
    ref = skimage.measure.regionprops_table(
        masks,
        properties=["moments", "moments_central", "moments_normalized", "moments_hu"],
    )
    for p in range(4):
        for q in range(4):
            # Scale the absolute floor by the raw-moment magnitude: raw moments set the size of the
            # terms summed for the central moments, so it bounds the cancellation round-off of the
            # (structurally near-zero) central moments too. `initial=1.0` floors it and covers the
            # empty-mask case.
            scale = numpy.nanmax(numpy.abs(ref[f"moments-{p}-{q}"]), initial=1.0)
            numpy.testing.assert_allclose(
                raw[:, p, q], ref[f"moments-{p}-{q}"], rtol=RTOL, atol=RTOL * scale
            )
            numpy.testing.assert_allclose(
                central[:, p, q],
                ref[f"moments_central-{p}-{q}"],
                rtol=RTOL,
                atol=RTOL * scale,
            )
            # normalized moments are O(1); NaN where p+q<2 in both
            assert numpy.array_equal(
                numpy.isnan(normalized[:, p, q]),
                numpy.isnan(ref[f"moments_normalized-{p}-{q}"]),
            ), f"NaN pattern mismatch normalized {p}{q}"
            numpy.testing.assert_allclose(
                normalized[:, p, q],
                ref[f"moments_normalized-{p}-{q}"],
                rtol=RTOL,
                atol=RTOL,
                equal_nan=True,
            )
    for k in range(7):  # Hu invariants are O(1)
        numpy.testing.assert_allclose(
            hu[:, k], ref[f"moments_hu-{k}"], rtol=RTOL, atol=RTOL
        )

    # inertia tensor + eigenvalues derived from the same central moments
    it_00, it_off, it_11, eig_0, eig_1 = inertia_2d(central)
    iref = skimage.measure.regionprops_table(
        masks, properties=["inertia_tensor", "inertia_tensor_eigvals"]
    )
    for got, key in [
        (it_00, "inertia_tensor-0-0"),
        (it_off, "inertia_tensor-0-1"),
        (it_off, "inertia_tensor-1-0"),
        (it_11, "inertia_tensor-1-1"),
        (eig_0, "inertia_tensor_eigvals-0"),
        (eig_1, "inertia_tensor_eigvals-1"),
    ]:
        scale = numpy.nanmax(numpy.abs(iref[key]), initial=1.0)
        numpy.testing.assert_allclose(got, iref[key], rtol=RTOL, atol=RTOL * scale)


def test_raw_moments_bit_exact():
    # raw spatial moments are integer-coordinate sums -> exactly equal to regionprops.
    masks = _generate_square_objects(256, 4)
    raw, *_ = spatial_moments_2d(masks)
    ref = skimage.measure.regionprops_table(masks, properties=["moments"])
    for p in range(4):
        for q in range(4):
            numpy.testing.assert_array_equal(raw[:, p, q], ref[f"moments-{p}-{q}"])


def test_moments_match_multi_object():
    _assert_moments_match(_generate_square_objects(256, 4))


def test_moments_match_empty():
    # No objects -> (0, ...) arrays; the shared assert covers it against regionprops.
    _assert_moments_match(numpy.zeros((20, 20), numpy.int32))


def test_moments_match_disk():
    # Non-trivial fixture: a disk has an off-grid centroid, so its central moments are genuine
    # near-total cancellations (many are structurally ~0). Axis-aligned rectangles hide this —
    # their odd central moments are bit-exactly 0 — so this is the case that actually exercises
    # the raw-magnitude-scaled tolerance in _assert_moments_match.
    _assert_moments_match(_generate_disk(240, 119.3, 121.7, 80))


def test_moments_match_edge_touching():
    masks = numpy.zeros((64, 64), numpy.int32)
    masks[0:20, 0:20] = 1
    masks[44:64, 44:64] = 2
    _assert_moments_match(masks)


def test_moments_single_pixel_object():
    # degenerate: central/normalized are 0/NaN; must match regionprops' handling.
    masks = numpy.zeros((32, 32), numpy.int32)
    masks[16, 16] = 1
    masks[5:15, 5:15] = 2
    _assert_moments_match(masks)


def test_get_sizeshape_wires_moment_features():
    # get_sizeshape exposes spatial_moments_2d under the public F_* names; the helper-vs-
    # regionprops accuracy is covered by the tests above, so this only checks the wiring.
    masks = _generate_square_objects(200, 3)
    pixels = numpy.random.default_rng(0).random(masks.shape)
    out = get_sizeshape(masks, pixels)
    raw, central, normalized, hu = spatial_moments_2d(masks)
    for p in range(3):  # spatial/central exposed for p in {0,1,2}, q in {0,1,2,3}
        for q in range(4):
            numpy.testing.assert_array_equal(
                out[f"SpatialMoment_{p}_{q}"], raw[:, p, q]
            )
            numpy.testing.assert_array_equal(
                out[f"CentralMoment_{p}_{q}"], central[:, p, q]
            )
    for p in range(4):  # normalized exposed for the full 4x4
        for q in range(4):
            numpy.testing.assert_array_equal(
                out[f"NormalizedMoment_{p}_{q}"], normalized[:, p, q]
            )
    for k in range(7):
        numpy.testing.assert_array_equal(out[f"HuMoment_{k}"], hu[:, k])
    it_00, it_off, it_11, eig_0, eig_1 = inertia_2d(central)
    numpy.testing.assert_array_equal(out["InertiaTensor_0_0"], it_00)
    numpy.testing.assert_array_equal(out["InertiaTensor_0_1"], it_off)
    numpy.testing.assert_array_equal(out["InertiaTensor_1_0"], it_off)
    numpy.testing.assert_array_equal(out["InertiaTensor_1_1"], it_11)
    numpy.testing.assert_array_equal(out["InertiaTensorEigenvalues_0"], eig_0)
    numpy.testing.assert_array_equal(out["InertiaTensorEigenvalues_1"], eig_1)


def _assert_axes_match(masks):
    # Option B: axis lengths / eccentricity / orientation are derived from the scatter central
    # moments instead of regionprops, so get_sizeshape requests no moments at all. They must still
    # match regionprops to round-off (orientation reported in degrees: radians * 180/pi).
    out = get_sizeshape(masks, masks.astype(float))
    ref = skimage.measure.regionprops_table(
        masks,
        properties=[
            "axis_major_length",
            "axis_minor_length",
            "eccentricity",
            "orientation",
        ],
    )
    numpy.testing.assert_allclose(
        out["MajorAxisLength"], ref["axis_major_length"], rtol=RTOL, atol=RTOL
    )
    numpy.testing.assert_allclose(
        out["MinorAxisLength"], ref["axis_minor_length"], rtol=RTOL, atol=RTOL
    )
    numpy.testing.assert_allclose(
        out["Eccentricity"], ref["eccentricity"], rtol=RTOL, atol=RTOL
    )
    numpy.testing.assert_allclose(
        out["Orientation"], ref["orientation"] * (180 / numpy.pi), rtol=RTOL, atol=RTOL
    )


def test_axes_match_single_object():
    masks = numpy.zeros((64, 64), numpy.int32)
    masks[18:50, 20:45] = 1  # rectangle -> nonzero orientation, distinct axis lengths
    _assert_axes_match(masks)


# Output features of the PyPI 0.1.19 release (2D, new_features + calculate_advanced on). Order is
# not part of the schema (downstream selects features by name), but the feature *set* is: a missing
# feature and an accidental extra (typo/dup) are both silent schema breaks. A genuine new feature is
# a deliberate one-line addition here.
_RELEASE_KEYS = [
    "Area",
    "BoundingBoxArea",
    "ConvexArea",
    "EquivalentDiameter",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "Eccentricity",
    "Orientation",
    "Center_X",
    "Center_Y",
    "BoundingBoxMinimum_X",
    "BoundingBoxMaximum_X",
    "BoundingBoxMinimum_Y",
    "BoundingBoxMaximum_Y",
    "FormFactor",
    "Extent",
    "Solidity",
    "Compactness",
    "EulerNumber",
    "MaximumRadius",
    "MeanRadius",
    "MedianRadius",
    "FilledArea",
    *[f"SpatialMoment_{p}_{q}" for p in range(3) for q in range(4)],
    *[f"CentralMoment_{p}_{q}" for p in range(3) for q in range(4)],
    *[f"NormalizedMoment_{p}_{q}" for p in range(4) for q in range(4)],
    *[f"HuMoment_{k}" for k in range(7)],
    "InertiaTensor_0_0",
    "InertiaTensor_0_1",
    "InertiaTensor_1_0",
    "InertiaTensor_1_1",
    "InertiaTensorEigenvalues_0",
    "InertiaTensorEigenvalues_1",
    "PerimeterCrofton",
]


def test_sizeshape_feature_set_matches_release():
    # The emitted feature set must equal the release set (order ignored): catches both a dropped
    # feature and an accidental extra. A genuine new feature means adding it to _RELEASE_KEYS.
    masks = numpy.zeros((40, 40), numpy.int32)
    masks[5:25, 5:25] = 1
    keys = set(get_sizeshape(masks, masks.astype(float)))
    expected = set(_RELEASE_KEYS)
    assert keys == expected, {
        "missing": sorted(expected - keys),
        "unexpected": sorted(keys - expected),
    }


def test_axes_clip_thin_objects_match_skimage():
    # A thin / 1px-wide object has almost no width, so the smaller eigenvalue of its inertia tensor
    # is ~0; float round-off can push that value slightly below zero. inertia_2d clips it to 0 (as
    # skimage does), so axis lengths and eccentricity match regionprops and are never NaN. Pre-clip,
    # ~4% of these gave NaN axis_minor / eccentricity > 1 — this loop guards that regression.
    rng = numpy.random.default_rng(1)
    for _ in range(50):
        masks = numpy.zeros((40, 40), numpy.int32)
        r0, c0 = rng.integers(2, 18, 2)
        length = int(rng.integers(6, 18))
        dr, dc = rng.integers(-2, 3, 2)
        for t in range(length):
            r, c = r0 + t * dr, c0 + t * dc
            if 0 <= r < 40 and 0 <= c < 40:
                masks[r, c] = 1
        if masks.max() == 0:
            continue
        out = get_sizeshape(masks, masks.astype(float))
        assert not numpy.isnan(out["MinorAxisLength"]).any()
        assert not numpy.isnan(out["Eccentricity"]).any()
        ref = skimage.measure.regionprops_table(
            masks, properties=["axis_minor_length", "eccentricity"]
        )
        # atol=1e-6 (looser than the global RTOL): a 1px line's minor axis is float noise (~1e-7)
        # in skimage vs exactly 0 from our clip; both mean "zero width", so compare only above
        # that floor.
        numpy.testing.assert_allclose(
            out["MinorAxisLength"], ref["axis_minor_length"], rtol=RTOL, atol=1e-6
        )
        numpy.testing.assert_allclose(
            out["Eccentricity"], ref["eccentricity"], rtol=RTOL, atol=1e-6
        )


def test_axes_match_single_pixel_and_line():
    # degenerate objects: single pixel (axes 0) and a 1-wide line (minor axis 0).
    masks = numpy.zeros((40, 40), numpy.int32)
    masks[20, 20] = 1
    masks[5:15, 8] = 2
    _assert_axes_match(masks)
