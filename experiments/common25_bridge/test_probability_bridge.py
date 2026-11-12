"""Manually specified synthetic probabilities/geometry only; no model or data IO."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "bridge_math", Path(__file__).with_name("probability_bridge.py")
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def grids(shape=(2, 3)):
    h, w = shape
    target = m.Grid(
        "sha256:" + "1" * 64,
        "sha256:" + "2" * 64,
        "train",
        ("synthetic", "native"),
        shape,
        (0, h - 1, 0, w - 1),
        shape,
        max(shape) / 25,
        25,
        "legacy25_raw",
        "3" * 64,
        "4" * 64,
        "5" * 64,
    )
    source = replace(
        target,
        pixels_per_degree=35,
        preprocessing_profile="legacy35_raw",
        prepared_shape=tuple(round(n * 1.4) for n in shape),
        manifest_sha256="6" * 64,
        content_sha256="7" * 64,
    )
    return source, target


def probabilities(shape, vessel=0.8):
    p = np.full((5, *shape), vessel, np.float32)
    p[1] = np.float32(0.3)
    p[2] = np.float32(0.7)
    return p


def test_identity_is_bit_exact_including_conditional_roundoff_and_signed_zero():
    _, g = grids((3, 5))
    p = probabilities(g.prepared_shape)
    p[0, 0, 0] = np.float32(-0.0)
    p[1] = np.float32(0.12345679)
    p[2] = np.float32(0.8765432)
    out = m.bridge(p, g, g)
    assert out.tobytes() == p.tobytes() and not np.shares_memory(out, p)
    out[0, 0, 0] = 1
    assert np.signbit(p[0, 0, 0])


def test_odd_independent_axes_ramp_has_half_pixel_manual_values():
    source, target = grids((2, 3))  # source3x4 -> target2x3, not an integer axis ratio.
    p = probabilities(source.prepared_shape)
    p[0] = np.array(
        [[0.1, 0.15, 0.2, 0.25], [0.2, 0.25, 0.3, 0.35], [0.3, 0.35, 0.4, 0.45]], np.float32
    )
    out = m.bridge(p, source, target)
    # Source centers y=(1/4,7/4), x=(1/6,3/2,17/6).
    expected = np.array([[2 / 15, 1 / 5, 4 / 15], [17 / 60, 7 / 20, 5 / 12]], np.float32)
    np.testing.assert_allclose(out[0], expected, rtol=0, atol=4e-8)


def test_edge_extension_is_clamped_not_zero_or_reflection():
    out = m._resize_plane(np.array([[0.1, 0.9]], np.float32), (1, 4))
    np.testing.assert_allclose(out, [[0.1, 0.3, 0.7, 0.9]], rtol=0, atol=6e-8)


def test_constants_survive_and_no_fov_masking_occurs_at_either_edge():
    source, target = grids((2, 3))
    p = probabilities(source.prepared_shape)
    out = m.bridge(p, source, target)
    np.testing.assert_allclose(out[[0, 3, 4]], 0.8, rtol=0, atol=1e-7)
    # A partial target FOV belongs to scorer; bridge must preserve even these pixels.
    target_fov = np.array([[True, False, False], [True, True, False]])
    np.testing.assert_allclose(out[0, ~target_fov], 0.8, rtol=0, atol=1e-7)
    assert np.count_nonzero((out[0] >= 0.5) & target_fov) == 3


def test_opposing_av_is_joint_interpolation_not_conditional_average():
    source, target = grids((1, 1))
    # Explicit small low-level two-pixel source is an operator test, not fake35 metadata.
    joint = np.array(
        [[[0.1, 0.9]], [[0.09, 0.09]], [[0.01, 0.81]], [[0.2, 0.8]], [[0.4, 0.6]]], np.float32
    )
    pooled = np.stack([m._resize_plane(x, (1, 1)) for x in joint])
    assert pooled[0, 0, 0] == pytest.approx(0.5)
    assert pooled[1, 0, 0] / pooled[0, 0, 0] == pytest.approx(0.18, abs=1e-7)
    # Exercise full bridge source1x3 -> target1x2: center x=.25 and1.75.
    source, target = grids((1, 2))
    p = probabilities(source.prepared_shape)
    p[0] = [[0.1, 0.9, 0.1]]
    p[1] = [[0.9, 0.1, 0.9]]
    p[2] = 1 - p[1]
    out = m.bridge(p, source, target)
    # (.75*.1*.9+.25*.9*.1)/(.75*.1+.25*.9)=.3, not .7.
    np.testing.assert_allclose(out[1], [[0.3, 0.3]], rtol=0, atol=8e-8)
    np.testing.assert_allclose(out[2], [[0.7, 0.7]], rtol=0, atol=8e-8)


def test_zero_vessel_yields_neutral_conditional_without_nan():
    source, target = grids()
    p = probabilities(source.prepared_shape, 0)
    out = m.bridge(p, source, target)
    assert np.isfinite(out).all() and not out[0].any()
    assert np.array_equal(out[1:3], np.full_like(out[1:3], 0.5))


@pytest.mark.parametrize(
    "field,value",
    [
        ("crop_position", (0, 0, 0, 2)),
        ("photo_id", "sha256:" + "8" * 64),
        ("split_group_id", "sha256:" + "8" * 64),
        ("canonical_owner", ("other", "native")),
        ("field_of_view_degrees", 0.2),
        ("geometry_policy_sha256", "8" * 64),
    ],
)
def test_geometry_or_lineage_mismatch_rejected(field, value):
    source, target = grids()
    with pytest.raises(ValueError):
        m.bridge(probabilities(source.prepared_shape), replace(source, **{field: value}), target)


def test_wrong_shape_dtype_or_test_split_rejected():
    source, target = grids()
    with pytest.raises(ValueError, match="FP32"):
        m.bridge(probabilities(source.prepared_shape).astype(np.float64), source, target)
    with pytest.raises(ValueError, match="shape"):
        m.bridge(
            probabilities(source.prepared_shape), replace(source, prepared_shape=(2, 3)), target
        )
    with pytest.raises(ValueError, match="TEST"):
        m.bridge(
            probabilities(source.prepared_shape),
            replace(source, split="test"),
            replace(target, split="test"),
        )


def test_partial_source_fov_boundary_is_not_premasked_before_interpolation():
    source, target = grids((1, 2))
    p = probabilities(source.prepared_shape)
    p[0] = [[0.2, 0.8, 0.2]]
    before = p.tobytes()
    # Hypothetical35FOV=[False,True,False] must NOT zero either .2 before resize.
    # Source center .25: .75*.2+.25*.8=.35; premasking would incorrectly give .2.
    out = m.bridge(p, source, target)
    np.testing.assert_allclose(out[0], [[0.35, 0.35]], rtol=0, atol=6e-8)
    assert p.tobytes() == before
    # The fixed25 scorer may count only the first pixel, but the bridge leaves
    # the second probability untouched. It never manufactures/reads a GT/FOV mask.
    target25_valid = np.array([[True, False]])
    assert out[0, ~target25_valid][0] == pytest.approx(0.35, abs=6e-8)
