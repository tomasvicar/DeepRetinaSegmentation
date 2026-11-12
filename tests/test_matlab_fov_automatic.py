"""Synthetic MATLAB source-stage goldens; no production profile activation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import binary_fill_holes

from deepretina.legacy.matlab_fov_source import (
    automatic_fov_source,
    automatic_fov_stages,
    source_close,
    source_disk10,
    source_histcounts250,
    source_rgb2gray,
    source_seed_components,
)

ROOT = Path(__file__).parent / "reference" / "matlab_fov_source"
META = json.loads((ROOT / "automatic-metadata.json").read_text())


@pytest.fixture(scope="module")
def arrays():
    path = ROOT / "automatic-synthetic.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == META["fixture_sha256"]
    with np.load(path, allow_pickle=False) as data:
        yield data


@pytest.mark.parametrize("case", META["rgb"])
def test_rgb_double_exact(arrays, case):
    np.testing.assert_array_equal(source_rgb2gray(arrays[case["rgb"]]), arrays[case["gray"]])


@pytest.mark.parametrize("case", META["morphology"])
def test_closing_and_four_connected_fill(arrays, case):
    np.testing.assert_array_equal(source_disk10(), arrays["footprint"])
    closed = source_close(arrays[case["mask"]])
    np.testing.assert_array_equal(closed, arrays[case["closed"]])
    np.testing.assert_array_equal(binary_fill_holes(closed), arrays[case["filled"]])


@pytest.mark.parametrize("case", META["components"])
def test_seed_membership_retains_background_label(arrays, case):
    np.testing.assert_array_equal(
        source_seed_components(arrays[case["mask"]], arrays[case["seeds"]]),
        arrays[case["selected"]],
    )


@pytest.mark.parametrize("case", META["histograms"])
def test_oracle_edges_only_histogram_counts(arrays, case):
    values = arrays[case["values"]]
    native_counts, native_edges = source_histcounts250(values)
    np.testing.assert_array_equal(native_counts, arrays[case["counts"]])
    np.testing.assert_array_equal(native_edges, arrays[case["edges"]])
    counts, edges = source_histcounts250(values, oracle_edges=arrays[case["edges"]])
    np.testing.assert_array_equal(counts, arrays[case["counts"]])
    np.testing.assert_array_equal(edges, arrays[case["edges"]])


@pytest.mark.parametrize("case", META["automatic"])
def test_automatic_source_stages_and_calibration(arrays, case):
    kwargs = {
        "dataset": case["dataset"],
        "histogram_edges": arrays[case["histogram_edges"]],
    }
    rgb = arrays[case["rgb"]]
    stages = automatic_fov_stages(rgb, **kwargs)
    for field in ["gray", "selected", "initial_closed", "closed", "filled"]:
        np.testing.assert_array_equal(stages[field], arrays[case[field]])
    assert stages["low_threshold"] == case["low"]
    assert stages["high_threshold"] == case["high"]
    if case["source_error"]:
        with pytest.raises(ValueError, match="paired boundary"):
            automatic_fov_source(rgb, **kwargs)
    else:
        result = automatic_fov_source(rgb, **kwargs)
        assert [
            result.calibration.center_row_one_based,
            result.calibration.center_col_one_based,
            result.calibration.radius_pixels,
        ] == case["calibration"]
        np.testing.assert_array_equal(result.processing_fov, arrays[case["processing_fov"]])
        np.testing.assert_array_equal(result.closed_mask, arrays[case["closed"]])
        assert result.diagnostics["histogram_backend"] == stages["histogram_backend"]
        assert result.diagnostics["multithresh_backend"] == "not_required"
    native = automatic_fov_stages(rgb, dataset=case["dataset"])
    np.testing.assert_array_equal(native["closed"], stages["closed"])
    assert native["histogram_backend"] == (
        "python_source_histcounts250" if stages["fallback_used"] else "not_required"
    )


def test_pre_holefill_boundary_changes_source_radius(arrays):
    case = next(c for c in META["automatic"] if c["name"] == "drive_large_hole")
    result = automatic_fov_source(arrays[case["rgb"]], dataset="drive")
    assert result.calibration.radius_pixels == 20
    assert not result.closed_mask[63, 79] and result.processing_fov[63, 79]


def test_unresolved_multithresh_is_explicit(arrays):
    case = META["automatic"][0]
    rgb = arrays[case["rgb"]]
    with pytest.raises(NotImplementedError, match="N=3"):
        automatic_fov_stages(rgb, dataset="stare")
    with pytest.raises(ValueError, match="exactly 3"):
        automatic_fov_stages(rgb, dataset="stare", multithresh_values=[0.1, 0.2])
    result = automatic_fov_stages(
        rgb,
        dataset="stare",
        multithresh_values=[0, 0.1, 0.2],
        histogram_edges=arrays[case["histogram_edges"]],
    )
    assert result["multithresh_backend"] == "supplied_oracle_vector"


@pytest.mark.parametrize("edges", [np.zeros(251), np.arange(250), np.full(251, np.nan)])
def test_reject_invalid_oracle_histogram_edges(edges):
    with pytest.raises(ValueError, match="251 finite"):
        source_histcounts250(np.array([0.0, 1.0]), oracle_edges=edges)
