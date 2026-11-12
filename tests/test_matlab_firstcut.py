"""First-cutoff required contract: source oracles, not full-vector certification."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from deepretina.legacy.matlab_firstcut import first_cutoff_source
from deepretina.legacy.matlab_fov_source import automatic_fov_source, automatic_fov_stages

ROOT = Path(__file__).parent / "reference" / "matlab_fov_source"
META = json.loads((ROOT / "firstcut-fixture-metadata.json").read_text())


@pytest.fixture(scope="module")
def histograms():
    path = ROOT / "firstcut-histograms.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == META["histogram_fixture_sha256"]
    with np.load(path, allow_pickle=False) as data:
        yield data


@pytest.mark.parametrize("case", META["cases"], ids=lambda c: f"{c['name']}-N{c['n']}")
def test_first_cutoff_bitwise_source_required_contract(histograms, case):
    row = case["histogram_row"]
    lo = histograms["minimums"][row]
    hi = histograms["maximums"][row]
    centers = lo + (hi - lo) * (np.arange(256) / 255)
    centers[0], centers[-1] = lo, hi
    values = np.repeat(centers, histograms["counts"][row])
    cutoff = first_cutoff_source(values, case["n"])
    assert cutoff.value == case["source_first_cutoff"]
    assert cutoff.effectiveness_status == case["effectiveness_status"]
    assert cutoff.matlab_parity_certified is False
    assert cutoff.full_multithresh_clone is False


@pytest.fixture(scope="module")
def fov_arrays():
    path = ROOT / "firstcut-fov-synthetic.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == META["fov_fixture_sha256"]
    with np.load(path, allow_pickle=False) as data:
        yield data


@pytest.mark.parametrize("case", META["fov_cases"], ids=lambda c: c["name"])
def test_explicit_backend_source_fov_stages_and_calibration(fov_arrays, case):
    image = fov_arrays[case["image"]]
    with pytest.raises(NotImplementedError, match="opt-in"):
        automatic_fov_stages(image, dataset=case["dataset"])
    args = {"dataset": case["dataset"], "threshold_backend": "source_first_cutoff"}
    stages = automatic_fov_stages(image, **args)
    factor = 0.1 if case["dataset"] == "diaretdb" else 1.0
    assert stages["first_cutoff"]["value"] * factor == case["initial_low"]
    assert stages["low_threshold"] == case["final_low"]
    for name in ["initial_closed", "closed", "filled"]:
        np.testing.assert_array_equal(stages[name], fov_arrays[case[name]])
    assert stages["multithresh_backend"] == "source_first_cutoff"
    assert stages["first_cutoff"]["full_multithresh_clone"] is False
    if case["source_error"]:
        with pytest.raises(ValueError):
            automatic_fov_source(image, **args)
    else:
        result = automatic_fov_source(image, **args)
        np.testing.assert_array_equal(result.calibration_boundary, fov_arrays[case["boundary"]])
        np.testing.assert_array_equal(result.processing_fov, fov_arrays[case["processing_fov"]])
        assert [
            result.calibration.center_row_one_based,
            result.calibration.center_col_one_based,
            result.calibration.radius_pixels,
        ] == case["calibration"]
        assert result.diagnostics["multithresh_backend"] == "source_first_cutoff"
        assert result.diagnostics["first_cutoff"] == stages["first_cutoff"]
        json.dumps(result.diagnostics, allow_nan=False)


def test_fractional_failed_search_is_not_rounded(histograms):
    case = next(
        c for c in META["cases"] if c["effectiveness_status"] == "negative_infinity" and c["n"] == 5
    )
    row = case["histogram_row"]
    lo = histograms["minimums"][row]
    hi = histograms["maximums"][row]
    centers = lo + (hi - lo) * np.arange(256) / 255
    centers[0], centers[-1] = lo, hi
    values = np.repeat(centers, histograms["counts"][row])
    cutoff = first_cutoff_source(values, 5)
    assert cutoff.bin_coordinate == 42.5
    assert cutoff.value == lo + (hi - lo) * (1 / 6)
    assert cutoff.value != lo + (hi - lo) * (42 / 255)
    assert asdict(cutoff)["effectiveness_status"] == "negative_infinity"


@pytest.mark.parametrize("n", [True, 0, 2, 6, 3.5])
def test_invalid_n_rejected(n):
    with pytest.raises(ValueError, match="integer N3/4/5"):
        first_cutoff_source(np.linspace(0, 1, 100), n)


@pytest.mark.parametrize(
    "data",
    [np.array([0, 1], np.float32), np.array([0, np.nan]), np.array([-1.0, 1.0]), np.array([])],
)
def test_invalid_gray_rejected(data):
    with pytest.raises(ValueError):
        first_cutoff_source(data, 3)


def test_degenerate_policy_stays_fail_closed():
    with pytest.raises(NotImplementedError, match="Degenerate"):
        first_cutoff_source(np.full((8, 8), 0.37, np.float64), 3)


def test_no_mixed_oracle_native_backend():
    image = np.full((8, 8, 3), 127, np.uint8)
    with pytest.raises(ValueError, match="not both"):
        automatic_fov_stages(
            image,
            dataset="stare",
            multithresh_values=[0.1, 0.2, 0.3],
            threshold_backend="source_first_cutoff",
        )
    with pytest.raises(ValueError, match="Choose oracle"):
        automatic_fov_stages(image, dataset="stare", threshold_backend="typo")
