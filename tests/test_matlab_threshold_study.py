"""Fixed250-bin source goldens and explicitly uncertified threshold candidate."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from deepretina.legacy.matlab_fov_source import implicit_histcounts250_edges

ROOT = Path(__file__).parent / "reference" / "matlab_fov_source"
HIST = json.loads((ROOT / "native-histogram-metadata.json").read_text())
MULTI = json.loads((ROOT / "multithresh-metadata.json").read_text())
PATH = Path(__file__).parents[1] / "reports/legacy/matlab-fov-source/nelder_mead_candidate.py"
SPEC = importlib.util.spec_from_file_location("fov_threshold_candidate", PATH)
CANDIDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CANDIDATE)


@pytest.mark.parametrize("case", HIST["cases"], ids=lambda c: c["name"])
def test_implicit250_bitwise_matlab_edges(case):
    edges = implicit_histcounts250_edges([case["minimum"], case["maximum"]])
    assert (
        hashlib.sha256(np.asarray(edges, dtype="<f8").tobytes()).hexdigest() == case["edges_sha256"]
    )


@pytest.fixture(scope="module")
def images():
    path = ROOT / "multithresh-synthetic.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MULTI["fixture_sha256"]
    with np.load(path, allow_pickle=False) as values:
        yield values


@pytest.mark.parametrize("case", MULTI["cases"], ids=lambda c: f"{c['name']}-N{c['n']}")
def test_local_candidate_keeps_measured_remaining_gap(images, case):
    image = images[case["name"]]
    if np.unique(image).size <= case["n"]:
        with pytest.raises(NotImplementedError, match="Degenerate"):
            CANDIDATE.candidate_multithresh(image, case["n"])
        return
    result = CANDIDATE.candidate_multithresh(image, case["n"])
    assert result["matlab_parity_certified"] is False
    lo, hi = image.min(), image.max()
    expected = (np.asarray(case["thresholds"]) - lo) / (hi - lo) * 255
    if case["name"] == "range_ramp" and case["n"] == 5:
        np.testing.assert_allclose(
            result["threshold_indices"] - expected, [0, 1, 1, 0, 0], atol=1e-12, rtol=0
        )
    else:
        np.testing.assert_allclose(result["threshold_indices"], expected, atol=1e-12, rtol=0)
    assert result["thresholds"][0] == case["thresholds"][0]
    assert result["effectiveness_status"] == case["effectiveness_status"]
    if result["effectiveness"] is not None and not (
        case["name"] == "range_ramp" and case["n"] == 5
    ):
        assert result["effectiveness"] == pytest.approx(case["effectiveness"], abs=1e-12)


def test_adjacent_double_histogram_boundary_is_not_epsilon_smoothed():
    first = implicit_histcounts250_edges([0.4, 0.65])
    adjacent = implicit_histcounts250_edges([0.4, np.nextafter(0.65, 0)])
    assert first[1] - first[0] != adjacent[1] - adjacent[0]
    assert first[-1] == 0.65025 and adjacent[-1] == 0.65


def test_unrepresentable_float_range_fails_without_epsilon():
    with pytest.raises(NotImplementedError, match="Unrepresentable"):
        implicit_histcounts250_edges([0, np.nextafter(0, 1)])
