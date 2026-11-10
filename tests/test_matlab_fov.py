"""Synthetic public-MATLAB oracle tests; no MATLAB or real photos at test runtime."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from deepretina.legacy.matlab_fov import binary_sobel_edges, calibrate_provided_fov

FIXTURE = Path(__file__).parent / "reference" / "matlab_fov"
METADATA = json.loads((FIXTURE / "metadata.json").read_text())


def test_fixture_integrity():
    assert (
        hashlib.sha256((FIXTURE / "synthetic.npz").read_bytes()).hexdigest()
        == METADATA["fixture_sha256"]
    )


@pytest.mark.parametrize("case", METADATA["cases"], ids=lambda c: c["name"])
def test_default_binary_sobel_and_calibration_against_matlab(case):
    with np.load(FIXTURE / "synthetic.npz", allow_pickle=False) as data:
        mask = data[case["mask"]]
        edges, stages = binary_sobel_edges(mask, return_stages=True)
        np.testing.assert_array_equal(edges, data[case["edges"]])
        for key in ["gx", "gy", "unthinned"]:
            np.testing.assert_array_equal(stages[key], data[case[key]])
        assert stages["threshold"] == pytest.approx(case["threshold"], abs=2e-16)
        if case["error"]:
            with pytest.raises(ValueError, match="paired edge"):
                calibrate_provided_fov(mask)
        else:
            result = calibrate_provided_fov(mask)
            assert result.center_row_one_based == case["center_row"]
            assert result.center_col_one_based == case["center_col"]
            assert result.radius_pixels == case["radius"]
            assert result.diameter_pixels == 2 * case["radius"]
            assert result.edge_pixels == edges.sum()


@pytest.mark.parametrize(
    "dataset", ["rimone", "rim-one", "erlangen", "erlangen_videodata", "roc", "riga"]
)
def test_special_source_branches_fail_closed(dataset):
    with pytest.raises(ValueError, match="special souradnice branch"):
        calibrate_provided_fov(np.ones((31, 41), bool), dataset=dataset)


@pytest.mark.parametrize(
    "mask",
    [
        np.ones((2, 3, 4)),
        np.ones((1, 10)),
        np.full((10, 10), np.nan),
        np.array([[0, 0.1, 0.2], [0, 0.2, 0.1]]),
        np.full((10, 10), -1),
    ],
)
def test_invalid_mask_rejected(mask):
    with pytest.raises(ValueError):
        binary_sobel_edges(mask)


def test_even_circle_rounds_positive_half_up_and_modes_choose_smallest():
    case = next(c for c in METADATA["cases"] if c["name"] == "circle_even")
    with np.load(FIXTURE / "synthetic.npz") as data:
        result = calibrate_provided_fov(data[case["mask"]])
    assert (result.center_row_one_based, result.center_col_one_based) == (31, 36)
