"""Optional source-special stages; public defaults stay unchanged."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from deepretina.legacy.matlab_fov import calibrate_provided_fov
from deepretina.legacy.matlab_fov_source import provided_fov_source, rimone_loader_mask

ROOT = Path(__file__).parent
FIXTURE = ROOT / "reference/matlab_fov_source"
METADATA = json.loads((FIXTURE / "metadata.json").read_text())
REPORT = json.loads(
    (ROOT.parent / "reports/legacy/matlab-fov-source/provided-comparison.json").read_text()
)


def test_fixture_sha256():
    assert (
        hashlib.sha256((FIXTURE / "synthetic.npz").read_bytes()).hexdigest()
        == METADATA["fixture_sha256"]
    )


@pytest.mark.parametrize("case", METADATA["cases"], ids=lambda c: c["name"])
def test_provided_source_branches_match_matlab(case):
    with np.load(FIXTURE / "synthetic.npz", allow_pickle=False) as stored:
        mask = stored[case["mask"]]
        original = mask.copy()
        if case["source_error"]:
            with pytest.raises(ValueError, match="paired boundary"):
                provided_fov_source(mask, dataset=case["dataset"])
        else:
            result = provided_fov_source(mask, dataset=case["dataset"])
            np.testing.assert_array_equal(result.calibration_boundary, stored[case["boundary"]])
            np.testing.assert_array_equal(
                result.processing_fov, stored[case["processing_fov"]].astype(bool)
            )
            np.testing.assert_array_equal(result.closed_mask, mask)
            assert (
                result.calibration.center_row_one_based,
                result.calibration.center_col_one_based,
                result.calibration.radius_pixels,
            ) == (case["row"], case["col"], case["radius"])
        np.testing.assert_array_equal(mask, original)


@pytest.mark.parametrize(
    "case", [c for c in REPORT["cases"] if c["name"].startswith("real_")], ids=lambda c: c["name"]
)
def test_real_source_shape_calibrations_without_real_photographs(case):
    shape = tuple(case["shape"])
    mask = rimone_loader_mask(shape) if case["dataset"] == "rimone" else np.ones(shape, bool)
    assert hashlib.sha256(mask.tobytes()).hexdigest() == case["mask_sha256"]
    result = provided_fov_source(mask, dataset=case["dataset"])
    expected = case["calibration"]
    assert result.calibration.center_row_one_based == expected["center_row_one_based"]
    assert result.calibration.center_col_one_based == expected["center_col_one_based"]
    assert result.calibration.radius_pixels == expected["radius_pixels"]
    assert result.processing_fov.all()


def test_rimone_loader_mask_is_left_frame_not_dense_fov():
    mask = rimone_loader_mask((79, 100))
    assert mask[[0, -1], :50].all() and mask[:, [0, 49]].all()
    assert not mask[1:-1, 1:49].any() and not mask[:, 50:].any()
    result = provided_fov_source(mask, dataset="rimone")
    np.testing.assert_array_equal(result.calibration_boundary, mask)
    assert result.processing_fov.all() and not mask.all()
    with pytest.raises(ValueError, match="even width"):
        rimone_loader_mask((79, 101))


@pytest.mark.parametrize("dataset", ["rimone", "erlangen", "roc", "riga"])
def test_existing_public_generic_guard_is_unchanged(dataset):
    with pytest.raises(ValueError, match="special souradnice"):
        calibrate_provided_fov(np.ones((79, 100), bool), dataset=dataset)


@pytest.mark.parametrize(
    "mask",
    [np.zeros((2, 3)), np.ones((1, 8)), np.full((10, 10), np.nan), np.array([[0, 0.3, 0.7]] * 3)],
)
def test_invalid_or_not_source_provided_shape_fails(mask):
    with pytest.raises(ValueError):
        provided_fov_source(mask)
