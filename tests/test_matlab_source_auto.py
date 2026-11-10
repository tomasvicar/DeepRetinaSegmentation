"""Optional automatic-source composition: no network or actual GT reader."""

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from deepretina.legacy import matlab_av
from deepretina.legacy import matlab_source_auto as module
from deepretina.legacy.matlab_fov_source import automatic_fov_source, automatic_fov_stages
from deepretina.legacy.matlab_resize import imresize_bicubic

ROOT = Path(__file__).parent / "reference" / "matlab_fov_source"
META = json.loads((ROOT / "source-auto-synthetic-metadata.json").read_text())


@pytest.fixture(scope="module")
def arrays():
    path = ROOT / "source-auto-synthetic.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == META["fixture_sha256"]
    with np.load(path, allow_pickle=False) as values:
        yield values


@pytest.mark.parametrize("case", META["half"], ids=lambda c: c["name"])
def test_half_factor_source_masks(arrays, case):
    kwargs = {"dataset": case["dataset"], "threshold_backend": "source_first_cutoff"}
    stages = automatic_fov_stages(arrays[case["image"]], **kwargs)
    assert stages["first_cutoff"]["value"] * 0.5 == case["initial_low"]
    assert stages["low_threshold"] == case["final_low"]
    for name in ["initial_closed", "closed", "filled"]:
        np.testing.assert_array_equal(stages[name], arrays[case[name]])
    if case["source_error"]:
        with pytest.raises(ValueError):
            automatic_fov_source(arrays[case["image"]], **kwargs)
    else:
        result = automatic_fov_source(arrays[case["image"]], **kwargs)
        np.testing.assert_array_equal(result.calibration_boundary, arrays[case["boundary"]])
        np.testing.assert_array_equal(result.processing_fov, arrays[case["processing_fov"]])
        assert [
            result.calibration.center_row_one_based,
            result.calibration.center_col_one_based,
            result.calibration.radius_pixels,
        ] == case["calibration"]


@pytest.mark.parametrize("case", META["chains"], ids=lambda c: c["name"])
def test_composed_chain_golden_and_no_second_calibration(arrays, case, monkeypatch):
    monkeypatch.setattr(
        matlab_av,
        "calibrate_provided_fov",
        lambda *_a, **_k: pytest.fail("Do not recalibrate the returned processing mask"),
    )
    image, vessel = arrays[case["image"]], arrays[case["vessel"]]
    before = image.copy(), vessel.copy()
    result = module.prepare_vesclass_matlab_source_auto(
        image, vessel, dataset=case["dataset"], field_of_view_degrees=case["degree"]
    )
    assert isinstance(result, module.PreparedMatlabSourceAuto)
    assert list(result.image.shape) == case["prepared_shape"]
    assert result.geometry.crop_position == (0, image.shape[0] - 1, 0, image.shape[1] - 1)
    np.testing.assert_array_equal(result.source_fov.processing_fov, arrays[case["processing_fov"]])
    np.testing.assert_array_equal(result.fov, arrays[case["resized_fov"]])
    np.testing.assert_array_equal(result.vessel, arrays[case["resized_vessel"]])
    assert [
        result.source_fov.calibration.center_row_one_based,
        result.source_fov.calibration.center_col_one_based,
        result.source_fov.calibration.radius_pixels,
    ] == case["calibration"]
    assert result.profile["fov_calibration"]["mode"] == "estimated_from_source_automatic_fov"
    assert result.profile["generic_recalibration_performed"] is False
    assert result.profile["matlab_parity_certified"] is False
    assert result.profile["historical_training_file_provenance_verified"] is False
    ys = np.array(case["probe_rows"])
    xs = np.array(case["probe_cols"])
    resized = imresize_bicubic(image.astype(float) / 255, result.geometry.rescale_factor)
    np.testing.assert_allclose(
        resized[ys[:, None], xs[None, :]], arrays[case["resized_probes"]], atol=4e-15, rtol=0
    )
    expected = arrays[case["stored_dicom12"]]
    difference = np.abs(result.image.astype(np.int32) - expected.astype(np.int32))
    assert difference.max() <= 1
    assert np.count_nonzero(difference) < expected.size * 0.001
    np.testing.assert_array_equal(image, before[0])
    np.testing.assert_array_equal(vessel, before[1])
    json.dumps(result.profile, allow_nan=False)


@pytest.mark.parametrize(
    "changes",
    [
        {"pixels_per_degree": 35},
        {"pixels_per_degree": True},
        {"storage": "other"},
        {"dataset": "riga"},
        {"dataset": "review"},
        {"dataset": "erlangen"},
        {"dataset": "rimone"},
        {"dataset": "unknown"},
        {"field_of_view_degrees": 0},
        {"field_of_view_degrees": float("nan")},
    ],
)
def test_invalid_request_rejected_before_fov(changes, monkeypatch):
    monkeypatch.setattr(
        module,
        "automatic_fov_source",
        lambda *_a, **_k: pytest.fail("Invalid calibration must fail before source FOV"),
    )
    args = {"dataset": "stare", **changes}
    with pytest.raises(ValueError):
        module.prepare_vesclass_matlab_source_auto(
            np.ones((16, 20, 3), np.uint8), np.zeros((16, 20), bool), **args
        )


def test_no_oracle_override_on_native_composed_api():
    parameters = inspect.signature(module.prepare_vesclass_matlab_source_auto).parameters
    assert "histogram_edges" not in parameters and "multithresh_values" not in parameters
    assert "threshold_backend" not in parameters


@pytest.mark.parametrize(
    "dataset,expected", [("eophtha", 40.0), ("ridb", 45.0), ("diaretdb0", 50.0), ("drishti", 30.0)]
)
def test_unambiguous_source_angle_defaults(dataset, expected, arrays, monkeypatch):
    from deepretina.legacy.preprocessing import ImageGeometry, PreparedVesClassProxy

    image = arrays[META["chains"][0]["image"]]
    source = automatic_fov_source(image, dataset="stare", threshold_backend="source_first_cutoff")
    monkeypatch.setattr(module, "automatic_fov_source", lambda *_a, **_k: source)
    called = []

    def prepare(rgb, vessel, **kwargs):
        called.append(kwargs)
        return PreparedVesClassProxy(
            rgb.copy(),
            vessel,
            kwargs["fov"],
            ImageGeometry(
                rgb.shape[:2], (0, rgb.shape[0] - 1, 0, rgb.shape[1] - 1), 1.0, rgb.shape[:2]
            ),
            {},
        )

    monkeypatch.setattr(module, "prepare_vesclass_matlab_calibrated", prepare)
    module.prepare_vesclass_matlab_source_auto(
        image, np.zeros(image.shape[:2], bool), dataset=dataset
    )
    assert called[0]["field_of_view_degrees"] == expected
    assert called[0]["pixels_per_degree"] == 25.0
    assert called[0]["fov_diameter_pixels"] == source.calibration.diameter_pixels
    assert called[0]["fov"] is source.processing_fov
