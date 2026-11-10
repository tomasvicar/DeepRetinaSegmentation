"""Public-API MATLAB numerical oracles and strict calibrated profile contracts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from deepretina.legacy.matlab_av import matlab_av_photometric, prepare_vesclass_matlab_calibrated
from deepretina.legacy.matlab_resize import imresize_bicubic, imresize_nearest_mask
from deepretina.legacy.preprocessing import quantize_vesclass_rgb

REFERENCE = Path(__file__).parent / "reference/matlab_resize"
METADATA = json.loads((REFERENCE / "metadata.json").read_text())


@pytest.mark.parametrize("case", METADATA["cases"], ids=lambda case: case["name"])
def test_scalar_scale_and_nearest_matlab_goldens(case):
    with np.load(REFERENCE / "synthetic.npz", allow_pickle=False) as fixture:
        image = fixture[case["input"]].astype(np.float64)
        expected = fixture[case["output"]]
        mask, nearest = fixture[case["mask"]], fixture[case["nearest"]]
    original = image.copy()
    actual = imresize_bicubic(image, case["scale"])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(imresize_nearest_mask(mask, actual.shape), nearest)
    assert actual.shape == tuple(np.ceil(np.array(image.shape) * case["scale"]).astype(int))


def test_fixture_identity():
    assert (
        hashlib.sha256((REFERENCE / "synthetic.npz").read_bytes()).hexdigest()
        == METADATA["fixture_sha256"]
    )


def test_matlab_constant_photometric_oracle_and_complete_calibrated_contract():
    # A MATLAB synthetic oracle gives0.51025normalized /2090stored for this case.
    image = np.full((200, 240, 3), 127, np.uint8)
    fov = np.ones((200, 240), bool)
    vessel = (np.indices(fov.shape).sum(axis=0) % 2) == 0
    normalized = matlab_av_photometric(image.astype(np.float64) / 255, fov)
    np.testing.assert_array_equal(
        quantize_vesclass_rgb(normalized), np.full(image.shape, 2090, np.uint16)
    )
    prepared = prepare_vesclass_matlab_calibrated(
        image, vessel, fov=fov, fov_diameter_pixels=200, field_of_view_degrees=8
    )
    assert prepared.image.dtype == np.uint16 and prepared.image.shape == image.shape
    np.testing.assert_array_equal(prepared.vessel, vessel)
    np.testing.assert_array_equal(prepared.fov, fov)
    assert prepared.geometry.original_shape == (200, 240)
    assert prepared.geometry.crop_position == (0, 199, 0, 239)
    assert prepared.metadata["matlab_parity_certified"] is False
    assert prepared.metadata["historical_training_file_provenance_verified"] is False
    assert prepared.metadata["photometry_pixels_per_degree"] == 25
    assert prepared.inputs().shape == (4, 200, 240)
    np.testing.assert_array_equal(prepared.inputs()[3], vessel.astype(np.float32))
    assert (image == 127).all() and fov.all()


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf])
def test_bad_resize_scale_rejected(scale):
    with pytest.raises(ValueError):
        imresize_bicubic(np.zeros((3, 4)), scale)


def test_calibration_and_threshold_requirements_are_explicit():
    image = np.full((20, 20, 3), 127, np.uint8)
    fov = np.ones((20, 20), bool)
    with pytest.raises(ValueError, match="paired edge"):
        prepare_vesclass_matlab_calibrated(image, fov, fov=fov, dataset="drive")
    with pytest.raises(ValueError, match="Threshold"):
        prepare_vesclass_matlab_calibrated(
            image, np.full(fov.shape, 0.3), fov=fov, fov_diameter_pixels=20, dataset="drive"
        )
    with pytest.raises(ValueError, match="known dataset"):
        prepare_vesclass_matlab_calibrated(image, fov, fov=fov, fov_diameter_pixels=20)
    with pytest.raises(ValueError, match="positive/finite"):
        prepare_vesclass_matlab_calibrated(
            image, fov, fov=fov, fov_diameter_pixels=0, dataset="drive"
        )


def test_generic_fov_estimate_and_explicit_calibration_prepare_identical_pixels():
    from deepretina.legacy.matlab_fov import calibrate_provided_fov

    rng = np.random.default_rng(27)
    image = rng.integers(30, 220, (200, 240, 3), np.uint8)
    y, x = np.indices(image.shape[:2])
    fov = (x - 120) ** 2 + (y - 100) ** 2 < 85**2
    vessel = ((x % 13) == 0) & fov
    calibration = calibrate_provided_fov(fov)
    automatic = prepare_vesclass_matlab_calibrated(image, vessel, fov=fov, field_of_view_degrees=8)
    explicit = prepare_vesclass_matlab_calibrated(
        image,
        vessel,
        fov=fov,
        field_of_view_degrees=8,
        fov_diameter_pixels=calibration.diameter_pixels,
    )
    np.testing.assert_array_equal(automatic.image, explicit.image)
    np.testing.assert_array_equal(automatic.inputs(), explicit.inputs())
    assert automatic.geometry == explicit.geometry
    info = automatic.metadata["fov_calibration"]
    assert info["mode"] == "estimated_from_provided_fov"
    assert info["method"] == "souradnice_generic_binary_sobel"
    assert info["center_row_one_based"] == calibration.center_row_one_based
    assert info["radius_pixels"] == calibration.radius_pixels
    assert explicit.metadata["fov_calibration"]["mode"] == "explicit_diameter"


def test_explicit_diameter_bypasses_fov_estimator_even_for_special_source(monkeypatch):
    import deepretina.legacy.matlab_av as profile

    def forbidden(*args, **kwargs):
        pytest.fail("Explicit diameter must take priority over automatic calibration")

    monkeypatch.setattr(profile, "calibrate_provided_fov", forbidden)
    image = np.full((200, 240, 3), 127, np.uint8)
    fov = np.ones((200, 240), bool)
    vessel = np.indices(fov.shape).sum(0) % 2 == 0
    result = profile.prepare_vesclass_matlab_calibrated(
        image,
        vessel,
        fov=fov,
        dataset="riga",
        field_of_view_degrees=8,
        fov_diameter_pixels=200,
    )
    assert result.metadata["fov_calibration"]["mode"] == "explicit_diameter"
    assert result.metadata["fov_calibration"]["center_row_one_based"] is None
