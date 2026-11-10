"""Saved native-training recipes must survive raw-image CLI prediction."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from deepretina.legacy.av import LegacyAVNet, checkpoint_payload, load_av
from deepretina.legacy.canonical_av import PROFILE, prepare_raw, validate_inputs
from deepretina.legacy.checkpoints import load_single_net
from deepretina.legacy.inference import predict_legacy
from deepretina.legacy.preprocessing import prepare_single_net
from deepretina.legacy.single_net import Unet


def policy():
    raw = (Path(__file__).parent / "reference/geometry_policy/policy.json").read_bytes()
    return {
        "format_version": 1,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "document_json": raw.decode(),
    }


def inputs():
    y, x = np.mgrid[:48, :64]
    rgb = np.stack((30 + 2 * x, 50 + 2 * y, 40 + x + y), axis=2).astype(np.uint8)
    fov = np.zeros((48, 64), bool)
    fov[4:44, 8:56] = True
    vessel = np.zeros((48, 64), np.uint8)
    vessel[10:38, 24:28] = 255
    vessel[0, 0] = 255  # supplied out-of-FOV pixels are excluded from this recipe
    return rgb, vessel, fov


def av_payload():
    model = LegacyAVNet((4, 2, 4), (4, 2), head_channels=3)
    result = checkpoint_payload(model)
    result["preprocessing"] = {
        "profile": PROFILE,
        "pixels_per_degree": 25,
        "input_values": "RGB/255-0.5 and binary vessel",
    }
    return result


def single_payload(normalized=False):
    architecture = {"filters": 2, "in_size": 3, "out_size": 1, "do": 0, "depth": 2}
    return {
        "format_version": 1,
        "family": "legacy_single_net",
        "architecture": architecture,
        "state_dict": Unet(**architecture).state_dict(),
        "channels": ["vessel"],
        "tiling": {"crop_size": 32, "border": 4},
        "preprocessing": {"normalized": normalized, "pixels_per_degree": 25},
    }


def save_images(tmp_path):
    rgb, vessel, fov = inputs()
    for name, array in [("image", rgb), ("vessel", vessel), ("fov", fov.astype(np.uint8) * 255)]:
        Image.fromarray(array).save(tmp_path / (name + ".png"))
    return rgb, vessel, fov


def test_raw_canonical_av_matches_training_input_and_masks_fov():
    rgb, vessel, fov = inputs()
    prepared, array, provenance = prepare_raw(
        rgb, vessel, av_payload(), fov=fov, field_of_view_degrees=2.56
    )
    canonical = prepare_single_net(rgb, fov=fov, field_of_view_degrees=2.56)
    expected_vessel = canonical.geometry.prepare_mask(vessel > 0) & canonical.fov
    np.testing.assert_array_equal(array[:3], canonical.tensor().numpy())
    np.testing.assert_array_equal(array[3], expected_vessel)
    assert prepared.geometry == canonical.geometry and provenance is None


@pytest.mark.parametrize("shape,angle", [((1934, 1956), 30), ((2000, 2992), 45), ((400, 600), 30)])
def test_native_canonical_preparation_uses_portable_camera_policy(shape, angle):
    h, w = shape
    rgb = np.empty((h, w, 3), np.uint8)
    rgb[..., 0] = np.arange(w)[None] % 128 + 20
    rgb[..., 1] = np.arange(h)[:, None] % 128 + 30
    rgb[..., 2] = rgb[..., 0] // 2 + rgb[..., 1] // 2
    vessel = np.zeros((h, w), np.uint8)
    vessel[h // 4 : 3 * h // 4, w // 2 : w // 2 + 12] = 1
    payload = av_payload()
    payload["geometry_policy"] = policy()
    prepared, array, provenance = prepare_raw(
        rgb, vessel, payload, dataset="drions" if h == 400 else "gamma", fov=np.ones((h, w), bool)
    )
    assert prepared.field_of_view_degrees == angle
    assert provenance["policy_applied"] is True
    assert provenance["native_shape_hw"] == [h, w]
    assert array.shape[1:] == prepared.image.shape[:2]


def test_actual_native_av_prediction_restores_crop_and_requires_explicit_vessel(tmp_path):
    save_images(tmp_path)
    checkpoint = tmp_path / "av.pt"
    torch.save(av_payload(), checkpoint)
    with pytest.raises(ValueError, match="vessel-mask"):
        predict_legacy(checkpoint, tmp_path / "image.png", tmp_path / "missing", device="cpu")
    assert not (tmp_path / "missing").exists()
    output = tmp_path / "prediction"
    predict_legacy(
        checkpoint,
        tmp_path / "image.png",
        output,
        device="cpu",
        vessel_mask_path=tmp_path / "vessel.png",
        fov_path=tmp_path / "fov.png",
        field_of_view_degrees=2.56,
    )
    meta = json.loads((output / "prediction.json").read_text())
    assert meta["preprocessing"]["profile"] == PROFILE
    assert meta["geometry"]["crop_position"] == [4, 43, 8, 55]
    with np.load(output / "probabilities.npz", allow_pickle=False) as data:
        assert data["probability"].shape == (2, 48, 64)
        assert np.isfinite(data["probability"]).all()
        assert not data["vessel_support"][0, 0]
        assert not data["probability"][:, :4].any()
    assert np.asarray(Image.open(output / "classes.png")).shape == (48, 64)


@pytest.mark.parametrize("mode", ["prepared", "prepared_inputs", "dicom_paths", "original_size"])
def test_canonical_profile_rejects_historical_input_modes_before_outputs(tmp_path, mode):
    save_images(tmp_path)
    path = tmp_path / "av.pt"
    torch.save(av_payload(), path)
    value = {
        "prepared": True,
        "prepared_inputs": np.zeros((4, 16, 16)),
        "dicom_paths": ("a", "b", "c", "d"),
        "original_size": (48, 64),
    }[mode]
    with pytest.raises(ValueError, match="historical prepared/DICOM"):
        predict_legacy(
            path, tmp_path / "image.png", tmp_path / "bad", device="cpu", **{mode: value}
        )
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "case", ["constant_rgb", "empty_vessel", "full_vessel", "historical_rgb", "nonfinite"]
)
def test_new_profile_does_not_modify_original_arithmetic_to_hide_bad_inputs(case):
    rng = np.random.default_rng(42)
    array = rng.uniform(-0.4, 0.4, (4, 16, 24)).astype(np.float32)
    array[3] = np.indices((16, 24))[1] % 2
    if case == "constant_rgb":
        array[1] = 0.1
    elif case == "empty_vessel":
        array[3] = 0
    elif case == "full_vessel":
        array[3] = 1
    elif case == "historical_rgb":
        array[:3] *= 4096
    else:
        array[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_inputs(array)


def test_single_net_uses_saved_normalized_choice_unless_explicit(tmp_path, monkeypatch):
    save_images(tmp_path)
    checkpoint = tmp_path / "single.pt"
    torch.save(single_payload(normalized=True), checkpoint)
    captured = []

    def observe(rgb, **kwargs):
        captured.append(kwargs["normalized"])
        return prepare_single_net(rgb, **{**kwargs, "normalized": False})

    monkeypatch.setattr("deepretina.legacy.inference.prepare_single_net", observe)
    for index, override in enumerate([None, False, True]):
        predict_legacy(
            checkpoint,
            tmp_path / "image.png",
            tmp_path / f"out{index}",
            device="cpu",
            normalized=override,
            fov_path=tmp_path / "fov.png",
            field_of_view_degrees=2.56,
        )
    assert captured == [True, False, True]


@pytest.mark.parametrize("family", ["single", "av"])
def test_tampered_native_policy_rejected_before_device_or_output(tmp_path, monkeypatch, family):
    p = single_payload() if family == "single" else av_payload()
    p["geometry_policy"] = copy.deepcopy(policy())
    p["geometry_policy"]["sha256"] = "0" * 64
    path = tmp_path / "bad.pt"
    torch.save(p, path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Device must not be resolved for a tampered policy")

    monkeypatch.setattr("deepretina.legacy.inference.resolve_device", forbidden)
    with pytest.raises(ValueError, match="hash"):
        predict_legacy(path, None, tmp_path / "output", device="cuda:0")
    with pytest.raises(ValueError, match="hash"):
        (load_single_net if family == "single" else load_av)(path, device="cuda:0")
    assert not (tmp_path / "output").exists()
