"""Synthetic CPU contracts for opt-in fixed-weight inference ensembling."""

import json
import random
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_unified import make_manifest
from test_unified_photometry import corpus  # noqa: F401

from deepretina import unified as u
from deepretina.legacy.preprocessing import ImageGeometry


class Pointwise(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.register_buffer("offset", torch.zeros(()))

    def forward(self, image):
        image = image * self.scale + self.offset
        return torch.cat(
            (image[:, :1], image[:, 1:2], -image[:, 1:2], image[:, 2:3], -image[:, 2:3]), 1
        )


@pytest.mark.parametrize("mode", u.TTA_OPERATORS)
def test_transforms_inverse_odd_asymmetric_rectangles(mode):
    image = torch.arange(3 * 5 * 9).reshape(3, 5, 9)
    for rotation, flip in u.TTA_OPERATORS[mode]:
        changed = u._tta_transform(image, rotation, flip)
        recovered = u._tta_inverse(changed.numpy(), rotation, flip)
        assert np.array_equal(recovered, image.numpy())


@torch.inference_mode()
@pytest.mark.parametrize("view", ["patch", "whole"])
def test_none_exact_old_arithmetic_and_rng(view):
    image = torch.linspace(0, 1, 3 * 35 * 49).reshape(3, 35, 49)
    model = Pointwise()
    cfg = u.UnifiedConfig(train_view=view, view_size=32)
    state = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
    old = u._predict_probabilities(model, image, "cpu", cfg, 32, 8)
    default = u.predict_probabilities(model, image, "cpu", cfg, 32, 8)
    explicit = u.predict_probabilities(model, image, "cpu", cfg, 32, 8, tta="none")
    assert np.array_equal(old, default) and np.array_equal(old, explicit)
    assert torch.equal(state[0], torch.get_rng_state()) and state[1] == random.getstate()
    assert all(np.array_equal(a, b) for a, b in zip(state[2], np.random.get_state()))
    assert u.tta_protocol("none") == {}


@pytest.mark.parametrize("mode", ["flips2", "rot4", "dihedral8"])
def test_actual_tiled_ensemble_inverse_alignment_float32_and_no_weight_mutation(mode):
    image = torch.linspace(0, 1, 3 * 35 * 49).reshape(3, 35, 49)
    model = Pointwise()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    expected = u.predict_probabilities(model, image, "cpu", tile_size=32, overlap=8)
    result = u.predict_probabilities(model, image, "cpu", tile_size=32, overlap=8, tta=mode)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
    assert result.dtype == np.float32
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_ensemble_weights_joint_av_by_vessel_and_zero_vessel(monkeypatch):
    predictions = []
    for vessel, artery in [(0.9, 0.8), (0.1, 0.1)]:
        p = np.zeros((5, 5, 9), np.float32)
        p[0] = vessel
        p[1] = artery
        p[2] = 1 - artery
        p[0, 2, 4] = 0
        predictions.append(p)
    monkeypatch.setattr(u, "_predict_probabilities", lambda *a: predictions.pop(0))
    result = u.predict_probabilities(Pointwise(), torch.zeros(3, 5, 9), "cpu", tta="flips2")
    assert result[1, 0, 0] == pytest.approx(0.73)
    assert result[1, 0, 0] != pytest.approx((0.8 + 0.1) / 2)
    assert np.array_equal(result[1:3, 2, 4], [0.5, 0.5])


def test_invalid_mode_fails_before_model(monkeypatch):
    monkeypatch.setattr(u, "_predict_probabilities", lambda *a: pytest.fail("model called"))
    with pytest.raises(ValueError, match="TTA mode"):
        u.predict_probabilities(None, None, "cpu", tta="bad")


def test_evaluation_preserves_gt_support_annotation_trace(tmp_path):
    manifest = make_manifest(tmp_path)
    model = Pointwise()
    model.supervised_channels = [True, True, True, False, False]
    baseline = u.evaluate_unified_model(
        model, manifest, config=u.UnifiedConfig(tile_size=32, overlap=8)
    )
    result = u.evaluate_unified_model(
        model, manifest, config=u.UnifiedConfig(tile_size=32, overlap=8), tta="dihedral8"
    )
    for key in ("photos", "records", "supervised_channels", "metric_support", "evaluation_policy"):
        assert result[key] == baseline[key]
    for old, new in zip(baseline["per_photo"], result["per_photo"]):
        for key in ("photo_id", "record_ids", "dataset", "channel_datasets"):
            assert old[key] == new[key]
        for channel in u.CHANNELS:
            assert (
                old["channels"][channel]["valid_pixels"] == new["channels"][channel]["valid_pixels"]
            )
            assert (
                old["channels"][channel]["tp"] + old["channels"][channel]["fn"]
                == new["channels"][channel]["tp"] + new["channels"][channel]["fn"]
            )
    assert "tta" not in baseline and result["tta"]["mode"] == "dihedral8"


def test_preprocessing_and_fov_once_outside_full_prepared_transforms(corpus, tmp_path, monkeypatch):  # noqa: F811
    _, raw, fov, _ = corpus
    cfg = u.UnifiedConfig(photometric_profile="clahe_hsv_v1", tile_size=32, overlap=8)
    geometry = ImageGeometry(
        raw.shape[:2], (0, raw.shape[0] - 1, 0, raw.shape[1] - 1), 1.0, raw.shape[:2]
    )
    prepared = SimpleNamespace(
        image=raw, fov=fov, geometry=geometry, field_of_view_degrees=45, fov_source="synthetic"
    )
    from deepretina.legacy import preprocessing

    preparation_calls, photometry_calls, views = [], [], []

    def prepare(*a, **kw):
        preparation_calls.append(1)
        return prepared

    monkeypatch.setattr(preprocessing, "prepare_single_net", prepare)
    photometry = u.apply_photometry

    def apply(*a, **kw):
        photometry_calls.append(kw["fov"].copy())
        return photometry(*a, **kw)

    monkeypatch.setattr(u, "apply_photometry", apply)
    expected = photometry(raw, cfg.photometric_profile, fov=fov, pixels_per_degree=25)
    tensor = torch.from_numpy(expected.transpose(2, 0, 1).copy()).float() / 255

    def predict(model, image, device, *a):
        views.append(image)
        return np.full((5, *image.shape[-2:]), 0.5, np.float32)

    monkeypatch.setattr(u, "_predict_probabilities", predict)
    model = SimpleNamespace(lineage={}, architecture_spec={})
    payload = {
        "config": asdict(cfg),
        "supervised_channels": [True] * 5,
        "checkpoint_sha256": "synthetic",
        "purpose": "functionality",
    }
    monkeypatch.setattr(u, "load_unified", lambda *a: (model, payload, "cpu"))
    u.predict_unified(
        tmp_path / "fake.pt", tmp_path / "image.png", tmp_path / "out", tta="dihedral8"
    )
    assert preparation_calls == [1] and len(photometry_calls) == 1
    assert np.array_equal(photometry_calls[0], fov)
    assert len(views) == 8
    for image, (rotation, flip) in zip(views, u.TTA_OPERATORS["dihedral8"]):
        assert torch.equal(image, u._tta_transform(tensor, rotation, flip))
    with np.load(tmp_path / "out" / "probabilities.npz") as saved:
        assert np.all(saved["probability"][[0, 3, 4]][:, ~fov] == 0)
        assert np.all(saved["probability"][1:3, ~fov] == 0.5)
    metadata = json.loads((tmp_path / "out" / "prediction.json").read_text())
    assert metadata["tta"]["original_standalone_certification"] is False
