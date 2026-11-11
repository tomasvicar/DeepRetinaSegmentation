"""Synthetic CPU contracts for opt-in fixed-weight inference ensembling."""

import json
import random
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from test_unified import make_manifest

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


@pytest.mark.parametrize("mode", list(u.TTA_OPERATORS))
def test_product_preparation_and_original_geometry_restore_once(tmp_path, monkeypatch, mode):
    raw = (np.arange(13 * 19 * 3) % 255).astype(np.uint8).reshape(13, 19, 3)
    fov = np.ones(raw.shape[:2], bool)
    fov[:2] = False
    Image.fromarray(raw).save(tmp_path / "image.png")
    cfg = u.UnifiedConfig(tile_size=32, overlap=8)
    geometry = ImageGeometry(raw.shape[:2], (0, 12, 0, 18), 1.0, raw.shape[:2])
    prepared = SimpleNamespace(
        image=raw, fov=fov, geometry=geometry, field_of_view_degrees=45, fov_source="synthetic"
    )
    from deepretina.legacy import preprocessing

    prepare_calls, restore_calls, views = [], [], []

    def prepare(*a, **kw):
        prepare_calls.append(kw)
        return prepared

    monkeypatch.setattr(preprocessing, "prepare_single_net", prepare)
    restore = u.restore_probabilities

    def restore_once(*a):
        restore_calls.append(1)
        return restore(*a)

    monkeypatch.setattr(u, "restore_probabilities", restore_once)

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
    output = tmp_path / ("prediction-" + mode)
    u.predict_unified(tmp_path / "fake.pt", tmp_path / "image.png", output, tta=mode)
    assert len(prepare_calls) == 1 and prepare_calls[0]["normalized"] is False
    assert restore_calls == [1]
    expected = torch.from_numpy(raw.transpose(2, 0, 1).copy()).float() / 255
    assert len(views) == len(u.TTA_OPERATORS[mode])
    for image, (rotation, flip) in zip(views, u.TTA_OPERATORS[mode]):
        assert torch.equal(image, u._tta_transform(expected, rotation, flip))
    with np.load(output / "probabilities.npz") as saved:
        assert np.all(saved["probability"][[0, 3, 4]][:, ~fov] == 0)
        assert np.all(saved["probability"][1:3, ~fov] == 0.5)
    metadata = json.loads((output / "prediction.json").read_text())
    assert ("tta" not in metadata) if mode == "none" else metadata["tta"]["mode"] == mode


@pytest.mark.parametrize(
    "precision,mode",
    [(p, m) for p in ["none", "bf16"] for m in u.TTA_OPERATORS if p == "bf16" or m != "none"],
)
def test_tta_fp32_under_ambient_autocast_and_bf16_metadata(precision, mode):
    cfg = u.UnifiedConfig(base_channels=2, depth=1, mixed_precision=precision)
    model = u.build_unified(cfg).eval()
    image = torch.linspace(0, 1, 3 * 35 * 49).reshape(3, 35, 49)
    expected = u.predict_probabilities(model, image, "cpu", cfg, 32, 8, tta=mode)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = u.predict_probabilities(model, image, "cpu", cfg, 32, 8, tta=mode)
    assert np.array_equal(actual, expected)
    assert actual.dtype == np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual[1:3].sum(0), 1, atol=1e-7)


@pytest.mark.parametrize("family", ["legacy_single_net", "legacy_av", "pilot_unet"])
@pytest.mark.parametrize("mode", ["flips2", "rot4", "dihedral8"])
def test_non_unified_models_reject_explicit_tta_before_inference(
    monkeypatch, family, mode, tmp_path
):
    from deepretina import cli

    monkeypatch.setattr(torch, "load", lambda *a, **kw: {"family": family})
    monkeypatch.setattr(cli, "_apply_legacy", lambda *a: pytest.fail("legacy path entered"))
    with pytest.raises(ValueError, match="requires a unified"):
        cli._apply_model(
            SimpleNamespace(tta=mode, command="predict", output=tmp_path / "out"), "unused.pt"
        )


@pytest.mark.parametrize("mode", ["flips2", "rot4", "dihedral8"])
def test_tta_preserves_occupied_prediction_output_guard(tmp_path, monkeypatch, mode):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("original")
    monkeypatch.setattr(u, "load_unified", lambda *a: pytest.fail("model loaded"))
    with pytest.raises(FileExistsError):
        u.predict_unified(tmp_path / "unused.pt", tmp_path / "unused.png", output, tta=mode)
    assert marker.read_text() == "original" and list(output.iterdir()) == [marker]


@pytest.mark.parametrize("mode", ["flips2", "rot4", "dihedral8"])
def test_whole_view_tta_preserves_pointwise_geometry(mode):
    image = torch.linspace(0, 1, 3 * 35 * 49).reshape(3, 35, 49)
    cfg = u.UnifiedConfig(train_view="whole", view_size=32)
    model = Pointwise()
    expected = u.predict_probabilities(model, image, "cpu", cfg, 32, 8)
    result = u.predict_probabilities(model, image, "cpu", cfg, 32, 8, tta=mode)
    assert result.shape == (5, 35, 49) and result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
