"""Product photometry integration: synthetic RGB and mocked pointwise inference only."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_unified_photometry import corpus as photometry_corpus

from deepretina import photometry as photo
from deepretina import unified as u
from deepretina.cli import parser
from deepretina.legacy.preprocessing import ImageGeometry

corpus = photometry_corpus


@pytest.mark.parametrize("profile", photo.PROFILES)
@pytest.mark.parametrize("tta", ["none", "flips2", "rot4", "dihedral8"])
@pytest.mark.parametrize("precision", ["none", "bf16"])
def test_saved_profile_applied_once_before_all_tta_views(
    corpus, tmp_path, monkeypatch, profile, tta, precision
):
    manifest, raw, fov, _ = corpus
    config = u.UnifiedConfig(
        photometric_profile=profile, mixed_precision=precision, tile_size=64, overlap=16
    )
    prepared = SimpleNamespace(
        image=raw.copy(),
        fov=fov,
        geometry=ImageGeometry(
            raw.shape[:2], (0, raw.shape[0] - 1, 0, raw.shape[1] - 1), 1.0, raw.shape[:2]
        ),
        field_of_view_degrees=45,
        fov_source="synthetic image-derived",
    )
    import deepretina.legacy.preprocessing as geometry

    monkeypatch.setattr(geometry, "prepare_single_net", lambda *a, **k: prepared)
    expected = photo.apply(raw, profile, fov=fov)
    calls = []

    def process(image, selected, **kwargs):
        calls.append((image.shape, selected))
        return photo.apply(image, selected, **kwargs)

    monkeypatch.setattr(u, "apply_photometry", process)
    model = SimpleNamespace(lineage={}, architecture_spec={})
    payload = {
        "config": asdict(config),
        "supervised_channels": [True] * 5,
        "purpose": "functionality",
        "checkpoint_sha256": "synthetic",
    }
    monkeypatch.setattr(u, "load_unified", lambda *a: (model, payload, torch.device("cpu")))
    seen = []

    def primitive(model, image, device, cfg, tile, overlap):
        # Mock model primitive sees each correctly transformed full prepared frame.
        assert image.dtype == torch.float32
        seen.append(image.clone())
        p = np.zeros((5, *image.shape[1:]), np.float32)
        p[0] = 0.1 + 0.8 * image[0].numpy()
        p[1] = 0.8
        p[2] = 0.2
        p[3] = image[1].numpy()
        p[4] = image[2].numpy()
        return p

    monkeypatch.setattr(u, "_predict_probabilities", primitive)
    out = tmp_path / "prediction"
    u.predict_unified(
        tmp_path / "mock.pt", manifest.parent / "image.png", out, device="cpu", tta=tta
    )
    assert calls == [(raw.shape, profile)]
    base = torch.from_numpy(expected.transpose(2, 0, 1).copy()).float() / 255
    assert len(seen) == len(u.TTA_OPERATORS[tta])
    for actual, (rotation, flip) in zip(seen, u.TTA_OPERATORS[tta]):
        assert torch.equal(actual, u._tta_transform(base, rotation, flip))
    with np.load(out / "probabilities.npz") as saved:
        probability = saved["probability"]
    # Pointwise predictions make inverse alignment exact up to FP32 summation.
    expected_probability = np.zeros((5, *raw.shape[:2]), np.float32)
    expected_probability[0] = 0.1 + 0.8 * base[0]
    expected_probability[1] = 0.8
    expected_probability[2] = 0.2
    expected_probability[3] = base[1]
    expected_probability[4] = base[2]
    expected_probability[[0, 3, 4]] *= fov[None]
    expected_probability[1:3, ~fov] = 0.5
    np.testing.assert_allclose(probability, expected_probability, rtol=2e-6, atol=2e-7)
    metadata = json.loads((out / "prediction.json").read_text())
    if profile == "raw":
        assert "photometry" not in metadata["preprocessing"]
    else:
        assert metadata["preprocessing"]["photometry"] == photo.contract(profile)


def test_train_cli_forwards_pinned_cache_without_changing_saved_config(tmp_path, monkeypatch):
    import deepretina.unified_cli as cli

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"photometric_profile": "clahe_hsv_v1"}))
    args = parser().parse_args(
        [
            "train-unified",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output",
            str(tmp_path / "run"),
            "--config",
            str(path),
            "--photometry-cache",
            str(tmp_path / "rgb"),
            "--expected-content-sha256",
            "canonical",
            "--expected-photometry-cache-fingerprint",
            "cache",
        ]
    )
    seen = []

    def train(manifest, output, cfg, **kwargs):
        seen.append((cfg, kwargs))
        return output / "last.pt"

    monkeypatch.setattr(cli, "train_unified", train)
    cli.train_from_args(args)
    assert seen[0][0].photometric_profile == "clahe_hsv_v1"
    assert seen[0][1]["photometry_cache_root"] == tmp_path / "rgb"
    assert seen[0][1]["expected_photometry_cache_fingerprint"] == "cache"
    assert "photometry_cache_root" not in asdict(seen[0][0])


@pytest.mark.parametrize("profile", photo.PROFILES[1:])
def test_missing_descriptor_or_backend_change_rejected_before_model(tmp_path, monkeypatch, profile):
    def forbidden(*a, **k):
        raise AssertionError("Model must not be constructed")

    monkeypatch.setattr(u, "build_unified", forbidden)
    payload = {
        "family": "unified",
        "format_version": 1,
        "channels": list(u.CHANNELS),
        "normalization": "RGB/255-0.5",
        "config": asdict(u.UnifiedConfig(photometric_profile=profile)),
        "state_dict": {},
        "preprocessing": {},
    }
    p = tmp_path / "model.pt"
    torch.save(payload, p)
    with pytest.raises(ValueError, match="photometry parameters/backend"):
        u.load_unified(p, "cpu")
    payload["preprocessing"]["photometry"] = photo.contract(profile)
    payload["preprocessing"]["photometry"]["backend_versions"]["numpy"] = "changed"
    torch.save(payload, p)
    with pytest.raises(ValueError, match="photometry parameters/backend"):
        u.load_unified(p, "cpu")


@pytest.mark.parametrize("profile", photo.PROFILES[1:])
def test_historical_benchmark_never_feeds_raw_rgb_to_saved_photometry(corpus, monkeypatch, profile):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import benchmark_latency as benchmark

    _, raw, fov, _ = corpus
    cfg = u.UnifiedConfig(photometric_profile=profile, tile_size=96, overlap=16)
    geometry = ImageGeometry(
        raw.shape[:2], (0, raw.shape[0] - 1, 0, raw.shape[1] - 1), 1.0, raw.shape[:2]
    )
    monkeypatch.setattr(
        benchmark,
        "prepare_single_net",
        lambda *a, **k: SimpleNamespace(image=raw, fov=fov, geometry=geometry),
    )
    expected = (
        torch.from_numpy(photo.apply(raw, profile, fov=fov).transpose(2, 0, 1).copy()).float() / 255
    )
    transforms = []
    predictions = []

    def transform(*args, **kwargs):
        transforms.append(True)
        return photo.apply(*args, **kwargs)

    def predict(model, image, device, config, tile, overlap):
        assert torch.equal(image, expected) and (tile, overlap) == (96, 16)
        predictions.append(True)
        return np.zeros((5, *raw.shape[:2]), np.float32)

    monkeypatch.setattr(benchmark, "apply_photometry", transform)
    monkeypatch.setattr(benchmark, "predict_probabilities", predict)
    operation, scope = benchmark.unified_timing_operation(
        raw, {}, SimpleNamespace(supervised_channels=[True] * 5), {"config": asdict(cfg)}, "cpu"
    )
    assert transforms == [True] and predictions == []
    for _ in range(3):
        operation()
    assert transforms == [True] and predictions == [True] * 3
    assert scope["photometry"] == photo.contract(profile) and "excluded" in scope["timing_scope"]


@pytest.mark.parametrize("field,value", [("contract_version", True), ("matlab_parity_claim", 0)])
def test_saved_descriptor_json_types_are_strict_before_model(tmp_path, monkeypatch, field, value):
    cfg = u.UnifiedConfig(photometric_profile="clahe_hsv_v1")
    payload = {
        "family": "unified",
        "format_version": 1,
        "channels": list(u.CHANNELS),
        "normalization": "RGB/255-0.5",
        "config": asdict(cfg),
        "state_dict": {},
        "preprocessing": {"photometry": photo.contract(cfg.photometric_profile)},
    }
    payload["preprocessing"]["photometry"][field] = value
    p = tmp_path / "mock.pt"
    torch.save(payload, p)
    monkeypatch.setattr(
        u, "build_unified", lambda *a, **k: pytest.fail("No model before descriptor admission")
    )
    with pytest.raises(ValueError, match="JSON types differ"):
        u.load_unified(p, "cpu")


def test_dropping_saved_profile_field_cannot_relabel_nonraw_weights_as_raw(tmp_path, monkeypatch):
    config = asdict(u.UnifiedConfig(photometric_profile="clahe_hsv_v1"))
    del config["photometric_profile"]
    payload = {
        "family": "unified",
        "format_version": 1,
        "channels": list(u.CHANNELS),
        "normalization": "RGB/255-0.5",
        "config": config,
        "state_dict": {},
        "preprocessing": {"photometry": photo.contract("clahe_hsv_v1")},
    }
    p = tmp_path / "mock.pt"
    torch.save(payload, p)
    monkeypatch.setattr(
        u, "build_unified", lambda *a, **k: pytest.fail("Never construct a relabeled model")
    )
    with pytest.raises(ValueError, match="photometry parameters/backend"):
        u.load_unified(p, "cpu")
