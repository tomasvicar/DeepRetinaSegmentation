"""Synthetic full-image photometry plus strict HPO TTA/provenance integration."""

import copy
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from test_unified_photometry import corpus as photometry_corpus
from test_unified_selection import case as canonical_case

from deepretina import photometry as photo
from deepretina import selection as gate
from deepretina import unified as u
from deepretina import validation_evidence as producer
from deepretina.completion_profiles import PROFILES
from deepretina.inference_tta import protocol_extension

corpus = photometry_corpus
case = canonical_case


@pytest.mark.parametrize("profile", photo.PROFILES)
@pytest.mark.parametrize("tta", ["none", "flips2", "rot4", "dihedral8"])
def test_full_profile_once_before_tta_and_metric_descriptor(corpus, monkeypatch, profile, tta):
    from types import SimpleNamespace

    manifest, raw, fov, _ = corpus
    cfg = u.UnifiedConfig(photometric_profile=profile, tile_size=64, overlap=16)
    expected = (
        torch.from_numpy(photo.apply(raw, profile, fov=fov).transpose(2, 0, 1).copy()).float() / 255
    )
    calls = []
    views = []

    def apply(*a, **kw):
        calls.append(a[0].shape)
        return photo.apply(*a, **kw)

    def primitive(model, image, *a):
        views.append(image.clone())
        p = np.zeros((5, *image.shape[1:]), np.float32)
        p[0] = 0.1 + 0.8 * image[0].numpy()
        p[1] = 0.8
        p[2] = 0.2
        p[3] = image[1].numpy()
        p[4] = image[2].numpy()
        return p

    monkeypatch.setattr(u, "apply_photometry", apply)
    monkeypatch.setattr(u, "_predict_probabilities", primitive)
    model = SimpleNamespace(lineage={}, supervised_channels=[True] * 5)
    metrics = u.evaluate_unified_model(model, manifest, split="train", config=cfg, tta=tta)
    assert calls == ([] if profile == "raw" else [raw.shape])
    assert len(views) == len(u.TTA_OPERATORS[tta])
    for actual, (rot, flip) in zip(views, u.TTA_OPERATORS[tta]):
        assert torch.equal(actual, u._tta_transform(expected, rot, flip))
    assert {k: v for k, v in metrics.items() if k == "tta"} == protocol_extension(tta)
    if profile == "raw":
        assert "photometry" not in metrics
    else:
        assert metrics["photometry"] == photo.contract(profile)
    assert metrics["photos"] == 1 and metrics["per_photo"][0]["record_ids"] == ["sample"]


def payload(profile):
    cfg = u.UnifiedConfig(photometric_profile=profile)
    return {"config": asdict(cfg), "preprocessing": {"photometry": photo.contract(profile)}}


def test_historical_raw_configuration_identity_and_protocol_schema_stay_exact():
    p = payload("raw")
    del p["config"]["photometric_profile"]
    p.pop("preprocessing")
    identity = gate.config_identity(p)
    assert "photometric_profile" not in identity["values"]
    assert "photometry" not in gate.inference_protocol(p) and "tta" not in gate.inference_protocol(
        p
    )
    p["config"]["photometric_profile"] = "raw"
    assert (
        gate.config_identity(p)["values"]["photometric_profile"] == "raw"
    )  # Never strip a saved field.


@pytest.mark.parametrize("profile", photo.PROFILES[1:])
@pytest.mark.parametrize("tta", ["none", "flips2", "rot4", "dihedral8"])
def test_protocol_binds_saved_photometry_without_losing_tta(profile, tta):
    p = payload(profile)
    original = copy.deepcopy(p)
    protocol = gate.inference_protocol(p, tta=tta)
    assert protocol["photometry"] == p["preprocessing"]["photometry"] and p == original
    assert {k: v for k, v in protocol.items() if k == "tta"} == protocol_extension(tta)


@pytest.mark.parametrize("change", ["missing", "backend", "boolean_version"])
def test_bad_nonraw_descriptor_fails_before_model_or_GT(change, tmp_path, monkeypatch):
    p = payload("clahe_hsv_v1")
    if change == "missing":
        p.pop("preprocessing")
    elif change == "backend":
        p["preprocessing"]["photometry"]["backend_versions"]["numpy"] = "changed"
    else:
        p["preprocessing"]["photometry"]["contract_version"] = True
    with pytest.raises(gate.SelectionError, match="photometry"):
        gate.inference_protocol(p)
    good = payload("clahe_hsv_v1")
    protocol = gate.inference_protocol(good)
    metrics = {} if change == "missing" else {"photometry": copy.deepcopy(protocol["photometry"])}
    if change == "backend":
        metrics["photometry"]["backend_versions"]["numpy"] = "changed"
    if change == "boolean_version":
        metrics["photometry"]["contract_version"] = True
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("Descriptor must fail before GT decode")
    )
    with pytest.raises(gate.SelectionError, match="photometry"):
        gate.validate_metrics(metrics, {}, protocol, u.UnifiedConfig(), tmp_path)


@pytest.mark.parametrize("mode", ["none", "rot4"])
def test_curated_source_still_has_no_new_TTA_completion_admission(case, tmp_path, mode):
    # This synthetic checkpoint's arbitrary new training identity cannot enter any typed completion gate.
    assert set(PROFILES) == {"standalone_0b", "standalone_72", "pool_6b", "pool_72_v3"}
    if mode == "none":
        return
    p = torch.load(case["checkpoint"], weights_only=True)
    p["config"]["photometric_profile"] = "clahe_hsv_v1"
    p["preprocessing"] = {"photometry": photo.contract("clahe_hsv_v1")}
    p["training_source_sha256"] = "f" * 64
    torch.save(p, case["checkpoint"])
    with pytest.raises(gate.SelectionError, match="source72"):
        producer.produce_validation_evidence(
            checkpoint=case["checkpoint"],
            manifest=case["manifest"],
            output=tmp_path / "no.json",
            tta=mode,
        )


def test_producer_forwards_admitted_cache_and_records_portable_identity(
    case, tmp_path, monkeypatch
):
    from test_selection_tta import prepared

    inputs = prepared(case, "rot4")
    p = torch.load(inputs["checkpoint"], weights_only=True)
    p["config"]["photometric_profile"] = "clahe_hsv_v1"
    p["preprocessing"] = {"photometry": photo.contract("clahe_hsv_v1")}
    torch.save(p, inputs["checkpoint"])
    fixture = json.loads(inputs["validation_evidence"].read_text())
    metrics = copy.deepcopy(fixture["metrics"])
    metrics["photometry"] = photo.contract("clahe_hsv_v1")
    identity = {"fingerprint": "reviewed-synthetic-cache", "profile": "clahe_hsv_v1"}

    class Cache:
        def __init__(self, root, manifest, profile, **kw):
            assert (
                root == tmp_path / "cache"
                and manifest == inputs["manifest"]
                and profile == "clahe_hsv_v1"
            )
            assert kw["expected_fingerprint"] == "reviewed-synthetic-cache"
            self.identity = identity

    monkeypatch.setattr(producer, "VerifiedPhotometryCache", Cache)

    def evaluate(*a, **kw):
        assert isinstance(kw["photometry_cache"], Cache)
        return metrics

    monkeypatch.setattr(producer, "evaluate_unified_model", evaluate)
    report = producer.produce_validation_evidence(
        checkpoint=inputs["checkpoint"],
        manifest=inputs["manifest"],
        output=tmp_path / "fresh.json",
        tta="rot4",
        photometry_cache_root=tmp_path / "cache",
        expected_photometry_cache_fingerprint="reviewed-synthetic-cache",
    )
    assert report["inference_protocol"]["photometry"] == photo.contract("clahe_hsv_v1")
    assert report["metrics"]["evaluation_runtime"]["photometry_cache"] == identity
    assert report["configuration"]["values"]["photometric_profile"] == "clahe_hsv_v1"
    assert report["kind"] == "unified_tta_validation_evidence"


@pytest.mark.parametrize("tta", ["none", "flips2", "rot4", "dihedral8"])
def test_header_only_photometry_boolean_alias_rejected_even_with_valid_metrics(tta):
    protocol = gate.inference_protocol(payload("clahe_hsv_v1"), tta=tta)
    valid_metrics = {"photometry": copy.deepcopy(protocol["photometry"])}
    changed = copy.deepcopy(protocol)
    changed["photometry"]["contract_version"] = True
    assert changed == protocol  # Ordinary Python equality is unsafe for typed descriptors.
    assert valid_metrics["photometry"]["contract_version"] == 1
    with pytest.raises(gate.SelectionError, match="photometry header"):
        gate._require_opt_in_header_types(changed, protocol)
    gate._require_opt_in_header_types(protocol, protocol)


@pytest.mark.parametrize("profile", photo.PROFILES[1:])
def test_existing_benchmark_callers_do_not_silently_feed_raw_to_nonraw(
    corpus, monkeypatch, profile
):
    from pathlib import Path
    from types import SimpleNamespace

    from deepretina.legacy.preprocessing import ImageGeometry

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import benchmark_latency as latency
    import benchmark_strong_pipeline as strong

    _, raw, fov, _ = corpus
    geometry = ImageGeometry(
        raw.shape[:2], (0, raw.shape[0] - 1, 0, raw.shape[1] - 1), 1.0, raw.shape[:2]
    )
    prepared = SimpleNamespace(image=raw, fov=fov, geometry=geometry)
    cfg = u.UnifiedConfig(photometric_profile=profile, tile_size=96, overlap=16)
    expected = (
        torch.from_numpy(photo.apply(raw, profile, fov=fov).transpose(2, 0, 1).copy()).float() / 255
    )
    calls = []
    predictions = []

    def transform(*a, **kw):
        calls.append(True)
        return photo.apply(*a, **kw)

    def predict(model, image, device, config, tile, overlap):
        assert torch.equal(image, expected) and (tile, overlap) == (96, 16)
        predictions.append(True)
        return np.zeros((5, *raw.shape[:2]), np.float32)

    for module in [strong, latency]:
        monkeypatch.setattr(module, "prepare_single_net", lambda *a, **k: prepared)
        monkeypatch.setattr(module, "apply_photometry", transform)
        monkeypatch.setattr(module, "predict_probabilities", predict)
    model = SimpleNamespace(supervised_channels=[True] * 5)
    strong.unified_operation(raw, fov, model, cfg, "cpu", {})
    assert calls == [True] and predictions == [True]
    calls.clear()
    predictions.clear()
    operation, scope = latency.unified_timing_operation(
        raw, {}, model, {"config": asdict(cfg)}, "cpu"
    )
    assert calls == [True] and predictions == []
    operation()
    operation()
    assert calls == [True] and predictions == [True, True] and "excluded" in scope["timing_scope"]
