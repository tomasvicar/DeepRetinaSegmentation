"""Portable preparation policy, original native dimensions, and old model paths."""

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from test_unified import make_manifest

from deepretina.geometry_policy import (
    capture_geometry_policy,
    geometry_rule,
    prediction_geometry,
    validate_geometry_policy,
)
from deepretina.unified import (
    CHANNELS,
    UnifiedConfig,
    build_unified,
    load_unified,
    predict_unified,
    train_unified,
)

POLICY = Path(__file__).parent / "reference/geometry_policy/policy.json"
SHA = "2591d8461b56433529105532c20203e0b17a48c781c1a74fd3a2901bcb301e3d"


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def embedded():
    raw = POLICY.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA
    return {"format_version": 1, "sha256": SHA, "document_json": raw.decode()}


def wrap(document):
    raw = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return {
        "format_version": 1,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "document_json": raw.decode(),
    }


@pytest.mark.parametrize(
    "shape,angle,method",
    [
        ((1934, 1956), 30, "native_shape_rule"),
        ((2000, 2992), 45, "native_shape_rule"),
        ((1956, 1934), 45, "explicit_unseen_shape_nominal_fallback"),
        ((256, 256), 45, "explicit_unseen_shape_nominal_fallback"),
    ],
)
def test_gamma_uses_exact_original_native_h_w_not_transpose_or_prepared_shape(shape, angle, method):
    effective, meta = prediction_geometry(embedded(), "gamma", shape)
    assert effective == angle and meta["policy_applied"] is True
    assert meta["declared_rule"]["camera_attribution_method"] == method
    assert meta["unseen_native_shape_fallback"] == (method != "native_shape_rule")
    assert meta["native_shape_hw"] == list(shape)
    assert meta["declared_rule"]["per_image_angle_measured"] is False


@pytest.mark.parametrize(
    "name,angle,nominal", [("drions", 30, True), ("idrid_disc", 50, False), ("biomisa", 45, True)]
)
def test_declared_flat_rules_retain_nominal_or_reported_qualification(name, angle, nominal):
    effective, meta = prediction_geometry(embedded(), name, (80, 96))
    assert effective == angle and meta["declared_rule"]["nominal_assumption"] is nominal
    assert meta["policy_sha256"] == SHA


def test_explicit_angle_wins_and_metadata_retains_unapplied_camera_rule():
    effective, meta = prediction_geometry(embedded(), "gamma", (1934, 1956), 42)
    assert effective == 42 and meta["policy_applied"] is False
    assert meta["selection"] == "explicit_field_of_view_degrees_override"
    assert meta["declared_rule"]["field_of_view_degrees"] == 30
    assert meta["override_measurement_claim"] is False


def test_old_checkpoints_and_legacy_datasets_do_not_gain_new_nominal_defaults():
    assert prediction_geometry(None, "gamma", (1934, 1956)) == (None, None)
    assert prediction_geometry(None, "drive", (80, 96), 42) == (42, None)
    angle, meta = prediction_geometry(embedded(), "drive", (80, 96))
    assert angle is None and meta["policy_applied"] is False
    assert prediction_geometry(embedded(), None, (1934, 1956))[0] is None


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x.update(format_version=2),
        lambda x: x.update(format_version=True),
        lambda x: x.update(document_json=x["document_json"] + " "),
        lambda x: x.update(sha256="0" * 64),
    ],
)
def test_embedded_version_or_exact_byte_hash_mismatch_fails(change):
    value = embedded()
    change(value)
    with pytest.raises(ValueError):
        validate_geometry_policy(value)


@pytest.mark.parametrize(
    "key,value",
    [("schema_version", 2), ("target_pixels_per_degree", 35), ("physical_measurement_claim", True)],
)
def test_document_schema_and_coordinate_contract_are_bound(key, value):
    doc = json.loads(embedded()["document_json"])
    doc[key] = value
    with pytest.raises(ValueError):
        validate_geometry_policy(wrap(doc))


def row():
    value = embedded()
    rule = geometry_rule(value, "drions", (32, 40))
    return {
        "dataset": "drions",
        "field_of_view_degrees": 30.0,
        "metadata": {
            "geometry_policy_sha256": SHA,
            "policy_sha256": SHA,
            "physical_calibration": rule,
            "original_geometry": {"original_shape": [32, 40]},
        },
    }


def test_capture_binds_sidecar_row_identity_rule_and_native_geometry(tmp_path):
    (tmp_path / "geometry-policy.json").write_bytes(POLICY.read_bytes())
    record = row()
    captured = capture_geometry_policy(tmp_path, [record])
    assert captured == embedded()
    for altered in [
        dict(record, field_of_view_degrees=45),
        dict(record, metadata={**record["metadata"], "policy_sha256": "f" * 64}),
        dict(
            record,
            metadata={**record["metadata"], "original_geometry": {"original_shape": [40, 32]}},
        ),
    ]:
        with pytest.raises(ValueError):
            capture_geometry_policy(tmp_path, [altered])
    (tmp_path / "geometry-policy.json").write_bytes(POLICY.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash"):
        capture_geometry_policy(tmp_path, [record])


def test_capture_ignores_orphan_sidecar_for_old_manifest_but_rejects_unbound_new_source(tmp_path):
    (tmp_path / "geometry-policy.json").write_text("not used by v1")
    assert capture_geometry_policy(tmp_path, [{"dataset": "drive"}]) is None
    with pytest.raises(ValueError):
        capture_geometry_policy(tmp_path, [{"dataset": "gamma"}])


def checkpoint(path, policy=None):
    cfg = UnifiedConfig(base_channels=2, depth=1, norm="group", tile_size=512, overlap=64)
    model = build_unified(cfg)
    payload = {
        "format_version": 1,
        "family": "unified",
        "channels": list(CHANNELS),
        "normalization": "RGB/255-0.5",
        "config": asdict(cfg),
        "state_dict": model.state_dict(),
        "purpose": "functionality",
        "supervised_channels": [True] * 5,
    }
    if policy is not None:
        payload["geometry_policy"] = policy
    torch.save(payload, path)
    return payload


def test_bad_checkpoint_policy_rejected_before_model_build(tmp_path):
    path = tmp_path / "bad.pt"
    policy = embedded()
    policy["sha256"] = "0" * 64
    checkpoint(path, policy)
    with patch("deepretina.unified.build_unified") as build:
        with pytest.raises(ValueError, match="hash"):
            load_unified(path, "cpu")
        build.assert_not_called()


def test_actual_raw_prediction_uses_embedded_policy_without_external_file_and_restores_native_frame(
    tmp_path,
):
    path = tmp_path / "portable.pt"
    checkpoint(path, embedded())
    rgb = np.full((80, 96, 3), 120, np.uint8)
    image = tmp_path / "raw.png"
    Image.fromarray(rgb).save(image)
    predict_unified(
        path,
        image,
        tmp_path / "output",
        device="cpu",
        dataset="drions",
        fov=np.ones(rgb.shape[:2], bool),
    )
    metadata = json.loads((tmp_path / "output/prediction.json").read_text())
    prep = metadata["preprocessing"]
    assert prep["field_of_view_degrees"] == 30 and prep["geometry_policy"]["policy_applied"] is True
    assert prep["geometry"]["original_shape"] == [80, 96]
    assert prep["geometry_policy"]["policy_sha256"] == SHA
    with np.load(tmp_path / "output/probabilities.npz") as data:
        assert data["probability"].shape == (5, 80, 96)
    assert Image.open(tmp_path / "output/vessel.png").size == (96, 80)


def test_synthetic_training_persists_policy_in_run_best_last_and_checks_resume_binding(tmp_path):
    manifest = make_manifest(tmp_path)
    rows = json.loads(manifest.read_text())
    for record in rows:
        if record["id"] == "train_disc_cup":
            record.update(row())
    manifest.write_text(json.dumps(rows))
    (tmp_path / "geometry-policy.json").write_bytes(POLICY.read_bytes())
    cfg = UnifiedConfig(
        epochs=2,
        epoch_samples=2,
        batch_size=2,
        patch_size=32,
        base_channels=2,
        depth=1,
        norm="group",
        workers=0,
        device="cpu",
        augmentation="basic",
        tile_size=32,
        overlap=8,
    )
    output = tmp_path / "run"
    validator = lambda *args, **kwargs: {"mean_dice": 0.5, "macro_dataset_mean_dice": 0.5}
    train_unified(manifest, output, cfg, stop_after_epoch=1, validation_fn=validator)
    assert json.loads((output / "run.json").read_text())["geometry_policy"] == embedded()
    for name in ["best.pt", "last.pt"]:
        saved = torch.load(output / name, weights_only=True)
        assert saved["geometry_policy"] == embedded()
    train_unified(manifest, output, cfg, resume=output / "last.pt", validation_fn=validator)
    changed = torch.load(output / "last.pt", weights_only=True)
    changed["geometry_policy"] = None
    torch.save(changed, output / "last.pt")
    with pytest.raises(ValueError, match="geometry policy changed"):
        train_unified(manifest, output, cfg, resume=output / "last.pt", validation_fn=validator)


@pytest.mark.parametrize(
    "dataset,shape",
    [
        ("gamma", (1934, 1956)),
        ("gamma", (2000, 2992)),
        ("gamma", (1956, 1934)),
        ("gamma", (32, 40)),
        ("drions", (400, 600)),
        ("idrid_disc", (2848, 4288)),
        ("biomisa", (992, 1550)),
    ],
)
def test_prediction_rule_matches_unmodified_corpus_preparer(dataset, shape):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "original_angle_for", POLICY.with_name("prepare_angle_for.py")
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    assert geometry_rule(embedded(), dataset, shape) == reference.angle_for(
        json.loads(embedded()["document_json"]), dataset, shape
    )


def test_policy_cannot_override_historical_dataset_rules():
    doc = json.loads(embedded()["document_json"])
    doc["policies"]["drive"] = copy.deepcopy(doc["policies"]["drions"])
    with pytest.raises(ValueError, match="new corpus-v2 source names"):
        validate_geometry_policy(wrap(doc))


def test_explicit_unknown_gamma_shape_records_unknown_without_claiming_fallback_was_applied():
    angle, meta = prediction_geometry(embedded(), "gamma", (256, 256), 30)
    assert angle == 30 and meta["unseen_native_shape"] is True
    assert meta["unseen_native_shape_fallback"] is False and meta["policy_applied"] is False
    assert meta["declared_rule"]["field_of_view_degrees"] == 45
