"""Scientific eligibility must not be inferred from missing provenance."""

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "benchmark_manifest", Path(__file__).resolve().parents[1] / "scripts/benchmark_manifest.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
audit = _module.audit


def inputs(tmp_path, mappings, rows=None):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(rows or [{"id": "row", "photo_id": "photo", "split": "test"}]))
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps({"images": {"old0": {"split": "train"}, "old1": {"split": "test"}}})
    )
    crosswalk = tmp_path / "crosswalk.json"
    crosswalk.write_text(json.dumps({"schema_version": 1, "mappings": mappings}))
    return manifest, {"universal": index}, crosswalk


def mapping(**kwargs):
    return {
        "photo_id": "photo",
        "family": "universal",
        "verified": True,
        "evidence": "reviewed immutable original_filename attribute",
        **kwargs,
    }


def test_missing_mapping_fails_closed(tmp_path):
    result = audit(*inputs(tmp_path, []))
    assert result["counts"]["gradient_unseen_all"] == 0
    assert result["counts"]["historically_untouched_all"] == 0


def test_legacy_test_is_gradient_unseen_but_tuning_exposed(tmp_path):
    result = audit(*inputs(tmp_path, [mapping(legacy_image_ids=["old1"])]))
    assert result["counts"]["gradient_unseen_all"] == 1
    assert result["counts"]["historically_untouched_all"] == 0


def test_any_original_training_alias_disqualifies(tmp_path):
    result = audit(*inputs(tmp_path, [mapping(legacy_image_ids=["old0", "old1"])]))
    assert result["counts"]["gradient_unseen_all"] == 0


def test_unknown_legacy_id_cannot_prove_absence(tmp_path):
    result = audit(*inputs(tmp_path, [mapping(legacy_image_ids=["missing"])]))
    assert result["counts"]["gradient_unseen_all"] == 0


def test_explicit_verified_absence_is_required(tmp_path):
    result = audit(*inputs(tmp_path, [mapping(legacy_image_ids=[])]))
    assert result["counts"]["historically_untouched_all"] == 0
    result = audit(
        *inputs(tmp_path, [mapping(legacy_image_ids=[], verified_absent_from_corpus=True)])
    )
    assert result["counts"]["historically_untouched_all"] == 1


def test_photo_cross_split_collision_disqualifies(tmp_path):
    rows = [
        {"id": "a", "photo_id": "photo", "split": "train"},
        {"id": "b", "photo_id": "photo", "split": "test"},
    ]
    result = audit(
        *inputs(tmp_path, [mapping(legacy_image_ids=[], verified_absent_from_corpus=True)], rows)
    )
    assert result["counts"]["split_collisions"] == 1
    assert result["counts"]["historically_untouched_all"] == 0


def test_patient_id_is_not_a_photo_identity(tmp_path):
    with pytest.raises(ValueError, match="photo_id"):
        audit(*inputs(tmp_path, [], [{"id": "a", "source_id": "patient", "split": "test"}]))
