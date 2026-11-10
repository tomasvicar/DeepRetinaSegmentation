"""Metadata contract checks without opening real reports, GT, or models."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "completed_val", Path(__file__).parents[1] / "scripts/compare_completed_val.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def matched():
    evidence = {
        "kind": "unified_validation_evidence",
        "metrics": {"split": "val", "photos": 770, "threshold": 0.5},
        "configuration": {"values": {"purpose": "full_data"}},
        "inference_protocol": {"threshold": 0.5},
        "data": {
            "manifest_sha256": "manifest",
            "prepared_content_sha256": "canonical",
            "eligible_photo_counts": {"train": 2221},
        },
        "checkpoint": {"sha256": "best", "epoch": 56},
        "source": {"training_source_sha256": "source"},
    }
    qa = {
        "terminal": True,
        "last_epoch": 60,
        "finite_best_last_weights_and_last_optimizer": True,
        "checkpoint_sha256": {"best.pt": "best"},
        "best_epoch": 56,
        "manifest_sha256": "manifest",
        "canonical_fingerprint": "canonical",
        "checkpoint_training_source_sha256": "source",
        "eligible_photos": 2221,
        "minimum_photo_visits": 60,
        "delivered_samples": 60 * 8192,
    }
    return evidence, qa


def test_completed60_can_select_earlier_best_epoch(matched):
    evidence, qa = matched
    assert m.unwrap(evidence)["checkpoint_sha256"] == "best"
    m.verify_completion(qa, evidence)


@pytest.mark.parametrize(
    "key,value",
    [
        ("terminal", False),
        ("last_epoch", 59),
        ("finite_best_last_weights_and_last_optimizer", False),
        ("best_epoch", 60),
        ("manifest_sha256", "different"),
        ("canonical_fingerprint", "different"),
        ("checkpoint_training_source_sha256", "different"),
        ("eligible_photos", 2220),
        ("minimum_photo_visits", 59),
        ("delivered_samples", 8192),
    ],
)
def test_wrong_completion_identity_or_budget_rejected(matched, key, value):
    evidence, qa = matched
    qa[key] = value
    with pytest.raises(ValueError):
        m.verify_completion(qa, evidence)


def test_threshold_and_split_must_be_actual_enriched_val(matched):
    evidence, _ = matched
    bad = copy.deepcopy(evidence)
    bad["metrics"]["split"] = "test"
    with pytest.raises(ValueError):
        m.unwrap(bad)
    evidence["inference_protocol"]["threshold"] = 0.35
    with pytest.raises(ValueError):
        m.unwrap(evidence)


def test_self_consistent_subset_completion_cannot_replace_manifest_domain(matched):
    evidence, qa = matched
    qa["eligible_photos"] = 100
    evidence["data"]["eligible_photo_counts"]["train"] = 100
    with pytest.raises(ValueError, match="coverage"):
        m.verify_completion(qa, evidence)
    m.verify_completion(
        qa, evidence, expected_train_photo_count=100
    )  # Count must come from bound trusted manifest in compare().


def test_eligible_training_ids_follow_frozen_reader_filters(tmp_path):
    import json

    rows = [
        {"photo_id": "p0", "split": "train", "task": "disc_cup"},
        {"photo_id": "p0", "split": "train", "task": "vessels"},
        {"photo_id": "val", "split": "val", "task": "av"},
        {"photo_id": "invalid", "split": "train", "task": "images"},
        {"photo_id": "excluded", "split": "train", "task": "av", "exclude_from_training": True},
        {
            "photo_id": "unusable",
            "split": "train",
            "task": "disc_cup",
            "usable_for_training": False,
        },
    ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(rows))
    assert m.eligible_train_photo_ids(path) == ["p0"]
