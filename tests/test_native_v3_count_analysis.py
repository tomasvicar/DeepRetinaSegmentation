"""Small metadata-only checks; genuine completed inputs supply the positive proof."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "native_counts", Path(__file__).parents[1] / "scripts/compare_native_v3_completed_val.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def actual_metrics():
    path = (
        Path(__file__).parents[1]
        / "reports/v3-completed-native-val-comparison/control0-training_best-counts.json"
    )
    return json.loads(path.read_text())


def test_saved_exact_count_view_preserves_all_integer_evidence():
    metrics = actual_metrics()
    result = MODULE.view(metrics, metrics["checkpoint_sha256"], {})
    assert result["per_photo"] == metrics["per_photo"]
    assert result["channels"] == metrics["channels"]
    assert result["kind"] != "unified_validation_evidence"


def test_count_view_rejects_partial_photo_membership():
    metrics = actual_metrics()
    metrics["per_photo"].pop()
    with pytest.raises(ValueError, match="fixed770"):
        MODULE.view(metrics, "unused", {})


def test_metadata_writer_never_overwrites(tmp_path):
    path = tmp_path / "evidence.json"
    MODULE.save(path, {"original": True})
    with pytest.raises(FileExistsError):
        MODULE.save(path, {"changed": True})
    assert json.loads(path.read_text()) == {"original": True}


@pytest.mark.parametrize("threshold", [0.35, None])
def test_native_nondefault_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        MODULE.require_native_metrics({"threshold": threshold})


def test_native_metric_support_cannot_be_guessed():
    with pytest.raises(ValueError, match="support"):
        MODULE.require_native_metrics({"threshold": 0.5, "metric_support": {}})
