"""Protect the original strict threshold and reference support contract."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "evaluate_legacy_disc", Path(__file__).parents[1] / "scripts/evaluate_legacy_disc.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_zero_logits_background_unknown_pixels_ignored():
    logits = np.array([[0.0, 1.0, -1.0, 1.0]])
    target = np.array([[1, 1, 0, 0]], dtype=bool)
    valid = np.array([[1, 1, 1, 0]], dtype=bool)
    assert module.disc_counts(logits, target, valid).tolist() == [1, 0, 1, 3]
    with pytest.raises(ValueError, match="Nonfinite"):
        module.disc_counts(np.array([[np.nan, 1, -1, 1]]), target, valid)


def test_canary_requires_exact_counts_and_annotation_source():
    counts = {"tp": 10, "fp": 2, "fn": 3, "valid_pixels": 100}
    report = {
        "variants": {
            "universal/raw": {
                "per_photo": [{"photo_id": "a", "channel_dataset": "refuge", "disc": counts.copy()}]
            }
        }
    }
    original = {
        "per_photo": [
            {
                "photo_id": "a",
                "channel_datasets": {"disc": "refuge"},
                "channels": {"disc": counts.copy()},
            }
        ]
    }
    assert module.compare_reference(report, original)["exact_per_photo_counts_match"]
    report["variants"]["universal/raw"]["per_photo"][0]["disc"]["fp"] += 1
    with pytest.raises(ValueError, match="exact counts differ"):
        module.compare_reference(report, original)
