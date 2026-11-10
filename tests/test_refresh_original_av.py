"""A new A/V reference must preserve every other original prediction count."""

import copy
import importlib.util
from pathlib import Path

import pytest


def module(monkeypatch):
    folder = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "refresh_reference", folder / "refresh_original_av.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture_reports():
    empty = {"tp": 0, "fp": 0, "fn": 0, "valid_pixels": 0}
    one = {name: dict(empty) for name in ("vessel", "artery", "vein", "disc", "cup")}
    one["vessel"] = {"tp": 10, "fp": 1, "fn": 2, "valid_pixels": 32}
    one["artery"] = {"tp": 2, "fp": 1, "fn": 2, "valid_pixels": 8}
    one["vein"] = {"tp": 3, "fp": 1, "fn": 1, "valid_pixels": 8}
    two = {name: dict(empty) for name in one}
    two["disc"] = {"tp": 4, "fp": 2, "fn": 1, "valid_pixels": 32}
    two["cup"] = {"tp": 2, "fp": 1, "fn": 1, "valid_pixels": 32}
    baseline = {
        "split": "val",
        "per_photo": [
            {
                "photo_id": "av",
                "channels": one,
                "channel_datasets": {name: "vascular" for name in ("vessel", "artery", "vein")},
            },
            {
                "photo_id": "disc",
                "channels": two,
                "channel_datasets": {name: "optic" for name in ("disc", "cup")},
            },
        ],
    }
    updates = [
        {
            "photo_id": "av",
            "channels": {
                "artery": {"tp": 3, "fp": 2, "fn": 1, "valid_pixels": 8},
                "vein": {"tp": 4, "fp": 1, "fn": 0, "valid_pixels": 8},
            },
            "calibration": {"diameter_pixels": 100},
            "conditional_counts": {
                "correct": 8,
                "scored_pixels": 8,
                "eligible_pixels": 8,
                "end_to_end_correct": 7,
            },
        }
    ]
    return baseline, updates


def test_refresh_preserves_nonvascular_heads_and_reaggregates_exactly(monkeypatch):
    api = module(monkeypatch)
    baseline, updates = fixture_reports()
    original = copy.deepcopy(baseline)
    result = api.summarize(baseline, updates)
    assert baseline == original
    assert result["per_photo"][1] == baseline["per_photo"][1]
    assert (
        result["per_photo"][0]["channels"]["vessel"]
        == baseline["per_photo"][0]["channels"]["vessel"]
    )
    assert result["channels"]["artery"]["dice"] == pytest.approx(6 / 9)
    assert result["by_dataset"]["vascular"]["artery"]["tp"] == 3
    assert result["channels"]["disc"]["tp"] == 4
    assert result["av_conditional_accuracy"] == 1
    assert result["av_end_to_end_accuracy"] == 7 / 8
    assert result["av_labeled_pixels"] == result["av_conditional_scored_pixels"] == 8


@pytest.mark.parametrize("defect", ["missing", "duplicate", "extra", "positive", "valid"])
def test_refresh_refuses_missing_or_mismatched_gt(monkeypatch, defect):
    api = module(monkeypatch)
    baseline, updates = fixture_reports()
    if defect == "missing":
        updates.clear()
    elif defect == "duplicate":
        updates.append(copy.deepcopy(updates[0]))
    elif defect == "extra":
        updates.append(dict(updates[0], photo_id="other"))
    elif defect == "positive":
        updates[0]["channels"]["artery"]["fn"] += 1
    else:
        updates[0]["channels"]["artery"]["valid_pixels"] += 1
    with pytest.raises(ValueError):
        api.summarize(baseline, updates)
