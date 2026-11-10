"""Count-only composition must not change GT support or any other head."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "compose_disc", Path(__file__).parents[1] / "scripts/compose_original_disc_reference.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    known = m.metrics({"tp": 2, "fp": 1, "fn": 2, "valid_pixels": 10})
    empty = m.metrics({"tp": 0, "fp": 0, "fn": 0, "valid_pixels": 0})
    rows = [
        {
            "photo_id": "one",
            "dataset": "a",
            "record_ids": ["a_disc"],
            "channel_datasets": {key: "a" for key in m.CHANNELS},
            "channels": {key: copy.deepcopy(known) for key in m.CHANNELS},
        },
        {
            "photo_id": "two",
            "dataset": "b",
            "record_ids": ["b_vessel"],
            "channel_datasets": {"vessel": "b"},
            "channels": {key: copy.deepcopy(empty) for key in m.CHANNELS},
        },
    ]
    rows[1]["channels"]["vessel"] = copy.deepcopy(known)
    pooled, grouped, macro, score = m.aggregate(rows, ["a", "b"])
    baseline = {
        "split": "val",
        "preview_only": False,
        "manifest_sha256": "manifest",
        "prepared_content_sha256": "content",
        "photos": 2,
        "per_photo": rows,
        "channels": pooled,
        "by_dataset": grouped,
        "macro_dataset_channel_dice": macro,
        "macro_dataset_mean_dice": score,
        "av_conditional_accuracy": 0.8,
    }
    native = {
        "split": "val",
        "preview_only": False,
        "manifest_sha256": "manifest",
        "prepared_content_sha256": "content",
        "variants": {
            "disc0/raw": {
                "profile": "raw",
                "all_weights_and_logits_finite": True,
                "checkpoint_sha256": "disc0",
                "per_photo": [
                    {
                        "photo_id": "one",
                        "record_ids": ["a_disc"],
                        "channel_dataset": "a",
                        "disc": m.metrics({"tp": 0, "fp": 0, "fn": 4, "valid_pixels": 10}),
                    }
                ],
            }
        },
    }
    return baseline, native


def test_only_disc_changes_with_no_added_cup_containment():
    baseline, native = fixture()
    before = copy.deepcopy(baseline)
    result = m.compose(baseline, native, "disc0/raw")
    assert baseline == before
    assert result["per_photo"][0]["channels"]["disc"]["tp"] == 0
    assert result["per_photo"][0]["channels"]["cup"]["tp"] == 2
    assert result["per_photo"][1] == baseline["per_photo"][1]
    assert result["av_conditional_accuracy"] == baseline["av_conditional_accuracy"]
    assert result["disc_checkpoint_sha256"] == "disc0"
    for channel in ("vessel", "artery", "vein", "cup"):
        assert result["channels"][channel] == baseline["channels"][channel]
        assert (
            result["macro_dataset_channel_dice"][channel]
            == baseline["macro_dataset_channel_dice"][channel]
        )


def test_changed_positive_gt_or_source_is_rejected():
    baseline, native = fixture()
    native["variants"]["disc0/raw"]["per_photo"][0]["disc"]["fn"] = 3
    with pytest.raises(ValueError, match="positive ground-truth"):
        m.compose(baseline, native, "disc0/raw")
    baseline, native = fixture()
    native["variants"]["disc0/raw"]["per_photo"][0]["channel_dataset"] = "wrong"
    with pytest.raises(ValueError, match="source differs"):
        m.compose(baseline, native, "disc0/raw")


def test_partial_native_support_is_rejected():
    baseline, native = fixture()
    native["variants"]["disc0/raw"]["per_photo"] = []
    with pytest.raises(ValueError, match="photo set differs"):
        m.compose(baseline, native, "disc0/raw")
