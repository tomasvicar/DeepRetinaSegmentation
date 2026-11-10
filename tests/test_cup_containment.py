"""Postprocessing changes cup predictions only and preserves reference support."""

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "cup_containment", ROOT / "scripts/evaluate_cup_containment.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_native_reproduction_requires_exact_integer_counts():
    counts = {"tp": 2, "fp": 1, "fn": 3, "valid_pixels": 9}
    m.exact(counts, dict(counts))
    with pytest.raises(ValueError, match="Native per-photo"):
        m.exact(counts, dict(counts, fp=2))


def test_intersection_can_remove_false_positives_and_true_positives_without_mutation():
    cup = np.array([[1, 1, 1, 0]], float)
    d0 = np.array([[1, 0, 0, 1]], float)
    d1 = np.ones_like(cup)
    target = np.array([[True, False, True, False]])
    valid = np.ones_like(target)
    original = cup.copy()
    r = m.cup_variants(cup, d0, d1, target, valid)
    assert r["native"]["tp"] == 2 and r["native"]["fp"] == 1
    assert r["disc0"]["tp"] == 1 and r["disc0"]["fp"] == 0 and r["disc0"]["fn"] == 1
    assert r["disc1"] == r["native"] and np.array_equal(cup, original)


def test_partial_cup_validity_never_scores_ignored_pixels():
    cup = np.ones((2, 3))
    d0 = np.zeros((2, 3))
    target = np.ones((2, 3), bool)
    valid = np.array([[True, False, False], [False, False, False]])
    r = m.cup_variants(cup, d0, cup, target, valid)
    assert r["disc0"]["valid_pixels"] == 1 and r["disc0"]["fn"] == 1
    assert r["native"]["tp"] == 1 and r["native"]["valid_pixels"] == 1


def test_empty_but_supervised_cup_keeps_real_false_positives_and_undefined_empty_dice():
    image = np.ones((2, 3))
    target = np.zeros((2, 3), bool)
    valid = np.ones_like(target)
    r = m.cup_variants(image, np.zeros_like(image), image, target, valid)
    assert r["native"]["fp"] == 6 and r["native"]["dice"] == 0
    assert r["disc0"]["fp"] == 0 and r["disc0"]["dice"] is None
    assert r["disc0"]["valid_pixels"] == 6


def test_no_cup_supervision_performs_no_forward():
    calls = []
    assert m.supported_forward(np.zeros((2, 3), bool), lambda: calls.append(1)) is None
    assert calls == []
    assert m.supported_forward(np.ones((2, 3), bool), lambda: 42) == 42


def test_complete_composition_changes_only_cup_and_rejects_changed_gt_support():
    value = {"tp": 1, "fp": 1, "fn": 1, "valid_pixels": 4}
    row = {
        "photo_id": "sample",
        "dataset": "x",
        "record_ids": ["first", "second"],
        "channel_datasets": {head: "x" for head in ("vessel", "artery", "vein", "disc", "cup")},
        "channels": {
            head: m.metrics(value) for head in ("vessel", "artery", "vein", "disc", "cup")
        },
    }
    pooled, grouped, macro, score = m.aggregate([row], ["x"])
    base = {
        "split": "val",
        "preview_only": False,
        "photos": 1,
        "per_photo": [row],
        "channels": pooled,
        "by_dataset": grouped,
        "macro_dataset_channel_dice": macro,
        "macro_dataset_mean_dice": score,
        "av_conditional_accuracy": 0.75,
    }
    original = copy.deepcopy(base)
    replacement = {"sample": dict(value, fp=0)}
    out = m.compose_cup(base, replacement, "disc0/raw")
    assert base == original and out["av_conditional_accuracy"] == 0.75
    for head in ("vessel", "artery", "vein", "disc"):
        assert out["channels"][head] == base["channels"][head]
        assert out["per_photo"][0]["channels"][head] == base["per_photo"][0]["channels"][head]
    assert out["channels"]["cup"]["fp"] == 0
    with pytest.raises(ValueError, match="GT support"):
        m.compose_cup(base, {"sample": dict(value, valid_pixels=5)}, "disc0/raw")
