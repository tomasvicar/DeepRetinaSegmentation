"""Synthetic count/mask fixtures; no production photos/checkpoints or forwards."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "cup_continuation", SCRIPTS / "continue_cup_containment.py"
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def metric(tp=0, fp=0, fn=0, valid=0):
    return helper.composer.metrics({"tp": tp, "fp": fp, "fn": fn, "valid_pixels": valid})


@pytest.fixture
def fixture():
    target = np.array([[1, 0], [0, 0]], bool)
    valid = np.ones((2, 2), bool)
    cup = np.array([[1.0, 1.0], [1.0, -1.0]])
    d0 = np.array([[1.0, -1.0], [1.0, -1.0]])
    d1 = np.array([[-1.0, 1.0], [-1.0, -1.0]])
    row = {
        "photo_id": "photo-a",
        "source": "toy",
        "record_ids": ["gt-a"],
        "selected_trace": [["gt-a", "disc_cup"]],
        "shape": [2, 2],
        "cup": helper.paired_counts(cup, d0, d1, target, valid),
        "native_disc": {"disc0": metric(2, 0, 0, 4), "disc1": metric(2, 0, 0, 4)},
        "native_counts_exact": True,
    }
    old = {
        "photo_id": "photo-a",
        "dataset": "toy",
        "record_ids": ["gt-a"],
        "channel_datasets": {"disc": "toy", "cup": "toy"},
        "channels": {name: metric() for name in ["vessel", "artery", "vein", "disc", "cup"]},
    }
    old["channels"]["cup"] = copy.deepcopy(row["cup"]["native"])
    old["channels"]["disc"] = metric(2, 0, 0, 4)
    lookup = {name: {"photo-a": copy.deepcopy(old)} for name in ["disc0", "disc1"]}
    payload = (json.dumps(row) + "\n").encode()
    return row, lookup, payload, target, valid, cup, d0, d1


def prefix(payload, lookup, **kwargs):
    return helper.verified_prefix(
        payload,
        expected_sha=kwargs.pop("expected_sha", helper.sha(payload)),
        eligible=["photo-a"],
        lookups=lookup,
        count=1,
        **kwargs,
    )


def test_pinned_prefix_proof_is_reused_not_relabelled_new(fixture):
    _row, lookup, payload, *_ = fixture
    result = prefix(payload, lookup)[0]
    assert result["phase"] == "reused_original_b413_prefix"
    assert result["original_prefix_native_counts_exact"]
    assert "native_counts_exact" not in result
    assert all(v["exact_counts"] for v in result["historical_native_count_comparison"].values())


@pytest.mark.parametrize(
    "defect",
    ["changed_bytes", "missing", "duplicate", "source", "record", "GT", "false_exact", "bad_disc"],
)
def test_prefix_hard_rejections(fixture, defect):
    row, lookup, payload, *_ = fixture
    old_sha = helper.sha(payload)
    if defect == "changed_bytes":
        with pytest.raises(ValueError, match="bytes"):
            prefix(payload + b" ", lookup, expected_sha=old_sha)
        return
    if defect == "missing":
        payload = b""
    elif defect == "duplicate":
        payload = payload + payload
    else:
        row = copy.deepcopy(row)
        if defect == "source":
            row["source"] = "other_source"
        elif defect == "record":
            row["record_ids"] = ["another_observer"]
        elif defect == "GT":
            row["cup"]["native"]["fn"] = 1
        elif defect == "false_exact":
            row["native_counts_exact"] = False
        elif defect == "bad_disc":
            row["native_disc"]["disc0"]["valid_pixels"] = 5
        payload = (json.dumps(row) + "\n").encode()
    with pytest.raises(ValueError):
        prefix(payload, lookup)


def test_decoded_GT_observer_shape_checks(fixture):
    row, _, _, target, valid, *_ = fixture
    disc_target = np.array([[1, 1], [0, 0]], bool)
    helper.check_sample(
        row, ["gt-a"], row["selected_trace"], [2, 2], target, valid, disc_target, valid
    )
    changed = target.copy()
    changed[0, 1] = True
    with pytest.raises(ValueError, match="GT"):
        helper.check_sample(
            row, ["gt-a"], row["selected_trace"], [2, 2], changed, valid, disc_target, valid
        )
    with pytest.raises(ValueError, match="observer"):
        helper.check_sample(
            row, ["gt-a"], [["other", "disc_cup"]], [2, 2], target, valid, disc_target, valid
        )


def test_same_forward_coupling_uses_fresh_native_even_with_historical_drift(fixture):
    row, lookup, _, _, _, cup, d0, d1 = fixture
    lookup["disc0"]["photo-a"]["channels"]["cup"] = metric(1, 1, 0, 4)
    lookup["disc1"]["photo-a"]["channels"]["cup"] = metric(1, 1, 0, 4)
    result = helper.checked_row(row, lookup)
    assert result["historical_native_count_comparison"]["cup"]["exact_counts"] is False
    assert result["historical_native_count_comparison"]["cup"]["fresh_minus_historical"]["fp"] == 1
    assert result["cup"]["native"]["fp"] == 2 and result["cup"]["disc0"]["fp"] == 1
    assert result["cup"]["disc1"]["tp"] == 0 and result["cup"]["disc1"]["fn"] == 1
    assert cup[0, 0] == 1 and d0[0, 0] == 1 and d1[0, 0] == -1
    broken = copy.deepcopy(row)
    broken["cup"]["disc0"] = metric(1, 3, 0, 4)
    with pytest.raises(ValueError, match="SAME fresh"):
        helper.checked_row(broken, lookup)


def test_invalid_and_empty_cup_support_are_not_false_supervision(fixture):
    _, _, _, target, valid, cup, d0, d1 = fixture
    valid[:] = False
    counts = helper.paired_counts(cup, d0, d1, target, valid)
    assert all(c["valid_pixels"] == 0 and c["dice"] is None for c in counts.values())
    valid[:] = True
    target[:] = False
    counts = helper.paired_counts(cup, d0, d1, target, valid)
    assert counts["native"]["tp"] == 0 and counts["native"]["fp"] == 3
    cup[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        helper.paired_counts(cup, d0, d1, target, valid)


def test_native_control_and_variant_keep_non_cup_and_historical_inputs_untouched(fixture):
    row, lookup, *_ = fixture
    base = {
        "split": "val",
        "photos": 1,
        "preview_only": False,
        "per_photo": [copy.deepcopy(lookup["disc0"]["photo-a"])],
        "by_dataset": {"toy": {}},
        "av_conditional_accuracy": 0.7,
        "policy": "historical policy",
    }
    base["per_photo"][0]["channels"]["cup"] = metric(1, 1, 0, 4)
    original = copy.deepcopy(base)
    row = helper.checked_row(
        row,
        {"disc0": {"photo-a": base["per_photo"][0]}, "disc1": {"photo-a": base["per_photo"][0]}},
    )
    row["phase"] = "new_continuation_forward"
    native = helper.compose_control(base, [row], "native", "disc0")
    clipped = helper.compose_control(base, [row], "disc0", "disc0")
    assert native["channels"]["cup"]["fp"] == 2 and clipped["channels"]["cup"]["fp"] == 1
    assert native["av_conditional_accuracy"] == clipped["av_conditional_accuracy"] == 0.7
    assert base == original
    for channel in ["vessel", "artery", "vein", "disc"]:
        assert native["channels"][channel] == clipped["channels"][channel]
    summary = helper.summarize_rows([row])
    assert summary["pooled"]["clipped"]["disc0"]["FP_removed_vs_fresh"] == 1
    assert summary["pooled"]["historical_drift"]["cup"]["photos_with_drift"] == 1


def test_no_overwrite_or_protected_tree_write(tmp_path):
    existing = tmp_path / "old"
    existing.mkdir()
    (existing / "launch.json").write_text("unchanged")
    with pytest.raises(ValueError, match="overwrite"):
        helper.ensure_new_output(existing, [])
    with pytest.raises(ValueError, match="protected"):
        helper.ensure_new_output(tmp_path / "canonical" / "new", [tmp_path / "canonical"])
    helper.ensure_new_output(tmp_path / "fresh", [tmp_path / "canonical"])
    path = existing / "launch.json"
    with pytest.raises(FileExistsError):
        helper.dump(path, {"changed": True})
    assert path.read_text() == "unchanged"


def test_failure_before_output_creation_cannot_mutate_existing_report(tmp_path, monkeypatch):
    existing = tmp_path / "old"
    existing.mkdir()
    (existing / "launch.json").write_text("{}")
    argv = ["continue"]
    for flag in ["source", "plan", "manifest", "reference-dir", "prefix", "prefix-terminal"]:
        argv.extend(["--" + flag, str(tmp_path / flag)])
    argv.extend(["--output", str(existing), "--expected-helper-sha256", "dummy"])
    monkeypatch.setattr(sys, "argv", argv)

    def reject(_):
        raise ValueError("preflight refused")

    monkeypatch.setattr(helper, "evaluate", reject)
    with pytest.raises(ValueError, match="preflight"):
        helper.main()
    assert not (existing / "failure.json").exists()


def test_wrong_frozen_source_rejected_before_data_or_models(tmp_path, monkeypatch):
    monkeypatch.setattr(
        helper.subprocess, "check_output", lambda *args, **kwargs: "87cf045-wrong-source"
    )
    with pytest.raises(ValueError, match="source commit differs"):
        helper.verify_frozen_source(tmp_path)
