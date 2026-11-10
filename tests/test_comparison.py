import copy
import hashlib
import json

import pytest

from deepretina.comparison import CHANNELS, compare_reports


def fixture_reports(tmp_path):
    rows = []
    reference, candidate = [], []
    for index in range(4):
        ident = f"p{index}"
        source = "a" if index < 2 else "b"
        rows.append(
            {
                "id": ident,
                "photo_id": ident,
                "source_id": f"patient{max(1, index)}",
                "split_group_id": f"patient{max(1, index)}",
                "dataset": source,
                "split": "val",
                "task": "av",
            }
        )
        entries = []
        for model in range(2):
            positives = [8, 8, 8, 1, 1] if model == 0 else [8, 4, 4, 9, 9]
            channels = {
                name: {"tp": tp, "fp": 10 - tp, "fn": 10 - tp, "valid_pixels": 20}
                for name, tp in zip(CHANNELS, positives, strict=True)
            }
            entries.append(
                {
                    "photo_id": ident,
                    "record_ids": [ident],
                    "dataset": source,
                    "channel_datasets": dict.fromkeys(CHANNELS, source),
                    "channels": channels,
                }
            )
        reference.append(entries[0])
        candidate.append(entries[1])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(rows))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    paths = []
    for name, entries in [("reference", reference), ("candidate", candidate)]:
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps({"split": "val", "manifest_sha256": digest, "per_photo": entries})
        )
        paths.append(path)
    return (*paths, manifest)


def test_aggregate_gain_does_not_hide_av_regression_and_groups_stay_paired(tmp_path):
    paths = fixture_reports(tmp_path)
    report = compare_reports(*paths, repeats=100)
    assert report["photos"] == 4 and report["split_groups"] == 3
    assert report["strata"] == {"a": 1, "b": 2}
    assert report["macro_dataset_mean_dice"]["candidate_minus_reference"] == pytest.approx(0.16)
    artery = report["channels"]["artery"]["pooled_dice"]
    assert artery["candidate_minus_reference"] == pytest.approx(-0.4)
    assert artery["paired_percentile_95ci"] == pytest.approx([-0.4, -0.4])
    assert not report["independent_quality_improvement_claim"]
    assert report == compare_reports(*paths, repeats=100)


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_photo",
        "positive_support",
        "valid_support",
        "source",
        "count_type",
        "variants",
        "split",
        "manifest",
    ],
)
def test_rejects_unpaired_or_inconsistent_reports(tmp_path, defect):
    paths = fixture_reports(tmp_path)
    obj = json.loads(paths[1].read_text())
    first = obj["per_photo"][0]
    if defect == "duplicate_photo":
        obj["per_photo"].append(copy.deepcopy(first))
    elif defect == "positive_support":
        first["channels"]["vessel"]["fn"] += 1
    elif defect == "valid_support":
        first["channels"]["vessel"]["valid_pixels"] += 1
    elif defect == "source":
        first["channel_datasets"]["vessel"] = "other"
    elif defect == "count_type":
        first["channels"]["vessel"]["tp"] = 8.0
    elif defect == "variants":
        first["record_ids"] = ["unknown"]
    elif defect == "split":
        obj["split"] = "test"
    elif defect == "manifest":
        obj["manifest_sha256"] = "wrong"
    paths[1].write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        compare_reports(*paths, repeats=100)


def test_manifest_group_leakage_is_rejected(tmp_path):
    paths = fixture_reports(tmp_path)
    rows = json.loads(paths[2].read_text())
    rows.append(dict(rows[0], id="extra_train", photo_id="extra_photo", split="train"))
    paths[2].write_text(json.dumps(rows))
    for path in paths[:2]:
        obj = json.loads(path.read_text())
        obj["manifest_sha256"] = hashlib.sha256(paths[2].read_bytes()).hexdigest()
        path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="crosses partitions"):
        compare_reports(*paths, repeats=100)


@pytest.mark.parametrize("field", ["macro_dataset_mean_dice", "channels"])
def test_rejects_summary_counts_that_do_not_match_photo_counts(tmp_path, field):
    paths = fixture_reports(tmp_path)
    obj = json.loads(paths[1].read_text())
    obj[field] = 0.9 if field == "macro_dataset_mean_dice" else obj["per_photo"][0]["channels"]
    paths[1].write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="disagree"):
        compare_reports(*paths, repeats=100)


def test_no_supported_output_has_explicitly_undefined_scores(tmp_path):
    paths = fixture_reports(tmp_path)
    for path in paths[:2]:
        obj = json.loads(path.read_text())
        obj["macro_dataset_mean_dice"] = None
        for row in obj["per_photo"]:
            row["channel_datasets"] = {}
            row["channels"] = {
                name: dict.fromkeys(("tp", "fp", "fn", "valid_pixels"), 0) for name in CHANNELS
            }
        path.write_text(json.dumps(obj))
    result = compare_reports(*paths, repeats=100)
    assert result["macro_dataset_mean_dice"]["candidate_minus_reference"] is None
    assert result["macro_dataset_mean_dice"]["paired_percentile_95ci"] is None
    assert result["macro_dataset_mean_dice"]["defined_bootstrap_draws"] == 0


def test_two_eyes_of_one_patient_are_resampled_together(tmp_path):
    paths = fixture_reports(tmp_path)
    rows = json.loads(paths[2].read_text())
    for row in rows:
        row["dataset"] = "a"
    paths[2].write_text(json.dumps(rows))
    digest = hashlib.sha256(paths[2].read_bytes()).hexdigest()
    for model, path in enumerate(paths[:2]):
        obj = json.loads(path.read_text())
        obj["manifest_sha256"] = digest
        for index, photo in enumerate(obj["per_photo"]):
            photo["dataset"] = "a"
            photo["channel_datasets"] = dict.fromkeys(CHANNELS, "a")
            tp = [10, 10, 0, 5][index] if model else 5
            photo["channels"]["vessel"] = {
                "tp": tp,
                "fp": 10 - tp,
                "fn": 10 - tp,
                "valid_pixels": 20,
            }
        path.write_text(json.dumps(obj))
    report = compare_reports(*paths, repeats=2000, seed=42)
    # Three patient draws have probability1/27 of all-positive/all-negative
    # patients: both tails exceed2.5%. Four independent photo draws would give
    # only1/256 probability of all-negative photos and a different lower bound.
    assert report["channels"]["vessel"]["pooled_dice"]["paired_percentile_95ci"] == pytest.approx(
        [-0.5, 0.5]
    )


def test_source_details_share_draws_and_preserve_all_regressions(tmp_path):
    paths = fixture_reports(tmp_path)
    rows = json.loads(paths[2].read_text())
    additions = [{**row, "id": row["id"] + "-dc", "task": "disc_cup"} for row in rows]
    paths[2].write_text(json.dumps(rows + additions))
    for path in paths[:2]:
        value = json.loads(path.read_text())
        value["manifest_sha256"] = hashlib.sha256(paths[2].read_bytes()).hexdigest()
        for entry in value["per_photo"]:
            entry["record_ids"].append(entry["photo_id"] + "-dc")
        path.write_text(json.dumps(value))
    result = compare_reports(*paths, repeats=100, include_sources=True)
    artery = result["by_dataset"]["a"]["artery"]
    assert artery["supported_photos"] == 2 and artery["supported_split_groups"] == 1
    assert artery["reference"]["tp"] == 16 and artery["candidate"]["tp"] == 8
    assert artery["candidate_minus_reference"]["dice"] == pytest.approx(-0.4)
    assert artery["paired_dice_percentile_95ci"] == pytest.approx([-0.4, -0.4])
    assert (
        result["macro_dataset_mean_dice"]
        == compare_reports(*paths, repeats=100)["macro_dataset_mean_dice"]
    )


def test_source_head_must_be_eligible_in_manifest(tmp_path):
    paths = fixture_reports(tmp_path)
    # Fixture rows only annotate AV; disc/cup attribution is intentionally impossible.
    with pytest.raises(ValueError, match="eligible"):
        compare_reports(*paths, repeats=100, include_sources=True)


def with_conditional_counts(paths):
    for index, path in enumerate(paths[:2]):
        report = json.loads(path.read_text())
        for entry in report["per_photo"]:
            correct = 16 if index == 0 else 8
            entry["av_conditional_counts"] = {
                "correct": correct,
                "scored_pixels": 20,
                "eligible_pixels": 20,
                "end_to_end_correct": correct,
            }
        path.write_text(json.dumps(report))


def test_conditional_av_uses_identical_stratified_group_draws_and_counts(tmp_path):
    paths = fixture_reports(tmp_path)
    with_conditional_counts(paths)
    report = compare_reports(*paths, repeats=100, include_av=True)
    conditional = report["av_accuracy"]["conditional_accuracy"]
    assert conditional["reference"] == 0.8 and conditional["candidate"] == 0.4
    assert conditional["paired_percentile_95ci"] == pytest.approx([-0.4, -0.4])
    assert report["av_accuracy"]["counts"]["reference"]["eligible_pixels"] == 80
    assert report["av_accuracy"]["by_dataset"]["b"]["counts"]["candidate"]["scored_pixels"] == 40
    assert (
        report["macro_dataset_mean_dice"]
        == compare_reports(*paths, repeats=100)["macro_dataset_mean_dice"]
    )


@pytest.mark.parametrize("defect", ["eligible", "scored", "correct", "end", "exclusive", "header"])
def test_invalid_or_different_conditional_av_domain_is_rejected(tmp_path, defect):
    paths = fixture_reports(tmp_path)
    with_conditional_counts(paths)
    v = json.loads(paths[1].read_text())
    entry = v["per_photo"][0]
    if defect == "eligible":
        entry["av_conditional_counts"]["eligible_pixels"] = 19
    if defect == "scored":
        entry["av_conditional_counts"]["scored_pixels"] = 19
    if defect == "correct":
        entry["av_conditional_counts"]["correct"] = 9
    if defect == "end":
        entry["av_conditional_counts"]["end_to_end_correct"] = 7
    if defect == "exclusive":
        entry["channels"]["artery"].update(tp=10, fp=5, fn=0)
        entry["channels"]["vein"].update(tp=10, fp=5, fn=0)
        entry["av_conditional_counts"].update(correct=20, end_to_end_correct=20)
    if defect == "header":
        v["av_conditional_accuracy"] = 0.9
    paths[1].write_text(json.dumps(v))
    with pytest.raises(ValueError, match="AV|decoder"):
        compare_reports(*paths, repeats=100, include_av=True)
