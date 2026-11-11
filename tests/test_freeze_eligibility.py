"""Synthetic metadata-only invariants; no model forward, real GT or TEST."""

import copy
from pathlib import Path

import pytest

from deepretina import freeze_eligibility as f
from deepretina.selection import SelectionError


@pytest.fixture
def coverage_case():
    groups = {f"p{i}": [] for i in range(2529)}
    for i in range(5700):
        groups[f"p{i % 2529}"].append({"id": f"a{i}", "dataset": f"s{i % 21}"})
    photos = dict.fromkeys(groups, 59)
    photos["p0"] += 59 * 8192 - sum(photos.values())
    coverage = {
        "eligible_photos": 2529,
        "seen_photos": 2529,
        "photo_visits": photos,
        "eligible_annotation_records": 5700,
        "contributing_annotation_records": 5700,
        "annotation_visits": {f"a{i}": 1 for i in range(5700)},
    }
    return {"epoch": 59, "coverage": coverage}, groups


def test_selected_own_weight_coverage_not_another_epoch(coverage_case):
    selected, groups = coverage_case
    result = f._coverage(selected, groups, "Selected BEST")
    assert result["epoch"] == 59 and result["delivered_samples"] == 59 * 8192
    assert result["contributing_annotation_records"] == 5700


@pytest.mark.parametrize("case", ["missing", "zero", "only_once", "budget", "foreign"])
def test_partial_or_false_own_coverage_rejected(coverage_case, case):
    selected, groups = copy.deepcopy(coverage_case)
    c = selected["coverage"]
    if case == "missing":
        c["annotation_visits"].pop("a0")
        c["contributing_annotation_records"] = 5699
    if case == "zero":
        c["annotation_visits"]["a0"] = 0
    if case == "only_once":
        c["photo_visits"] = dict.fromkeys(groups, 1)
    if case == "budget":
        c["photo_visits"]["p0"] += 1
    if case == "foreign":
        c["annotation_visits"]["alien"] = c["annotation_visits"].pop("a0")
    with pytest.raises(SelectionError):
        f._coverage(selected, groups, "Selected BEST")


@pytest.mark.parametrize(
    "family,role", [("pool_72_v3", "training_best"), ("pool_6b", "best"), ("standalone_0b", "last")]
)
def test_unsupported_family_or_ambiguous_role_rejects_before_any_file(family, role, monkeypatch):
    monkeypatch.setattr(f.gate, "json_snapshot", lambda _: pytest.fail("No file should be read"))
    missing = Path("not-an-input")
    with pytest.raises(SelectionError):
        f.check_freeze_eligibility(
            checkpoint=missing,
            best_checkpoint=missing,
            last_checkpoint=missing,
            manifest=missing,
            metrics_csv=missing,
            training_summary=missing,
            completion_qa=missing,
            expected_qa_sha256="0" * 64,
            family=family,
            checkpoint_role=role,
        )


def test_external_qa_pin_is_required_and_cannot_be_filename_only(tmp_path):
    p = tmp_path / "qa.json"
    p.write_text('{"kind":"standalone_unified_completion_qa"}')
    with pytest.raises(SelectionError):
        f._pinned_json(p, "", "QA")
    with pytest.raises(SelectionError):
        f._pinned_json(p, "0" * 64, "QA")
