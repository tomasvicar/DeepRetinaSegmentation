"""Synthetic all2529/5700/770 metadata only; no real model/data/TEST reads."""

import copy
import sys
from pathlib import Path

import pytest
from test_frozen_completed_val_comparison import case  # noqa: F401 - shared synthetic fixture

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_frozen_completed_val as common
import compare_terminal60_val as terminal

from deepretina import completion_roles as roles


@pytest.fixture
def full(case):  # noqa: F811 - pytest fixture imported explicitly above
    values, paths = case
    rows = values["manifest"]
    template = next(r for r in rows if r["split"] == "train")
    for i in range(1, 2529):
        r = copy.deepcopy(template)
        r.update(
            photo_id=f"tr{i:04}",
            source_id=f"tr{i:04}",
            split_group_id=f"tr{i:04}",
            id=f"tr{i:04}-av",
        )
        rows.append(r)
    # Eligible annotation variants, never encoded duplicates or invented photos.
    for i in range(5700 - sum(r["split"] == "train" for r in rows)):
        r = copy.deepcopy(template)
        r["id"] = f"extra-observer-{i:04}"
        rows.append(r)
    manifest_raw = common.canonical(rows)
    groups, counts, original_inventory, unified_inventory = common.inventory(manifest_raw)
    e = values["candidate"]
    q = values["completion_qa"]
    original = values["reference"]
    e["data"].update(
        manifest_sha256=common.sha(manifest_raw),
        eligible_photo_counts=counts,
        split_inventory_sha256=unified_inventory,
    )
    original.update(
        manifest_sha256=common.sha(manifest_raw),
        eligible_photo_counts=counts,
        inventory_sha256=original_inventory,
    )
    e["checkpoint"].update(sha256="f" * 64, epoch=60, steps=60 * 2048)
    e["source"]["training_source_sha256"] = roles.TRAINING
    q.update(
        kind="trusted_pool_completion_qa",
        schema_version=1,
        manifest_sha256=common.sha(manifest_raw),
        source_commit=roles.SOURCE,
        execution_code_sha256=roles.EXECUTION,
        checkpoint_training_source_sha256=roles.TRAINING,
        steps=60 * 2048,
        eligible_photos=2529,
        best_epoch=52,
        geometry_policy_sha256=original["geometry_policy_sha256"],
        eligible_annotation_records=5700,
        contributing_annotation_records=5700,
        all_eligible_annotations_contributed=True,
        minimum_annotation_visits=2,
        eligible_annotation_ids_sha256=roles.ids_sha(
            {r["id"] for rs in groups["train"].values() for r in rs}
        ),
        eligible_photo_ids_sha256=roles.ids_sha(groups["train"]),
    )
    values["candidate_context"]["checkpoint_role"] = "fixed_terminal60"
    save(values, paths)
    return values, paths


def save(values, paths):
    e = values["candidate"]
    o = values["reference"]
    q = values["completion_qa"]
    for key, report in [("candidate", e), ("reference", o)]:
        c = values[key + "_context"]
        c["report_sha256"] = common.sha(common.canonical(report))
        c["bindings"] = {k: report[k] for k in c["bindings"]}
    values["candidate_context"]["completion_qa_sha256"] = common.sha(common.canonical(q))
    for name, p in paths.items():
        p.write_bytes(common.canonical(values[name]))


def run(full):
    _, p = full
    return terminal.compare(
        **p,
        trusted_reference_context_sha256=common.sha(p["reference_context"].read_bytes()),
        trusted_candidate_context_sha256=common.sha(p["candidate_context"].read_bytes()),
        repeats=100,
    )


def test_terminal60_accepts_actual_last_and_preserves_best52(full):
    result = run(full)
    assert result["checkpoint_role"] == "fixed_terminal60"
    assert result["baseline_only"] is True and result["selection_authorized"] is False
    assert result["test_evaluation_enabled"] is False
    assert result["candidate_checkpoint"]["epoch"] == 60
    assert result["original_completion_best_epoch"] == 52
    assert result["original_completion_checkpoint_sha256"] == {
        "best.pt": "e" * 64,
        "last.pt": "f" * 64,
    }


@pytest.mark.parametrize(
    "change",
    [
        "best52",
        "last_hash",
        "role_missing",
        "role_wrong",
        "annotation",
        "source",
        "geometry",
        "steps",
    ],
)
def test_rehashed_context_cannot_replace_terminal_contract(full, change):
    values, p = full
    e = values["candidate"]
    q = values["completion_qa"]
    if change == "best52":
        e["checkpoint"].update(sha256="e" * 64, epoch=52, steps=52 * 2048)
    if change == "last_hash":
        q["checkpoint_sha256"]["last.pt"] = "a" * 64
    if change == "role_missing":
        values["candidate_context"].pop("checkpoint_role")
    if change == "role_wrong":
        values["candidate_context"]["checkpoint_role"] = "best"
    if change == "annotation":
        q["contributing_annotation_records"] = 5699
    if change == "source":
        q["source_commit"] = "0" * 40
    if change == "geometry":
        q["geometry_policy_sha256"] = "0" * 64
    if change == "steps":
        q["steps"] -= 1
    save(values, p)
    with pytest.raises(ValueError):
        run(full)


def test_default_best_route_does_not_accept_terminal_role_or_relabel_hash(full):
    values, p = full
    kwargs = dict(
        **p,
        trusted_reference_context_sha256=common.sha(p["reference_context"].read_bytes()),
        trusted_candidate_context_sha256=common.sha(p["candidate_context"].read_bytes()),
        repeats=100,
    )
    with pytest.raises(ValueError, match="role"):
        common.compare(**kwargs)
    values["candidate_context"].pop("checkpoint_role")
    save(values, p)
    kwargs["trusted_candidate_context_sha256"] = common.sha(p["candidate_context"].read_bytes())
    with pytest.raises(ValueError, match="selected checkpoint"):
        common.compare(**kwargs)
