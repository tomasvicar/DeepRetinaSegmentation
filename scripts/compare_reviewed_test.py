"""Compare saved, externally reviewed TEST reports using metadata/counts only.

No selection/heldout module import, checkpoint loading, GT access or model inference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from deepretina.comparison import CHANNELS, compare_reports
from deepretina.completion_profiles import PROFILES, ROLES

TASK_CHANNELS = {
    "vessels": ("vessel",),
    "av": ("vessel", "artery", "vein"),
    "disc_cup": ("disc", "cup"),
}
BOUNDARIES = {
    "test_evaluation_enabled": False,
    "prior_blindness_proven": False,
    "reviewer_identity_authenticated": False,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def timestamp(value):
    result = datetime.fromisoformat(value)
    require(
        result.tzinfo is not None and result <= datetime.now(UTC), "Invalid/future review timestamp"
    )
    return result


def digest_value(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def inventory(manifest_bytes):
    document = parse(manifest_bytes)
    rows = document["records"] if isinstance(document, dict) else document
    groups = {split: {} for split in ("train", "val", "test")}
    ids, partitions = set(), {}
    for row in rows:
        require(
            row["id"] not in ids and row["split"] in groups, "Duplicate record/unknown partition"
        )
        ids.add(row["id"])
        for key in ("photo_id", "source_id", "split_group_id"):
            value = row.get(key)
            require(isinstance(value, str) and value, "Require explicit portable identity: " + key)
            require(
                partitions.setdefault((key, value), row["split"]) == row["split"],
                "Identity crosses partitions",
            )
        if (
            row["task"] not in TASK_CHANNELS
            or not row.get("usable_for_training", True)
            or row.get("exclude_from_training", False)
        ):
            continue
        metadata = row.get("metadata", {})
        require(
            not row.get("preparation_preview", metadata.get("preparation_preview", False))
            and row.get("preparation_max_size") is None
            and row.get("preprocessing_profile", metadata.get("preprocessing_profile"))
            == "legacy25_raw"
            and row.get("pixels_per_degree", metadata.get("pixels_per_degree")) == 25,
            "Require full canonical prepared metadata",
        )
        groups[row["split"]].setdefault(row["photo_id"], []).append(row)
    require(all(groups.values()), "Missing eligible TRAIN/VAL/TEST metadata")
    groups = {
        s: {p: sorted(records, key=lambda r: r["id"]) for p, records in sorted(photos.items())}
        for s, photos in groups.items()
    }
    original_inventory = {
        s: {p: [r["id"] for r in records] for p, records in photos.items()}
        for s, photos in groups.items()
    }
    unified_inventory = {
        s: [
            {
                "photo_id": p,
                "records": [
                    {k: r[k] for k in ("id", "dataset", "source_id", "task")} for r in records
                ],
            }
            for p, records in photos.items()
        ]
        for s, photos in groups.items()
    }
    return (
        groups,
        {s: len(p) for s, p in groups.items()},
        sha(canonical(original_inventory)),
        sha(canonical(unified_inventory)),
    )


def check_photo_inventory(report, groups):
    entries = report["per_photo"]
    require(
        report["split"] == "test"
        and report["photos"] == len(groups)
        and len(entries) == len(groups),
        "Require full dynamic TEST photo inventory",
    )
    lookup = {p["photo_id"]: p for p in entries}
    require(len(lookup) == len(entries) and set(lookup) == set(groups), "TEST photo IDs differ")
    for pid, rows in groups.items():
        photo = lookup[pid]
        require(photo["record_ids"] == [r["id"] for r in rows], "TEST annotation record IDs differ")
        tasks = [task for task in ("av", "vessels") if any(r["task"] == task for r in rows)][:1]
        if any(r["task"] == "disc_cup" for r in rows):
            tasks.append("disc_cup")
        sources = {}
        for task in tasks:
            first = next(r for r in rows if r["task"] == task)
            sources.update({channel: first["dataset"] for channel in TASK_CHANNELS[task]})
        require(
            photo["dataset"] == rows[0]["dataset"] and photo["channel_datasets"] == sources,
            "TEST source attribution differs from manifest annotation policy",
        )
        require(set(photo["channels"]) == set(CHANNELS), "Missing TEST output counts")


def check_v2_eligibility(selection, candidate, groups):
    proof = selection["freeze_eligibility"]
    role = selection["checkpoint_role"]
    cp = selection["checkpoint"]
    completion = selection["training_completion"]
    require(
        candidate.get("checkpoint_role") == role and candidate.get("freeze_eligibility") == proof,
        "Stage2 selected role/typed eligibility differs from frozen decision",
    )
    require(
        proof.get("schema_version") == 1
        and proof.get("kind") == "checked_freeze_eligibility"
        and proof.get("family") in PROFILES
        and proof.get("checkpoint_role") == role
        and proof.get("selection_created") is False
        and proof.get("test_evaluation_enabled") is False,
        "Require typed eligibility schema and explicit completion family",
    )
    kind, commit, execution, training = PROFILES[proof["family"]]
    require(
        proof["original_qa_kind"] == kind
        and digest_value(proof["original_qa_sha256"])
        and proof["input_sha256"]["qa"] == proof["original_qa_sha256"]
        and proof["source_profile"]
        == {
            "commit": commit,
            "execution_code_sha256": execution,
            "training_source_sha256": training,
        }
        and selection["source"]["training_source_sha256"] == training,
        "Typed original family/source/QA pin differs",
    )
    require(
        proof["checkpoint_sha256"] == cp["sha256"]
        and proof["checkpoint_epoch"] == cp["epoch"]
        and proof["checkpoint_steps"] == cp["steps"]
        and proof["selected_training_score"] == cp["training_selection_score"]
        and proof["training_completion"] == completion
        and proof["canonical_data"] == selection["canonical_data"],
        "Typed selected bytes/own epoch score/completion/data differs",
    )
    require(
        proof["training_best"]["epoch"] == completion["best_epoch"]
        and proof["training_best"]["score"] == completion["best_training_selection_score"]
        and digest_value(proof["training_best"]["sha256"]),
        "True original BEST provenance differs",
    )
    require(
        proof["raw_training_configuration_sha256"]
        == sha(canonical(proof["raw_training_configuration"])),
        "Original raw training configuration hash differs",
    )
    # Explicit known compatibility defaults, never deletion/relabeling of saved fields.
    normalized = {k: v for k, v in proof["raw_training_configuration"].items() if k != "device"}
    normalized.setdefault("vascular_sampling_weight", 1.0)
    normalized.setdefault("mixed_precision", "none")
    require(
        normalized == cp["configuration"]["values"], "Raw training versus runtime config differs"
    )
    require(
        proof["actual_evaluation_source"]
        == {k: v for k, v in selection["source"].items() if k != "training_source_sha256"},
        "Eligibility evaluation runtime differs from frozen source lease",
    )
    photos = sorted(groups["train"])
    records = sorted({r["id"] for rows in groups["train"].values() for r in rows})
    for name, epoch in (("terminal_coverage", 60), ("selected_checkpoint_coverage", cp["epoch"])):
        view = proof[name]
        require(
            view["epoch"] == epoch
            and view["eligible_photos"] == view["seen_photos"] == len(photos)
            and view["eligible_annotation_records"]
            == view["contributing_annotation_records"]
            == len(records)
            and type(view["minimum_photo_visits"]) is int
            and view["minimum_photo_visits"] >= epoch
            and type(view["minimum_annotation_visits"]) is int
            and view["minimum_annotation_visits"] >= 1
            and view["delivered_samples"] == epoch * 8192
            and view["eligible_photo_ids_sha256"] == sha(canonical(photos) + b"\n")
            and view["eligible_annotation_ids_sha256"] == sha(canonical(records) + b"\n")
            and digest_value(view["photo_visits_sha256"])
            and digest_value(view["annotation_visits_sha256"]),
            "Typed terminal/selected OWN full TRAIN coverage differs",
        )
    terminal = proof["original_completion"]
    require(digest_value(terminal["sha256"]), "Original completion outcome byte pin missing")
    if proof["family"] == "pool_6b":
        require(
            terminal["kind"] == "original_pool_COMPLETE_receipt"
            and digest_value(terminal["original_task_sha256"])
            and digest_value(terminal["original_decision_sha256"]),
            "Original pool completion provenance differs",
        )
    else:
        require(
            terminal["kind"] == "original_standalone_supervisor_terminal"
            and digest_value(terminal["reviewed_protocol_sha256"]),
            "Original standalone outcome provenance differs",
        )


def checked_candidate(
    original, policy, candidate, selection, *, manifest_bytes, policy_sha, selection_sha
):
    require(
        policy["schema_version"] == 1 and policy["kind"] == "frozen_original_reference_policy",
        "Unsupported original policy",
    )
    require(
        type(selection.get("schema_version")) is int
        and selection["schema_version"] in (1, 2)
        and selection["kind"] == "unified_heldout_selection",
        "Unsupported unified selection",
    )
    require(
        policy["selected_unified_selection_sha256"]
        == selection_sha
        == candidate["selection_sha256"],
        "Unified selection review pin differs",
    )
    require(
        original["schema_version"] == 1
        and original["family"] == "frozen_original_reference"
        and original["preview_only"] is False
        and original["split"] == "test"
        and original["reviewed_policy_sha256"] == policy_sha,
        "Require policy-bound original TEST report",
    )
    require(
        candidate["schema_version"] == selection["schema_version"]
        and candidate["kind"] == "unified_reviewed_heldout_evaluation",
        "Require reviewed stage2 unified TEST wrapper",
    )
    require(
        original["policy_spec"] == policy["policy"] and original["source"] == policy["source"],
        "Original recipe/source differs from its own policy",
    )
    models = policy["policy"]["model_sha256"]
    require(
        set(models) == {"universal", "vessel", "disc", "av"}
        and all(digest_value(v) for v in models.values()),
        "Original four-checkpoint policy missing",
    )
    for name, key in [
        ("universal", "checkpoint_sha256"),
        ("vessel", "vessel_checkpoint_sha256"),
        ("disc", "disc_checkpoint_sha256"),
        ("av", "av_checkpoint_sha256"),
    ]:
        require(original[key] == models[name], "Original checkpoint differs from policy: " + name)
    require(
        digest_value(policy["validation_report_sha256"]) and bool(policy["rationale"].strip()),
        "Original reviewed VAL provenance missing",
    )
    decision = selection["decision"]
    require(
        bool(decision["reviewer"].strip()) and bool(decision["rationale"].strip()),
        "Missing actual selection decision",
    )
    decided = timestamp(decision["recorded_at_utc"])
    require(
        timestamp(selection["validation_evidence"]["evaluated_at_utc"]) <= decided,
        "VAL evidence postdates selection",
    )
    require(
        decided <= timestamp(policy["frozen_at_utc"]) <= timestamp(original["evaluated_at_utc"])
        and decided <= timestamp(candidate["evaluated_at_utc"]),
        "TEST/review chronology differs",
    )
    require(selection["boundaries"] == BOUNDARIES, "Stage1 boundary declaration differs")
    context = candidate["execution_context"]
    require(
        context
        == {
            "route": "explicit_reviewed_stage2_script",
            "explicit_execution_requested": True,
            "stage1_test_evaluation_enabled": False,
            "prior_blindness_proven": False,
            "reviewer_identity_authenticated": False,
            "new_model_selection_performed": False,
        },
        "Require explicit reviewed stage2 execution context",
    )
    for key, candidate_key in [
        ("decision", "selection_decision"),
        ("training_completion", "training_completion"),
        ("validation_evidence", "validation_evidence"),
        ("source", "source"),
        ("canonical_data", "canonical_data"),
        ("inference_protocol", "inference_protocol"),
    ]:
        require(
            candidate[candidate_key] == selection[key],
            "Unified report differs from own selection: " + key,
        )
    require(
        digest_value(selection["source"]["evaluation_source_sha256"])
        and digest_value(selection["source"]["training_source_sha256"]),
        "Missing selected evaluation/training source byte identities",
    )
    checkpoint = selection["checkpoint"]
    require(digest_value(checkpoint["sha256"]), "Invalid selected checkpoint SHA")
    require(
        candidate["checkpoint"]
        == {
            k: checkpoint[k]
            for k in (
                ("sha256", "epoch", "steps", "lineage", "training_selection_score")
                if selection["schema_version"] == 2
                else ("sha256", "epoch", "steps", "lineage")
            )
        }
        and candidate["configuration"] == checkpoint["configuration"],
        "Selected checkpoint/configuration differs",
    )
    config = checkpoint["configuration"]["values"]
    require(
        checkpoint["configuration"]["sha256"] == sha(canonical(config)),
        "Configuration byte identity differs",
    )
    require(
        config["purpose"] == "full_data"
        and config["epochs"] == 60
        and config["epoch_samples"] == 8192
        and config["coverage_first"] is True
        and config["legacy_initialization"] is None
        and config["allow_historical_exposure"] is False
        and config["max_steps"] is None
        and type(config["batch_size"]) is int
        and config["batch_size"] > 0,
        "Require full-data completed60 training configuration",
    )
    completion = selection["training_completion"]
    require(
        completion["planned_epochs"]
        == completion["completed_epochs"]
        == completion["last_epoch"]
        == 60
        and completion["last_steps"] == 60 * 8192 // config["batch_size"]
        and type(checkpoint["epoch"]) is int
        and 1 <= checkpoint["epoch"] <= 60
        and checkpoint["steps"] == checkpoint["epoch"] * 8192 // config["batch_size"],
        "Actual completed60/best checkpoint metadata differs",
    )
    role = selection.get("checkpoint_role", "training_best")
    require(role in ROLES, "Unknown selected checkpoint role")
    if selection["schema_version"] == 1 or role == "training_best":
        require(
            completion["best_epoch"] == checkpoint["epoch"]
            and completion["best_training_selection_score"]
            == checkpoint["training_selection_score"],
            "Macro BEST completion/selected epoch score differs",
        )
    else:
        require(
            checkpoint["epoch"] == 60
            and checkpoint["sha256"] == completion["last_checkpoint_sha256"]
            and checkpoint["steps"] == completion["last_steps"],
            "Fixed terminal role must select actual LAST60 bytes",
        )
    require(
        all(
            digest_value(completion[k])
            for k in (
                "last_checkpoint_sha256",
                "metrics_csv_sha256",
                "training_summary_sha256",
                "coverage_sha256",
            )
        )
        and digest_value(selection["validation_evidence"]["sha256"]),
        "Missing completed-run/VAL byte pins",
    )
    protocol = selection["inference_protocol"]
    threshold = protocol["threshold"]
    require(
        type(threshold) in (int, float) and math.isfinite(threshold) and 0 < threshold < 1,
        "Invalid frozen threshold",
    )
    require(
        protocol["geometry"] == "canonical_prepared"
        and protocol["normalization"] == "RGB/255-0.5"
        and protocol["preprocessing_profile"] == config["preprocessing_profile"] == "legacy25_raw"
        and protocol["pixels_per_degree"] == config["pixels_per_degree"] == 25
        and protocol["channels"] == checkpoint["channels"] == list(CHANNELS)
        and (
            protocol["mode"] == "tiled"
            if config["train_view"] == "patch"
            else protocol["mode"] == "whole"
        ),
        "Frozen inference geometry/config differs",
    )
    # Tile size/overlap may be explicit reviewed overrides; equality to the
    # selected protocol is enforced above rather than to training config defaults.
    metrics = candidate["metrics"]
    require(
        metrics["family"] == "unified"
        and metrics["purpose"] == "full_data"
        and metrics["split"] == "test"
        and metrics["threshold"] == threshold
        and metrics["supervised_channels"] == [True] * 5
        and metrics["metric_support"] == protocol["metric_support"]
        and metrics["evaluation_policy"] == protocol["annotation_policy"],
        "Require frozen-protocol TEST metrics",
    )
    groups, counts, original_inventory, unified_inventory = inventory(manifest_bytes)
    if selection["schema_version"] == 2:
        check_v2_eligibility(selection, candidate, groups)
    data = selection["canonical_data"]
    require(
        data["manifest_sha256"] == sha(manifest_bytes)
        and digest_value(data["prepared_content_sha256"])
        and data["fingerprint_schema"] == 2
        and data["eligible_photo_counts"] == counts
        and data["split_inventory_sha256"] == unified_inventory,
        "Unified canonical metadata inventory differs",
    )
    original_data = policy["data"]
    require(
        original_data["manifest_sha256"] == data["manifest_sha256"]
        and original_data["prepared_content_sha256"] == data["prepared_content_sha256"]
        and original_data["eligible_photo_counts"] == counts
        and original_data["inventory_sha256"] == original_inventory
        and all(original.get(k) == v for k, v in original_data.items()),
        "Original canonical metadata differs from reviewed policy/shared domain",
    )
    require(
        metrics["records"] == sum(map(len, groups["test"].values())),
        "Dynamic TEST annotation count differs",
    )
    check_photo_inventory(original, groups["test"])
    check_photo_inventory(metrics, groups["test"])
    for key, expected in {
        "manifest_sha256": data["manifest_sha256"],
        "prepared_content_sha256": data["prepared_content_sha256"],
        "checkpoint_sha256": checkpoint["sha256"],
    }.items():
        require(
            key not in metrics or metrics[key] == expected,
            "Existing TEST metric identity contradicts reviewed wrapper: " + key,
        )
    derived = copy.deepcopy(metrics)
    derived.update(
        manifest_sha256=data["manifest_sha256"],
        prepared_content_sha256=data["prepared_content_sha256"],
        checkpoint_sha256=checkpoint["sha256"],
    )
    return derived


def compare(
    *,
    original_report,
    original_policy,
    trusted_original_policy_sha256,
    candidate_report,
    selection,
    trusted_selection_sha256,
    manifest,
):
    paths = {
        k: Path(v)
        for k, v in {
            "original_report": original_report,
            "original_policy": original_policy,
            "candidate_report": candidate_report,
            "selection": selection,
            "manifest": manifest,
        }.items()
    }
    blobs = {k: p.read_bytes() for k, p in paths.items()}
    require(
        sha(blobs["original_policy"]) == trusted_original_policy_sha256
        and sha(blobs["selection"]) == trusted_selection_sha256,
        "Externally reviewed artifact bytes differ",
    )
    values = {k: parse(v) for k, v in blobs.items() if k != "manifest"}
    derived = checked_candidate(
        values["original_report"],
        values["original_policy"],
        values["candidate_report"],
        values["selection"],
        manifest_bytes=blobs["manifest"],
        policy_sha=trusted_original_policy_sha256,
        selection_sha=trusted_selection_sha256,
    )
    derived_bytes = canonical(derived) + b"\n"
    with tempfile.TemporaryDirectory(prefix="retina-reviewed-test-") as directory:
        root = Path(directory)
        for name, raw in [
            ("original.json", blobs["original_report"]),
            ("candidate-view.json", derived_bytes),
            ("manifest.json", blobs["manifest"]),
        ]:
            (root / name).write_bytes(raw)
        result = compare_reports(
            root / "original.json",
            root / "candidate-view.json",
            root / "manifest.json",
            repeats=2000,
            seed=42,
            include_sources=True,
            include_av=True,
        )
    require(
        all(p.read_bytes() == blobs[k] for k, p in paths.items()),
        "Input changed during metadata comparison",
    )
    result.update(
        kind="reviewed_paired_test_comparison",
        whole_input_sha256={k: sha(v) for k, v in blobs.items()},
        trusted_original_policy_sha256=trusted_original_policy_sha256,
        trusted_selection_sha256=trusted_selection_sha256,
        derived_candidate_view_sha256=sha(derived_bytes),
        candidate_wrapped_report_sha256=sha(blobs["candidate_report"]),
        metadata_only=True,
        prior_blindness_proven=False,
        reviewer_identity_authenticated=False,
        selection_or_protocol_changed=False,
        input_view_qualification="candidate_report_sha256 identifies derived metrics view; candidate_wrapped_report_sha256 and whole_input_sha256 identify complete original wrapper bytes",
        pipeline_source_binding="Each pipeline bound to its own reviewed policy/selection; source identities may differ",
        original_checkpoint_sha256=values["original_policy"]["policy"]["model_sha256"],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "original-report",
        "original-policy",
        "candidate-report",
        "selection",
        "manifest",
        "output",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("trusted-original-policy-sha256", "trusted-selection-sha256"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    result = compare(
        original_report=args.original_report,
        original_policy=args.original_policy,
        trusted_original_policy_sha256=args.trusted_original_policy_sha256,
        candidate_report=args.candidate_report,
        selection=args.selection,
        trusted_selection_sha256=args.trusted_selection_sha256,
        manifest=args.manifest,
    )
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
