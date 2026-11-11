"""Metadata-only fixed terminal baseline role, independent of best/selection gates."""

import hashlib
import json

ROLE = "fixed_terminal60"
SOURCE = "6b3dfad605a256e904a877efcdcc1f3cc6135dd0"
EXECUTION = "cb5c4f98e0181a13b166c128a632faf502de1a3bf770fbee4a5b7fa649e651cd"
TRAINING = "9075246883f938ba80b4fe282b365da10ddb8c834bd3106d846465ac79ce9473"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ids_sha(values):
    raw = (
        json.dumps(sorted(values), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def verify_terminal60(qa, evidence, train_groups):
    """Verify actual last60; never substitute its hash for the recorded best."""
    cp = evidence["checkpoint"]
    data = evidence["data"]
    cfg = evidence["configuration"]["values"]
    annotations = {r["id"] for rows in train_groups.values() for r in rows}
    require(
        qa.get("kind") == "trusted_pool_completion_qa"
        and qa.get("schema_version") == 1
        and qa.get("terminal") is True
        and type(qa["last_epoch"]) is int
        and qa["last_epoch"] == 60
        and qa.get("finite_best_last_weights_and_last_optimizer") is True,
        "Require actual typed finite terminal60 QA",
    )
    require(
        type(cp["epoch"]) is int
        and cp["epoch"] == 60
        and cp["sha256"] == qa["checkpoint_sha256"]["last.pt"]
        and type(cp["steps"]) is int
        and type(qa["steps"]) is int
        and type(qa["best_epoch"]) is int
        and 1 <= qa["best_epoch"] <= 60
        and cp["steps"] == qa["steps"] == 60 * 8192 // cfg["batch_size"],
        "Fixed terminal60 evidence must identify actual last.pt/epoch/full steps",
    )
    require(
        qa["source_commit"] == SOURCE
        and qa["execution_code_sha256"] == EXECUTION
        and qa["checkpoint_training_source_sha256"]
        == evidence["source"]["training_source_sha256"]
        == TRAINING
        and qa["manifest_sha256"] == data["manifest_sha256"]
        and qa["canonical_fingerprint"] == data["prepared_content_sha256"],
        "Terminal source/data binding differs",
    )
    require(
        len(train_groups) == data["eligible_photo_counts"]["train"] == qa["eligible_photos"] == 2529
        and type(qa["minimum_photo_visits"]) is int
        and qa["minimum_photo_visits"] >= 60
        and qa["delivered_samples"] == 60 * 8192,
        "Incomplete actual2529 full-budget photo coverage",
    )
    require(
        len(annotations)
        == qa["eligible_annotation_records"]
        == qa["contributing_annotation_records"]
        == 5700
        and qa["all_eligible_annotations_contributed"] is True
        and type(qa["minimum_annotation_visits"]) is int
        and qa["minimum_annotation_visits"] >= 1
        and qa["eligible_annotation_ids_sha256"] == ids_sha(annotations)
        and qa["eligible_photo_ids_sha256"] == ids_sha(train_groups),
        "Terminal weights must cover all5700 actual manifest annotations",
    )
