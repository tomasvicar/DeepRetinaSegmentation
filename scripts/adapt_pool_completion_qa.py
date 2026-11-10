"""Translate trusted terminal V2 pool evidence to completion QA; metadata only.

Does not validate tensors again: inherits finite-state attestations from the exact
caller-trusted pool receipt. Never opens checkpoints, photographs or target arrays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

SOURCE = "6b3dfad605a256e904a877efcdcc1f3cc6135dd0"
EXECUTION = "cb5c4f98e0181a13b166c128a632faf502de1a3bf770fbee4a5b7fa649e651cd"
TRAINING = "9075246883f938ba80b4fe282b365da10ddb8c834bd3106d846465ac79ce9473"
MANIFEST = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
CONTENT = "00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52"
POLICY = "2591d8461b56433529105532c20203e0b17a48c781c1a74fd3a2901bcb301e3d"
CACHE = "18c3af6ce64e23c896d7befc52e5ba8e8ebda8e4058c2f156574fa755480766e"
SOURCE_PHOTOS = {
    "aria": 87,
    "avrdb": 59,
    "biomisa": 45,
    "chasedb1": 16,
    "drhagis": 23,
    "drions": 110,
    "drishti": 40,
    "drive": 16,
    "erlangen": 63,
    "g1020": 609,
    "gamma": 99,
    "hrf": 27,
    "idrid_disc": 54,
    "iostar": 18,
    "onhsd": 53,
    "refuge": 400,
    "riga": 510,
    "rimone": 95,
    "rite": 16,
    "stare": 12,
    "uoadr": 193,
}
DATASETS = set(SOURCE_PHOTOS)
METADATA = (
    "metrics.csv",
    "coverage.json",
    "run.json",
    "training_summary.json",
    "validation.json",
    "search_trial.json",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def trusted(raw, expected, label):
    require(sha(raw) == expected, "Trusted original bytes differ: " + label)
    return json.loads(raw)


def adapt(
    receipt_bytes,
    task_bytes,
    decision_bytes,
    manifest_bytes=b"",
    *,
    trusted_receipt_sha256,
    trusted_task_sha256,
    trusted_decision_sha256,
    milestone,
):
    receipt = trusted(receipt_bytes, trusted_receipt_sha256, "receipt")
    # Reject partial stages before opening any milestone evidence.
    q = receipt["checkpoint_qa"]
    require(
        receipt["status"] == "COMPLETE"
        and receipt["actual_epoch"] == receipt["target_epoch"] == 60
        and q["status"] == "PASS"
        and q["actual_epoch"] == 60
        and q["completed_budget"] is True,
        "Require actual terminal E60 receipt, never a partial rung",
    )
    task = trusted(task_bytes, trusted_task_sha256, "task")
    decision = trusted(decision_bytes, trusted_decision_sha256, "decision")
    require(
        decision["action"] == "finished"
        and decision["trial_state"] == "COMPLETE"
        and decision["receipt_sha256"] == sha(canonical(receipt)),
        "Require bound completed coordinator decision",
    )
    for key in ("study_id", "trial_number", "task_id", "lease"):
        require(
            receipt[key] == task[key] == decision[key], "Terminal task/decision differs: " + key
        )
    for key in ("worker_id", "profile_sha256", "plan_sha256", "config_sha256", "gpu_uuid"):
        require(receipt[key] == task[key] == q[key], "Receipt identity differs: " + key)
    require(
        task["study_kind"] == "finite_v2_controlled"
        and task["target_epoch"] == 60
        and task["previous_epoch"] == 27
        and task["resume"] is True,
        "Require finite V2 final rung",
    )
    plan = task["plan"]
    config = plan["candidates"][task["trial"]]
    require(
        set(plan["candidates"]) == {task["trial"]}
        and sha(canonical(plan)) == task["plan_sha256"]
        and sha(canonical(config)) == task["config_sha256"],
        "Exact plan/config hash differs",
    )
    pins = {
        "source_commit": SOURCE,
        "execution_code_sha256": EXECUTION,
        "training_source_sha256": TRAINING,
        "manifest_sha256": MANIFEST,
        "prepared_content_sha256": CONTENT,
        "geometry_policy_sha256": POLICY,
    }
    for key, expected in pins.items():
        require(plan[key] == expected, "Frozen V2 identity differs: " + key)
        qkey = "canonical_sha256" if key == "prepared_content_sha256" else key
        require(q[qkey] == expected, "QA identity differs: " + key)
    require(
        task["source_commit"] == SOURCE
        and task["execution_code_sha256"] == EXECUTION
        and task["canonical_sha256"] == CONTENT,
        "Task source/data differs",
    )
    require(
        plan["training_photos"] == 2529
        and plan["training_annotation_records"] == 5700
        and set(plan["datasets"]) == DATASETS
        and len(plan["datasets"]) == 21,
        "Require exact V2 TRAIN support inventory",
    )
    cache = plan["derived_cache"]
    if cache is not None:
        require(
            cache
            == {
                "schema": 1,
                "canonical_fingerprint": CONTENT,
                "fingerprint": CACHE,
                "allow_canonical_fallback": False,
            },
            "V2 cache identity differs",
        )
    require(
        task["cache_sha256"] == q["cache_sha256"] == (CACHE if cache else None),
        "Profile cache differs",
    )
    require(
        config["epochs"] == 60
        and config["epoch_samples"] == 8192
        and config["coverage_first"] is True
        and config["purpose"] == "full_data"
        and config["legacy_initialization"] is None
        and config["allow_historical_exposure"] is False
        and config["max_steps"] is None
        and config["require_all_heads"] is True,
        "Require unchanged full-data budget/lineage",
    )
    steps = 60 * 8192 // config["batch_size"]
    require(
        q["finite_weights"] is True
        and q["finite_optimizer"] is True
        and q["scheduler_T_max"] == q["scheduler_last_epoch"] == 60
        and q["steps"] == steps
        and q["delivered_samples"] == 60 * 8192,
        "Incomplete finite optimizer/scheduler/sample attestation",
    )
    run_result = receipt["run_result"]
    require(
        run_result["returncode"] == 0
        and run_result["target_epoch"] == 60
        and run_result["test_evaluation_performed"] is False,
        "Run did not finish clean E60 without TEST",
    )
    proof = q["independent_milestone"]
    require(
        task["output"] == q["output"] == run_result["output"] == proof["source_run"],
        "Original run path differs",
    )
    for key in ("source_commit", "profile_sha256", "plan_sha256", "gpu_uuid", "worker_id"):
        require(run_result["worker"][key] == task[key], "Original run worker differs: " + key)
    require(
        proof["status"] == "independent_copy_sha_verified_readonly"
        and proof["hardlinks_used"] is False,
        "Require independent immutable milestone proof",
    )
    root = Path(milestone)
    require(
        str(root) == proof["destination"]
        and not root.is_symlink()
        and root.is_dir()
        and not root.stat().st_mode & 0o222,
        "Require original read-only milestone directory",
    )
    require(
        set(proof["sha256"]) == {"best.pt", "last.pt", *METADATA}, "Milestone inventory differs"
    )
    require(
        proof["sha256"]["best.pt"] == q["best_checkpoint_sha256"]
        and proof["sha256"]["last.pt"] == q["checkpoint_sha256"],
        "Milestone checkpoint attestation differs",
    )
    require(
        all(
            isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
            for value in proof["sha256"].values()
        ),
        "Milestone requires full SHA256 identities",
    )
    documents = {}
    for name in (*METADATA, "copy-verification.json"):
        path = root / name
        require(
            not path.is_symlink() and path.is_file() and not path.stat().st_mode & 0o222,
            "Require read-only original metadata: " + name,
        )
        raw = path.read_bytes()
        if name != "copy-verification.json":
            require(sha(raw) == proof["sha256"][name], "Milestone metadata bytes differ: " + name)
        documents[name] = raw
    require(
        json.loads(documents["copy-verification.json"]) == proof, "Independent copy receipt differs"
    )
    require(
        documents["metrics.csv"].decode() == q["csv_text"]
        and sha(documents["metrics.csv"]) == q["csv_sha256"],
        "Actual CSV bytes differ",
    )
    rows = list(csv.DictReader(io.StringIO(q["csv_text"])))
    require(
        len(rows) == 60 and [int(r["epoch"]) for r in rows] == list(range(1, 61)),
        "Require contiguous epochs1..60",
    )
    scores = []
    for epoch, row in enumerate(rows, 1):
        require(all(math.isfinite(float(v)) for v in row.values()), "Nonfinite CSV")
        require(int(row["steps"]) == epoch * 8192 // config["batch_size"], "CSV steps differ")
        score = float(row["val_selection_score"])
        require(
            0 <= score <= 1 and score == float(row["val_macro_dataset_mean_dice"]),
            "CSV objective differs",
        )
        scores.append(score)
    best_score, best_epoch = max(scores), scores.index(max(scores)) + 1
    require(
        q["best_epoch"] == best_epoch
        and q["metric"] == receipt["metric"] == decision["value"] == best_score,
        "Selected best epoch/score differs",
    )
    require(sha(manifest_bytes) == MANIFEST, "Trusted V2 manifest bytes differ")
    manifest_document = json.loads(manifest_bytes)
    manifest_rows = (
        manifest_document["records"] if isinstance(manifest_document, dict) else manifest_document
    )
    train_rows = [
        r
        for r in manifest_rows
        if r["split"] == "train"
        and r["task"] in {"vessels", "av", "disc_cup"}
        and r.get("usable_for_training", True)
        and not r.get("exclude_from_training", False)
    ]
    annotation_ids = {r["id"] for r in train_rows}
    photo_ids = {r["photo_id"] for r in train_rows}
    require(
        len(train_rows) == len(annotation_ids) == 5700
        and len(photo_ids) == 2529
        and all(isinstance(value, str) and value for value in annotation_ids | photo_ids),
        "Trusted manifest eligible TRAIN identity inventory differs",
    )
    source_photos = {
        name: {r["photo_id"] for r in train_rows if r["dataset"] == name} for name in DATASETS
    }
    require(
        {r["dataset"] for r in train_rows} == DATASETS
        and {name: len(ids) for name, ids in source_photos.items()} == SOURCE_PHOTOS,
        "Trusted manifest source-photo inventory differs",
    )
    coverage = json.loads(documents["coverage.json"])
    visits = coverage["photo_visits"]
    require(
        coverage["eligible_photos"]
        == coverage["seen_photos"]
        == len(visits)
        == q["eligible_photos"]
        == q["seen_photos"]
        == 2529
        and coverage["eligible_annotation_records"] == 5700
        and set(visits) == photo_ids,
        "V2 photo/annotation support differs",
    )
    require(
        all(type(v) is int and v >= 60 for v in visits.values())
        and sum(visits.values()) == 60 * 8192
        and min(visits.values()) == q["minimum_photo_visits"],
        "Actual full-budget photo coverage differs",
    )
    require(
        set(coverage["by_dataset"]) == DATASETS
        and all(
            coverage["by_dataset"][name] == {"eligible_photos": count, "seen_photos": count}
            for name, count in SOURCE_PHOTOS.items()
        ),
        "Incomplete actual21-source coverage",
    )
    annotation_visits = coverage["annotation_visits"]
    require(
        len(annotation_visits) == coverage["contributing_annotation_records"] == 5700
        and set(annotation_visits) == annotation_ids
        and all(type(v) is int and v > 0 for v in annotation_visits.values()),
        "Require actual positive contributions from every eligible TRAIN annotation ID",
    )
    run = json.loads(documents["run.json"])
    for key in (
        "manifest_sha256",
        "prepared_content_sha256",
        "training_source_sha256",
        "derived_cache",
    ):
        require(run[key] == plan[key], "Run provenance differs: " + key)
    from deepretina.geometry_policy import validate_geometry_policy

    validate_geometry_policy(run["geometry_policy"])
    require(
        run["config"] == config and run["geometry_policy"]["sha256"] == POLICY,
        "Run config/policy differs",
    )
    summary = json.loads(documents["training_summary.json"])
    require(
        summary["completed_budget"] is True
        and summary["steps"] == steps
        and summary["best_validation_score"] == best_score
        and summary["selection_metric"] == config["selection_metric"],
        "Terminal summary differs",
    )
    validation = json.loads(documents["validation.json"])
    require(
        validation["macro_dataset_mean_dice"] == validation["selection_score"] == best_score,
        "Best VAL metadata differs",
    )
    marker = json.loads(documents["search_trial.json"])
    require(
        marker
        == {
            "plan_sha256": task["plan_sha256"],
            "trial": task["trial"],
            "execution_code_sha256": EXECUTION,
            "manifest_sha256": MANIFEST,
        },
        "Run marker differs",
    )
    return {
        "kind": "trusted_pool_completion_qa",
        "schema_version": 1,
        "terminal": True,
        "last_epoch": 60,
        "finite_best_last_weights_and_last_optimizer": True,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "steps": steps,
        "delivered_samples": 60 * 8192,
        "eligible_photos": 2529,
        "minimum_photo_visits": min(visits.values()),
        "by_dataset": coverage["by_dataset"],
        "eligible_annotation_records": 5700,
        "all_eligible_annotations_contributed": True,
        "minimum_annotation_visits": min(annotation_visits.values()),
        "annotation_visits_total": sum(annotation_visits.values()),
        "eligible_annotation_ids_sha256": sha(canonical(sorted(annotation_ids))),
        "eligible_photo_ids_sha256": sha(canonical(sorted(photo_ids))),
        "contributing_annotation_records": len(annotation_visits),
        "checkpoint_sha256": {
            "best.pt": q["best_checkpoint_sha256"],
            "last.pt": q["checkpoint_sha256"],
        },
        "source_commit": SOURCE,
        "execution_code_sha256": EXECUTION,
        "checkpoint_training_source_sha256": TRAINING,
        "manifest_sha256": MANIFEST,
        "canonical_fingerprint": CONTENT,
        "geometry_policy_sha256": POLICY,
        "derived_cache": cache,
        "plan_sha256": task["plan_sha256"],
        "original_receipt_sha256": trusted_receipt_sha256,
        "original_task_sha256": trusted_task_sha256,
        "original_decision_sha256": trusted_decision_sha256,
        "original_receipt_text": receipt_bytes.decode(),
        "original_task": task,
        "original_decision": decision,
        "independent_milestone": proof,
        "sidecar_sha256": {name: sha(raw) for name, raw in documents.items()},
        "attestation_scope": "Metadata adapter of caller-trusted pool CPU checkpoint QA; finite tensor/optimizer checks inherited, not rerun; checkpoint files and GT never opened",
        "no_gpu_forward": True,
        "no_test_evaluation": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("receipt", "task", "decision", "manifest", "milestone", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("receipt", "task", "decision"):
        parser.add_argument("--trusted-" + key + "-sha256", required=True)
    args = parser.parse_args()
    result = adapt(
        args.receipt.read_bytes(),
        args.task.read_bytes(),
        args.decision.read_bytes(),
        args.manifest.read_bytes(),
        trusted_receipt_sha256=args.trusted_receipt_sha256,
        trusted_task_sha256=args.trusted_task_sha256,
        trusted_decision_sha256=args.trusted_decision_sha256,
        milestone=args.milestone,
    )
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
