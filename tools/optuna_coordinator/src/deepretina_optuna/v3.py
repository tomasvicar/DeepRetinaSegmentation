"""Isolated adaptive corpus-v3 contracts; count-only objective, no ML imports."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from . import v2

KIND = "adaptive_v3_unified"
SOURCE = "72fb321187eca517823083aca80cedace9605570"
EXECUTION = "347f78ffd2f4a352c0870faad34327945f7d1c64b865dacb7b7d86e5ca4e0aaa"
TRAINING = "3725ad32a76449b2a3383485cbf14ef050a75f945b4c25e98f4b5b511a332f28"
CACHE = "18c3af6ce64e23c896d7befc52e5ba8e8ebda8e4058c2f156574fa755480766e"
REFERENCE = json.loads(Path(__file__).with_name("v3_reference.json").read_text())
SEARCH = {
    "lr": {"kind": "float", "low": 1e-4, "high": 8e-4, "log": True},
    "weight_decay": {"kind": "float", "low": 1e-6, "high": 1e-3, "log": True},
    "vascular_sampling_weight": {"kind": "float", "low": 1.0, "high": 3.0, "log": True},
    "base_channels": {"kind": "categorical", "choices": [12, 16]},
    "norm": {"kind": "categorical", "choices": ["batch", "group"]},
    "augmentation": {"kind": "categorical", "choices": ["basic", "mild", "strong"]},
}
CONTROL = dict(v2.CONTROL, vascular_sampling_weight=2.0, mixed_precision="bf16")
PLAN = {
    "datasets": [
        "aria",
        "avrdb",
        "biomisa",
        "chasedb1",
        "drhagis",
        "drions",
        "drishti",
        "drive",
        "erlangen",
        "g1020",
        "gamma",
        "hrf",
        "idrid_disc",
        "iostar",
        "onhsd",
        "refuge",
        "riga",
        "rimone",
        "rite",
        "stare",
        "uoadr",
    ],
    "derived_cache": {
        "schema": 1,
        "canonical_fingerprint": "00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52",
        "fingerprint": "18c3af6ce64e23c896d7befc52e5ba8e8ebda8e4058c2f156574fa755480766e",
        "allow_canonical_fallback": False,
    },
    "execution_code_sha256": "347f78ffd2f4a352c0870faad34327945f7d1c64b865dacb7b7d86e5ca4e0aaa",
    "manifest_sha256": "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767",
    "prepared_content_sha256": "00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52",
    "purpose": "full_data_validation_search",
    "schema_version": 1,
    "source_commit": "72fb321187eca517823083aca80cedace9605570",
    "training_photos": 2529,
    "training_source_sha256": "3725ad32a76449b2a3383485cbf14ef050a75f945b4c25e98f4b5b511a332f28",
    "geometry_policy_sha256": "2591d8461b56433529105532c20203e0b17a48c781c1a74fd3a2901bcb301e3d",
    "training_annotation_records": 5700,
    "validation_photos": 770,
    "test_photos": 856,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    ).hexdigest()


def is_adaptive(p):
    return p.get("study_kind") == KIND


def make_protocol(deadline_utc, control_worker_id, profiles):
    p = {"rungs": [3, 9, 27, 60], "direction": "maximize", "plan_template": copy.deepcopy(PLAN)}
    p.update(
        schema_version=3,
        study_kind=KIND,
        study_id="leipzig-adaptive-v3",
        deadline_utc=deadline_utc,
        control_worker_id=control_worker_id,
        max_new_configs=24,
        maximum_meta_gpus=5,
        maximum_own_gpus=0,
        config_template=copy.deepcopy(CONTROL),
        search_space=copy.deepcopy(SEARCH),
        seed_candidate={k: CONTROL[k] for k in SEARCH},
        profiles=copy.deepcopy(profiles),
        objective=copy.deepcopy(REFERENCE),
        sampler={
            "name": "TPESampler",
            "seed": 42,
            "constant_liar": True,
            "multivariate": True,
            "n_startup_trials": 8,
            "n_ei_candidates": 24,
        },
        pruner={
            "name": "SuccessiveHalvingPruner",
            "min_resource": 27,
            "reduction_factor": 3,
            "min_early_stopping_rate": 0,
            "bootstrap_count": 0,
        },
        control_policy="Actual fresh control0 protected60;3/9diagnostic;first native pruning27",
    )
    p["plan_template"]["derived_cache"] = {
        "schema": 1,
        "canonical_fingerprint": v2.CONTENT,
        "fingerprint": CACHE,
        "allow_canonical_fallback": False,
    }
    p["qualifications"] = [
        "Independent fresh BF16 study; no imported historical baseline",
        "Objective uses actual rung LAST six-task ratio; COMPLETE only actual60/full annotations",
        "No TEST; root must freeze deadline/resources after actual staging/benchmark",
    ]
    return p


def validate_protocol(p):
    require(p.get("schema_version") == 3 and p["study_kind"] == KIND, "V3 schema/kind differs")
    require(
        digest(p["search_space"]) == digest(SEARCH)
        and digest(p["config_template"]) == digest(CONTROL),
        "V3 exact search/fixed recipe differs",
    )
    require(
        digest(p["seed_candidate"]) == digest({k: CONTROL[k] for k in SEARCH}),
        "V3 seed control differs",
    )
    require(digest(p["objective"]) == digest(REFERENCE), "V3 six-task reference/formula differs")
    require(
        type(p["max_new_configs"]) is int
        and 1 <= p["max_new_configs"] <= 24
        and "manual_baseline" not in p,
        "V3 budget/baseline differs",
    )
    require(
        p["sampler"]
        == {
            "name": "TPESampler",
            "seed": 42,
            "constant_liar": True,
            "multivariate": True,
            "n_startup_trials": 8,
            "n_ei_candidates": 24,
        },
        "V3 sampler differs",
    )
    require(
        p["pruner"]
        == {
            "name": "SuccessiveHalvingPruner",
            "min_resource": 27,
            "reduction_factor": 3,
            "min_early_stopping_rate": 0,
            "bootstrap_count": 0,
        },
        "V3 delayed pruning differs",
    )
    require(digest(p["plan_template"]) == digest(PLAN), "V3 source/data/policy/cache plan differs")
    require(
        type(p["maximum_meta_gpus"]) is int
        and 0 <= p["maximum_meta_gpus"] <= 5
        and type(p["maximum_own_gpus"]) is int
        and 0 <= p["maximum_own_gpus"] <= 3,
        "V3 GPU caps differ",
    )
    require(
        all(profile["cache"] is True for profile in p["profiles"].values()),
        "V3 requires strict cache per profile",
    )


def _counts(value):
    names = ("tp", "fp", "fn", "valid_pixels")
    require(
        all(type(value[k]) is int and value[k] >= 0 for k in names), "Invalid integer pixel counts"
    )
    require(
        value["tp"] + value["fp"] + value["fn"] <= value["valid_pixels"], "Impossible pixel counts"
    )
    return {k: value[k] for k in names}


def _dice(c):
    denominator = 2 * c["tp"] + c["fp"] + c["fn"]
    return 2 * c["tp"] / denominator if denominator else None


def support_inventory(metrics):
    return sorted(
        [
            {
                "photo_id": p["photo_id"],
                "dataset": p["dataset"],
                "record_ids": sorted(p["record_ids"]),
                "channel_datasets": p["channel_datasets"],
                "channels": {
                    k: {"positive": v["tp"] + v["fn"], "valid_pixels": v["valid_pixels"]}
                    for k, v in p["channels"].items()
                },
                "av_support": {
                    k: p["av_conditional_counts"][k] for k in ("eligible_pixels", "scored_pixels")
                },
            }
            for p in metrics["per_photo"]
        ],
        key=lambda p: p["photo_id"],
    )


def objective_metrics(metrics):
    require(metrics["split"] == "val" and metrics["photos"] == 770, "Actual VAL770 required")
    require(
        metrics["threshold"] == 0.5
        and metrics["family"] == "unified"
        and metrics["purpose"] == "full_data",
        "Fixed float32 VAL recipe required",
    )
    photos = metrics["per_photo"]
    ids = [p["photo_id"] for p in photos]
    require(
        len(ids) == len(set(ids)) == 770
        and digest(sorted(ids)) == REFERENCE["validation_photo_ids_sha256"],
        "VAL photo inventory differs",
    )
    require(
        digest(support_inventory(metrics)) == REFERENCE["validation_support_sha256"],
        "VAL GT/source/record support differs",
    )
    result = {}
    for name in REFERENCE["counts"]:
        pooled = _counts(metrics["channels"][name])
        summed = {key: sum(_counts(p["channels"][name])[key] for p in photos) for key in pooled}
        require(pooled == summed, "Pooled/per-photo count mismatch: " + name)
        value = _dice(pooled)
        require(
            value is not None and value == metrics["channels"][name]["dice"],
            "Derived Dice differs: " + name,
        )
        result[name] = value
    correct = scored = 0
    for p in photos:
        c = p["av_conditional_counts"]
        require(
            all(
                type(c[k]) is int and c[k] >= 0
                for k in ("correct", "scored_pixels", "eligible_pixels", "end_to_end_correct")
            ),
            "Invalid conditional counts",
        )
        require(
            c["end_to_end_correct"] <= c["correct"] <= c["scored_pixels"] <= c["eligible_pixels"],
            "Impossible conditional counts",
        )
        correct += c["correct"]
        scored += c["scored_pixels"]
    require(
        scored > 0 and correct / scored == metrics["av_conditional_accuracy"],
        "Derived conditional AV differs",
    )
    result["av_conditional_accuracy"] = correct / scored
    return result


def objective_score(metrics):
    values = objective_metrics(metrics)
    refs = {k: _dice(c) for k, c in REFERENCE["counts"].items()}
    c = REFERENCE["conditional_av_counts"]
    refs["av_conditional_accuracy"] = c["correct"] / c["scored_pixels"]
    return min(values[k] / refs[k] for k in refs)


def validate_rung(q, task, macro_score, macro_epoch):
    require(q.get("kind") == "v3_rung_evidence", "Typed V3 LAST evidence required")
    require(
        q["last_checkpoint_sha256"] == q["checkpoint_sha256"], "LAST checkpoint binding differs"
    )
    metrics = q["last_validation_metrics"]
    require(
        q["last_validation_metrics_sha256"] == digest(metrics), "LAST metric bytes digest differs"
    )
    require(
        q["macro_best_score"] == macro_score
        and q["macro_best_epoch"] == q["best_epoch"] == macro_epoch,
        "Macro CSV best provenance differs",
    )
    score = objective_score(metrics)
    require(
        type(q["metric"]) is float and math.isfinite(q["metric"]) and q["metric"] == score,
        "LAST six-task objective differs",
    )
    if task["target_epoch"] == 60:
        coverage = q["coverage"]
        annotations = coverage["annotation_visits"]
        photos = coverage["photo_visits"]
        require(
            len(annotations) == 5700
            and all(type(v) is int and v > 0 for v in annotations.values())
            and digest(sorted(annotations)) == REFERENCE["eligible_train_record_ids_sha256"],
            "Exact all5700 positive TRAIN annotations required",
        )
        require(
            len(photos) == 2529
            and all(type(v) is int and v >= 60 for v in photos.values())
            and digest(sorted(photos)) == REFERENCE["train_photo_ids_sha256"]
            and sum(photos.values()) == 491520,
            "Exact all2529 full-budget TRAIN photos required",
        )
        require(
            coverage["eligible_annotation_records"]
            == coverage["contributing_annotation_records"]
            == 5700,
            "Annotation coverage header differs",
        )
    return score
