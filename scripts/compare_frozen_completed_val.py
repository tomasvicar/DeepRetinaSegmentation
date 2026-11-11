"""Strict metadata-only completed VAL comparison for frozen_original_reference.

Both pipelines require their OWN externally pinned review context. This route
never opens GT, weights or TEST reports and does not weaken the older cup gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

from compare_completed_val import unwrap, verify_completion
from compare_reviewed_test import canonical, digest_value, inventory, parse, require, sha, timestamp

from deepretina.comparison import CHANNELS, compare_reports

DATA_KEYS = ("manifest_sha256", "prepared_content_sha256", "eligible_photo_counts")
MODEL_KEYS = {
    "universal": "checkpoint_sha256",
    "vessel": "vessel_checkpoint_sha256",
    "disc": "disc_checkpoint_sha256",
    "av": "av_checkpoint_sha256",
}


def check_context(context, report, raw, *, family, bindings):
    require(
        context.get("schema_version") == 1 and context.get("kind") == family,
        "Unsupported externally reviewed VAL context",
    )
    require(context.get("report_sha256") == sha(raw), "Reviewed report bytes differ")
    require(bool(context.get("rationale", "").strip()), "Missing reviewed context rationale")
    require(
        timestamp(report["evaluated_at_utc"]) <= timestamp(context["reviewed_at_utc"]),
        "Review predates actual VAL evidence",
    )
    require(
        context.get("bindings") == {key: report[key] for key in bindings},
        "Report source/data/protocol differs from own reviewed context",
    )


def check_inventory(report, groups):
    require(
        report["split"] == "val" and report["photos"] == len(groups) == 770,
        "Require complete770 VAL inventory",
    )
    entries = report["per_photo"]
    lookup = {p["photo_id"]: p for p in entries}
    require(
        len(lookup) == len(entries) == 770 and set(lookup) == set(groups), "VAL photo IDs differ"
    )
    for pid, rows in groups.items():
        photo = lookup[pid]
        require(photo["record_ids"] == [r["id"] for r in rows], "VAL annotation IDs differ")
        sources = {}
        tasks = [t for t in ("av", "vessels") if any(r["task"] == t for r in rows)][:1]
        if any(r["task"] == "disc_cup" for r in rows):
            tasks.append("disc_cup")
        for task in tasks:
            row = next(r for r in rows if r["task"] == task)
            names = {
                "vessels": ("vessel",),
                "av": ("vessel", "artery", "vein"),
                "disc_cup": ("disc", "cup"),
            }[task]
            sources.update({name: row["dataset"] for name in names})
        require(
            photo["dataset"] == rows[0]["dataset"] and photo["channel_datasets"] == sources,
            "VAL annotation-source policy differs",
        )
        require(set(photo["channels"]) == set(CHANNELS), "Missing VAL head counts")


def checked(original, evidence, completion, manifest_bytes, *, checkpoint_role="best"):
    require(
        original.get("schema_version") == 1
        and original.get("family") == "frozen_original_reference"
        and original.get("preview_only") is False
        and original.get("split") == "val",
        "Require actual flat frozen original VAL family",
    )
    require(
        original.get("reviewed_policy_sha256") is None,
        "VAL report unexpectedly claims TEST reviewed policy",
    )
    candidate = unwrap(evidence)
    require(evidence.get("schema_version") == 1, "Unsupported enriched VAL schema")
    groups, counts, original_inventory, unified_inventory = inventory(manifest_bytes)
    data = evidence["data"]
    require(
        data["manifest_sha256"] == sha(manifest_bytes)
        and digest_value(data["prepared_content_sha256"])
        and data["eligible_photo_counts"] == counts
        and data["split_inventory_sha256"] == unified_inventory
        and data["fingerprint_schema"] == 2,
        "Candidate canonical inventory differs",
    )
    require(
        all(original[key] == data[key] for key in DATA_KEYS)
        and original["inventory_sha256"] == original_inventory,
        "Original canonical inventory differs",
    )
    require(digest_value(original["photos_sha256"]), "Missing original photos.json byte identity")
    require(
        candidate["records"] == sum(map(len, groups["val"].values())), "VAL record count differs"
    )
    for report in (original, candidate):
        check_inventory(report, groups["val"])
    if checkpoint_role == "best":
        verify_completion(completion, evidence, expected_train_photo_count=counts["train"])
    elif checkpoint_role == "fixed_terminal60":
        from deepretina.completion_roles import verify_terminal60

        require(
            completion["geometry_policy_sha256"] == original["geometry_policy_sha256"]
            and digest_value(completion["geometry_policy_sha256"]),
            "Terminal geometry policy differs",
        )
        verify_terminal60(completion, evidence, groups["train"])
    else:
        raise ValueError("Unsupported checkpoint comparison role")
    require(
        all(
            type(completion[k]) is int
            for k in (
                "last_epoch",
                "best_epoch",
                "eligible_photos",
                "minimum_photo_visits",
                "delivered_samples",
            )
        )
        and all(digest_value(completion["checkpoint_sha256"][k]) for k in ("best.pt", "last.pt")),
        "Missing integer terminal coverage/checkpoint byte identities",
    )
    cfg = evidence["configuration"]
    require(cfg["sha256"] == sha(canonical(cfg["values"])), "Configuration hash differs")
    cfg = cfg["values"]
    require(
        cfg["epochs"] == 60
        and cfg["epoch_samples"] == 8192
        and cfg["coverage_first"] is True
        and cfg["legacy_initialization"] is None
        and cfg["allow_historical_exposure"] is False
        and cfg["max_steps"] is None
        and type(cfg["batch_size"]) is int
        and cfg["batch_size"] > 0,
        "Require independent full-data completed60 configuration",
    )
    cp = evidence["checkpoint"]
    require(
        digest_value(cp["sha256"])
        and type(cp["epoch"]) is int
        and 1 <= cp["epoch"] <= 60
        and cp["steps"] == cp["epoch"] * 8192 // cfg["batch_size"],
        "Selected checkpoint step/epoch differs",
    )
    protocol = evidence["inference_protocol"]
    require(
        protocol["threshold"] == candidate["threshold"] == 0.5
        and protocol["geometry"] == "canonical_prepared"
        and protocol["normalization"] == "RGB/255-0.5"
        and protocol["preprocessing_profile"] == cfg["preprocessing_profile"] == "legacy25_raw"
        and protocol["pixels_per_degree"] == cfg["pixels_per_degree"] == 25
        and protocol["channels"] == list(CHANNELS)
        and protocol["metric_support"] == candidate["metric_support"]
        and protocol["annotation_policy"] == candidate["evaluation_policy"]
        and candidate["family"] == "unified"
        and candidate["purpose"] == "full_data"
        and candidate["supervised_channels"] == [True] * 5,
        "Require fixed .5 canonical unified VAL protocol",
    )
    require(
        digest_value(evidence["source"]["evaluation_source_sha256"])
        and digest_value(evidence["source"]["training_source_sha256"])
        and isinstance(evidence["source"].get("runtime"), dict)
        and bool(evidence["source"]["runtime"])
        and digest_value(original["source"]["runtime"]["evaluation_source_sha256"])
        and original["source"]["runtime"]["runtime"]
        and original["source"]["scripts"]
        and all(digest_value(v) for v in original["source"]["scripts"].values()),
        "Missing actual per-pipeline source/runtime identities",
    )
    policy = original["policy_spec"]
    require(
        policy["name"]
        == "original-raw-vessel-calibrated-provided-fov-av-disc0-universal-cup-in-disc0-v1"
        and policy["single_net"]
        == "raw RGB/255 minus0.5; native checkpoint tiling; strict logits>0"
        and policy["cup"]
        == "universal/raw cup>0 intersect SAME-forward dedicated disc0/raw>0 in canonical frame before any NN restoration"
        and policy["gt"]
        == "UnifiedDataset full_data: one photo once; deterministic first sorted annotation per task; AV vessel GT priority; unknown validity excluded"
        and policy["av"]
        == "vesclass_matlab_provided_fov_calibrated; canonical predicted vessel and provided FOV NN-restored to native frame; strict AV ties→0; original NN restore_classes then canonical prepare_mask"
        and set(policy["model_sha256"]) == set(MODEL_KEYS)
        and all(
            digest_value(policy["model_sha256"][k]) and original[v] == policy["model_sha256"][k]
            for k, v in MODEL_KEYS.items()
        ),
        "Original declared four-model recipe/checkpoint identity differs",
    )
    return candidate


def compare(
    *,
    reference,
    candidate,
    manifest,
    completion_qa,
    reference_context,
    trusted_reference_context_sha256,
    candidate_context,
    trusted_candidate_context_sha256,
    repeats=2000,
    seed=42,
    checkpoint_role="best",
):
    paths = {
        k: Path(v)
        for k, v in {
            "reference": reference,
            "candidate": candidate,
            "manifest": manifest,
            "completion_qa": completion_qa,
            "reference_context": reference_context,
            "candidate_context": candidate_context,
        }.items()
    }
    blobs = {k: p.read_bytes() for k, p in paths.items()}
    require(
        sha(blobs["reference_context"]) == trusted_reference_context_sha256
        and sha(blobs["candidate_context"]) == trusted_candidate_context_sha256,
        "External VAL context SHA differs",
    )
    values = {k: parse(v) for k, v in blobs.items() if k != "manifest"}
    original, evidence = values["reference"], values["candidate"]
    check_context(
        values["reference_context"],
        original,
        blobs["reference"],
        family="completed_val_original_context",
        bindings=(
            "source",
            "policy_spec",
            *DATA_KEYS,
            "inventory_sha256",
            "geometry_policy_sha256",
            "photos_sha256",
        ),
    )
    check_context(
        values["candidate_context"],
        evidence,
        blobs["candidate"],
        family="completed_val_unified_context",
        bindings=("source", "configuration", "inference_protocol", "data", "checkpoint"),
    )
    require(
        values["candidate_context"]["completion_qa_sha256"] == sha(blobs["completion_qa"]),
        "Externally reviewed completion QA differs",
    )
    require(
        checkpoint_role in {"best", "fixed_terminal60"}
        and values["candidate_context"].get("checkpoint_role", "best") == checkpoint_role,
        "External candidate context must bind explicit checkpoint role",
    )
    derived = checked(
        original,
        evidence,
        values["completion_qa"],
        blobs["manifest"],
        checkpoint_role=checkpoint_role,
    )
    with tempfile.TemporaryDirectory(prefix="retina-frozen-val-") as directory:
        root = Path(directory)
        for name, raw in [
            ("reference", blobs["reference"]),
            ("candidate", canonical(derived)),
            ("manifest", blobs["manifest"]),
        ]:
            (root / name).write_bytes(raw)
        result = compare_reports(
            root / "reference",
            root / "candidate",
            root / "manifest",
            repeats=repeats,
            seed=seed,
            include_sources=True,
            include_av=True,
        )
    require(
        all(p.read_bytes() == blobs[k] for k, p in paths.items()),
        "Inputs changed during comparison",
    )
    result.update(
        kind="completed_val_fixed_terminal60_frozen_original_comparison"
        if checkpoint_role == "fixed_terminal60"
        else "completed_val_frozen_original_comparison",
        checkpoint_role=checkpoint_role,
        baseline_only=checkpoint_role == "fixed_terminal60",
        selection_authorized=False,
        original_completion_checkpoint_sha256=copy.deepcopy(
            values["completion_qa"]["checkpoint_sha256"]
        ),
        original_completion_best_epoch=values["completion_qa"]["best_epoch"],
        reference_family=original["family"],
        input_artifact_sha256={k: sha(v) for k, v in blobs.items()},
        reference_policy=copy.deepcopy(original["policy_spec"]),
        reference_source=original["source"],
        candidate_source=evidence["source"],
        candidate_configuration=evidence["configuration"],
        candidate_checkpoint=evidence["checkpoint"],
        candidate_inference_protocol=evidence["inference_protocol"],
        completed_training_epochs=60,
        metadata_only=True,
        test_evaluation_enabled=False,
        pipeline_source_binding="Each pipeline bound to its OWN externally SHA-pinned reviewed context; identities need not match",
        prior_blindness_proven=False,
        reviewer_identity_authenticated=False,
    )
    result["qualifications"] += [
        "VAL pipeline comparison only; no TEST execution or selection authorization.",
        "Historical original training exposure is incompletely recoverable; source photometry retains measured residuals.",
        "This original flat report is not the older composed cup-clipped report family.",
        "External contexts bind declared reviewed bytes; reviewer identity is not authenticated by hashes.",
    ]
    if checkpoint_role == "fixed_terminal60":
        result["qualifications"].append(
            "Fixed terminal60 all-annotation baseline; not the VAL-selected best checkpoint or a final selection/TEST authorization. Original best/last QA identities are preserved."
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in (
        "reference",
        "candidate",
        "manifest",
        "completion-qa",
        "reference-context",
        "candidate-context",
        "output",
    ):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("trusted-reference-context-sha256", "trusted-candidate-context-sha256"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    a = vars(p.parse_args())
    output = a.pop("output")
    a["repeats"] = a.pop("draws")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Preserve existing comparison")
    result = compare(**a)
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
