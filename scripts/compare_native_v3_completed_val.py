"""Compare saved genuine pool72 final60 VAL counts; never run model/GT loading."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from deepretina.comparison import CHANNELS, compare_reports

MANIFEST = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
CANONICAL = "00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52"
SOURCE72 = "72fb321187eca517823083aca80cedace9605570"


def load(path):
    blob = Path(path).read_bytes()
    return json.loads(blob), hashlib.sha256(blob).hexdigest()


def save(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def view(metrics, checkpoint, provenance):
    if metrics["split"] != "val" or len(metrics["per_photo"]) != 770:
        raise ValueError("Only the fixed770 VAL domain is supported")
    # Deliberately omit paths/runtime strings. Preserve every integer and source ID.
    return {
        "kind": "derived_native_saved_VAL_count_view_not_fresh_validation_evidence",
        "split": "val",
        "manifest_sha256": MANIFEST,
        "prepared_content_sha256": CANONICAL,
        "checkpoint_sha256": checkpoint,
        "macro_dataset_mean_dice": metrics["macro_dataset_mean_dice"],
        "channels": metrics["channels"],
        "per_photo": [
            {
                key: row[key]
                for key in (
                    "photo_id",
                    "dataset",
                    "record_ids",
                    "channel_datasets",
                    "channels",
                    "av_conditional_counts",
                )
            }
            for row in metrics["per_photo"]
        ],
        "provenance": provenance,
        "NN_forward_performed": False,
        "TEST_evaluated": False,
    }


def require_native_metrics(metrics):
    if metrics.get("threshold") != 0.5:
        raise ValueError("Native saved evaluation threshold must be0.5")
    expected = {
        "vessel": "annotated_valid_pixels_within_field_of_view",
        "artery": "reference_class_labeled_vessel_pixels",
        "vein": "reference_class_labeled_vessel_pixels",
        "disc": "annotated_valid_pixels_within_field_of_view",
        "cup": "annotated_valid_pixels_within_field_of_view",
    }
    if metrics.get("metric_support") != expected:
        raise ValueError("Native saved metric support differs")


def pool_views(spec, output):
    root = Path(spec["root"])
    receipt, receipt_sha = load(root / "original-receipt.json")
    qa = receipt["checkpoint_qa"]
    if receipt["status"] != "COMPLETE" or receipt["run_result"]["returncode"] != 0:
        raise ValueError("Original genuine COMPLETE trainer receipt required")
    result = {}
    for role, proof_path in spec["proofs"].items():
        proof, proof_sha = load(proof_path)
        if (
            proof["kind"] != "checked_freeze_eligibility"
            or proof["family"] != "pool_72_v3"
            or proof["checkpoint_role"] != role
            or proof["original_qa_sha256"] != receipt_sha
            or proof["source_profile"]["commit"] != SOURCE72
            or proof["canonical_data"]["manifest_sha256"] != MANIFEST
            or proof["canonical_data"]["prepared_content_sha256"] != CANONICAL
            or proof["training_completion"]["completed_epochs"] != 60
        ):
            raise ValueError("Actual typed pool72 final60 proof/pins required")
        coverage = proof["selected_checkpoint_coverage"]
        if (
            coverage["eligible_annotation_records"] != 5700
            or coverage["contributing_annotation_records"] != 5700
        ):
            raise ValueError("Selected role needs all5700 annotations")
        cfg = proof["raw_training_configuration"]
        for key, expected in {
            "tile_size": 512,
            "overlap": 64,
            "pixels_per_degree": 25.0,
            "preprocessing_profile": "legacy25_raw",
            "train_view": "patch",
        }.items():
            if cfg[key] != expected:
                raise ValueError("Different native inference protocol")
        if role == "fixed_terminal60":
            metrics = qa["last_validation_metrics"]
            expected_sha = qa["last_checkpoint_sha256"]
        elif role == "training_best":
            metrics, _ = load(root / "milestone/validation.json")
            expected_sha = qa["best_checkpoint_sha256"]
        else:
            raise ValueError("Unsupported role")
        require_native_metrics(metrics)
        if proof["checkpoint_sha256"] != expected_sha:
            raise ValueError("Selected role SHA mismatch")
        path = output / (spec["name"] + "-" + role + "-counts.json")
        save(
            path,
            view(
                metrics,
                expected_sha,
                {
                    "receipt_sha256": receipt_sha,
                    "typed_eligibility_sha256": proof_sha,
                    "checkpoint_role": role,
                    "checkpoint_epoch": proof["checkpoint_epoch"],
                    "original_index_sha256": proof["input_sha256"]["pool_index"],
                    "training_source_commit": SOURCE72,
                    "native_saved_counts_verified_against_actual_checkpoint_by_typed_gate": True,
                    "fresh_final_runtime_validation": False,
                    "training_configuration": cfg,
                    "native_inference": {
                        "tile_size": 512,
                        "overlap": 64,
                        "threshold": 0.5,
                        "precision": "float32_no_autocast",
                        "preprocessing": "legacy25_raw",
                    },
                },
            ),
        )
        result[role] = path
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference-spec", required=True)
    parser.add_argument("--pool-spec", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if load(args.manifest)[1] != MANIFEST:
        raise ValueError("Wrong fixed manifest")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    references = {}
    for name, spec in load(args.reference_spec)[0].items():
        report, sha = load(spec["path"])
        if sha != spec["sha256"]:
            raise ValueError("Reference input SHA mismatch")
        metrics = report.get("metrics", report)
        data = report.get("data", report)
        if data["manifest_sha256"] != MANIFEST or data["prepared_content_sha256"] != CANONICAL:
            raise ValueError("Reference canonical pin mismatch")
        checkpoint = report.get("checkpoint", {}).get("sha256", report.get("checkpoint_sha256"))
        path = output / (name + "-counts.json")
        save(
            path,
            view(
                metrics,
                checkpoint,
                {
                    "original_report_sha256": sha,
                    "reference_role": spec["role"],
                    "original_policy": report.get("policy_spec"),
                    "original_inference_protocol": report.get("inference_protocol"),
                },
            ),
        )
        references[name] = (spec["role"], path)
    summary = {"scope": "exploratory_saved_native_VAL_counts_not_final_selection", "pairs": {}}
    for spec_path in args.pool_spec:
        spec = load(spec_path)[0]
        for role, candidate in pool_views(spec, output).items():
            for name, (reference_role, reference) in references.items():
                if reference_role not in {"both", role}:
                    continue
                result = compare_reports(
                    reference,
                    candidate,
                    args.manifest,
                    repeats=2000,
                    seed=42,
                    include_sources=True,
                    include_av=True,
                )
                assert result["photos"] == 770 and result["split_groups"] == 612
                ratios = [
                    result["channels"][head]["pooled_dice"]["candidate"]
                    / result["channels"][head]["pooled_dice"]["reference"]
                    for head in CHANNELS
                ]
                av = result["av_accuracy"]["conditional_accuracy"]
                ratios.append(av["candidate"] / av["reference"])
                result["worst6_pooled_ratio_to_this_reference"] = min(ratios)
                name_pair = spec["name"] + "-" + role + "-vs-" + name
                save(output / (name_pair + ".json"), result)
                summary["pairs"][name_pair] = {
                    k: result[k]
                    for k in (
                        "macro_dataset_mean_dice",
                        "channels",
                        "av_accuracy",
                        "worst6_pooled_ratio_to_this_reference",
                    )
                }
    save(output / "summary.json", summary)


if __name__ == "__main__":
    main()
