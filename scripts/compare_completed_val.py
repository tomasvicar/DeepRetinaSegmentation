"""Metadata-only comparison of enriched completed VAL against a declared original policy.

No image/label/model files are opened. Input SHA and exact manifest pairing remain
mandatory. This is interim exploratory VAL evidence, never held-out evaluation.
"""

import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path

from deepretina.comparison import compare_reports


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unwrap(value):
    if value.get("kind") != "unified_validation_evidence" or value["metrics"]["split"] != "val":
        raise ValueError("Require enriched VAL evidence")
    if (
        value["metrics"]["photos"] != 770
        or value["configuration"]["values"]["purpose"] != "full_data"
    ):
        raise ValueError("Require full-data all770 VAL evidence")
    if value["inference_protocol"]["threshold"] != value["metrics"]["threshold"]:
        raise ValueError("Threshold metadata differs")
    result = copy.deepcopy(value["metrics"])
    result.update(
        manifest_sha256=value["data"]["manifest_sha256"],
        prepared_content_sha256=value["data"]["prepared_content_sha256"],
        checkpoint_sha256=value["checkpoint"]["sha256"],
    )
    return result


def verify_completion(qa, evidence):
    if (
        qa.get("terminal") is not True
        or qa.get("last_epoch") != 60
        or qa.get("finite_best_last_weights_and_last_optimizer") is not True
    ):
        raise ValueError("Require recorded terminal finite completed60 QA")
    if (
        qa["checkpoint_sha256"]["best.pt"] != evidence["checkpoint"]["sha256"]
        or qa["best_epoch"] != evidence["checkpoint"]["epoch"]
    ):
        raise ValueError("Completed run selected checkpoint differs")
    if (
        qa["manifest_sha256"] != evidence["data"]["manifest_sha256"]
        or qa["canonical_fingerprint"] != evidence["data"]["prepared_content_sha256"]
        or qa["checkpoint_training_source_sha256"] != evidence["source"]["training_source_sha256"]
    ):
        raise ValueError("Completion source/data identity differs")
    if (
        qa["eligible_photos"] != 2221
        or qa["minimum_photo_visits"] < 60
        or qa["delivered_samples"] != 60 * 8192
    ):
        raise ValueError("Incomplete full-budget photo coverage")


def compare(reference_path, evidence_path, manifest, *, completion_qa_path, repeats=2000, seed=42):
    input_paths = {
        "reference": reference_path,
        "candidate": evidence_path,
        "manifest": manifest,
        "completion_qa": completion_qa_path,
    }
    before = {key: sha(path) for key, path in input_paths.items()}
    reference = json.loads(Path(reference_path).read_text())
    evidence = json.loads(Path(evidence_path).read_text())
    candidate = unwrap(evidence)
    completion = json.loads(Path(completion_qa_path).read_text())
    verify_completion(completion, evidence)
    component = reference.get("fresh_paired_cup_component", {})
    if (
        reference.get("family") != "optional_fresh_paired_cup_experiment"
        or reference.get("av_preprocessing_profile") != "vesclass_matlab_provided_fov_calibrated"
        or reference.get("single_net_normalized") is not False
        or reference.get("split") != "val"
        or reference.get("photos") != 770
        or reference.get("preview_only") is not False
        or reference.get("disc_reference_variant") != "disc0/raw"
        or component.get("cup_variant") != "disc0"
        or component.get("reference_context") != "disc0"
        or component.get("policy")
        != "Fresh native cup strictlogit>0 vs cup intersect SAMEforward fresh dedicated disc strictlogit>0; no GTused for clipping"
    ):
        raise ValueError("Require completed strongest calibrated disc0/cup-clipped VAL reference")
    if reference["prepared_content_sha256"] != candidate["prepared_content_sha256"]:
        raise ValueError("Canonical data pin differs")
    if candidate["threshold"] not in (0.5, 0.35):
        raise ValueError("Only requested fixed.5 and exploratory.35 comparisons")
    with tempfile.TemporaryDirectory(prefix="retina-count-pair-") as temporary:
        flat = Path(temporary) / "candidate.json"
        flat.write_text(json.dumps(candidate, allow_nan=False))
        result = compare_reports(
            reference_path,
            flat,
            manifest,
            repeats=repeats,
            seed=seed,
            include_sources=True,
            include_av=True,
        )
    result["completion_qa_sha256"] = sha(completion_qa_path)
    result["completed_training_epochs"] = completion["last_epoch"]
    result["selected_checkpoint_epoch"] = evidence["checkpoint"]["epoch"]
    result["candidate_enriched_evidence_sha256"] = sha(evidence_path)
    result["candidate_flat_metrics_role"] = (
        "Metadata-only unwrap of enriched evidence; SHA differs from original enriched producer artifact"
    )
    result["candidate_threshold"] = candidate["threshold"]
    result["comparison_role"] = (
        "main_fixed_threshold_0.5"
        if candidate["threshold"] == 0.5
        else "exploratory_same_VAL_selected_threshold_0.35"
    )
    result["candidate_checkpoint_metadata"] = evidence["checkpoint"]
    result["candidate_configuration_sha256"] = evidence["configuration"]["sha256"]
    result["candidate_evaluation_source_sha256"] = evidence["source"]["evaluation_source_sha256"]
    result["candidate_inference_protocol"] = evidence["inference_protocol"]
    result["reference_policy"] = {
        "disc_variant": reference["disc_reference_variant"],
        "cup_component": component,
        "av_profile": reference["av_preprocessing_profile"],
        "checkpoint_sha256": {
            key: reference[key]
            for key in [
                "checkpoint_sha256",
                "vessel_checkpoint_sha256",
                "av_checkpoint_sha256",
                "disc_checkpoint_sha256",
            ]
        },
    }
    result["prepared_content_sha256"] = candidate["prepared_content_sha256"]
    result["corpus_scope"] = {
        "version": "v1",
        "training_registered_dense_selectors": 17,
        "observed_VAL_manifest_dataset_labels": len(
            {
                row["dataset"]
                for row in json.loads(Path(manifest).read_text())
                if row["split"] == "val"
            }
        ),
        "selected_VAL_head_annotation_sources": len(result["by_dataset"]),
        "source_alias_qualification": "DRIVE/RITE shared photos remain one photo/group; AV-priority vessel/head annotation is attributed to RITE, so DRIVE is a canonical stratum but not an additional selected head source",
        "all_available_annotated_archives_claim": False,
        "qualification": "Unregistered DRIONS-DB and GAMMA Disc_Cup_Mask were subsequently identified; separate corpusv2 audit does not change immutable v1/HPO/VAL770",
    }
    result["script_sha256"] = sha(__file__)
    result["reads"] = (
        "Saved JSON evidence and manifest metadata only; no ground truth, input images, trained weights, forward or TEST"
    )
    result["qualifications"] += [
        "This completed baseline is not a final HPO selection; ongoing study candidates are separate",
        "Original cup clipping was selected using this VAL and trades false positives against true positives; original model exposure is incompletely recoverable",
        "Calibrated original preprocessing and native backend numerical drift differ from unified inference; this is a declared pipeline comparison, not an architecture-only experiment",
        "Source intervals are descriptive without multiple-comparison adjustment; known split groups do not prove all patients independent",
        "Threshold0.35 was selected from six values on this same VAL; its unadjusted intervals do not account for threshold selection",
    ]
    if before != {key: sha(path) for key, path in input_paths.items()}:
        raise ValueError("Input evidence/manifest changed during metadata comparison")
    result["input_artifact_sha256"] = before
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "manifest", "output", "completion-qa"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Preserve prior comparison")
    result = compare(
        a.reference,
        a.candidate,
        a.manifest,
        completion_qa_path=a.completion_qa,
        repeats=a.draws,
        seed=a.seed,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["macro_dataset_mean_dice"]))


if __name__ == "__main__":
    main()
