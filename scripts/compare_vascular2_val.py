"""Metadata-only paired VAL analysis for the genuine standalone vascular2 run.

This analysis consumes existing byte-pinned evidence, not model/GT files. It is
neither a completion verifier nor a selection/TEST authorization mechanism.
"""

import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path

from compare_completed_val import unwrap
from compare_frozen_completed_val import check_inventory
from compare_reviewed_test import inventory

from deepretina.comparison import CHANNELS, compare_reports

MANIFEST = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
CANONICAL = "00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52"
ORIGINAL = "182ad346b7bda554ae58e76acbc3f17d99228a30bd19cfa335c89534e9f66836"
CONTROL = "b3b7d7e947d85b1c7d0334c8234cf44cc79c19bd1d1498ba4a101f74c0b157d7"
SUMMARY = "93a16a3afee078d74c49f20b8d2695e25762b4d8c81cceac4e2ef11eff6987e0"
SOURCE = "0b8762e667c997a2dfd129dc65988c1f26f6b2ad"
TRAINING = "85441afac96f47165df55505d06a08742183b9686fb1c608225166810a23a00f"
PROTOCOL = "f4ca4a1aaa7f2a01ad3f6dc818e5a61087fd299cd9ad436941b938bbfd76ec8c"
REPORTS = {
    "fixed_terminal60": "8a4ea1732da9ca4bfb356eca494b42492dc6d9ae92d257279c3250afbed0f0ee",
    "best": "ddc9677f100a61d2d0bc8d7fa0ee3f9212e5142c1329d7f96e9ce6698f6a3fc1",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_candidate(evidence, summary, role):
    """Check actual standalone family/role pins; never convert into older pool QA."""
    require(role in REPORTS, "Unknown checkpoint role")
    require(
        summary["kind"] == "public_standalone_completion_summary"
        and summary["original_qa_kind"] == "standalone_unified_completion_qa"
        and summary["terminal"] is True
        and summary["last_epoch"] == 60
        and summary["finite_best_last_weights_and_last_optimizer"] is True
        and summary["source_commit"] == SOURCE
        and summary["checkpoint_training_source_sha256"] == TRAINING
        and summary["reviewed_protocol_sha256"] == PROTOCOL,
        "Require original typed standalone completion provenance",
    )
    coverage = summary[
        "last_checkpoint_coverage" if role == "fixed_terminal60" else "best_checkpoint_coverage"
    ]
    key = "last.pt" if role == "fixed_terminal60" else "best.pt"
    require(
        evidence["checkpoint"]["sha256"] == summary["checkpoint_sha256"][key]
        and evidence["checkpoint"]["epoch"] == coverage["epoch"]
        and coverage["eligible_photos"] == coverage["seen_photos"] == 2529
        and coverage["eligible_annotation_records"]
        == coverage["contributing_annotation_records"]
        == 5700
        and coverage["all_eligible_annotations_contributed"] is True,
        "Selected checkpoint identity/own annotation coverage differs",
    )
    require(
        evidence["source"]["training_source_sha256"] == TRAINING
        and evidence["configuration"] == summary["configuration"]
        and evidence["data"]["manifest_sha256"] == summary["manifest_sha256"] == MANIFEST
        and evidence["data"]["prepared_content_sha256"]
        == summary["canonical_fingerprint"]
        == CANONICAL,
        "Source/config/canonical identity differs",
    )
    cfg = evidence["configuration"]["values"]
    require(
        cfg["epochs"] == 60
        and cfg["epoch_samples"] == 8192
        and cfg["vascular_sampling_weight"] == 2.0
        and cfg["vessel_loss_weight"] == cfg["vessel_pos_weight"] == cfg["av_weight"] == 1.0
        and evidence["checkpoint"]["steps"] == coverage["epoch"] * 8192 // cfg["batch_size"],
        "Original sampling/budget contract differs",
    )
    protocol = evidence["inference_protocol"]
    require(
        protocol["threshold"] == 0.5
        and protocol["tile_size"] == 512
        and protocol["overlap"] == 64
        and protocol["precision"] == "float32_no_autocast"
        and protocol["cup_containment"] == "cup_and_disc",
        "Frozen inference protocol differs",
    )
    return unwrap(evidence)


def validate_source_aggregates(report):
    """Recompute declared source/head integer headers from attributed photo counts."""
    sums = {
        source: {head: dict.fromkeys(("tp", "fp", "fn", "valid_pixels"), 0) for head in CHANNELS}
        for source in report["by_dataset"]
    }
    for photo in report["per_photo"]:
        for head, source in photo["channel_datasets"].items():
            require(source in sums, "Missing declared source count header")
            for key in sums[source][head]:
                sums[source][head][key] += photo["channels"][head][key]
    for source, heads in sums.items():
        for head, counts in heads.items():
            require(
                all(
                    report["by_dataset"][source][head][key] == value
                    for key, value in counts.items()
                ),
                "Source aggregate counts disagree with per-photo attribution",
            )


def objective(result):
    """V3 worst6ratio: minimum of five POOLED Dice ratios and conditional AV ratio."""
    metrics = {h: result["channels"][h]["pooled_dice"] for h in CHANNELS}
    metrics["conditional_av"] = result["av_accuracy"]["conditional_accuracy"]
    ratios = {k: v["candidate"] / v["reference"] for k, v in metrics.items()}
    return {
        "ratios": ratios,
        "minimum_ratio": min(ratios.values()),
        "bottleneck": min(ratios, key=ratios.get),
        "definition": "min(candidate/reference for five pooled Dice heads plus conditional AV accuracy)",
        "bootstrap_ci_for_minimum_ratio": None,
    }


def compare(manifest, directory, original, control, *, repeats=2000):
    directory = Path(directory)
    paths = {
        "manifest": Path(manifest),
        "original": Path(original),
        "control": Path(control),
        "summary": directory / "standalone-completion-summary.json",
    }
    paths.update({role: directory / (role + "-validation.json") for role in REPORTS})
    blobs = {k: p.read_bytes() for k, p in paths.items()}
    pins = {
        "manifest": MANIFEST,
        "original": ORIGINAL,
        "control": CONTROL,
        "summary": SUMMARY,
        **REPORTS,
    }
    require(all(digest(blobs[k]) == v for k, v in pins.items()), "Input byte identity differs")
    groups, counts, _, _ = inventory(blobs["manifest"])
    require(counts == {"train": 2529, "val": 770, "test": 856}, "Canonical inventory differs")
    originals = json.loads(blobs["original"])
    require(originals["family"] == "frozen_original_reference", "Original family differs")
    summary = json.loads(blobs["summary"])
    views = {"original": originals, "control_terminal60": unwrap(json.loads(blobs["control"]))}
    for role in REPORTS:
        views[role] = checked_candidate(json.loads(blobs[role]), summary, role)
    for value in views.values():
        require(
            value["manifest_sha256"] == MANIFEST and value["prepared_content_sha256"] == CANONICAL,
            "Comparison domain differs",
        )
        check_inventory(value, groups["val"])
        validate_source_aggregates(value)
    output = {}
    with tempfile.TemporaryDirectory(prefix="vascular2-paired-counts-") as temp:
        tmp = Path(temp)
        for name, value in views.items():
            (tmp / name).write_text(json.dumps(value, allow_nan=False))
        for role in REPORTS:
            for reference in ("original", "control_terminal60"):
                value = compare_reports(
                    tmp / reference,
                    tmp / role,
                    paths["manifest"],
                    repeats=repeats,
                    seed=42,
                    include_sources=True,
                    include_av=True,
                )
                require(value["photos"] == 770 and value["split_groups"] == 612, "Grouping differs")
                value.update(
                    kind="standalone_vascular2_paired_val_analysis",
                    checkpoint_role=role,
                    full_input_sha256={k: digest(v) for k, v in blobs.items()},
                    candidate_checkpoint=json.loads(blobs[role])["checkpoint"],
                    candidate_inference_protocol=json.loads(blobs[role])["inference_protocol"],
                    original_standalone_qa_kind=summary["original_qa_kind"],
                    original_standalone_qa_sha256=summary["original_qa_sha256"],
                    source_commit=SOURCE,
                    reviewed_standalone_protocol_sha256=PROTOCOL,
                    selected_checkpoint_coverage=copy.deepcopy(
                        summary[
                            "last_checkpoint_coverage"
                            if role == "fixed_terminal60"
                            else "best_checkpoint_coverage"
                        ]
                    ),
                    analysis_helper_sha256=digest(Path(__file__).read_bytes()),
                    support_and_integer_aggregate_checks="exact per-photo source/record/valid/GT-positive pairing; pooled and source headers reaggregated",
                    metadata_only=True,
                    test_evaluation_performed=False,
                    selection_authorized=False,
                )
                if reference == "original":
                    value["v3_worst6ratio"] = objective(value)
                value["qualifications"].extend(
                    [
                        "Actual standalone provenance, not rewritten as historical 6b/pool QA.",
                        "BESTE59 and terminalLASTE60 are separate roles; comparison does not select either.",
                        "Original models have incompletely recoverable historical training exposure and distinct preprocessing/decoding.",
                        "Control60 used fresh CPU evaluation; vascular2 used CUDA; backend/thread differences are not causally isolated.",
                        "612 known split groups are not proof of complete patient independence; no seed-level uncertainty.",
                        "VAL used for development; source/head intervals are descriptive without multiplicity correction.",
                    ]
                )
                output[role + "-vs-" + reference] = value
    require(
        all(p.read_bytes() == blobs[k] for k, p in paths.items()), "Inputs changed during analysis"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--original", default="reports/frozen-original-val-nearest/validation.json")
    parser.add_argument("--control", default="reports/v2-control60-terminal-val/validation.json")
    args = parser.parse_args()
    results = compare(args.manifest, args.directory, args.original, args.control)
    for name, value in results.items():
        with (Path(args.directory) / (name + ".json")).open("x") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")


if __name__ == "__main__":
    main()
