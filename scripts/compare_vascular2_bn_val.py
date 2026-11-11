"""Metadata-only paired research VAL for derived vascular2 BatchNorm buffers."""

import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path

from compare_completed_val import unwrap
from compare_frozen_completed_val import check_inventory
from compare_reviewed_test import inventory

from deepretina.comparison import compare_reports

PARENT = "81113e7e74d86a3d51994754895d18e90b2ee8d54d5a8660a8d5535096451b7b"
PARENT_REPORT = "ddc9677f100a61d2d0bc8d7fa0ee3f9212e5142c1329d7f96e9ce6698f6a3fc1"
ORIGINAL = "182ad346b7bda554ae58e76acbc3f17d99228a30bd19cfa335c89534e9f66836"
MANIFEST = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
CONTENT = "00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, text):
    if not condition:
        raise ValueError(text)


def derived_view(report, calibration):
    require(
        report["kind"] == "vascular2_bn_recalibration_research_validation"
        and report["parent_checkpoint_sha256"] == PARENT
        and report["training_selection_score"] is None
        and report["selection_authorized"] is False
        and report["test_evaluation_enabled"] is False,
        "Require actual derived research evidence, no inherited score or selection receipt",
    )
    proof = report["derivation"]
    require(
        proof["independently_checked_non_bn_bitidentical"] is True
        and proof["parent_gradient_epoch"] == 59
        and proof["completed_parent_run_epochs"] == 60
        and proof["calibration_gradient_updates"] == 0,
        "Wrong derivation proof/parent role",
    )
    require(
        calibration["parent_sha256"] == PARENT
        and calibration["derived_sha256"] == report["checkpoint_sha256"]
        and calibration["recipe"] == report["calibration_recipe"]
        and calibration["delivered_samples"] == 8192
        and len(calibration["photo_visits"]) == 2529
        and sum(calibration["photo_visits"].values()) == 8192
        and min(calibration["photo_visits"].values()) >= 1
        and len(calibration["changed_bn_keys"]) == 54
        and calibration["non_bn_state_bitidentical"] is True
        and calibration["gradient_updates"] == 0,
        "Calibration state/coverage differs",
    )
    require(
        report["data"]["manifest_sha256"] == MANIFEST
        and report["data"]["prepared_content_sha256"] == CONTENT
        and report["inference_protocol"]["threshold"] == 0.5
        and report["inference_protocol"]["tile_size"] == 512
        and report["inference_protocol"]["overlap"] == 64,
        "Research inference/data differs",
    )
    view = copy.deepcopy(report["metrics"])
    view.update(
        manifest_sha256=MANIFEST,
        prepared_content_sha256=CONTENT,
        checkpoint_sha256=report["checkpoint_sha256"],
    )
    return view


def compare(
    report_path,
    calibration_path,
    parent_path,
    original_path,
    manifest_path,
    *,
    trusted_report_sha256,
    trusted_calibration_sha256,
):
    paths = {
        "research_validation": Path(report_path),
        "calibration": Path(calibration_path),
        "parent_validation": Path(parent_path),
        "original_validation": Path(original_path),
        "manifest": Path(manifest_path),
    }
    blobs = {k: p.read_bytes() for k, p in paths.items()}
    pins = {
        "research_validation": trusted_report_sha256,
        "calibration": trusted_calibration_sha256,
        "parent_validation": PARENT_REPORT,
        "original_validation": ORIGINAL,
        "manifest": MANIFEST,
    }
    require(all(sha(blobs[k]) == v for k, v in pins.items()), "Externally supplied bytes differ")
    values = {k: json.loads(v) for k, v in blobs.items() if k != "manifest"}
    report = values["research_validation"]
    require(
        report["calibration_sha256"] == pins["calibration"], "Calibration/report binding differs"
    )
    parent = values["parent_validation"]
    require(parent["checkpoint"]["sha256"] == PARENT, "Wrong original BEST59 parent")
    require(
        {k: v for k, v in report["configuration"].items() if k != "device"}
        == parent["configuration"]["values"],
        "Gradient configuration changed",
    )
    groups, counts, _, _ = inventory(blobs["manifest"])
    require(counts == {"train": 2529, "val": 770, "test": 856}, "Canonical inventory differs")
    views = {
        "parent": unwrap(parent),
        "original": values["original_validation"],
        "derived": derived_view(report, values["calibration"]),
    }
    for view in views.values():
        check_inventory(view, groups["val"])
    result = {}
    with tempfile.TemporaryDirectory(prefix="bn-counts-") as directory:
        root = Path(directory)
        for name, view in views.items():
            (root / name).write_text(json.dumps(view, allow_nan=False))
        for reference in ["parent", "original"]:
            comparison = compare_reports(
                root / reference,
                root / "derived",
                paths["manifest"],
                repeats=2000,
                seed=42,
                include_sources=True,
                include_av=True,
            )
            require(
                comparison["photos"] == 770 and comparison["split_groups"] == 612,
                "VAL grouping differs",
            )
            comparison.update(
                kind="vascular2_bn_paired_research_val",
                measured_input_sha256=pins,
                analysis_helper_sha256=sha(Path(__file__).read_bytes()),
                parent_checkpoint_sha256=PARENT,
                derived_checkpoint_sha256=report["checkpoint_sha256"],
                selection_authorized=False,
                test_evaluation_performed=False,
                original_parent_completion_does_not_certify_derived_weights=True,
                qualifications_additional=[
                    "TRAIN labels inform fixed valid-mask/foreground sampler; only BN buffers update, no gradients.",
                    "GTX1060 derived FP32 versus RTX4070 parent fresh VAL; backend differences not causally isolated.",
                    "Exploratory development VAL, incomplete historical original exposure,612known groups not proven independent patients.",
                    "No seed uncertainty or multiplicity correction; no final model selection or TEST authorization.",
                ],
            )
            if reference == "original":
                six = {
                    h: comparison["channels"][h]["pooled_dice"]
                    for h in ("vessel", "artery", "vein", "disc", "cup")
                }
                six["conditional_av"] = comparison["av_accuracy"]["conditional_accuracy"]
                ratios = {k: z["candidate"] / z["reference"] for k, z in six.items()}
                comparison["v3_worst6ratio"] = {
                    "ratios": ratios,
                    "minimum_ratio": min(ratios.values()),
                    "bottleneck": min(ratios, key=ratios.get),
                    "definition": "minimum of five pooled Dice plus conditional AV candidate/original ratios",
                }
            result[reference] = comparison
    require(
        all(p.read_bytes() == blobs[k] for k, p in paths.items()),
        "Inputs changed during count analysis",
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "report",
        "calibration",
        "parent",
        "original",
        "manifest",
        "output-prefix",
        "trusted-report-sha256",
        "trusted-calibration-sha256",
    ]:
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    result = compare(
        a.report,
        a.calibration,
        a.parent,
        a.original,
        a.manifest,
        trusted_report_sha256=a.trusted_report_sha256,
        trusted_calibration_sha256=a.trusted_calibration_sha256,
    )
    for name, report in result.items():
        with Path(a.output_prefix + "-vs-" + name + ".json").open("x") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")


if __name__ == "__main__":
    main()
