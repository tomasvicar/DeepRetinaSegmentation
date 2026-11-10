"""Offline paired E8/reference comparison; saved counts only, no inference."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if sys.flags.optimize:
        raise RuntimeError("Run with Python assertions enabled for this fixed audit protocol")
    parser = argparse.ArgumentParser(description=__doc__)
    for key in [
        "reference-directory",
        "candidate-directory",
        "manifest",
        "comparison-module",
        "output",
    ]:
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.comparison_module) == (
        "10cbf98db427be2d2e841b59af7b3fd6777ef1835ec30148a2eca3430a392086"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("saved_counts_comparison", args.comparison_module)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    manifest_sha = sha(args.manifest)
    assert manifest_sha == "529edba7743eb5b9175da429c2124d53f48c419b090a284b5650fd68f1fdcf69"
    expected_references = {
        "disc0": "2ce2a1891fac58999e903bb50b84a3378572d9ebea6ad0e430f55a535f8f0792",
        "disc1": "e8b0c48437b8b3c341228b57cf83e5567d5263b15bb86e4570b3d42fb5d99091",
    }
    results = []
    supports = None
    started = time.perf_counter()
    for name in ["lr0003", "baseline", "strong"]:
        candidate = args.candidate_directory / f"{name}-e008-counts.json"
        actual = json.loads(candidate.read_text())
        assert actual["selection_epoch"] == 8 and len(actual["per_photo"]) == 770
        assert actual["manifest_sha256"] == manifest_sha
        for variant, expected in expected_references.items():
            reference = args.reference_directory / f"original-calibrated-av-{variant}-raw-val.json"
            assert sha(reference) == expected
            raw = json.loads(reference.read_text())
            assert raw["disc_reference_variant"] == variant + "/raw"
            assert (
                raw["disc_checkpoint_sha256"]
                and raw["disc_checkpoint_sha256"] != raw["checkpoint_sha256"]
            )
            assert raw["disc_composition"]["replaced_supported_disc_photos"] == 691
            if supports is None:
                supports = {
                    channel: {
                        "valid_pixels": values["valid_pixels"],
                        "reference_positive_pixels": values["tp"] + values["fn"],
                        "supported_photos": sum(
                            entry["channels"][channel]["valid_pixels"] > 0
                            for entry in raw["per_photo"]
                        ),
                    }
                    for channel, values in raw["channels"].items()
                }
            report = helper.compare_reports(
                reference, candidate, args.manifest, repeats=2000, seed=42
            )
            assert report["photos"] == 770 and report["split_groups"] == 612
            report.update(
                candidate_name=name,
                candidate_epoch=8,
                reference_source_commit="84f899b",
                reference_disc_variant=raw["disc_reference_variant"],
                reference_disc_role=raw["disc_reference_role"],
                reference_is_composite=True,
                reference_component_checkpoints={
                    "vessel_dedicated": raw["vessel_checkpoint_sha256"],
                    "artery_vein": raw["av_checkpoint_sha256"],
                    "disc_dedicated": raw["disc_checkpoint_sha256"],
                    "cup_universal": raw["checkpoint_sha256"],
                },
                reference_av_profile=raw["av_preprocessing_profile"],
                reference_disc_profile=raw["disc_preprocessing_profile"],
                reference_disc_composition=raw["disc_composition"],
                historical_exposure_independence_claim=False,
                patients_proven_independent=False,
                no_new_inference=True,
                no_test_evaluation=True,
                conditional_av_accuracy_point_only={
                    "reference": raw["av_conditional_accuracy"],
                    "candidate": actual["av_conditional_accuracy"],
                    "difference": actual["av_conditional_accuracy"]
                    - raw["av_conditional_accuracy"],
                    "paired_ci": None,
                    "reason": "Per-photo ungated conditional-correct/scored counts absent in saved reference and candidate reports; gated TP/FP/FN cannot recover them.",
                },
            )
            target = args.output / f"{name}-e008-vs-{variant}-raw.json"
            target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            results.append(
                {
                    "candidate": name,
                    "reference": variant + "/raw",
                    "comparison_report": target.name,
                    "comparison_sha256": sha(target),
                    "macro": report["macro_dataset_mean_dice"],
                    "pooled_channels": {
                        channel: value["pooled_dice"]
                        for channel, value in report["channels"].items()
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "candidate": name,
                        "reference": variant,
                        "photos": report["photos"],
                        "split_groups": report["split_groups"],
                        "macro": report["macro_dataset_mean_dice"],
                    }
                ),
                flush=True,
            )
    summary = {
        "performed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol_sha256": sha(__file__),
        "comparison_module_sha256": sha(args.comparison_module),
        "comparison_module_source_commit": "fb11844ccab7f578e5075708dc44cbe27f6ebbc5",
        "manifest_sha256": manifest_sha,
        "reference_commit": "84f899b",
        "bootstrap_draws": 2000,
        "seed": 42,
        "unit": "source-stratified reviewed split_group_id; not proven independent patients",
        "photos": 770,
        "split_groups": 612,
        "common_ground_truth_support": supports,
        "new_model_inference": False,
        "test_evaluation": False,
        "missing_reports": [],
        "missing_metric_counts": [
            "Per-photo ungated conditional AV correct/scored counts; conditional accuracy gets point differences only, no paired CI."
        ],
        "results": results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
