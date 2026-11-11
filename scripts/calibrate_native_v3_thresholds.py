"""One probability inference per VAL photo, predeclared scalar thresholds; no TEST.

With TTA, one inference comprises the declared2/4/8 neural views, not one view.
Counts are exploratory inference-ablation evidence, never ordinary freeze evidence.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import calibrate_vascular2_thresholds as counters
import numpy as np
import torch

from deepretina import selection as gate
from deepretina import unified as u
from deepretina.comparison import compare_reports
from deepretina.inference_tta import protocol_extension

THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
BEST = "21c1b3c6126494b8534a022b2256ada38687c9194120f7be63bf66879687bdfb"
ORIGINAL = "182ad346b7bda554ae58e76acbc3f17d99228a30bd19cfa335c89534e9f66836"
SOURCE_EVALUATION = "198fdd476505ab38437f8668945d24c39dfb344a5d8fd96a4bb2c9d3a7400d05"
COUNTERS = "7be1227ac7d41247f847c169aaf43a061b0c4ac60363a9346b506ca8495bf147"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(photos, threshold):
    report = counters.aggregate(photos, threshold)
    report.update(
        kind="native_pool72_BEST54_scalar_threshold_VAL_ablation",
        checkpoint_sha256=BEST,
        prepared_content_sha256="00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52",
        ordinary_freeze_evidence=False,
    )
    return report


def threshold_counts(probability, target, valid, rows, trace, photo_id, supported):
    require(
        probability.dtype == np.float32 and np.isfinite(probability).all(),
        "Require finite FP32 inference",
    )
    result = {
        t: counters.photo_counts(probability, target, valid, rows, trace, photo_id, supported, t)
        for t in THRESHOLDS
    }
    av = {
        (
            r["av_conditional_counts"]["correct"],
            r["av_conditional_counts"]["scored_pixels"],
            r["av_conditional_counts"]["eligible_pixels"],
        )
        for r in result.values()
    }
    require(len(av) == 1, "Conditional A/V support/accuracy changed with scalar threshold")
    return result


def native_counter_parity(
    model, dataset, index, sample, probability, config, device, tta, expected
):
    """Native318 counter at0.5, injected SAME probability; zero extra NN calls."""
    rows = dataset.groups[index]

    class OnePhoto:
        def __init__(self):
            self.groups = [rows]
            self.photo_ids = [dataset.photo_ids[index]]

        def sample(self, *args, **kwargs):
            return sample

        def __len__(self):
            return 1

    one = OnePhoto()
    one.rows = rows
    with (
        patch.object(u, "UnifiedDataset", return_value=one),
        patch.object(u, "predict_probabilities", return_value=probability),
    ):
        native = u.evaluate_unified_model(
            model, dataset.manifest, "val", device, config, 0.5, tta=tta
        )
    current = aggregate([expected], 0.5)
    for key in (
        "channels",
        "by_dataset",
        "per_photo",
        "macro_dataset_mean_dice",
        "av_conditional_accuracy",
        "av_accuracy",
    ):
        require(native[key] == current[key], "Same-forward native0.5 counter parity failed:" + key)
    require(
        ({"tta": native["tta"]} if "tta" in native else {}) == protocol_extension(tta),
        "Native TTA descriptor differs",
    )


def worst6(report, original):
    ratios = {
        c: report["channels"][c]["dice"] / original["channels"][c]["dice"] for c in u.CHANNELS
    }
    ratios["conditional_av"] = (
        report["av_conditional_accuracy"] / original["av_conditional_accuracy"]
    )
    return min(ratios.values()), ratios


def run(profile):
    require(profile["execution_authorized"] is True, "False packet; root review required")
    require(profile["tta"] in ("none", "rot4", "dihedral8"), "Root must choose one explicit mode")
    require(
        profile["thresholds"] == list(THRESHOLDS)
        and profile["tile_size"] == 512
        and profile["overlap"] == 64,
        "Predeclared scalar/grid protocol differs",
    )
    require(
        sha(__file__) == profile["science_helper_sha256"] and sha(counters.__file__) == COUNTERS,
        "Science/counter helpers changed",
    )
    require(
        gate.evaluation_source_identity()["evaluation_source_sha256"] == SOURCE_EVALUATION,
        "Use exact frozen318 API source",
    )
    require(
        sha(profile["checkpoint"]) == BEST
        and sha(profile["original"]) == ORIGINAL
        and sha(profile["manifest"]) == counters.MANIFEST,
        "Input bytes differ",
    )
    require(
        torch.get_num_threads() == 2 and torch.get_num_interop_threads() == 1,
        "Actual CPU2/interop1 runtime differs",
    )
    require(profile["science_maximum_seconds"] == 1650, "Bounded science budget differs")
    output = Path(profile["output"])
    require(not output.exists(), "Preserve previous result directory")
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + profile["science_maximum_seconds"]
    model, payload, device = u.load_unified(profile["checkpoint"], profile["device"])
    config = u.inference_config(payload, tile_size=512, overlap=64)
    require(
        payload["epoch"] == 54
        and payload["training_source_sha256"]
        == "3725ad32a76449b2a3383485cbf14ef050a75f945b4c25e98f4b5b511a332f28",
        "Wrong genuine pool selected model",
    )
    data, groups = gate.data_identity(profile["manifest"], payload)
    require(
        len(groups["val"]) == 770
        and len({r["split_group_id"] for rows in groups["val"].values() for r in rows}) == 612,
        "Require770VAL/612knownsplitgroups",
    )
    dataset = u.UnifiedDataset(profile["manifest"], "val", config)
    before = counters.scoped_fingerprint(dataset)
    gathered = {t: [] for t in THRESHOLDS}
    original = json.loads(Path(profile["original"]).read_text())
    for index, rows in enumerate(dataset.groups):
        require(time.monotonic() < deadline, "Science deadline reached; preserve partial journal")
        sample = dataset.sample(index, random_annotator=False, return_trace=True)
        image, target, valid, trace = sample
        probability = u.predict_probabilities(
            model, image, device, config, 512, 64, tta=profile["tta"]
        )
        counts = threshold_counts(
            probability,
            target,
            valid,
            rows,
            trace,
            dataset.photo_ids[index],
            model.supervised_channels,
        )
        native_counter_parity(
            model, dataset, index, sample, probability, config, device, profile["tta"], counts[0.5]
        )
        for threshold, count in counts.items():
            gathered[threshold].append(count)
        with (output / "progress.jsonl").open("a") as f:
            f.write(json.dumps({"photos": index + 1, "counts": counts}, allow_nan=False) + "\n")
        del probability
    require(counters.scoped_fingerprint(dataset) == before, "VAL asset bytes changed")
    require(
        gate.evaluation_source_identity()["evaluation_source_sha256"] == SOURCE_EVALUATION,
        "Source changed during sweep",
    )
    scores = []
    for threshold, photos in gathered.items():
        report = aggregate(photos, threshold)
        score, ratios = worst6(report, original)
        report.update(
            worst6ratio=score,
            ratios=ratios,
            tta_protocol=protocol_extension(profile["tta"]),
            original_training_configuration=payload["config"],
            actual_inference_protocol=gate.inference_protocol(
                payload, threshold=threshold, tta=profile["tta"]
            ),
            canonical_data=data,
        )
        counters.save(output / f"threshold-{threshold:.2f}.json", report)
        scores.append(
            {
                "threshold": threshold,
                "worst6ratio": score,
                "macro": report["macro_dataset_mean_dice"],
                "pooled_dice": {c: report["channels"][c]["dice"] for c in u.CHANNELS},
                "conditional_av": report["av_conditional_accuracy"],
            }
        )
    selected = max(scores, key=lambda x: (x["worst6ratio"], -abs(x["threshold"] - 0.5)))
    for threshold in THRESHOLDS:
        if threshold == 0.5:
            continue
        require(time.monotonic() < deadline, "Deadline before bootstrap; preserve completed counts")
        comparison = compare_reports(
            output / "threshold-0.50.json",
            output / f"threshold-{threshold:.2f}.json",
            profile["manifest"],
            repeats=2000,
            seed=42,
            include_sources=True,
            include_av=True,
        )
        require(comparison["split_groups"] == 612, "Paired bootstrap group inventory differs")
        counters.save(output / f"paired-vs-half-{threshold:.2f}.json", comparison)
    require(time.monotonic() < deadline, "Deadline before original comparison")
    original_pair = compare_reports(
        profile["original"],
        output / f"threshold-{selected['threshold']:.2f}.json",
        profile["manifest"],
        repeats=2000,
        seed=42,
        include_sources=True,
        include_av=True,
    )
    require(original_pair["split_groups"] == 612, "Original pairing groups differ")
    counters.save(output / "selected-vs-original-paired.json", original_pair)
    require(time.monotonic() < deadline, "Deadline before completed summary")
    counters.save(
        output / "summary.json",
        {
            "scores": scores,
            "VAL_exploratory_selected_threshold": selected["threshold"],
            "selection_rule": "max pooled six-ratio minimum vsoriginal; ties closest.5",
            "default_threshold_changed": False,
            "tta": profile["tta"],
            "inference_calls_per_photo": 1,
            "network_views_per_photo": len(u.TTA_OPERATORS[profile["tta"]]),
            "same_forward_native_half_counter_parity_all770": True,
            "conditional_AV_threshold_invariant": True,
            "photos": 770,
            "known_split_groups": 612,
            "bootstrap_draws": 2000,
            "bootstrap_seed": 42,
            "worst6_minimum_ratio_CI": None,
            "interval_scope": "paired pooled/source head Dice and conditionalAV, not nonlinearminimum",
            "seconds": time.monotonic() - started,
            "checkpoint_sha256": BEST,
            "original_sha256": ORIGINAL,
            "VAL_encoded_fingerprint": before,
            "scope": "threshold ablation only; NOT ordinary frozen selection evidence",
            "qualifications": [
                "AdaptiveVAL grid selection/intervals have multiplicity and selection bias",
                "Known612split groups are not proven complete independent patients",
                "Global scalar changes vessel/AVgating ANDdisc/cup; inspect adverse source/head intervals",
                "Historical original exposure/preprocessing/backend qualification remains",
            ],
            "TEST_evaluated": False,
        },
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", required=True)
    p.add_argument("--trusted-profile-sha256", required=True)
    a = p.parse_args(argv)
    require(sha(a.profile) == a.trusted_profile_sha256, "Externally pinned profile differs")
    run(json.loads(Path(a.profile).read_text()))


if __name__ == "__main__":
    main()
