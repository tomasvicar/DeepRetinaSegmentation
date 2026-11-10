"""Replace only original disc counts in a calibrated five-head VAL reference.

No prediction, model import, hierarchy, geometry change or test evaluation.
Preserve every vascular/cup per-photo value and recompute aggregates from counts.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

CHANNELS = ("vessel", "artery", "vein", "disc", "cup")
COUNT_KEYS = ("tp", "fp", "fn", "valid_pixels")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(counts):
    tp, fp, fn, valid = (int(counts[k]) for k in COUNT_KEYS)
    if min(tp, fp, fn, valid) < 0 or tp + fp + fn > valid:
        raise ValueError("Invalid segmentation confusion counts")

    def ratio(a, b):
        return a / b if b else None

    return {
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "valid_pixels": valid,
    }


def aggregate(rows, datasets):
    zero = lambda: {channel: {key: 0 for key in COUNT_KEYS} for channel in CHANNELS}
    totals, grouped = zero(), {name: zero() for name in datasets}
    for row in rows:
        for channel in CHANNELS:
            counts = row["channels"][channel]
            metrics(counts)  # Validate counts even on unsupported heads.
            source = row["channel_datasets"].get(channel)
            if source is None and any(counts[key] for key in COUNT_KEYS):
                raise ValueError("Supported channel has no annotation-source attribution")
            if source is not None:
                grouped.setdefault(source, zero())
            for key in COUNT_KEYS:
                value = int(counts[key])
                totals[channel][key] += value
                if source is not None:
                    grouped[source][channel][key] += value
    pooled = {channel: metrics(counts) for channel, counts in totals.items()}
    by_dataset = {
        name: {channel: metrics(counts) for channel, counts in values.items()}
        for name, values in grouped.items()
    }
    macro = {}
    for channel in CHANNELS:
        scores = [
            values[channel]["dice"]
            for values in by_dataset.values()
            if values[channel]["dice"] is not None
        ]
        macro[channel] = float(np.mean(scores)) if scores else None
    scores = [score for score in macro.values() if score is not None]
    return pooled, by_dataset, macro, float(np.mean(scores)) if scores else None


def compose(baseline, native, variant):
    if baseline["split"] != "val" or native["split"] != "val":
        raise ValueError("Validation-only composition")
    if baseline["preview_only"] or native["preview_only"]:
        raise ValueError("Complete reports required, not previews")
    if baseline["manifest_sha256"] != native["manifest_sha256"]:
        raise ValueError("Manifest identity differs")
    if baseline["prepared_content_sha256"] != native["prepared_content_sha256"]:
        raise ValueError("Prepared content identity differs")
    chosen = native["variants"][variant]
    if chosen["profile"] != "raw" or not chosen["all_weights_and_logits_finite"]:
        raise ValueError("Require finite verified original raw disc variant")
    selected = {row["photo_id"]: row for row in chosen["per_photo"]}
    if len(selected) != len(chosen["per_photo"]):
        raise ValueError("Duplicate native photo IDs")
    base_ids = [row["photo_id"] for row in baseline["per_photo"]]
    if len(set(base_ids)) != len(base_ids) or baseline["photos"] != len(base_ids):
        raise ValueError("Invalid baseline photo inventory")
    supported = {
        row["photo_id"]
        for row in baseline["per_photo"]
        if row["channels"]["disc"]["valid_pixels"] > 0
    }
    if set(selected) != supported:
        raise ValueError("Native disc photo set differs from baseline supported photos")
    old = aggregate(baseline["per_photo"], baseline["by_dataset"])
    if old[0] != baseline["channels"] or old[1] != baseline["by_dataset"]:
        raise ValueError("Baseline source/pooled counts cannot be reproduced")
    result = copy.deepcopy(baseline)
    for row in result["per_photo"]:
        if row["photo_id"] not in selected:
            continue
        candidate = selected[row["photo_id"]]
        original = row["channels"]["disc"]
        counts = candidate["disc"]
        if row["record_ids"] != candidate["record_ids"]:
            raise ValueError("Annotation record inventory differs")
        if row["channel_datasets"]["disc"] != candidate["channel_dataset"]:
            raise ValueError("Disc annotation source differs")
        if original["valid_pixels"] != counts["valid_pixels"]:
            raise ValueError("Disc valid support differs")
        if original["tp"] + original["fn"] != counts["tp"] + counts["fn"]:
            raise ValueError("Disc positive ground-truth support differs")
        row["channels"]["disc"] = metrics(counts)
    pooled, by_dataset, macro, score = aggregate(result["per_photo"], result["by_dataset"])
    for before, after in zip(baseline["per_photo"], result["per_photo"], strict=True):
        for channel in ("vessel", "artery", "vein", "cup"):
            if before["channels"][channel] != after["channels"][channel]:
                raise ValueError("A non-disc per-photo metric was changed")
        for key in ("photo_id", "dataset", "record_ids", "channel_datasets"):
            if before[key] != after[key]:
                raise ValueError("Photo/source/annotation metadata changed")
    result.update(
        channels=pooled,
        by_dataset=by_dataset,
        macro_dataset_channel_dice=macro,
        macro_dataset_mean_dice=score,
        disc_checkpoint_sha256=chosen["checkpoint_sha256"],
        disc_preprocessing_profile="legacy25_raw",
        disc_reference_variant=variant,
        disc_reference_selection_metric="macro_dataset_disc_dice",
        disc_reference_role="primary_macro"
        if variant == "disc0/raw"
        else "pooled_winner_alternative",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--native-disc", type=Path, required=True)
    parser.add_argument("--variant", choices=["disc0/raw", "disc1/raw"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous evidence; choose a fresh output")
    baseline = json.loads(args.baseline.read_text())
    native = json.loads(args.native_disc.read_text())
    result = compose(baseline, native, args.variant)
    result["original_universal_report_sha256"] = baseline.get("baseline_report_sha256")
    result["baseline_report_sha256"] = sha256(args.baseline)
    result["disc_reference_report_sha256"] = sha256(args.native_disc)
    result["disc_reference_helper_sha256"] = native["script_sha256"]
    result["disc_reference_runtime_proof"] = {
        key: native[key]
        for key in (
            "source_commit",
            "script_sha256",
            "seconds",
            "threads",
            "device",
            "python",
            "torch",
            "disc_photos",
            "manifest_sha256",
            "prepared_content_sha256",
            "command",
        )
    }
    result["disc_reference_runtime_proof"].update(
        all_weights_and_logits_finite=native["variants"][args.variant][
            "all_weights_and_logits_finite"
        ],
        architecture=native["variants"][args.variant]["architecture"],
        tiling=native["variants"][args.variant]["tiling"],
    )
    result["disc_composition"] = {
        "script_sha256": sha256(__file__),
        "performed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "replaced_supported_disc_photos": len(native["variants"][args.variant]["per_photo"]),
        "preserved_all_baseline_photo_ids": True,
        "non_disc_per_photo_and_av_accuracy_unchanged": True,
        "original_strictlogit_gt_zero": True,
        "added_hierarchy_or_containment": False,
        "new_predictions": 0,
        "test_evaluation_performed": False,
    }
    result["disc_reference_qualifications"] = [
        "Exposure-qualified original composite reference; no independent improvement claim",
        "Validation-selected original disc variant; primary macro and pooled-winner alternative both reported",
        "Calibrated V/AV and original cup source values preserved exactly; no containment added",
        "Whole-pipeline metrics are recomputed from per-photo counts on unchanged original GT support",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "disc_variant": args.variant,
                "disc": result["channels"]["disc"],
                "macro_dataset_channel_dice": result["macro_dataset_channel_dice"],
                "macro_dataset_mean_dice": result["macro_dataset_mean_dice"],
            }
        )
    )


if __name__ == "__main__":
    main()
