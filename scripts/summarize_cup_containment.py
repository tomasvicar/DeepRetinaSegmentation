"""Summarize completed optional cup containment counts without model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_original_disc_reference import metrics
from evaluate_cup_containment import CANONICAL, KEYS, REF_SHA, exact, sha


def summarize(result, references):
    result, references = Path(result), Path(references)
    report = json.loads((result / "report.json").read_text())
    rows = report["per_photo"]
    if (
        report["mode"] != "full"
        or report["processed"] != 612
        or len(rows) != 612
        or len({row["photo_id"] for row in rows}) != 612
        or report["canonical_fingerprint"] != CANONICAL
        or not report["native_per_photo_counts_exact"]
        or not all(row["native_counts_exact"] for row in rows)
    ):
        raise ValueError("Complete exact-count-gated canonical full VAL required")
    groups = {"pooled": rows}
    for source in sorted({row["source"] for row in rows}):
        groups[source] = [row for row in rows if row["source"] == source]
    changes = {}
    for source, group in groups.items():
        totals = {
            variant: metrics({key: sum(row["cup"][variant][key] for row in group) for key in KEYS})
            for variant in ("native", "disc0", "disc1")
        }
        native = totals["native"]
        variants = {}
        for name in ("disc0", "disc1"):
            clipped = totals[name]
            if (
                clipped["valid_pixels"] != native["valid_pixels"]
                or clipped["tp"] + clipped["fn"] != native["tp"] + native["fn"]
                or clipped["tp"] > native["tp"]
                or clipped["fp"] > native["fp"]
            ):
                raise ValueError("Intersection or fixed GT support invariant failed")
            variants[name] = {
                "counts_and_metrics": clipped,
                "true_positives_removed": native["tp"] - clipped["tp"],
                "false_positives_removed": native["fp"] - clipped["fp"],
                "dice_delta": clipped["dice"] - native["dice"],
            }
        changes[source] = {
            "photos": len(group),
            "gt_positive_pixels": native["tp"] + native["fn"],
            "native": native,
            "variants": variants,
        }
    composites = {}
    for name in ("disc0", "disc1"):
        native_path = references / f"original-calibrated-av-{name}-raw-val.json"
        if sha(native_path) != REF_SHA[name]:
            raise ValueError("Strong original reference hash differs")
        native = json.loads(native_path.read_text())
        clipped_path = result / f"cup-and-{name}-raw-val.json"
        clipped = json.loads(clipped_path.read_text())
        if len(native["per_photo"]) != 770 or len(clipped["per_photo"]) != 770:
            raise ValueError("All 770 reference photos must remain present")
        for before, after in zip(native["per_photo"], clipped["per_photo"], strict=True):
            for key in ("photo_id", "dataset", "record_ids", "channel_datasets"):
                if before[key] != after[key]:
                    raise ValueError("Photo or annotation inventory changed")
            for channel in ("vessel", "artery", "vein", "disc"):
                if before["channels"][channel] != after["channels"][channel]:
                    raise ValueError("Non-cup head changed")
        exact(native["channels"]["cup"], changes["pooled"]["native"])
        exact(clipped["channels"]["cup"], changes["pooled"]["variants"][name]["counts_and_metrics"])
        composites[name] = {
            "native_reference_sha256": sha(native_path),
            "clipped_composite_sha256": sha(clipped_path),
            "native_pooled": native["channels"],
            "clipped_pooled": clipped["channels"],
            "native_source_macro": native["macro_dataset_channel_dice"],
            "clipped_source_macro": clipped["macro_dataset_channel_dice"],
            "native_five_head_macro": native["macro_dataset_mean_dice"],
            "clipped_five_head_macro": clipped["macro_dataset_mean_dice"],
        }
    return {
        "split": "val",
        "canonical_fingerprint": CANONICAL,
        "full_report_sha256": sha(result / "report.json"),
        "all_reference_photos": 770,
        "cup_supported_photos": 612,
        "cup_unsupported_no_forwards": 158,
        "changes": changes,
        "composites": composites,
        "elapsed_seconds": report["elapsed_seconds"],
        "max_rss_kib": report["max_rss_kib"],
        "historical_exposure_caveat_unchanged": True,
        "original_defaults_changed": False,
        "test_evaluation": False,
        "scope": "Optional validation-only predicted-mask postprocessing; no independent quality claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.result, args.reference_dir)
    with args.output.open("x") as output:
        output.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
